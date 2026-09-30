#!/usr/bin/env python3
"""Run one cold-cache OpenIE throughput case on a fixed MuSiQue corpus sample.

Set PATHCONDRAG_LLM_MAX_IN_FLIGHT before starting Python. Use the same corpus,
sample size, and seed in separate output directories to compare limits:

  PATHCONDRAG_LLM_MAX_IN_FLIGHT=4 conda run -n rag python \
    scripts/benchmark_openie_concurrency.py --output-dir outputs/openie_limit4
  PATHCONDRAG_LLM_MAX_IN_FLIGHT=8 conda run -n rag python \
    scripts/benchmark_openie_concurrency.py --output-dir outputs/openie_limit8

This exercises only NER and triple extraction. It does not build embeddings or
an index, and it never starts a second benchmark case automatically.
Run one case at a time against the local vLLM server so other clients do not
confound its throughput or memory use.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathcondrag.information_extraction.openie_openai import OpenIE  # noqa: E402
from pathcondrag.llm.openai_gpt import CacheOpenAI, LLM_MAX_IN_FLIGHT  # noqa: E402
from pathcondrag.utils.config_utils import BaseConfig  # noqa: E402
from pathcondrag.utils.misc_utils import compute_mdhash_id  # noqa: E402


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_corpus(corpus: list[dict[str, Any]], sample_size: int, seed: int):
    """Return corpus-order documents from one seed-controlled sample."""
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("corpus must be a nonempty list")
    if not 1 <= sample_size <= len(corpus):
        raise ValueError(f"sample size must be between 1 and {len(corpus)}")

    indices = sorted(random.Random(seed).sample(range(len(corpus)), sample_size))
    chunks = {}
    selected = []
    for corpus_index in indices:
        row = corpus[corpus_index]
        if not isinstance(row, dict):
            raise ValueError(f"corpus row {corpus_index} must be an object")
        title = str(row.get("title", ""))
        passage = title + "\n" + str(row.get("text") or row.get("paragraph_text") or "")
        chunk_id = compute_mdhash_id(passage, prefix="chunk-")
        if chunk_id in chunks:
            raise ValueError("sampled duplicate passages; choose another seed or sample size")
        chunks[chunk_id] = {"content": passage}
        selected.append({
            "corpus_index": corpus_index,
            "chunk_id": chunk_id,
            "title": title,
            "passage_sha256": _sha256_bytes(passage.encode("utf-8")),
        })
    return chunks, selected


class TimedOpenIE(OpenIE):
    """Record per-chunk time/results without changing production prompts or flow."""

    def __init__(self, llm_model: CacheOpenAI, max_workers: int):
        super().__init__(llm_model=llm_model, max_workers=max_workers)
        self._result_lock = threading.Lock()
        self.ner_results = {}
        self.triple_results = {}
        self.ner_seconds = {}
        self.triple_seconds = {}

    def ner(self, chunk_key: str, passage: str):
        start = time.perf_counter()
        try:
            result = super().ner(chunk_key, passage)
            with self._result_lock:
                self.ner_results[chunk_key] = result
            return result
        finally:
            with self._result_lock:
                self.ner_seconds[chunk_key] = time.perf_counter() - start

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: list[str]):
        start = time.perf_counter()
        try:
            result = super().triple_extraction(chunk_key, passage, named_entities)
            with self._result_lock:
                self.triple_results[chunk_key] = result
            return result
        finally:
            with self._result_lock:
                self.triple_seconds[chunk_key] = time.perf_counter() - start


def collect_rows(selected: list[dict[str, Any]], extractor: TimedOpenIE):
    """Make output independent of completion order, including partial failures."""
    rows = []
    for item in selected:
        chunk_id = item["chunk_id"]
        ner = extractor.ner_results.get(chunk_id)
        triple = extractor.triple_results.get(chunk_id)
        rows.append({
            **item,
            "ner": asdict(ner) if ner is not None else None,
            "triple": asdict(triple) if triple is not None else None,
            "ner_seconds": extractor.ner_seconds.get(chunk_id),
            "triple_seconds": extractor.triple_seconds.get(chunk_id),
        })
    return rows


def check_run(rows: list[dict[str, Any]], request_stats: dict[str, int]):
    """Report every observable parsing/request problem; fail on cold-cache errors."""
    problems = []
    parse_retries = 0
    for row in rows:
        chunk_id = row["chunk_id"]
        for stage in ("ner", "triple"):
            result = row[stage]
            if result is None:
                problems.append(f"{chunk_id}: missing {stage} result")
            elif result["metadata"].get("error"):
                problems.append(f"{chunk_id}: {stage}: {result['metadata']['error']}")
        ner = row["ner"]
        if ner is not None and ner["metadata"].get("ner_max_tokens_used") == 1024:
            parse_retries += 1

    if request_stats.get("cache_hits", 0):
        problems.append(f"cold cache had {request_stats['cache_hits']} hit(s)")
    if request_stats.get("retries", 0):
        problems.append(f"LLM had {request_stats['retries']} transient HTTP/network retry(ies)")
    if request_stats.get("failures", 0):
        problems.append(f"LLM had {request_stats['failures']} terminal request failure(s)")
    return problems, parse_retries


def check_llm_endpoint(base_url: str, model_name: str) -> None:
    """Fail before scheduling OpenIE if the local model endpoint is unavailable."""
    response = httpx.get(base_url.rstrip("/") + "/models", timeout=10)
    response.raise_for_status()
    data = response.json().get("data", [])
    served = {str(item.get("id")) for item in data if isinstance(item, dict)}
    if model_name not in served:
        raise RuntimeError(f"model {model_name!r} is not served by {base_url}: {sorted(served)}")


def run(args: argparse.Namespace) -> int:
    if "qwen3" not in args.llm_name.lower():
        raise ValueError("this benchmark requires a Qwen3 served model with thinking disabled")
    if args.openie_workers < 1:
        raise ValueError("openie_workers must be positive")
    raw_limit = os.environ.get("PATHCONDRAG_LLM_MAX_IN_FLIGHT", "")
    if not raw_limit.isdecimal() or int(raw_limit) != LLM_MAX_IN_FLIGHT:
        raise ValueError("set a positive PATHCONDRAG_LLM_MAX_IN_FLIGHT before starting Python")
    if not 1 <= LLM_MAX_IN_FLIGHT <= 8:
        raise ValueError("benchmark LLM in-flight limit must be between 1 and 8")
    raw_http_connections = os.environ.get("HIPPO_HTTP_MAX_CONNECTIONS", "500")
    if not raw_http_connections.isdecimal() or int(raw_http_connections) < LLM_MAX_IN_FLIGHT:
        raise ValueError("HIPPO_HTTP_MAX_CONNECTIONS must be at least the LLM in-flight limit")
    expected_env = {
        "HIPPO_OPENIE_MAX_WORKERS": args.openie_workers,
        "HIPPO_OPENIE_NER_WORKERS": args.openie_workers,
        "HIPPO_OPENIE_TRIPLE_WORKERS": args.openie_workers,
        "HIPPO_OPENIE_NER_MAX_TOKENS": 512,
        "HIPPO_OPENIE_TRIPLE_MAX_TOKENS": 2048,
    }
    for key, expected in expected_env.items():
        raw = os.environ.get(key)
        if raw is not None and raw != str(expected):
            raise ValueError(f"{key}={raw!r} would change the fixed benchmark setting {expected}")

    corpus_path = Path(args.corpus_path).resolve()
    corpus_bytes = corpus_path.read_bytes()
    corpus = json.loads(corpus_bytes)
    chunks, selected = select_corpus(corpus, args.sample_size, args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    config = BaseConfig(
        save_dir=str(output_dir),
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        temperature=0.0,
        max_new_tokens=2048,
        seed=None,
    )
    llm = CacheOpenAI.from_experiment_config(config)
    thinking = llm.llm_config.generate_params["extra_body"]["chat_template_kwargs"]["enable_thinking"]
    if thinking is not False:
        raise RuntimeError("Qwen3 thinking mode must be disabled")
    extractor = TimedOpenIE(llm, max_workers=args.openie_workers)

    print(
        f"[openie] documents={len(chunks)} seed={args.seed} "
        f"workers={args.openie_workers} max_in_flight={LLM_MAX_IN_FLIGHT} "
        f"output={output_dir}",
        flush=True,
    )
    start = time.perf_counter()
    exception = None
    try:
        extractor.batch_openie(chunks)
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start

    rows = collect_rows(selected, extractor)
    request_stats = llm.get_request_stats()
    problems, parse_retries = check_run(rows, request_stats)
    if exception is not None:
        problems.append(exception)
    if request_stats["http_attempts"] == 0:
        problems.append("no HTTP requests; cold-cache benchmark did not exercise vLLM")

    parsed_payload = [
        {"chunk_id": row["chunk_id"],
         "entities": row["ner"]["unique_entities"] if row["ner"] else None,
         "triples": row["triple"]["triples"] if row["triple"] else None}
        for row in rows
    ]
    parsed_sha256 = _sha256_bytes(
        json.dumps(parsed_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    code_paths = (
        Path(__file__),
        ROOT / "src/pathcondrag/llm/openai_gpt.py",
        ROOT / "src/pathcondrag/information_extraction/openie_openai.py",
        ROOT / "src/pathcondrag/prompts/templates/ner.py",
        ROOT / "src/pathcondrag/prompts/templates/triple_extraction.py",
    )
    report = {
        "corpus_path": str(corpus_path),
        "corpus_sha256": _sha256_bytes(corpus_bytes),
        "code_sha256": {
            str(path.relative_to(ROOT)): _sha256_bytes(path.read_bytes()) for path in code_paths
        },
        "sample_indices": [item["corpus_index"] for item in selected],
        "sample_size": len(selected),
        "seed": args.seed,
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "temperature": 0.0,
        "seed_for_generation": None,
        "thinking_enabled": False,
        "max_new_tokens": 2048,
        "ner_initial_max_tokens": 512,
        "ner_parse_retry_max_tokens": 1024,
        "triple_max_tokens": 2048,
        "openie_workers": args.openie_workers,
        "llm_max_in_flight": LLM_MAX_IN_FLIGHT,
        "http_max_connections": int(raw_http_connections),
        "elapsed_seconds": elapsed,
        "documents_per_second": len(rows) / elapsed if elapsed > 0 else None,
        "request_stats": request_stats,
        "ner_parse_retries_to_1024": parse_retries,
        "parsed_output_sha256": parsed_sha256,
        "problems": problems,
        "complete": not problems,
    }
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if not problems else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", default="/root/datasets/musique_corpus.json")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8035/v1")
    parser.add_argument("--openie-workers", type=int, default=8)
    parser.add_argument("--output-dir", required=True,
                        help="New directory for the cold-cache results and SQLite cache")
    args = parser.parse_args()
    check_llm_endpoint(args.llm_base_url, args.llm_name)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
