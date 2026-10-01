"""Strict, model-free preparation and reporting for the seven retrieval cases."""

import hashlib
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from .common import http_status_counts, read_json, write_json


MODEL_DIR = "qwen3-8b__root_models_Qwen3-Embedding-8B"
EMBEDDING_MODEL = "/root/models/Qwen3-Embedding-8B"
CASES = {
    "hipporag2": None,
    "pathcondrag_original": 0,
    "exp1_correctness": 1,
    "exp2_evidence_candidates": 2,
    "exp3_prefix_coverage": 3,
    "exp4_dependency_binding": 4,
    "exp5_verified_beam": 5,
}
ASSETS = (
    "graph.pickle",
    "chunk_embeddings/vdb_chunk.parquet",
    "entity_embeddings/vdb_entity.parquet",
    "fact_embeddings/vdb_fact.parquet",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_hashes(index):
    return {name: sha256(Path(index) / MODEL_DIR / name) for name in ASSETS}


def cache_hashes(directory):
    directory = Path(directory)
    return {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.name.endswith(".lock")
    }


def passage_text(item):
    return item["title"] + "\n" + (item.get("text") or item.get("paragraph_text", ""))


def gold_docs(sample):
    return set(passage_text(p) for p in sample["paragraphs"] if p.get("is_supporting") is not False)


def validated_dataset(data_path, corpus_path):
    data, corpus = read_json(data_path), read_json(corpus_path)
    require(isinstance(data, list) and bool(data), "dataset must be a non-empty list")
    require(isinstance(corpus, list) and bool(corpus), "corpus must be a non-empty list")
    docs = set(passage_text(p) for p in corpus)
    hops = []
    ids = set()
    for i, sample in enumerate(data):
        require(isinstance(sample.get("question"), str) and bool(sample["question"].strip()),
                f"dataset[{i}] lacks a question")
        decomposition = sample.get("question_decomposition")
        require(isinstance(decomposition, list) and len(decomposition) in (2, 3, 4)
                and all(isinstance(s, dict) and isinstance(s.get("question"), str)
                        and bool(s["question"].strip()) for s in decomposition),
                f"dataset[{i}] lacks valid 2/3/4-hop annotations")
        hop = len(decomposition)
        match = re.match(r"([234])hop\d*(?:__|_)", str(sample.get("id", "")))
        require(match is not None and int(match[1]) == hop, f"dataset[{i}] hop ID mismatch")
        require(sample["id"] not in ids, f"duplicate sample ID: {sample['id']}")
        ids.add(sample["id"])
        gold = gold_docs(sample)
        require(bool(gold) and gold <= docs, f"dataset[{i}] gold passages absent from corpus")
        hops.append(hop)
    return data, docs, hops


def load_indices(path, total):
    indices = read_json(path)
    if isinstance(indices, dict):
        indices = indices.get("selected_indices")
    require(isinstance(indices, list) and bool(indices), "indices must be a non-empty JSON list")
    require(all(type(i) is int and 0 <= i < total for i in indices), "indices are out of range")
    require(len(set(indices)) == len(indices), "indices contain duplicates")
    return indices


def prepare(args):
    out, source = Path(args.out_root).resolve(), Path(args.source_index).resolve()
    require(source != out and source not in out.parents and out not in source.parents,
            "source index and output directory must be separate")
    data, docs, hops = validated_dataset(args.data_path, args.corpus_path)
    require(0 <= args.sample_size <= len(data), "sample_size must be between 0 and dataset size")
    if args.sample_indices_file:
        indices = load_indices(args.sample_indices_file, len(data))
        require(not args.sample_size or len(indices) == args.sample_size, "sample_size/indices mismatch")
    elif not args.sample_size:
        indices = list(range(len(data)))
    elif args.sample_size == 2:
        require(2 in hops and 4 in hops, "two-question smoke requires both 2-hop and 4-hop samples")
        indices = [hops.index(2), hops.index(4)]
    else:
        indices = sorted(random.Random(args.sample_seed).sample(range(len(data)), args.sample_size))
    names = args.cases.split() if args.cases else list(CASES)
    require(bool(names) and len(names) == len(set(names)) and all(n in CASES for n in names),
            "CASES contains unknown or duplicate case names")
    names = [n for n in CASES if n in names]
    identity = read_json(source / MODEL_DIR / "index_manifest.json")
    require(identity.get("embedding", {}).get("model_name") == EMBEDDING_MODEL,
            "source index embedding model is not Qwen3-Embedding-8B")
    require(identity.get("embedding", {}).get("provider") == "transformers", "source embedding provider mismatch")
    require(identity.get("openie", {}).get("identity", {}).get("model_name") == "qwen3-8b",
            "source extraction model mismatch")
    chunk_ids = set(read_json(source / MODEL_DIR / "chunk_metadata.json"))
    expected_ids = {"chunk-" + hashlib.md5(doc.encode()).hexdigest() for doc in docs}
    require(chunk_ids == expected_ids, "source index does not contain exactly the declared corpus")
    hashes = asset_hashes(source)
    require((source / "llm_cache").is_dir(), "Hippo source llm_cache is missing")
    initial_cache = cache_hashes(source / "llm_cache")
    require(bool(initial_cache), "Hippo source llm_cache is empty")
    manifest = {
        "schema_version": 1,
        "source_index": str(source), "model_dir": MODEL_DIR,
        "data_path": str(Path(args.data_path).resolve()),
        "corpus_path": str(Path(args.corpus_path).resolve()),
        "data_sha256": sha256(args.data_path), "corpus_sha256": sha256(args.corpus_path),
        "source_asset_sha256": hashes, "initial_cache_sha256": initial_cache,
        "indexed_docs": len(docs), "dataset_size": len(data),
        "sample_size_requested": args.sample_size, "sample_seed": args.sample_seed,
        "selected_indices": indices, "selected_sample_ids": [data[i]["id"] for i in indices],
        "benchmark_hops": [hops[i] for i in indices],
        "hop_distribution": dict(Counter(str(hops[i]) for i in indices)),
        "dataset_hop_distribution": dict(Counter(map(str, hops))),
        "cases": names,
        "runtime": {"embedding_model_name": EMBEDDING_MODEL, "embedding_batch_size": 4,
                    "llm_name": "qwen3-8b", "llm_prefetch_workers": 8, "openie_max_workers": 8,
                    "max_new_tokens": 2048, "enable_thinking": False,
                    "retrieval_top_k": 200, "result_top_k": 10, "candidate_output_top_k": 200},
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    snapshot = out / "initial_llm_cache"
    if manifest_path.exists():
        require(read_json(manifest_path) == manifest, "existing manifest differs; use a new OUT_ROOT")
        require(cache_hashes(snapshot) == initial_cache, "initial LLM cache snapshot changed")
        require(load_indices(out / "selected_indices.json", len(data)) == indices, "saved indices changed")
    else:
        require(not snapshot.exists(), "partial cache snapshot exists; use a new OUT_ROOT")
        shutil.copytree(source / "llm_cache", snapshot, ignore=shutil.ignore_patterns("*.lock"))
        require(cache_hashes(snapshot) == initial_cache, "source LLM cache changed during snapshot")
        write_json(out / "selected_indices.json", indices)
        write_json(manifest_path, manifest)
    print(f"[prepared] n={len(indices)} hops={manifest['hop_distribution']} cases={names}", flush=True)


def manifest_at(root):
    return read_json(Path(root) / "manifest.json")


def initialize_case(args):
    root, name = Path(args.out_root).resolve(), args.name
    manifest = manifest_at(root)
    require(name in manifest["cases"], f"case not selected: {name}")
    source = Path(manifest["source_index"])
    require(asset_hashes(source) == manifest["source_asset_sha256"], "source graph/embeddings changed")
    case = root / "cases" / name
    require(not (case / "validated.ok").exists(), "validated case must not be overwritten")
    require(not case.exists(), f"incomplete case exists: {case}; inspect it and select a new OUT_ROOT")
    case.mkdir(parents=True)
    index = case / "index"
    shutil.copytree(source, index, ignore=shutil.ignore_patterns("*.lock", "metrics*.json", "eval_results*"))
    shutil.rmtree(index / "llm_cache")
    shutil.copytree(root / "initial_llm_cache", index / "llm_cache")
    require(asset_hashes(index) == manifest["source_asset_sha256"], "copied graph/embeddings differ")
    require(cache_hashes(index / "llm_cache") == manifest["initial_cache_sha256"], "copied LLM cache differs")
    write_json(case / "before.json", {"asset_sha256": asset_hashes(index),
                                      "initial_cache_sha256": cache_hashes(index / "llm_cache")})
    print(f"[initialized] {name}: isolated Hippo index and identical cache snapshot", flush=True)


def ready(args):
    case = Path(args.out_root) / "cases" / args.name
    if not (case / "validated.ok").is_file():
        return 1
    marker = read_json(case / "validated.ok")
    require(marker == {"result_sha256": sha256(case / "result.json"),
                       "report_sha256": sha256(case / "report.json"),
                       "manifest_sha256": sha256(Path(args.out_root) / "manifest.json")},
            f"validated files changed: {args.name}")
    require(asset_hashes(case / "index") == manifest_at(args.out_root)["source_asset_sha256"],
            f"validated index changed: {args.name}")
    print(f"[skip] {args.name}: previously validated", flush=True)
    return 0


def recall(gold, docs):
    return len(gold & set(docs)) / len(gold)


def validate_result(result, manifest, name, data, corpus):
    indices = manifest["selected_indices"]
    require(result.get("selected_indices") == indices, "selected indices mismatch")
    require(result.get("sample_size_effective") == len(indices), "sample count mismatch")
    require(result.get("result_top_k") == 10 and result.get("candidate_output_top_k") == 200,
            "result/candidate export budgets mismatch")
    config = result.get("runtime_config") or {}
    for key in ("embedding_batch_size", "llm_prefetch_workers", "openie_max_workers"):
        require(config.get(key) == manifest["runtime"][key], f"runtime_config.{key} mismatch")
    require(config.get("embedding_model_name") == EMBEDDING_MODEL, "runtime embedding model mismatch")
    require(config.get("llm_name") == "qwen3-8b", "runtime LLM model mismatch")
    require(config.get("max_new_tokens") == 2048, "max_new_tokens must remain 2048")
    if CASES[name] is not None:
        require(config.get("improvement_stage") == CASES[name], "improvement stage mismatch")
        require(result.get("hop_source") == "benchmark", "hop_source must be benchmark")
        require(result.get("hop_distribution") == manifest["hop_distribution"], "hop distribution mismatch")
    rows = result.get("results")
    require(isinstance(rows, list) and len(rows) == len(indices), "result rows mismatch")
    measurements = []
    for position, (row, index) in enumerate(zip(rows, indices)):
        sample, hop = data[index], manifest["benchmark_hops"][position]
        require(row.get("query_index") == index and row.get("sample_id") == sample["id"],
                f"row {position}: sample identity mismatch")
        require(row.get("benchmark_hops") == hop and row.get("question") == sample["question"],
                f"row {position}: question/hop mismatch")
        docs, candidates = row.get("docs"), row.get("candidate_docs")
        require(isinstance(docs, list) and len(docs) == 10 and len(set(docs)) == 10,
                f"row {position}: need 10 unique result documents")
        require(isinstance(candidates, list) and len(candidates) == 200 and len(set(candidates)) == 200,
                f"row {position}: need 200 unique candidate documents")
        require(docs == candidates[:10], f"row {position}: result/candidate prefixes differ")
        require(set(candidates) <= corpus, f"row {position}: document absent from corpus")
        for key, count in (("doc_scores", 10), ("candidate_doc_scores", 200)):
            scores = row.get(key)
            require(isinstance(scores, list) and len(scores) == count
                    and all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores),
                    f"row {position}: invalid {key}")
        gold = gold_docs(sample)
        require(set(row.get("gold_docs") or []) == gold, f"row {position}: exported gold docs mismatch")
        metrics = {f"Recall@{k}": recall(gold, candidates[:k]) for k in (1, 2, 5, 10, 20, 200)}
        published_row = row.get("retrieval_metrics") or {}
        require(all(isinstance(published_row.get(key), (int, float))
                    and abs(published_row[key] - value) <= 0.00011 for key, value in metrics.items()),
                f"row {position}: per-question recall differs")
        ranks = row.get("gold_document_ranks")
        expected_ranks = {doc: candidates.index(doc) + 1 if doc in candidates else None for doc in gold}
        require(isinstance(ranks, list) and len(ranks) == len(gold)
                and all(isinstance(item, dict) and "doc" in item and "rank" in item for item in ranks)
                and {item["doc"]: item["rank"] for item in ranks} == expected_ranks,
                f"row {position}: gold document ranks differ")
        all5, all10 = gold <= set(docs[:5]), gold <= set(docs)
        require(row.get("all_gold_in_top5") is all5 and row.get("all_gold_in_top10") is all10,
                f"row {position}: full-chain flags mismatch")
        require(isinstance(row.get("retrieval_trace"), dict), f"row {position}: missing retrieval trace")
        measurements.append({"query_index": index, "hops": hop,
                             "metrics": metrics, "all_gold_top5": all5, "all_gold_top10": all10})
    metrics = {key: sum(m["metrics"][key] for m in measurements) / len(measurements)
               for key in measurements[0]["metrics"]}
    published = result.get("retrieval_metrics") or {}
    for key, value in metrics.items():
        require(isinstance(published.get(key), (int, float)) and abs(published[key] - value) <= 0.00011,
                f"{key}: published metric differs from macro exact-string recall")
    return measurements, metrics


def report(args):
    root, name = Path(args.out_root), args.name
    manifest = manifest_at(root)
    require(name in manifest["cases"], "case not selected")
    require(sha256(manifest["data_path"]) == manifest["data_sha256"], "dataset changed")
    require(sha256(manifest["corpus_path"]) == manifest["corpus_sha256"], "corpus changed")
    data, corpus, _ = validated_dataset(manifest["data_path"], manifest["corpus_path"])
    case = root / "cases" / name
    result = read_json(case / "result.json")
    measurements, metrics = validate_result(result, manifest, name, data, corpus)
    before = read_json(case / "before.json")
    after = asset_hashes(case / "index")
    require(before["asset_sha256"] == after == manifest["source_asset_sha256"], "graph/embeddings changed")
    require(asset_hashes(manifest["source_index"]) == manifest["source_asset_sha256"], "source index changed")
    require(before["initial_cache_sha256"] == manifest["initial_cache_sha256"], "initial cache mismatch")
    require(args.log_start >= 0 and args.log_end >= args.log_start, "vLLM log rotated/truncated")
    status = http_status_counts(args.vllm_log, args.log_start, args.log_end)
    require(not any(c != "200" for c in status), f"non-200 HTTP responses: {status}")
    stats = result.get("llm_request_stats")
    require(isinstance(stats, dict) and "failures" in stats, "LLM request failure accounting is missing")
    require(stats.get("failures") == 0, f"LLM request failures: {stats}")
    require(stats.get("max_in_flight") == 8, "HTTP in-flight ceiling must be 8")
    require(not stats.get("http_attempts") or bool(status), "HTTP attempts have no vLLM status accounting")
    stratified = {}
    for hop in (2, 3, 4):
        subset = [m for m in measurements if m["hops"] == hop]
        if subset:
            stratified[str(hop)] = {
                "n_samples": len(subset),
                "retrieval_metrics": {key: sum(m["metrics"][key] for m in subset) / len(subset) for key in metrics},
                "all_gold_top5": sum(m["all_gold_top5"] for m in subset) / len(subset),
                "all_gold_top10": sum(m["all_gold_top10"] for m in subset) / len(subset),
            }
    output = {
        "name": name, "improvement_stage": CASES[name], "validated": True,
        "seconds": args.elapsed, "retrieval_seconds": result.get("retrieval_seconds"),
        "n_samples": len(measurements), "retrieval_metrics": metrics,
        "all_gold_top5": sum(m["all_gold_top5"] for m in measurements) / len(measurements),
        "all_gold_top10": sum(m["all_gold_top10"] for m in measurements) / len(measurements),
        "hop_distribution": manifest["hop_distribution"], "stratified": stratified,
        "llm_request_stats": stats, "http_status_in_log": status,
        "vllm_log_slice": {"path": args.vllm_log, "start": args.log_start, "end": args.log_end},
        "asset_sha256_before": before["asset_sha256"], "asset_sha256_after": after,
        "per_question": measurements,
    }
    write_json(case / "report.json", output)
    write_json(case / "validated.ok", {"result_sha256": sha256(case / "result.json"),
                                      "report_sha256": sha256(case / "report.json"),
                                      "manifest_sha256": sha256(root / "manifest.json")})
    print(f"[validated] {name} n={len(measurements)} sec={args.elapsed} R@5={metrics['Recall@5']:.4f}", flush=True)


def summary(args):
    root, manifest = Path(args.out_root), manifest_at(args.out_root)
    reports = {}
    for name in manifest["cases"]:
        args.name = name
        require(ready(args) == 0, f"case is incomplete: {name}")
        reports[name] = read_json(root / "cases" / name / "report.json")
    compact = {
        name: {key: value for key, value in report.items() if key != "per_question"}
        for name, report in reports.items()
    }
    write_json(root / "comparison.json", {"manifest": str(root / "manifest.json"), "cases": compact})
    table = ["| Case | R@1 | R@2 | R@5 | R@10 | All gold@5 | Seconds | HTTP attempts |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in reports.items():
        values = [f"{item['retrieval_metrics'][f'Recall@{k}']:.4f}" for k in (1, 2, 5, 10)]
        table.append(f"| {name} | " + " | ".join(values)
                     + f" | {item['all_gold_top5']:.4f} | {item['seconds']} | {item['llm_request_stats'].get('http_attempts', 0)} |")
    (root / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    write_json(root / "completed.ok", {"cases": list(reports), "manifest_sha256": sha256(root / "manifest.json")})
    print("\n".join(table), flush=True)


def register_commands(subparsers):
    parser = subparsers.add_parser("improvement-prepare", help="validate corpus/index and freeze sample/cache identity")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-indices-file")
    parser.add_argument("--cases", default="")
    parser.set_defaults(handler=prepare)
    for command, handler in (("improvement-initialize-case", initialize_case), ("improvement-ready", ready)):
        parser = subparsers.add_parser(command)
        parser.add_argument("--out-root", required=True)
        parser.add_argument("--name", choices=list(CASES), required=True)
        parser.set_defaults(handler=handler)
    parser = subparsers.add_parser("improvement-report", help="strict post-run integrity and retrieval validation")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--name", choices=list(CASES), required=True)
    parser.add_argument("--elapsed", type=int, required=True)
    parser.add_argument("--vllm-log", required=True)
    parser.add_argument("--log-start", type=int, required=True)
    parser.add_argument("--log-end", type=int, required=True)
    parser.set_defaults(handler=report)
    parser = subparsers.add_parser("improvement-summary", help="summarize only validated completed cases")
    parser.add_argument("--out-root", required=True)
    parser.set_defaults(handler=summary)
