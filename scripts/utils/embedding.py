"""Data preparation and reports for the paired embedding experiments."""

import gc
import json
import logging
import os
import random
import sys
from pathlib import Path

from .common import read_json, write_json


def release_memory(args):
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[gpu] cleared")


def prepare_smoke_data(args):
    source = Path(args.datasets_dir)
    output = Path(args.output_dir)
    samples = read_json(source / "musique.json")
    rng = random.Random(args.sample_seed)
    indices = sorted(rng.sample(range(len(samples)), k=min(args.sample_size, len(samples))))
    selected = [samples[index] for index in indices]
    docs = {}
    for sample in selected:
        for paragraph in sample.get("paragraphs") or []:
            title = paragraph.get("title") or ""
            text = paragraph.get("text") or paragraph.get("paragraph_text") or ""
            docs[title + "\n" + text] = {"title": title, "text": text}
    corpus = list(docs.values())
    write_json(output / "musique.json", selected)
    write_json(output / "musique_corpus.json", corpus)
    (output / "selected_indices.json").write_text(json.dumps(indices), encoding="utf-8")
    print(f"[smoke-data] samples={len(selected)} corpus={len(corpus)} idxs={indices}")


def print_metrics(args):
    result = read_json(args.result)
    print(f"[{args.method}]", result.get("retrieval_metrics"), result.get("qa_metrics"))
    if args.method == "hippo":
        print("[hippo] emb", result.get("embedding_name"), "n", result.get("n_samples"),
              "docs", result.get("n_docs"))


def pair_summary(args):
    output = Path(args.output_dir)
    hippo = read_json(output / "hipporag2_musique" / "metrics.json")
    path = read_json(output / "pathcondrag" / "result.json")
    summary = {
        "mode": args.mode,
        "tag": args.tag,
        "embedding": args.embedding,
        "llm": args.llm,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_max_seq_len": 2048,
        "hipporag2": {
            "retrieval": hippo.get("retrieval_metrics"),
            "qa": hippo.get("qa_metrics"),
            "n_docs": hippo.get("n_docs"),
            "n_samples": hippo.get("n_samples"),
        },
        "pathcondrag_pc3": {
            "retrieval": path.get("retrieval_metrics"),
            "qa": path.get("qa_metrics"),
        },
    }
    write_json(output / "pair_summary.json", summary)
    print("[summary]", json.dumps(summary, indent=2))


def evaluate_hippo_instruction_fix(args):
    """Run Hippo retrieval against the seeded index, without rebuilding it."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("HIPPORAG_KNN_DEVICE", "cpu")
    hippo_root = Path(args.hippo_root)
    sys.path.insert(0, str(hippo_root / "src"))
    sys.path.insert(0, str(hippo_root))

    from main import get_gold_docs
    from hipporag.HippoRAG import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    logging.basicConfig(level=logging.INFO)
    dataset_dir = Path(args.datasets_dir)
    samples = read_json(dataset_dir / "musique.json")
    corpus = read_json(dataset_dir / "musique_corpus.json")
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_docs = get_gold_docs(samples, "musique")
    config = BaseConfig(
        save_dir=args.save_dir,
        dataset="musique",
        llm_name=args.llm,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding,
        embedding_provider="transformers",
        embedding_batch_size=args.embedding_batch_size,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=5,
        max_new_tokens=2048,
        temperature=0.0,
        openie_mode="online",
        synonymy_edge_topk=50,
        synonymy_edge_query_batch_size=128,
        synonymy_edge_key_batch_size=1024,
        rerank_dspy_file_path=str(
            hippo_root / "src" / "hipporag" / "prompts" / "dspy_prompts"
            / "filter_llama3.3-70B-Instruct.json"
        ),
    )
    with HippoRAG(global_config=config) as rag:
        rag.index(docs)
        _, retrieval_metrics = rag.retrieve(queries=queries, gold_docs=gold_docs)
    metrics = {
        "dataset": "musique",
        "method": "hipporag2",
        "eval_mode": "retrieve",
        "note": "instrfix: Qwen3 uses Hippo task prompts via ST prompt=",
        "n_samples": len(samples),
        "n_docs": len(docs),
        "llm_name": args.llm,
        "embedding_name": args.embedding,
        "retrieval_metrics": retrieval_metrics or {},
    }
    output = Path(args.save_dir) / "metrics_retrieve.json"
    write_json(output, metrics)
    print("[hippo]", json.dumps(retrieval_metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {output}")


def instruction_fix_summary(args):
    output = Path(args.output_dir)
    hippo = read_json(output / "hipporag2_musique" / "metrics_retrieve.json")
    path = read_json(output / "pathcondrag" / "result.json")
    old_path = Path(args.source_hippo).parent / "pair_summary.json"
    old = read_json(old_path) if old_path.is_file() else {}
    summary = {
        "note": "instrfix retrieve-only; index reused from emb_ablation_qwen3emb8b_full",
        "embedding": args.embedding,
        "llm": args.llm,
        "hipporag2_retrieve": hippo.get("retrieval_metrics"),
        "pathcondrag_pc3_retrieve": path.get("retrieval_metrics"),
        "baseline_before_fix": {
            "hipporag2": (old.get("hipporag2") or {}).get("retrieval"),
            "pathcondrag_pc3": (old.get("pathcondrag_pc3") or {}).get("retrieval"),
        },
    }
    write_json(output / "pair_summary_retrieve.json", summary)
    print("[summary]", json.dumps(summary, indent=2, ensure_ascii=False))


def distinct_summary(args):
    output = Path(args.output_dir)
    hippo = read_json(output / "hipporag2_musique" / "metrics.json")
    path = read_json(output / "pathcondrag" / "result.json")
    summary = {
        "embedding": "/root/models/Qwen3-Embedding-8B",
        "embedding_batch_size": 4,
        "hipporag2": {"retrieval": hippo["retrieval_metrics"], "qa": hippo["qa_metrics"]},
        "pathcondrag_pc3": {"retrieval": path["retrieval_metrics"], "qa": path["qa_metrics"]},
    }
    # Retain this runner's existing ASCII serialization for its summary.
    write_json(output / "pair_summary.json", summary, ensure_ascii=True)
    print(json.dumps(summary, indent=2))


def register_commands(subparsers):
    parser = subparsers.add_parser("embedding-release-memory", help="Collect unused memory between runs")
    parser.set_defaults(handler=release_memory)

    parser = subparsers.add_parser("embedding-prepare-smoke", help="Build a small MuSiQue corpus")
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.set_defaults(handler=prepare_smoke_data)

    parser = subparsers.add_parser("embedding-print-metrics", help="Print paired experiment metrics")
    parser.add_argument("--result", required=True)
    parser.add_argument("--method", choices=("hippo", "pcr"), required=True)
    parser.set_defaults(handler=print_metrics)

    parser = subparsers.add_parser("embedding-pair-summary", help="Write embedding ablation summary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--embedding-batch-size", type=int, required=True)
    parser.set_defaults(handler=pair_summary)

    parser = subparsers.add_parser("embedding-eval-hippo-instrfix", help="Evaluate Hippo with fixed query instructions")
    parser.add_argument("--hippo-root", required=True)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--embedding-batch-size", type=int, required=True)
    parser.set_defaults(handler=evaluate_hippo_instruction_fix)

    parser = subparsers.add_parser("embedding-instrfix-summary", help="Compare instruction fix to prior results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-hippo", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--llm", required=True)
    parser.set_defaults(handler=instruction_fix_summary)

    parser = subparsers.add_parser("embedding-distinct-summary", help="Write batch-four paired summary")
    parser.add_argument("--output-dir", required=True)
    parser.set_defaults(handler=distinct_summary)
