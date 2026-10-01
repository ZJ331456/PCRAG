"""Validation and reports for the retrieval component ablation runners."""

from pathlib import Path

from .common import read_json, write_json


RECALL_KEYS = ("Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20")
QWEN_MODEL = "/root/models/Qwen3-Embedding-8B"
QWEN_INDEX_DIR = "qwen3-8b__root_models_Qwen3-Embedding-8B"
COMPONENT_CASES = (
    "wo_qcappr", "wo_eba", "wo_idf", "wo_iter", "wo_qd", "wo_pcqd", "wo_pathset",
)
ACTIVE_COMPONENTS = (
    "use_qcappr", "use_eba", "use_entity_idf_index", "use_iterative_retrieval",
    "use_query_decomposition", "use_path_conditioned_qd", "use_path_set_optimization",
)


def validate_qwen_source(args):
    result = read_json(args.source_result)
    config = result["runtime_config"]
    expected = {
        "embedding_model_name": args.embedding_model,
        "embedding_batch_size": 4,
        "retrieval_top_k": 200,
        "hop_force_max": 2,
        "hop_multi_min_signals": 2,
        "use_qcappr": True,
        "use_eba": True,
        "use_entity_idf_index": True,
        "use_bridge_cache_index": True,
        "use_iterative_retrieval": True,
        "use_query_decomposition": True,
        "use_path_conditioned_qd": True,
        "use_path_set_optimization": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(
                f"Source result mismatch: {key}={config.get(key)!r}, expected {value!r}"
            )
    if result.get("sample_size_effective") != 1000 or result.get("indexed_docs") != 11656:
        raise SystemExit("Source result is not the completed 1000-question full-corpus run")
    manifest = read_json(Path(args.source_index) / QWEN_INDEX_DIR / "index_manifest.json")
    if manifest["embedding"]["model_name"] != args.embedding_model:
        raise SystemExit("Source index embedding does not match the requested Qwen3-Embedding-8B")
    print("[verified] Existing PathCondRAG Qwen3 index and full-run configuration")
    return 0


def qwen_result_complete(args):
    result = read_json(args.result)
    config = result.get("runtime_config", {})
    complete = (
        result.get("sample_size_effective") == 1000
        and result.get("eval_mode") == "retrieve"
        and result.get("retrieval_metrics", {}).get("Recall@20") is not None
        and config.get("embedding_batch_size") == 4
        and config.get("embedding_model_name") == QWEN_MODEL
    )
    return 0 if complete else 1


def print_qwen_result(args):
    result = read_json(args.result)
    metrics = result["retrieval_metrics"]
    config = result["runtime_config"]
    print(
        "[done]", args.name, {key: metrics.get(key) for key in RECALL_KEYS},
        "active=", {key: config[key] for key in ACTIVE_COMPONENTS}, flush=True,
    )
    return 0


def write_qwen_summary(args):
    root = Path(args.out_root)
    reference = read_json(args.source_result)
    reference_metrics = reference["retrieval_metrics"]
    summary = {
        "source_index": str(root.parent / "pathcondrag" / "index"),
        "embedding_model": QWEN_MODEL,
        "embedding_batch_size": 4,
        "full_reference_existing_result": {
            key: reference_metrics.get(key) for key in RECALL_KEYS
        },
        "ablations": {},
    }
    for name in COMPONENT_CASES:
        result = read_json(root / "results" / f"{name}.json")
        metrics = result["retrieval_metrics"]
        summary["ablations"][name] = {
            "retrieval": {key: metrics.get(key) for key in RECALL_KEYS},
            "delta_vs_existing_full": {
                key: round(metrics[key] - reference_metrics[key], 4) for key in RECALL_KEYS
            },
            "module_usage": result.get("retrieval_diagnostics", {}).get("module_usage", {}),
        }
    path = root / "summary.json"
    write_json(path, summary)
    print(f"[summary] {path}", flush=True)
    return 0


def pc3_result_complete(args):
    result = read_json(args.result)
    metrics = result.get("retrieval_metrics") or {}
    return 0 if metrics.get("Recall@5") is not None else 1


def print_pc3_result(args):
    result = read_json(args.result)
    metrics = result.get("retrieval_metrics") or {}
    print("[metrics]", args.name, {key: metrics.get(key) for key in RECALL_KEYS})
    return 0


def write_pc3_summary(args):
    result_dir = Path(args.result_dir)
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        result = read_json(path)
        metrics = result.get("retrieval_metrics") or {}
        rows.append((path.stem, {key: metrics.get(key) for key in RECALL_KEYS}))
    print("\n===== ABLATION SUMMARY (retrieve) =====")
    header = f"{'case':20} " + " ".join(f"{key:>10}" for key in RECALL_KEYS)
    print(header)
    print("-" * len(header))
    for name, metrics in rows:
        print(
            f"{name:20} " + " ".join(
                f"{(metrics[key] if metrics[key] is not None else float('nan')):10.4f}"
                for key in RECALL_KEYS
            )
        )
    path = result_dir.parent / "summary.json"
    write_json(
        path, [{"case": name, "retrieval": metrics} for name, metrics in rows],
        ensure_ascii=True,
    )
    print(f"\n[saved] {path}")
    return 0


def register_commands(subparsers):
    """Register the stdlib-only helpers used by both ablation shell scripts."""
    parser = subparsers.add_parser("ablation-qwen-validate-source")
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.set_defaults(handler=validate_qwen_source)

    for command, handler in (
        ("ablation-qwen-result-complete", qwen_result_complete),
        ("ablation-qwen-print-result", print_qwen_result),
        ("ablation-pc3-result-complete", pc3_result_complete),
        ("ablation-pc3-print-result", print_pc3_result),
    ):
        parser = subparsers.add_parser(command)
        parser.add_argument("--result", required=True)
        if command.endswith("print-result"):
            parser.add_argument("--name", required=True)
        parser.set_defaults(handler=handler)

    parser = subparsers.add_parser("ablation-qwen-summary")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-result", required=True)
    parser.set_defaults(handler=write_qwen_summary)

    parser = subparsers.add_parser("ablation-pc3-summary")
    parser.add_argument("--result-dir", required=True)
    parser.set_defaults(handler=write_pc3_summary)
