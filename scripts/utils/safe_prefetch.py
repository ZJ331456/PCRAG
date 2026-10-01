"""Validate and compare bounded LLM prefetch benchmark outputs."""

from pathlib import Path

from .common import http_status_counts, read_json, write_json


def report(args):
    """Check a completed run and save its timing, metrics, and HTTP report."""
    result = read_json(args.result)
    expected = args.sample_size
    if not (
        len(result["results"])
        == len(result["selected_indices"])
        == result["sample_size_effective"]
        == expected
    ):
        raise SystemExit(
            f"Unexpected completed sample count: {result['sample_size_effective']} != {expected}"
        )
    if result.get("hop_source") != "benchmark":
        raise SystemExit("Benchmark hop source was not recorded")
    status = http_status_counts(args.vllm_log, args.log_start, args.log_end, clamp=False)
    run_report = {
        "seconds": args.elapsed,
        "retrieval_seconds": result.get("retrieval_seconds"),
        "llm_request_stats": result.get("llm_request_stats", {}),
        "retrieval_metrics": result["retrieval_metrics"],
        "hop_distribution": result.get("hop_distribution"),
        "hop_counter": result["retrieval_diagnostics"].get("hop_counter"),
        "qd_used": result["retrieval_diagnostics"].get("qd_used_count"),
        "pcqd_used": result["retrieval_diagnostics"].get("pcqd_used_count"),
        "http_status_in_log": status,
    }
    write_json(Path(args.case_dir) / "report.json", run_report, ensure_ascii=True)
    print("[done]", Path(args.case_dir).name, run_report, flush=True)
    if any(code != "200" for code in status):
        raise SystemExit("Non-200 LLM response detected in vLLM log")
    if run_report["llm_request_stats"].get("failures", 0):
        raise SystemExit("Client reported terminal LLM request failure")
    if run_report["llm_request_stats"].get("http_attempts", 0) > 0 and not status:
        raise SystemExit(
            "Client made LLM requests but no HTTP statuses were found in the vLLM log"
        )


def compare(args):
    """Compare top-five documents and scores against the serial baseline."""
    root = Path(args.run_root)
    cases = sorted(root.glob("workers_*/result.json"))
    summary = {}
    baseline_path = root / "workers_1" / "result.json"
    if not baseline_path.exists():
        raise SystemExit("Missing workers_1 baseline")
    baseline = read_json(baseline_path)
    samples = read_json(args.samples)
    for path in cases:
        result = read_json(path)
        if not (
            len(result["results"])
            == len(result["selected_indices"])
            == result["sample_size_effective"]
        ):
            raise SystemExit(f"Invalid result length: {path}")
        all_gold_top5 = 0
        for row, sample_index in zip(result["results"], result["selected_indices"]):
            sample = samples[sample_index]
            gold = {
                p["title"] + "\n" + (p.get("text") or p.get("paragraph_text", ""))
                for p in sample["paragraphs"]
                if p.get("is_supporting") is not False
            }
            all_gold_top5 += int(bool(gold) and gold.issubset(set(row["docs"][:5])))
        summary[path.parent.name] = {
            "seconds": read_json(path.parent / "report.json")["seconds"],
            "retrieval_seconds": result.get("retrieval_seconds"),
            "Recall@1": result["retrieval_metrics"]["Recall@1"],
            "Recall@2": result["retrieval_metrics"]["Recall@2"],
            "Recall@5": result["retrieval_metrics"]["Recall@5"],
            "Recall@10": result["retrieval_metrics"]["Recall@10"],
            "all_gold_top5": all_gold_top5 / len(result["results"]),
            "qd_used": result["retrieval_diagnostics"].get("qd_used_count"),
            "pcqd_used": result["retrieval_diagnostics"].get("pcqd_used_count"),
            "llm_request_stats": result.get("llm_request_stats", {}),
        }
        if result["selected_indices"] != baseline["selected_indices"]:
            raise SystemExit("Cases used different question subsets")
        case = summary[path.parent.name]
        case["same_ordered_top5_vs_workers_1"] = sum(
            a["docs"][:5] == b["docs"][:5]
            for a, b in zip(result["results"], baseline["results"])
        )
        case["same_top5_scores_vs_workers_1"] = sum(
            a["doc_scores"][:5] == b["doc_scores"][:5]
            for a, b in zip(result["results"], baseline["results"])
        )
        case["different_question_indices_vs_workers_1"] = [
            idx
            for idx, (a, b) in enumerate(zip(result["results"], baseline["results"]))
            if a["docs"][:5] != b["docs"][:5]
        ]
    write_json(root / "comparison.json", summary, ensure_ascii=True)
    print("[comparison]", summary, flush=True)


def register_commands(subparsers):
    parser = subparsers.add_parser(
        "safe-prefetch-report", help="Validate one bounded LLM prefetch benchmark run"
    )
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--elapsed", required=True, type=int)
    parser.add_argument("--sample-size", required=True, type=int)
    parser.add_argument("--vllm-log", required=True, type=Path)
    parser.add_argument("--log-start", required=True, type=int)
    parser.add_argument("--log-end", required=True, type=int)
    parser.set_defaults(handler=report)

    parser = subparsers.add_parser(
        "safe-prefetch-compare", help="Compare bounded prefetch runs with workers_1"
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.set_defaults(handler=compare)
