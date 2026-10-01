"""Validate and summarize the retrieval LLM concurrency ablation runs."""

import json
from pathlib import Path

from .common import http_status_counts, read_json, write_json


CASE_NAMES = (
    "pathcondrag_bs8",
    "pathcondrag_bs8_fixedhop",
    "pathcondrag_bs4",
    "pathcondrag_bs2",
)


def report(args):
    """Save one run's metrics and reject incomplete or failed results."""
    result = read_json(args.result)
    expected = args.sample_size
    n = int(result.get("sample_size_effective") or len(result.get("results") or []))
    if expected > 0 and n != expected:
        raise SystemExit(f"[{args.name}] sample mismatch: {n} != {expected}")
    if expected == 0 and n < 1000:
        raise SystemExit(f"[{args.name}] expected full (~1000), got {n}")

    want = "benchmark" if args.hop_mode == "benchmark" else "estimated"
    if result.get("hop_source") != want:
        raise SystemExit(
            f"[{args.name}] hop_source={result.get('hop_source')} want={want}"
        )

    status = {}
    if args.vllm_log.is_file():
        status = http_status_counts(args.vllm_log, args.log_start, args.log_end)
    diagnostics = result.get("retrieval_diagnostics") or {}
    run_report = {
        "name": args.name,
        "llm_prefetch_workers": args.workers,
        "hop_mode": args.hop_mode,
        "hop_source": result.get("hop_source"),
        "seconds": args.elapsed,
        "retrieval_seconds": result.get("retrieval_seconds"),
        "n_samples": n,
        "retrieval_metrics": result.get("retrieval_metrics"),
        "hop_distribution": result.get("hop_distribution"),
        "hop_counter": diagnostics.get("hop_counter"),
        "qd_used": diagnostics.get("qd_used_count"),
        "pcqd_used": diagnostics.get("pcqd_used_count"),
        "llm_request_stats": result.get("llm_request_stats", {}),
        "http_status_in_log": status,
    }
    write_json(args.case_dir / "report.json", run_report)
    print(
        "[done]", run_report["name"],
        f"sec={run_report['seconds']}",
        f"R@5={(run_report.get('retrieval_metrics') or {}).get('Recall@5')}",
        f"hop={run_report.get('hop_counter')}",
        flush=True,
    )
    if any(code != "200" for code in status):
        raise SystemExit(f"[{args.name}] non-200 in vLLM log: {status}")
    stats = run_report["llm_request_stats"] or {}
    if stats.get("failures", 0):
        raise SystemExit(f"[{args.name}] LLM failures={stats.get('failures')}")


def summarize(args):
    """Collect completed reports in the existing experiment case order."""
    root = args.out_root
    summary = {"out_root": str(root), "cases": {}}
    for name in CASE_NAMES:
        report_path = root / name / "report.json"
        if report_path.is_file():
            summary["cases"][name] = read_json(report_path)
    write_json(root / "comparison.json", summary)
    print("[comparison]", json.dumps({
        name: {
            "workers": report.get("llm_prefetch_workers"),
            "hop": report.get("hop_source"),
            "sec": report.get("seconds"),
            "R@5": (report.get("retrieval_metrics") or {}).get("Recall@5"),
        }
        for name, report in summary["cases"].items()
    }, indent=2, ensure_ascii=False))


def register_commands(subparsers):
    parser = subparsers.add_parser(
        "prefetch-report", help="Validate and report one LLM concurrency ablation run"
    )
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--hop-mode", required=True, choices=("benchmark", "fixed"))
    parser.add_argument("--elapsed", required=True, type=int)
    parser.add_argument("--sample-size", required=True, type=int)
    parser.add_argument("--vllm-log", required=True, type=Path)
    parser.add_argument("--log-start", required=True, type=int)
    parser.add_argument("--log-end", required=True, type=int)
    parser.set_defaults(handler=report)

    parser = subparsers.add_parser(
        "prefetch-summary", help="Summarize completed LLM concurrency ablation reports"
    )
    parser.add_argument("--out-root", required=True, type=Path)
    parser.set_defaults(handler=summarize)
