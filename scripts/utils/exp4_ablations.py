"""Prepare, isolate and validate the four families of exp4 contribution controls."""
import hashlib
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from .common import http_status_counts, read_json, write_json
from . import improvement_experiments as shared

CASES = {
    "exp4_abla1_budget_dag": ("budget_dag", "literal", "coverage", 4),
    "exp4_abla1_budget_qd": ("budget_qd", "literal", "coverage", 4),
    "exp4_abla1_budget_iterative": ("budget_iterative", "literal", "coverage", 4),
    "exp4_abla2_fixed_coverage": ("fixed_pool", "literal", "coverage", 3),
    "exp4_abla2_fixed_binding": ("fixed_pool", "literal", "coverage", 4),
    "exp4_abla3_string": ("validation", "string", "coverage", 4),
    "exp4_abla3_literal": ("validation", "literal", "coverage", 4),
    "exp4_abla3_relation": ("validation", "relation", "coverage", 4),
    "exp4_abla4_coverage": ("selection", "literal", "coverage", 4),
    "exp4_abla4_ancestor": ("selection", "literal", "ancestor", 4),
    "exp4_abla4_joint": ("selection", "literal", "joint", 4),
}
require, sha256 = shared.require, shared.sha256


def chunk_hash(text):
    return "chunk-" + hashlib.md5(text.encode()).hexdigest()


def manifest_at(root):
    return read_json(Path(root) / "ablation_manifest.json")


def selected_cases(value):
    requested = value.split() if value else list(CASES)
    require(len(requested) == len(set(requested)) and requested and all(n in CASES for n in requested),
            "Unknown or duplicated ablation case")
    if any(n in requested for n in ("exp4_abla4_ancestor", "exp4_abla4_joint")):
        requested = list(dict.fromkeys([*requested, "exp4_abla4_coverage"]))
    return [n for n in CASES if n in requested]


def freeze_records(pool_result, budget_result, data, indices, corpus):
    """Only observed candidate identities/scores and call counts reach retrieval."""
    pool_rows = {r["query_index"]: r for r in pool_result["results"]}
    budget_rows = {r["query_index"]: r for r in budget_result["results"]}
    records = []
    for index in indices:
        sample = data[index]
        p, b = pool_rows[index], budget_rows[index]
        require(p["sample_id"] == b["sample_id"] == sample["id"]
                and p["question"] == b["question"] == sample["question"], "Reference sample mismatch")
        docs, scores = p["candidate_docs"], p["candidate_doc_scores"]
        require(len(docs) == len(set(docs)) == 200 and set(docs) <= corpus,
                "Frozen pool must contain 200 distinct corpus documents")
        require(len(scores) == 200 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in scores),
                "Invalid frozen pool scores")
        evidence = b["retrieval_trace"].get("evidence") or {}
        counts = [evidence.get("llm_plan_calls", 0), evidence.get("llm_verification_calls", 0)]
        require(all(type(v) is int and v >= 0 for v in counts), "Invalid reference call counts")
        records.append({"question": sample["question"], "sample_id": sample["id"], "query_index": index,
                        "pool": [{"doc_hash": chunk_hash(doc), "score": float(score)}
                                 for doc, score in zip(docs, scores)],
                        "evidence_call_budget": sum(counts)})
    require(len({r["query_index"] for r in records}) == len(records), "Duplicate sample indices in ablation inputs")
    return {"schema_version": 1, "records": records}


def prepare(args):
    root, ref, source = map(lambda x: Path(x).resolve(), (args.out_root, args.reference_root, args.source_index))
    original = shared.manifest_at(ref)
    require(original.get("index_build_status") == "ready", "Reference shared index is not ready")
    require(source == Path(original["source_index"]).resolve(), "Use the reference run's shared Hippo index")
    require(shared.index_ready(SimpleNamespace(out_root=str(ref))) == 0, "Shared index identity changed")
    data, corpus, hops = shared.validated_dataset(original["data_path"], original["corpus_path"])
    available = original["selected_indices"]
    if args.sample_indices_file:
        indices = shared.load_indices(args.sample_indices_file, len(data))
        require(set(indices) <= set(available), "Selected questions absent from reference run")
        require(not args.sample_size or len(indices) == args.sample_size, "Sample size mismatch")
    elif args.sample_size == 0:
        indices = available
    elif args.sample_size == 2:
        indices = [next(i for i in available if hops[i] == h) for h in (2, 4)]
    else:
        require(0 < args.sample_size <= len(available), "Invalid sample size")
        indices = sorted(random.Random(args.sample_seed).sample(available, args.sample_size))
    provenance = {}
    reference_results = []
    for name in ("exp3_prefix_coverage", "exp4_dependency_binding"):
        require(shared.ready(SimpleNamespace(out_root=str(ref), name=name)) == 0, "Reference case not validated")
        path = ref / "cases" / name / "result.json"
        provenance[name] = {"path": str(path), "sha256": sha256(path)}
        reference_results.append(read_json(path))
    inputs = freeze_records(*reference_results, data, indices, corpus)
    names = selected_cases(args.cases)
    manifest = {
        "schema_version": 1, "reference_root": str(ref), "source_index": str(source),
        "source_asset_sha256": original["source_asset_sha256"],
        "initial_cache_path": str(ref / "initial_llm_cache"),
        "initial_cache_sha256": original["initial_cache_sha256"],
        "data_path": original["data_path"], "corpus_path": original["corpus_path"],
        "data_sha256": original["data_sha256"], "corpus_sha256": original["corpus_sha256"],
        "selected_indices": indices, "benchmark_hops": [hops[i] for i in indices],
        "hop_distribution": dict(Counter(str(hops[i]) for i in indices)),
        "runtime": original["runtime"], "reference_results": provenance,
        "cases": names, "case_specs": {n: list(CASES[n]) for n in names},
        "budget_scope": "additional_evidence_module_logical_calls_including_semantic_repairs",
        "budget_total": sum(r["evidence_call_budget"] for r in inputs["records"]),
        "frozen_pool_source": "exp3_prefix_coverage",
    }
    root.mkdir(parents=True, exist_ok=True)
    path, inputs_path = root / "ablation_manifest.json", root / "ablation_inputs.json"
    if path.exists():
        existing = read_json(path)
        require({k: v for k, v in existing.items() if k != "inputs_sha256"} == manifest,
                "Existing ablation manifest differs; use a separate smoke directory")
        require(read_json(inputs_path) == inputs and sha256(inputs_path) == existing["inputs_sha256"],
                "Frozen ablation inputs changed")
    else:
        require(not inputs_path.exists(), "Partial ablation inputs already exist")
        write_json(inputs_path, inputs)
        manifest["inputs_sha256"] = sha256(inputs_path)
        write_json(path, manifest)
        write_json(root / "ablation_selected_indices.json", indices)
    print(f"[prepared-ablation] questions={len(indices)} cases={len(names)} added_call_allowance={manifest['budget_total']}", flush=True)


def verify_source(manifest):
    require(shared.asset_hashes(manifest["source_index"]) == manifest["source_asset_sha256"], "Shared index changed")
    require(shared.cache_hashes(manifest["initial_cache_path"]) == manifest["initial_cache_sha256"], "Initial cache snapshot changed")


def trajectory_digest(result):
    import json
    fields = ("plan", "routes", "branch_scores", "llm_plan_calls", "llm_verification_calls")
    traces = [{"query_index": row["query_index"], **{
        key: (row["retrieval_trace"].get("evidence") or {}).get(key) for key in fields}}
        for row in result["results"]]
    return hashlib.sha256(json.dumps(traces, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def selection_cache_at(root):
    root = Path(root)
    info = read_json(root / "ablation_selection_cache_manifest.json")
    path = root / "ablation_selection_cache"
    require(shared.cache_hashes(path) == info["cache_sha256"], "Shared selection response snapshot changed")
    require(sha256(root / "cases" / "exp4_abla4_coverage" / "result.json") == info["coverage_result_sha256"],
            "Selection trajectory reference changed")
    return path, info


def initialize_case(args):
    root, name = Path(args.out_root), args.name
    manifest = manifest_at(root)
    require(name in manifest["cases"], "Case not selected")
    verify_source(manifest)
    case, source = root / "cases" / name, Path(manifest["source_index"])
    require(not case.exists(), "Incomplete existing case; inspect it or use a new output directory")
    cache_source, cache_hashes = Path(manifest["initial_cache_path"]), manifest["initial_cache_sha256"]
    cache_seed_kind = "initial_index_snapshot"
    if CASES[name][0] == "selection" and CASES[name][2] != "coverage":
        require(ready(SimpleNamespace(out_root=str(root), name="exp4_abla4_coverage")) == 0,
                "Run the shared selection exploration before ancestor/joint")
        cache_source, cache_info = selection_cache_at(root)
        cache_hashes = cache_info["cache_sha256"]
        cache_seed_kind = "shared_selection_trajectory"
    model_source = source / shared.MODEL_DIR
    model_target = case / "index" / shared.MODEL_DIR
    model_target.mkdir(parents=True)
    assets = set(shared.ASSETS)
    for path in model_source.rglob("*"):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        relative = path.relative_to(model_source)
        target = model_target / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if str(relative) in assets:
            os.link(path, target)
        else:
            shutil.copy2(path, target)
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith(".lock"):
            shutil.copy2(path, case / "index" / path.name)
    shutil.copytree(cache_source, case / "index" / "llm_cache",
                    ignore=shutil.ignore_patterns("*.lock"))
    require(shared.asset_hashes(case / "index") == manifest["source_asset_sha256"], "Index copy mismatch")
    require(shared.cache_hashes(case / "index" / "llm_cache") == cache_hashes, "Cache copy mismatch")
    write_json(case / "before.json", {"asset_sha256": manifest["source_asset_sha256"],
                                      "initial_cache_sha256": cache_hashes, "cache_seed_kind": cache_seed_kind})
    print(f"[initialized-ablation] {name}: shared immutable graph/vectors; isolated metadata and cache", flush=True)


def ready(args):
    root, name = Path(args.out_root), args.name
    case = root / "cases" / name
    if not (case / "validated.ok").exists():
        return 1
    require(read_json(case / "validated.ok") == {
        "result_sha256": sha256(case / "result.json"), "report_sha256": sha256(case / "report.json"),
        "manifest_sha256": sha256(root / "ablation_manifest.json")}, "Validated ablation files changed")
    manifest = manifest_at(root)
    require(shared.asset_hashes(case / "index") == manifest["source_asset_sha256"], "Validated index changed")
    verify_source(manifest)
    print(f"[skip-ablation] {name}", flush=True)
    return 0


def validate_controls(result, manifest, name, inputs):
    mode, binding, selection, stage = CASES[name]
    cfg = result["runtime_config"]
    for key, expected in (("evidence_ablation_mode", mode), ("evidence_binding_mode", binding),
                          ("evidence_selection_mode", selection), ("improvement_stage", stage)):
        require(cfg.get(key) == expected, f"{key} mismatch")
    diagnostics = {"logical_calls": 0, "allowance": 0, "under_budget_questions": 0,
                   "fixed_pool_questions": 0, "selection_policy_counts": {},
                   "search_count": 0, "semantic_failure_events": 0,
                   "logical_prompt_tokens": 0, "logical_completion_tokens": 0}
    for row, record in zip(result["results"], inputs["records"]):
        require(row["query_index"] == record["query_index"], "Input/result identity mismatch")
        trace = row["retrieval_trace"].get("evidence") or {}
        control = trace.get("ablation") or {}
        require(control.get("mode") == mode and control.get("binding_mode") == binding
                and control.get("selection_mode") == selection, "Missing/mismatched ablation trace")
        require(control.get("global_query_index") == record["query_index"],
                "Runtime matched the wrong ablation sample identity")
        calls = control.get("llm_calls")
        require(type(calls) is int and calls >= 0, "Missing logical-call accounting")
        diagnostics["logical_calls"] += calls
        diagnostics["search_count"] += int(trace.get("search_count", 0))
        diagnostics["semantic_failure_events"] += len(trace.get("semantic_failures", []))
        diagnostics["logical_prompt_tokens"] += int(control.get("logical_prompt_tokens", 0))
        diagnostics["logical_completion_tokens"] += int(control.get("logical_completion_tokens", 0))
        if mode.startswith("budget_"):
            allowance = record["evidence_call_budget"]
            require(control.get("call_budget") == allowance and calls <= allowance,
                    "Per-question additional LLM budget exceeded")
            if mode in ("budget_qd", "budget_iterative"):
                require(calls == allowance, "Budget control did not spend its query-generation allowance")
            diagnostics["allowance"] += allowance
            diagnostics["under_budget_questions"] += calls < allowance
        if mode == "fixed_pool":
            require({chunk_hash(doc) for doc in row["candidate_docs"]}
                    == {p["doc_hash"] for p in record["pool"]}, "Fixed Top200 pool changed")
            diagnostics["fixed_pool_questions"] += 1
        if selection in ("ancestor", "joint"):
            diag = trace.get("selection_diagnostics") or {}
            require(diag.get("closure_valid") is True, "Final prefix violated ancestor support constraint")
            key = str(diag.get("policy", selection))
            diagnostics["selection_policy_counts"][key] = diagnostics["selection_policy_counts"].get(key, 0) + 1
        if mode == "selection":
            require(control.get("selection_shared_branches") is True and control.get("shared_beam_width") == 3,
                    "Selection groups must use the same width-three exploration")
    return diagnostics


def report(args):
    root, name = Path(args.out_root), args.name
    manifest = manifest_at(root)
    require(name in manifest["cases"], "Case not selected")
    verify_source(manifest)
    for field in ("data", "corpus"):
        require(sha256(manifest[field + "_path"]) == manifest[field + "_sha256"], f"{field} changed")
    require(sha256(root / "ablation_inputs.json") == manifest["inputs_sha256"], "Frozen inputs changed")
    data, corpus, _ = shared.validated_dataset(manifest["data_path"], manifest["corpus_path"])
    case = root / "cases" / name
    result = read_json(case / "result.json")
    measurements, metrics = shared.validate_result(result, manifest, name, data, corpus, expected_stage=CASES[name][3])
    diagnostics = validate_controls(result, manifest, name, read_json(root / "ablation_inputs.json"))
    before = read_json(case / "before.json")
    require(before["asset_sha256"] == shared.asset_hashes(case / "index") == manifest["source_asset_sha256"], "Index mutated")
    replay = CASES[name][0] == "selection" and CASES[name][2] != "coverage"
    if replay:
        _, cache_info = selection_cache_at(root)
        require(before["cache_seed_kind"] == "shared_selection_trajectory"
                and before["initial_cache_sha256"] == cache_info["cache_sha256"], "Selection cache replay mismatch")
        require(trajectory_digest(result) == cache_info["trajectory_sha256"],
                "Selection groups generated different branches; comparison would be confounded")
        diagnostics["trajectory_sha256"] = cache_info["trajectory_sha256"]
        diagnostics["shared_exploration_replayed"] = True
        diagnostics["shared_exploration_cost_case"] = "exp4_abla4_coverage"
    else:
        require(before["initial_cache_sha256"] == manifest["initial_cache_sha256"], "Initial cache mismatch")
    require(args.log_start >= 0 and args.log_end >= args.log_start, "vLLM log rotated")
    statuses = http_status_counts(args.vllm_log, args.log_start, args.log_end)
    stats = result.get("llm_request_stats") or {}
    require(stats.get("failures") == 0 and stats.get("max_in_flight") == 8, "LLM failure/concurrency mismatch")
    require(not any(code != "200" for code in statuses), f"Non-200 LLM responses: {statuses}")
    require(not stats.get("http_attempts") or statuses, "HTTP attempts have no server status records")
    stratified = {}
    for hop in (2, 3, 4):
        subset = [m for m in measurements if m["hops"] == hop]
        if subset:
            stratified[str(hop)] = {"n_samples": len(subset),
                "retrieval_metrics": {key: sum(m["metrics"][key] for m in subset) / len(subset) for key in metrics},
                "all_gold_top5": sum(m["all_gold_top5"] for m in subset) / len(subset)}
    output = {"name": name, "validated": True, "case_spec": list(CASES[name]),
        "n_samples": len(measurements), "seconds": args.elapsed, "retrieval_seconds": result.get("retrieval_seconds"),
        "retrieval_metrics": metrics, "all_gold_top5": sum(m["all_gold_top5"] for m in measurements) / len(measurements),
        "stratified": stratified, "llm_request_stats": stats, "http_status_in_log": statuses,
        "control_diagnostics": diagnostics, "asset_sha256_before": before["asset_sha256"],
        "asset_sha256_after": shared.asset_hashes(case / "index"),
        "vllm_log_slice": {"path": args.vllm_log, "start": args.log_start, "end": args.log_end}}
    write_json(case / "report.json", output)
    if name == "exp4_abla4_coverage":
        snapshot = root / "ablation_selection_cache"
        if snapshot.exists():
            _, existing = selection_cache_at(root)
            require(existing["trajectory_sha256"] == trajectory_digest(result), "Selection replay trajectory changed")
        else:
            pending = root / "ablation_selection_cache.pending"
            require(not pending.exists(), "Partial selection response snapshot; inspect it before retrying")
            shutil.copytree(case / "index" / "llm_cache", pending, ignore=shutil.ignore_patterns("*.lock"))
            write_json(root / "ablation_selection_cache_manifest.json", {
                "cache_sha256": shared.cache_hashes(pending), "coverage_result_sha256": sha256(case / "result.json"),
                "trajectory_sha256": trajectory_digest(result), "shared_exploration_cost_case": name})
            pending.rename(snapshot)
    write_json(case / "validated.ok", {"result_sha256": sha256(case / "result.json"),
        "report_sha256": sha256(case / "report.json"), "manifest_sha256": sha256(root / "ablation_manifest.json")})
    print(f"[validated-ablation] {name} n={len(measurements)} R@5={metrics['Recall@5']:.4f} added_calls={diagnostics['logical_calls']}", flush=True)


def summary(args):
    root, manifest = Path(args.out_root), manifest_at(args.out_root)
    reports = {}
    for name in manifest["cases"]:
        require(ready(SimpleNamespace(out_root=str(root), name=name)) == 0, "Incomplete ablation")
        reports[name] = read_json(root / "cases" / name / "report.json")
    write_json(root / "ablation_comparison.json", {"manifest": str(root / "ablation_manifest.json"), "cases": reports})
    lines = ["| Case | R@1 | R@2 | R@5 | R@10 | All gold@5 | Seconds | Added logical calls | HTTP attempts |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in reports.items():
        values = " | ".join(f"{r['retrieval_metrics'][f'Recall@{k}']:.4f}" for k in (1, 2, 5, 10))
        lines.append(f"| {name} | {values} | {r['all_gold_top5']:.4f} | {r['seconds']} | {r['control_diagnostics']['logical_calls']} | {r['llm_request_stats']['http_attempts']} |")
    (root / "ablation_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "ablations_completed.ok", {"cases": list(reports), "manifest_sha256": sha256(root / "ablation_manifest.json")})
    print("\n".join(lines), flush=True)


def list_cases(args):
    for name in selected_cases(args.cases):
        print("|".join(map(str, (name, *CASES[name]))))


def register_commands(subparsers):
    p = subparsers.add_parser("exp4-ablation-prepare")
    for flag in ("out-root", "reference-root", "source-index"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--sample-size", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--sample-indices-file", default="")
    p.add_argument("--cases", default="")
    p.set_defaults(handler=prepare)
    for name, handler in (("initialize-case", initialize_case), ("ready", ready), ("report", report)):
        p = subparsers.add_parser("exp4-ablation-" + name)
        p.add_argument("--out-root", required=True)
        p.add_argument("--name", required=True, choices=list(CASES))
        if name == "report":
            p.add_argument("--elapsed", type=int, required=True)
            p.add_argument("--vllm-log", required=True)
            p.add_argument("--log-start", type=int, required=True)
            p.add_argument("--log-end", type=int, required=True)
        p.set_defaults(handler=handler)
    p = subparsers.add_parser("exp4-ablation-summary")
    p.add_argument("--out-root", required=True)
    p.set_defaults(handler=summary)
    p = subparsers.add_parser("exp4-ablation-list")
    p.add_argument("--cases", default="")
    p.set_defaults(handler=list_cases)
