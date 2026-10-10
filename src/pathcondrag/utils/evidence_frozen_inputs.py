"""Freeze label-free finalizer inputs for an exact, paired selection experiment.

This evaluation helper does not replace retrieval in production. A normal
legacy run captures the boundary immediately before the outermost finalizer.
Replay runs the actual finalizer and semantic review from that same boundary,
without recomputing embeddings, graph scores or upstream LLM decisions.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np


SCHEMA = 1
STATE_KEYS = ("evidence_candidates", "evidence_trace", "_evidence_plan",
              "_evidence_winning_bindings", "_evidence_beams")
LABEL_KEYS = frozenset(("gold_docs", "gold_answers", "gold_document_ranks",
                       "retrieval_metrics", "all_gold_in_top5", "all_gold_in_top10"))
OUTPUT_KEYS = frozenset(("finalizer_input_hash", "evidence_scoring_input_sha256",
                        "dependency_scoring", "dependency_joint_selection",
                        "improvement_dag_package", "support_semantic_veto",
                        "improvement_support_semantic_veto"))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def passage_order_sha256(keys):
    return _digest(json.dumps(list(keys), ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8"))


def runtime_config_sha256(config):
    values = deepcopy(config)
    for key in ("save_dir", "evidence_scoring_mode"):
        values.pop(key, None)
    return _digest(json.dumps(values, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8"))


def experiment_metadata(manifest, mode):
    return {"mode": mode, "query_count": manifest["query_count"],
            "payload_sha256": manifest["payload_sha256"],
            "passage_order_sha256": manifest["passage_order_sha256"],
            "upstream_recomputed": mode == "capture",
            "timing_scope": "normal retrieval" if mode == "capture"
                else "finalizer and semantic review only"}


def _pure_inputs(value):
    """Refuse labels and runtime objects such as futures or model instances."""
    if isinstance(value, dict):
        if any(key in LABEL_KEYS for key in value):
            raise ValueError("Benchmark labels are not permitted in frozen inputs")
        for key, item in value.items():
            _pure_inputs(key)
            _pure_inputs(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _pure_inputs(item)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            for item in value.flat:
                _pure_inputs(item)
    elif not isinstance(value, (str, bytes, bool, int, float, complex, np.generic, type(None))):
        raise ValueError(f"Non-data object in frozen inputs: {type(value).__name__}")


def _input_hash(record):
    from ..evidence_dependency_scoring import scoring_input_sha256
    return scoring_input_sha256(record["query"], record["ids"], record["scores"], record["state"])


def _check_runtime(runtime, capture=False):
    cfg = runtime.cfg
    mode = getattr(cfg, "evidence_scoring_mode", "legacy")
    if getattr(cfg, "improvement_stage", 0) != 4:
        raise ValueError("Frozen inputs require improvement_stage=4")
    if mode not in (("legacy",) if capture else ("legacy", "dependency_joint")):
        raise ValueError(f"Unsupported frozen-input scoring mode: {mode}")
    flags = getattr(runtime, "improvements", frozenset())
    if not {"plan_prune", "dag_package", "support_semantic_veto"}.issubset(flags):
        raise ValueError("Frozen inputs require plan_prune, dag_package and support_semantic_veto")


class EvidenceInputCapture:
    def __init__(self):
        self.records = []
        self.passage_keys = None

    def before_finalize(self, runtime, query, ids, scores, ctx, state):
        _check_runtime(runtime, capture=True)
        keys = list(runtime.rag.passage_node_keys)
        if self.passage_keys is not None and self.passage_keys != keys:
            raise ValueError("Passage order changed during capture")
        self.passage_keys = keys
        trace = state.get("evidence_trace", {})
        if (any(key in trace for key in OUTPUT_KEYS)
                or trace.get("selected_prefix") or trace.get("covered_goals")):
            raise ValueError("Capture must occur before any evidence finalizer")
        record = deepcopy({"query": query, "query_idx": state.get("query_idx", len(self.records)),
                           "ids": ids, "scores": scores, "ctx": ctx,
                           "state": {key: state[key] for key in STATE_KEYS if key in state}})
        _pure_inputs(record)
        if record["query_idx"] != len(self.records):
            raise ValueError("Captured query order differs from retrieval order")
        record["input_sha256"] = _input_hash(record)
        self.records.append(record)

    def write(self, directory, baseline_result):
        """Publish only after the normal evaluation successfully wrote its result."""
        directory, baseline_result = Path(directory), Path(baseline_result).resolve()
        raw = baseline_result.read_bytes()
        result = json.loads(raw)
        rows = result.get("results", [])
        if not rows or len(rows) != len(self.records):
            raise ValueError("Captured input count differs from baseline result count")
        for record, row in zip(self.records, rows):
            evidence = row.get("retrieval_trace", {}).get("evidence", {})
            if (record["query"] != row.get("question")
                    or record["input_sha256"] != evidence.get("finalizer_input_hash")):
                raise ValueError("Captured raw input hash does not match baseline export")
            record["upstream_trace"] = deepcopy({key: value for key, value in
                row.get("retrieval_trace", {}).items() if key != "evidence"})
        diagnostics = deepcopy(result.get("retrieval_diagnostics", {}))
        diagnostics.pop("support_semantic_veto", None)
        payload = {"records": self.records, "retrieval_diagnostics": diagnostics}
        _pure_inputs(payload)
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        manifest = {"schema": SCHEMA, "payload_sha256": _digest(blob),
                    "baseline_result": str(baseline_result),
                    "passage_order_sha256": passage_order_sha256(self.passage_keys),
                    "passage_count": len(self.passage_keys), "query_count": len(rows),
                    "queries": [row["question"] for row in rows],
                    "selected_indices": result["selected_indices"],
                    "benchmark_hops": [row.get("benchmark_hops") for row in rows],
                    "input_sha256": [record["input_sha256"] for record in self.records],
                    "runtime_config_sha256": runtime_config_sha256(result["runtime_config"]),
                    "label_free_payload": True,
                    "scope": "frozen upstream finalizer experiment; replay time excludes upstream retrieval"}
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "manifest.json").exists() or (directory / "inputs.pkl").exists():
            raise FileExistsError(f"Frozen input snapshot already exists: {directory}")
        result["frozen_evidence_inputs"] = experiment_metadata(manifest, "capture")
        baseline_temporary = baseline_result.with_name(baseline_result.name + ".frozen.tmp")
        baseline_temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        baseline_temporary.replace(baseline_result)
        manifest["baseline_result_sha256"] = _digest(baseline_result.read_bytes())
        temporary = directory / "inputs.pkl.tmp"
        temporary.write_bytes(blob)
        temporary.replace(directory / "inputs.pkl")
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(directory / "manifest.json")
        return manifest


def load_frozen_inputs(directory, queries, passage_keys, hop_overrides=None, runtime_config=None):
    """Validate own trusted experiment files before loading their pickle payload."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("label_free_payload"):
        raise ValueError("Unsupported frozen-input manifest")
    if list(queries) != manifest.get("queries"):
        raise ValueError("Replay queries or query order differ from capture")
    if (len(passage_keys) != manifest.get("passage_count")
            or passage_order_sha256(passage_keys) != manifest.get("passage_order_sha256")):
        raise ValueError("Replay passage index order differs from capture")
    expected_hops = manifest.get("benchmark_hops")
    actual_hops = list(hop_overrides) if hop_overrides is not None else [None] * len(queries)
    if actual_hops != expected_hops:
        raise ValueError("Replay benchmark hop overrides differ from capture")
    if runtime_config is not None and runtime_config_sha256(runtime_config) != manifest.get("runtime_config_sha256"):
        raise ValueError("Replay runtime configuration differs from capture")
    baseline = Path(manifest["baseline_result"]).read_bytes()
    if _digest(baseline) != manifest.get("baseline_result_sha256"):
        raise ValueError("Baseline result changed after capture")
    blob = (directory / "inputs.pkl").read_bytes()
    if _digest(blob) != manifest.get("payload_sha256"):
        raise ValueError("Frozen input payload digest mismatch")
    payload = pickle.loads(blob)
    _pure_inputs(payload)
    records = payload.get("records", [])
    if len(records) != manifest.get("query_count") or len(records) != len(queries):
        raise ValueError("Frozen input record count mismatch")
    for index, record in enumerate(records):
        if (record.get("query_idx") != index or record.get("query") != queries[index]
                or _input_hash(record) != manifest["input_sha256"][index]
                or record.get("input_sha256") != manifest["input_sha256"][index]):
            raise ValueError("Frozen input identity or raw scorer hash mismatch")
    return payload, manifest


def replay_retrieve(rag, directory, queries, num_to_retrieve=None, gold_docs=None,
                    *, postprocess=None, solution_class=None, recall_class=None):
    """Execute real evidence selection/review; labels enter only final evaluation."""
    started = time.time()
    if not rag.ready_to_retrieve:
        rag.prepare_retrieval_objects()
    runtime = rag.evidence_runtime
    _check_runtime(runtime)
    payload, manifest = load_frozen_inputs(directory, queries, rag.passage_node_keys,
        getattr(rag, "_query_hop_overrides", None), asdict(runtime.cfg))
    if solution_class is None:
        from .misc_utils import QuerySolution
        solution_class = QuerySolution
    if postprocess is None:
        from ..evidence_support_semantic_veto import postprocess_support_semantic_veto
        postprocess = postprocess_support_semantic_veto
    limit = num_to_retrieve if num_to_retrieve is not None else rag.global_config.retrieval_top_k
    rag.retrieval_diagnostics = deepcopy(payload["retrieval_diagnostics"])
    solutions, review_inputs = [], []
    for frozen in payload["records"]:
        record = deepcopy(frozen)
        if _input_hash(record) != record["input_sha256"]:
            raise ValueError("Input copy changed scorer hash")
        ids, scores, evidence = runtime.finalize(record["query"], record["ids"],
            record["scores"], record["ctx"], record["state"])
        if evidence.get("finalizer_input_hash") != record["input_sha256"]:
            raise ValueError("Actual replay finalizer changed the frozen raw-input hash")
        trace = deepcopy(record["upstream_trace"])
        trace["evidence"] = evidence
        solution = solution_class(question=record["query"],
            docs=[rag.chunk_embedding_store.get_row(rag.passage_node_keys[int(doc_id)])["content"]
                  for doc_id in ids[:limit]], doc_scores=np.asarray(scores)[:limit], retrieval_trace=trace)
        solutions.append(solution)
        review_inputs.append((solution, np.asarray(ids[:limit]).copy()))
    rag.retrieval_diagnostics["support_semantic_veto"] = postprocess(runtime, review_inputs)
    rag.retrieval_diagnostics["frozen_evidence_replay"] = {
        "enabled": True, "query_count": len(solutions), "upstream_recomputed": False,
        "extra_embedding_calls": 0, "exact_input_hashes": True,
        "payload_sha256": manifest["payload_sha256"],
        "timing_scope": "finalizer and semantic review only"}
    rag.all_retrieval_time += time.time() - started
    if gold_docs is None:
        return solutions
    if recall_class is None:
        from ..evaluation.retrieval_eval import RetrievalRecall
        recall_class = RetrievalRecall
    overall, _ = recall_class(global_config=rag.global_config).calculate_metric_scores(
        gold_docs=gold_docs, retrieved_docs=[solution.docs for solution in solutions],
        k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200])
    return solutions, overall


@contextmanager
def frozen_evidence_runtime(capture=None, replay=None):
    """Temporarily install evaluation hooks and restore them even after errors."""
    from ..evidence_improvements import ImprovedEvidenceRetrieval
    from ..pathcondrag import PathCondRAG
    if bool(capture) == bool(replay):
        raise ValueError("Specify exactly one capture or replay mode")
    had_local_finalize = "finalize" in ImprovedEvidenceRetrieval.__dict__
    original_finalize, original_retrieve = ImprovedEvidenceRetrieval.finalize, PathCondRAG.retrieve
    collector = EvidenceInputCapture() if capture else None
    if capture:
        def finalize(runtime, query, ids, scores, ctx, state):
            collector.before_finalize(runtime, query, ids, scores, ctx, state)
            return original_finalize(runtime, query, ids, scores, ctx, state)
        ImprovedEvidenceRetrieval.finalize = finalize
    else:
        def retrieve(rag, queries, num_to_retrieve=None, gold_docs=None):
            return replay_retrieve(rag, replay, queries, num_to_retrieve, gold_docs)
        PathCondRAG.retrieve = retrieve
    try:
        yield collector
    finally:
        # finalize is normally inherited; removing the temporary override keeps
        # the original method resolution order exactly as it was.
        if capture:
            if had_local_finalize:
                ImprovedEvidenceRetrieval.finalize = original_finalize
            else:
                del ImprovedEvidenceRetrieval.finalize
        PathCondRAG.retrieve = original_retrieve
