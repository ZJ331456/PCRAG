"""CPU checks for exact upstream capture and actual candidate execution."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_frozen_evidence_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package
frozen = importlib.import_module(PACKAGE + ".utils.evidence_frozen_inputs")
improvements = importlib.import_module(PACKAGE + ".evidence_improvements")


@dataclass
class Config:
    save_dir: str = "control"
    improvement_stage: int = 4
    evidence_scoring_mode: str = "legacy"
    evidence_improvements: str = "plan_prune,dag_package,support_semantic_veto"
    retrieval_top_k: int = 3
    max_new_tokens: int = 2048
    llm_prefetch_workers: int = 8


def rag_fixture(config=None):
    config = config or Config()
    documents = {"p0": "Work X\nWork X was written by Alpha.",
                 "p1": "Other\nUnrelated text.",
                 "p2": "Alpha\nAlpha was born in Rome."}
    rag = SimpleNamespace(pcrag_config=config, global_config=config,
        passage_node_keys=list(documents), ready_to_retrieve=True,
        chunk_embedding_store=SimpleNamespace(get_row=lambda key: {"content": documents[key]}),
        retrieval_diagnostics={}, all_retrieval_time=0., _query_hop_overrides=[2])
    rag.get_query_embeddings = lambda *_args: (_ for _ in ()).throw(AssertionError("Embeddings recomputed"))
    rag.evidence_runtime = improvements.ImprovedEvidenceRetrieval(rag)
    return rag


def inputs():
    return {"query": "Where was the writer of Work X born?", "query_idx": 0,
            "ids": np.array([0, 1, 2], dtype=np.int32),
            "scores": np.array([.8, .5, .2], dtype=np.float32),
            "ctx": {"hops": 2, "tuple": (1, 2), "set": {3, 4}},
            "state": {"query_idx": 0, "evidence_trace": {
                "plan": [], "bindings": {}, "branch_scores": [], "selected_prefix": []},
                "evidence_candidates": {0: {"base_score": .8, "verified": [], "goals": {}, "sources": []}},
                "_evidence_plan": [], "_evidence_winning_bindings": {}, "_evidence_beams": []}}


class FrozenEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "snapshot"
        self.baseline = Path(self.temporary.name) / "result.json"
        self.rag = rag_fixture()
        self.raw = inputs()

    def capture(self):
        collector = frozen.EvidenceInputCapture()
        item = self.raw
        collector.before_finalize(self.rag.evidence_runtime, item["query"], item["ids"],
                                  item["scores"], item["ctx"], item["state"])
        digest = collector.records[0]["input_sha256"]
        self.baseline.write_text(json.dumps({"runtime_config": asdict(self.rag.pcrag_config),
            "selected_indices": [47], "retrieval_diagnostics": {"total_queries": 1,
                "support_semantic_veto": {"old": True}, "qd_used_count": 1},
            "results": [{"question": item["query"], "benchmark_hops": 2,
                "gold_docs": ["THIS MUST NOT ENTER THE PICKLE"],
                "retrieval_metrics": {"Recall@5": 1.0},
                "retrieval_trace": {"static_sub_questions": ["Who wrote Work X?"],
                    "evidence": {"finalizer_input_hash": digest,
                        "improvement_support_semantic_veto": {"old": True}}}}]}))
        manifest = collector.write(self.directory, self.baseline)
        return collector, manifest

    def test_round_trip_preserves_float_bits_types_and_has_no_gold_or_baseline_ranking(self):
        _collector, manifest = self.capture()
        payload, _ = frozen.load_frozen_inputs(self.directory, [self.raw["query"]],
                                               self.rag.passage_node_keys, [2])
        record = payload["records"][0]
        self.assertEqual(record["scores"].dtype, np.dtype("float32"))
        self.assertEqual(record["scores"].tobytes(), self.raw["scores"].tobytes())
        self.assertEqual(record["ids"].dtype, np.dtype("int32"))
        self.assertEqual(record["ctx"]["tuple"], (1, 2))
        self.assertEqual(record["ctx"]["set"], {3, 4})
        self.assertEqual(list(record["state"]["evidence_candidates"]), [0])
        self.assertNotIn("support_semantic_veto", payload["retrieval_diagnostics"])
        self.assertNotIn("evidence", record["upstream_trace"])
        self.assertNotIn(b"THIS MUST NOT ENTER THE PICKLE", (self.directory / "inputs.pkl").read_bytes())
        self.assertEqual(json.loads(self.baseline.read_text())["frozen_evidence_inputs"]["mode"], "capture")
        self.assertEqual(frozen._digest(self.baseline.read_bytes()), manifest["baseline_result_sha256"])

    def test_capture_deep_copy_isolated_from_later_finalizer_mutation(self):
        collector, _ = self.capture()
        self.raw["ids"][0] = 2
        self.raw["state"]["evidence_trace"]["selected_prefix"].append({"doc_id": 2})
        self.raw["ctx"]["set"].add(9)
        self.assertEqual(collector.records[0]["ids"][0], 0)
        self.assertEqual(collector.records[0]["state"]["evidence_trace"]["selected_prefix"], [])
        self.assertEqual(collector.records[0]["ctx"]["set"], {3, 4})

    def test_real_finalizer_runs_and_review_mutates_real_solutions_once(self):
        self.capture()
        self.rag.evidence_runtime.cfg.evidence_scoring_mode = "dependency_joint"
        calls = []
        def review(runtime, pairs):
            calls.append(pairs)
            self.assertEqual(pairs[0][0].retrieval_trace["static_sub_questions"], ["Who wrote Work X?"])
            self.assertIn("dependency_joint_selection", pairs[0][0].retrieval_trace["evidence"])
            solution, ids = pairs[0]
            solution.docs[0], solution.docs[1] = solution.docs[1], solution.docs[0]
            solution.retrieval_trace["evidence"]["actual_new_review"] = True
            return {"actual": 1}
        solutions = frozen.replay_retrieve(self.rag, self.directory, [self.raw["query"]],
            postprocess=review, solution_class=SimpleNamespace)
        self.assertEqual(len(calls), 1)
        self.assertEqual(solutions[0].docs[0], "Other\nUnrelated text.")
        self.assertTrue(solutions[0].retrieval_trace["evidence"]["actual_new_review"])
        self.assertEqual(self.rag.retrieval_diagnostics["support_semantic_veto"], {"actual": 1})
        self.assertEqual(self.rag.retrieval_diagnostics["qd_used_count"], 1)
        self.assertGreater(self.rag.all_retrieval_time, 0.)

    def test_gold_only_reaches_metric_after_selection_and_review(self):
        self.capture()
        events = []
        def review(_runtime, _pairs):
            events.append("review")
            return {}
        class Metric:
            def __init__(self, global_config):
                pass
            def calculate_metric_scores(self, gold_docs, retrieved_docs, k_list):
                events.append("metrics")
                if gold_docs != [["gold-only-for-metric"]]:
                    raise AssertionError("Metric label lost")
                return {"Recall@5": .25}, []
        _, metric = frozen.replay_retrieve(self.rag, self.directory, [self.raw["query"]],
            gold_docs=[["gold-only-for-metric"]], postprocess=review,
            solution_class=SimpleNamespace, recall_class=Metric)
        self.assertEqual(events, ["review", "metrics"])
        self.assertEqual(metric, {"Recall@5": .25})

    def test_actual_semantic_review_executes_and_does_not_modify_snapshot(self):
        self.capture()
        before = (self.directory / "inputs.pkl").read_bytes()
        review = importlib.import_module(PACKAGE + ".evidence_support_semantic_veto")
        # Empty plans cannot form a source-validated proposal and legitimately
        # abstain without HTTP. The real review still records its invariants.
        solutions = frozen.replay_retrieve(self.rag, self.directory, [self.raw["query"]],
            postprocess=review.postprocess_support_semantic_veto, solution_class=SimpleNamespace)
        diagnostic = solutions[0].retrieval_trace["evidence"]["improvement_support_semantic_veto"]
        self.assertFalse(diagnostic["accepted"])
        self.assertTrue(diagnostic["top200_set_preserved"])
        self.assertEqual(self.rag.retrieval_diagnostics["support_semantic_veto"]["questions"], 1)
        self.assertEqual(before, (self.directory / "inputs.pkl").read_bytes())

    def test_query_index_hops_and_config_mismatch_rejected_before_unpickle(self):
        self.capture()
        with patch.object(frozen.pickle, "loads", side_effect=AssertionError("Unpickle too early")):
            for queries, keys, hops, config in [(["wrong query"], self.rag.passage_node_keys, [2], None),
                    ([self.raw["query"]], list(reversed(self.rag.passage_node_keys)), [2], None),
                    ([self.raw["query"]], self.rag.passage_node_keys, [3], None),
                    ([self.raw["query"]], self.rag.passage_node_keys, None, None),
                    ([self.raw["query"]], self.rag.passage_node_keys, [2], {"max_new_tokens": 1024})]:
                with self.assertRaises(ValueError):
                    frozen.load_frozen_inputs(self.directory, queries, keys, hops, config)

    def test_tampered_payload_rejected_before_unpickle(self):
        self.capture()
        with (self.directory / "inputs.pkl").open("ab") as stream:
            stream.write(b"tampering")
        with patch.object(frozen.pickle, "loads", side_effect=AssertionError("Unpickle too early")):
            with self.assertRaisesRegex(ValueError, "digest"):
                frozen.load_frozen_inputs(self.directory, [self.raw["query"]], self.rag.passage_node_keys, [2])

    def test_changed_baseline_rejected_before_unpickle(self):
        self.capture()
        self.baseline.write_text(self.baseline.read_text() + " ")
        with patch.object(frozen.pickle, "loads", side_effect=AssertionError("Unpickle too early")):
            with self.assertRaisesRegex(ValueError, "Baseline result changed"):
                frozen.load_frozen_inputs(self.directory, [self.raw["query"]], self.rag.passage_node_keys, [2])

    def test_finalizer_and_review_errors_propagate(self):
        self.capture()
        with patch.object(self.rag.evidence_runtime, "finalize", side_effect=RuntimeError("selector failed")):
            with self.assertRaisesRegex(RuntimeError, "selector failed"):
                frozen.replay_retrieve(self.rag, self.directory, [self.raw["query"]],
                    postprocess=lambda *_: {}, solution_class=SimpleNamespace)
        with self.assertRaisesRegex(RuntimeError, "HTTP failed"):
            frozen.replay_retrieve(self.rag, self.directory, [self.raw["query"]],
                postprocess=lambda *_: (_ for _ in ()).throw(RuntimeError("HTTP failed")),
                solution_class=SimpleNamespace)

    def test_labels_futures_and_already_finalized_trace_refused(self):
        for field, value in [("gold_docs", ["label"]), ("future", object())]:
            state = deepcopy(self.raw["state"])
            state["evidence_trace"][field] = value
            with self.assertRaises(ValueError):
                frozen.EvidenceInputCapture().before_finalize(self.rag.evidence_runtime,
                    self.raw["query"], self.raw["ids"], self.raw["scores"], self.raw["ctx"], state)
        for field, value in [("selected_prefix", [{"doc_id": 0}]), ("covered_goals", {"a": 1}),
                             ("finalizer_input_hash", "already finalized")]:
            state = deepcopy(self.raw["state"])
            state["evidence_trace"][field] = value
            with self.assertRaisesRegex(ValueError, "before any"):
                frozen.EvidenceInputCapture().before_finalize(self.rag.evidence_runtime,
                    self.raw["query"], self.raw["ids"], self.raw["scores"], self.raw["ctx"], state)

    def test_evaluation_hooks_restore_method_resolution_after_failure(self):
        original = improvements.ImprovedEvidenceRetrieval.finalize
        had_local = "finalize" in improvements.ImprovedEvidenceRetrieval.__dict__
        class FakeRag:
            def retrieve(self, *args, **kwargs):
                return "normal retrieval"
        module = ModuleType(PACKAGE + ".pathcondrag")
        module.PathCondRAG = FakeRag
        normal_retrieve = FakeRag.retrieve
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(RuntimeError, "experiment failed"):
                with frozen.frozen_evidence_runtime(capture=self.directory) as collector:
                    item = self.raw
                    self.rag.evidence_runtime.finalize(item["query"], item["ids"], item["scores"],
                                                       item["ctx"], item["state"])
                    self.assertEqual(len(collector.records), 1)
                    self.assertEqual(collector.records[0]["state"]["evidence_trace"]["selected_prefix"], [])
                    raise RuntimeError("experiment failed")
            self.assertIs(improvements.ImprovedEvidenceRetrieval.finalize, original)
            self.assertEqual("finalize" in improvements.ImprovedEvidenceRetrieval.__dict__, had_local)
            with frozen.frozen_evidence_runtime(replay=self.directory):
                self.assertIsNot(FakeRag.retrieve, normal_retrieve)
            self.assertIs(FakeRag.retrieve, normal_retrieve)


if __name__ == "__main__":
    unittest.main()
