import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src/pathcondrag"
PACKAGE = "_planning_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("evidence_retrieval")
planning = load_module("evidence_planning")


class Engine(planning.PlannerImprovementMixin, planning.AdaptiveSearchMixin, base.EvidenceRetrieval):
    def __init__(self, rag, improvements):
        self.improvements = frozenset(improvements)
        super().__init__(rag)


class FakeRAG:
    def __init__(self, responses):
        self.pcrag_config = SimpleNamespace(improvement_stage=4)
        self.responses = iter(responses)
        self.calls = []
        self.passage_node_keys = list(range(8))
        self.documents = {i: "No requested fact is stated in this document." for i in range(8)}
        self.documents[0] = "Alpha directed Film A in 1990."
        self.documents[4] = "Beta directed Film B in 1991."
        self.chunk_embedding_store = SimpleNamespace(get_row=lambda key: {"content": self.documents[key]})
        self.llm_model = SimpleNamespace(infer=self.infer)

    def infer(self, messages, temperature):
        self.calls.append(messages)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value) if isinstance(value, dict) else value, {"finish_reason": "stop"}

    @staticmethod
    def _extract_llm_text(value):
        return str(value)

    def dense_passage_retrieval(self, query):
        return np.arange(8), np.linspace(1.0, 0.1, 8)


def comparison_plan():
    return {"nodes": [
        {"id": "a1", "question": "Who directed Film A?", "depends_on": [], "answer_type": "person"},
        {"id": "a2", "question": "When was ${a1.answer} born?", "depends_on": ["a1"], "answer_type": "date"},
        {"id": "b1", "question": "Who directed Film B?", "depends_on": [], "answer_type": "person"},
        {"id": "b2", "question": "When was ${b1.answer} born?", "depends_on": ["b1"], "answer_type": "date"},
    ]}


def answer(name, doc_id, quote):
    return {"hypotheses": [{"answer": name, "doc_id": f"D{doc_id}",
                            "evidence": quote, "confidence": 0.9}]}


class PlanningImprovementTests(unittest.TestCase):
    def test_two_depth_hint_accepts_four_atomic_parallel_nodes(self):
        rag = FakeRAG([comparison_plan()])
        engine = Engine(rag, {"planning"})
        nodes, reason = engine._plan("Whose director was born earlier, Film A or Film B?", 2)
        self.assertIsNone(reason)
        self.assertEqual(len(nodes), 4)
        diagnostics = next(iter(engine._plan_diagnostics.values()))
        self.assertEqual(diagnostics["depth_hint"], 2)
        self.assertEqual(diagnostics["actual_depth"], 2)
        self.assertEqual(diagnostics["attempts"], 1)
        prompt = str(rag.calls[0])
        self.assertIn("single passage", prompt)
        self.assertIn("Do not compress", prompt)
        self.assertIn("NOT the total number", prompt)
        self.assertNotIn("gold", prompt)

    def test_disabling_planning_preserves_original_node_limit(self):
        rag = FakeRAG([comparison_plan(), comparison_plan()])
        nodes, reason = Engine(rag, set())._plan("Compare directors", 2)
        self.assertEqual(nodes, [])
        self.assertEqual(reason, "invalid_node_count")
        self.assertEqual(len(rag.calls), 2)

    def test_depth_limit_is_independent_from_total_node_capacity(self):
        payload = {"nodes": []}
        for index in range(5):
            deps = [f"s{index}"] if index else []
            question = f"What is ${{s{index}.answer}}?" if index else "Who wrote Work X?"
            payload["nodes"].append({"id": f"s{index+1}", "question": question, "depends_on": deps})
        self.assertEqual(planning.validate_retrieval_plan(payload, 6, 4)[1], "dependency_depth_exceeded")
        self.assertIsNone(planning.validate_retrieval_plan(payload, 6, 5)[1])

    def test_comparison_is_not_an_extra_retrieval_node(self):
        payload = comparison_plan()
        payload["nodes"].append({"id": "final", "question": "Compare ${a2.answer} and ${b2.answer}",
                                 "depends_on": ["a2", "b2"], "answer_type": "comparison"})
        self.assertEqual(planning.validate_retrieval_plan(payload, 6, 4)[1],
                         "derived_comparison_is_not_retrieval")

    def test_planning_repair_and_http_failure_are_bounded(self):
        rag = FakeRAG([{"nodes": []}, comparison_plan()])
        engine = Engine(rag, {"planning"})
        self.assertEqual(len(engine._plan("Compare", 2)[0]), 4)
        self.assertEqual(engine._plan_diagnostics["Compare"]["attempts"], 2)
        rag = FakeRAG([ConnectionError("HTTP unavailable")])
        with self.assertRaisesRegex(ConnectionError, "HTTP unavailable"):
            Engine(rag, {"planning"})._plan("Compare", 2)
        self.assertEqual(len(rag.calls), 1)


class AdaptiveSearchTests(unittest.TestCase):
    def docs(self, engine):
        return engine._verification_documents(list(range(8)))

    def test_successful_first_batch_does_not_spend_extra_request(self):
        rag = FakeRAG([answer("Alpha", 0, "Alpha directed Film A in 1990.")])
        engine = Engine(rag, {"adaptive"})
        accepted, rejected, reason = engine._verify("Who directed Film A?", "person", self.docs(engine))
        self.assertEqual(accepted[0]["answer"], "Alpha")
        self.assertEqual(len(rag.calls), 1)
        diagnostics = next(v for k, v in engine._verification_diagnostics.items() if k.endswith("0,1,2,3,4,5"))
        self.assertEqual(diagnostics["adaptive_extra_batches"], 0)

    def test_missing_fact_checks_only_the_next_three_documents(self):
        rag = FakeRAG([{"hypotheses": []}, answer("Beta", 4, "Beta directed Film B in 1991.")])
        engine = Engine(rag, {"adaptive"})
        accepted, rejected, reason = engine._verify("Who directed Film B?", "person", self.docs(engine))
        self.assertIsNone(reason)
        self.assertEqual(accepted[0]["doc_id"], 4)
        self.assertEqual(len(rag.calls), 2)
        self.assertNotIn("[D4]", str(rag.calls[0]))
        self.assertIn("[D4]", str(rag.calls[1]))
        diagnostic = engine._verification_diagnostics["Who directed Film B?::0,1,2,3,4,5"]
        self.assertEqual(diagnostic["attempts"], 2)
        self.assertEqual(diagnostic["adaptive_extra_batches"], 1)
        self.assertEqual(diagnostic["verified_document_batches"], [[0, 1, 2], [3, 4, 5]])

    def test_empty_fallback_is_not_recursive(self):
        rag = FakeRAG([{"hypotheses": []}, {"hypotheses": []}])
        engine = Engine(rag, {"adaptive"})
        accepted, rejected, reason = engine._verify("Who wrote Missing?", "person", self.docs(engine))
        self.assertEqual(accepted, [])
        self.assertIsNone(reason)
        self.assertEqual(len(rag.calls), 2)

    def test_protocol_or_http_failure_never_triggers_evidence_fallback(self):
        rag = FakeRAG(["invalid JSON", "invalid JSON"])
        engine = Engine(rag, {"adaptive"})
        self.assertEqual(engine._verify("Who?", "person", self.docs(engine))[2], "invalid_json_object")
        self.assertEqual(len(rag.calls), 2)
        rag = FakeRAG([ConnectionError("HTTP failed")])
        engine = Engine(rag, {"adaptive"})
        with self.assertRaisesRegex(ConnectionError, "HTTP failed"):
            engine._verify("Who?", "person", self.docs(engine))
        self.assertEqual(len(rag.calls), 1)

    def test_two_beams_only_when_distinct_verified_scores_are_close(self):
        engine = Engine(FakeRAG([]), {"adaptive"})
        beams = [
            {"bindings": {"s1": "Alpha"}, "proofs": {}, "qualities": [.9]},
            {"bindings": {"s1": "Beta"}, "proofs": {}, "qualities": [.88]},
            {"bindings": {"s1": "Gamma"}, "proofs": {}, "qualities": [.4]},
        ]
        self.assertEqual(len(engine._prune_beams(beams, 2)), 2)
        self.assertEqual(len(engine._prune_beams([beams[0], beams[2]], 2)), 1)
        duplicate = dict(beams[0])
        self.assertEqual(len(engine._prune_beams([beams[0], duplicate], 2)), 1)
        self.assertEqual(engine.max_searches, 20)

    def test_disabling_adaptive_keeps_three_documents_and_one_beam(self):
        engine = Engine(FakeRAG([]), {"planning"})
        self.assertEqual(list(self.docs(engine)), [0, 1, 2])
        self.assertEqual(engine.beam_width, 1)


if __name__ == "__main__":
    unittest.main()
