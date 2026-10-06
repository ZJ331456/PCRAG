"""CPU integration checks for composed evidence retrieval improvements."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_evidence_improvements_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name,
                                                  ROOT / "src/pathcondrag" / (name.replace(".", "/") + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load("evidence_retrieval")
binding = load("evidence_binding")
planning = load("evidence_planning")
selection = load("evidence_selection")
improved = load("evidence_improvements")
config = load("config")


class FakeRAG:
    def __init__(self, flags="", responses=()):
        self.pcrag_config = config.PCRAGConfig(improvement_stage=4, evidence_improvements=flags)
        self.responses = iter(responses)
        self.calls = []
        self.documents = ["Work X\nAlpha wrote Work X in 1990.",
                          "Alpha\nAlpha was born in Rome in 1960.",
                          "Other A\nAn unrelated passage.", "Other B\nAnother unrelated passage.",
                          "Sergio Nasca\nSergio Nasca was an Italian film director.",
                          "Other C\nMore irrelevant text."]
        self.passage_node_keys = [f"d{i}" for i in range(len(self.documents))]
        self.chunk_embedding_store = SimpleNamespace(get_row=lambda key: {"content": self.documents[int(key[1:])]})
        self.llm_model = SimpleNamespace(infer=self.infer)
        self.searches = []

    def infer(self, messages, temperature):
        self.calls.append(copy.deepcopy(messages))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return (json.dumps(result) if isinstance(result, dict) else result, {"finish_reason": "stop"})

    @staticmethod
    def _extract_llm_text(response):
        return str(response)

    def dense_passage_retrieval(self, question):
        self.searches.append(question)
        order = [1, 0, 2, 3, 4, 5] if "Alpha" in question else [0, 2, 1, 3, 4, 5]
        return np.asarray(order), np.asarray([.9, .8, .7, .6, .5, .4])


PLAN = {"nodes": [{"id": "s1", "question": "Who wrote Work X?", "depends_on": [], "answer_type": "person"},
                  {"id": "s2", "question": "Where was ${s1.answer} born?", "depends_on": ["s1"], "answer_type": "place"}]}
ALPHA = {"hypotheses": [{"answer": "Alpha", "doc_id": "D0", "evidence": "Alpha wrote Work X in 1990.", "confidence": .9}]}
ROME = {"hypotheses": [{"answer": "Rome", "doc_id": "D1", "evidence": "Alpha was born in Rome in 1960.", "confidence": .9}]}
TYPED = {"hypotheses": [{"answer": "Italian", "doc_id": "D4", "evidence": "Sergio Nasca was an Italian film director.",
                          "confidence": .9, "answer_entity": "Sergio Nasca", "answer_relation": "nationality",
                          "subject_evidence": "Sergio Nasca was an Italian film director."}]}


def state():
    return {"query_idx": 0, "query": "Where was the writer of Work X born?",
            "base": (np.arange(6), np.asarray([.9, .8, .7, .6, .5, .4]), {"hops": 2}),
            "static_sub_questions": [], "pcqd_sub_questions": []}


class ImprovementIntegrationTests(unittest.TestCase):
    def test_disabled_all_exact_original_replay(self):
        left, right = FakeRAG(responses=[PLAN, ALPHA, ROME]), FakeRAG(responses=[PLAN, ALPHA, ROME])
        original, composed = base.EvidenceRetrieval(left), improved.ImprovedEvidenceRetrieval(right)
        a, b = state(), state()
        original.process_window([a])
        composed.process_window([b])
        self.assertEqual(left.calls, right.calls)
        self.assertEqual(left.searches, right.searches)
        self.assertEqual(a["evidence_trace"], b["evidence_trace"])
        result_a = original.finalize(a["query"], *a["base"], a)
        result_b = composed.finalize(b["query"], *b["base"], b)
        np.testing.assert_array_equal(result_a[0], result_b[0])
        np.testing.assert_array_equal(result_a[1], result_b[1])
        self.assertEqual(result_a[2], result_b[2])

    def test_adaptive_fallback_uses_binding_on_extra_documents(self):
        rag = FakeRAG("adaptive,binding", [{"hypotheses": []}, TYPED])
        engine = improved.ImprovedEvidenceRetrieval(rag)
        docs = engine._verification_documents(list(range(6)))
        accepted, rejected, reason = engine._verify("What nationality was Sergio Nasca?", "nationality", docs)
        self.assertIsNone(reason)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["doc_id"], 4)
        self.assertTrue(accepted[0]["relation_supported"])
        self.assertEqual(len(rag.calls), 2)
        self.assertIn("answer_entity", rag.calls[0][0]["content"])
        self.assertIn("answer_entity", rag.calls[1][0]["content"])
        diag = engine._verification_diagnostics["What nationality was Sergio Nasca?::0,1,2,3,4,5"]
        self.assertEqual(diag["attempts"], 2)
        self.assertEqual(diag["verified_document_batches"], [[0, 1, 2], [3, 4, 5]])

    def test_extra_batch_rejects_typed_relation_error(self):
        bad = copy.deepcopy(TYPED)
        bad["hypotheses"][0]["answer_relation"] = "birth_date"
        rag = FakeRAG("adaptive,binding", [{"hypotheses": []}, bad, {"hypotheses": []}])
        engine = improved.ImprovedEvidenceRetrieval(rag)
        accepted, rejected, _ = engine._verify("What nationality was Sergio Nasca?", "nationality",
                                             engine._verification_documents(list(range(6))))
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "requested_relationship_mismatch")
        self.assertEqual(len(rag.calls), 3)

    def test_supported_first_batch_does_not_trigger_extra(self):
        typed = copy.deepcopy(TYPED)
        typed["hypotheses"][0]["doc_id"] = "D0"
        rag = FakeRAG("adaptive,binding", [typed])
        rag.documents[0] = rag.documents[4]
        engine = improved.ImprovedEvidenceRetrieval(rag)
        accepted, _, _ = engine._verify("What nationality was Sergio Nasca?", "nationality",
                                       engine._verification_documents(list(range(6))))
        self.assertTrue(accepted)
        self.assertEqual(len(rag.calls), 1)

    def test_invalid_protocol_does_not_spend_fallback_budget(self):
        rag = FakeRAG("adaptive,binding", ["not json", "still not json"])
        engine = improved.ImprovedEvidenceRetrieval(rag)
        accepted, _, reason = engine._verify("Question?", "person", engine._verification_documents(list(range(6))))
        self.assertFalse(accepted)
        self.assertEqual(reason, "invalid_json_object")
        self.assertEqual(len(rag.calls), 2)

    def test_http_failure_propagates_through_composed_verifier(self):
        rag = FakeRAG("adaptive,binding", [ConnectionError("HTTP unavailable")])
        engine = improved.ImprovedEvidenceRetrieval(rag)
        with self.assertRaisesRegex(ConnectionError, "HTTP unavailable"):
            engine._verify("Question?", "person", engine._verification_documents(list(range(6))))

    def test_document_budget_and_all_flag_composition(self):
        ordinary = improved.ImprovedEvidenceRetrieval(FakeRAG())
        upgraded = improved.ImprovedEvidenceRetrieval(FakeRAG("planning,selection,binding,closure,adaptive"))
        self.assertEqual(len(ordinary._verification_documents(list(range(6)))), 3)
        self.assertEqual(len(upgraded._verification_documents(list(range(6)))), 6)
        self.assertEqual(ordinary.beam_width, 1)
        self.assertEqual(upgraded.beam_width, 2)
        self.assertEqual(upgraded.improvements, frozenset({"planning", "selection", "binding", "closure", "adaptive"}))

    def test_config_validation_and_normalized_flags(self):
        valid = config.PCRAGConfig(improvement_stage=4, evidence_improvements="binding, planning,binding")
        self.assertEqual(valid.evidence_improvements, "binding,planning")
        for kwargs in [dict(improvement_stage=4, evidence_improvements="fake"),
                       dict(improvement_stage=0, evidence_improvements="binding"),
                       dict(improvement_stage=4, evidence_improvements="binding", evidence_ablation_mode="validation"),
                       dict(evidence_plan_node_budget=9), dict(evidence_plan_depth_budget=5),
                       dict(evidence_selection_top_k=21), dict(evidence_adaptive_mode="unlimited"),
                       dict(evidence_plan_validation="guess_refs"),
                       dict(evidence_binding_validation="unchecked"),
                       dict(evidence_binding_validation="strict_relation")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                config.PCRAGConfig(**kwargs)

    def test_all_flags_never_read_answer_or_support_annotations(self):
        protected = {"gold_docs", "gold_answers", "answer", "supporting_facts", "question_decomposition", "type"}

        class GuardedState(dict):
            def __getitem__(self, key):
                if key in protected:
                    raise AssertionError("Algorithm read a benchmark annotation")
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key in protected:
                    raise AssertionError("Algorithm read a benchmark annotation")
                return super().get(key, default)

        alpha = copy.deepcopy(ALPHA)
        alpha["hypotheses"][0].update(answer_entity="Work X", answer_relation="author",
                                       subject_evidence="Alpha wrote Work X in 1990.")
        rome = copy.deepcopy(ROME)
        rome["hypotheses"][0].update(answer_entity="Alpha", answer_relation="birth_place",
                                      subject_evidence="Alpha was born in Rome in 1960.")
        rag = FakeRAG("planning,selection,binding,closure,adaptive", [PLAN, alpha, rome])
        engine = improved.ImprovedEvidenceRetrieval(rag)
        guarded = GuardedState(state())
        for key in protected:
            guarded[key] = "forbidden_annotation"
        engine.process_window([guarded])
        ids, _, trace = engine.finalize(guarded["query"], *guarded["base"], guarded)
        self.assertEqual(trace["bindings"], {"s1": "Alpha", "s2": "Rome"})
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("forbidden_annotation", str(rag.calls))


if __name__ == "__main__":
    unittest.main()
