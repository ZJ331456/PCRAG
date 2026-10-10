"""CPU checks for scoring safety, provenance, and fixed-candidate controls."""
import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_dependency_scoring_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package
base = importlib.import_module(PACKAGE + ".evidence_retrieval")
scorer = importlib.import_module(PACKAGE + ".evidence_dependency_scoring")
config_module = importlib.import_module(PACKAGE + ".config")


class FakeRAG:
    def __init__(self, mode="dependency", count=205):
        self.pcrag_config = SimpleNamespace(
            improvement_stage=4, evidence_budget=5, evidence_scoring_mode=mode)
        self.passage_node_keys = list(range(count))
        self.documents = {i: f"Page {i}\nAn unrelated passage about ordinary topics."
                          for i in range(count)}
        self.documents[7] = "Work X\nWork X was written by Alpha."
        self.documents[8] = "Alpha\nAlpha was born in Rome."
        self.chunk_embedding_store = SimpleNamespace(
            get_row=lambda key: {"content": self.documents[key]})

    def dense_passage_retrieval(self, query):
        raise AssertionError("The scorer must not retrieve documents")


def fixture():
    plan = [{"id": "s1", "question": "Who wrote Work X?", "depends_on": [],
             "answer_type": "person"},
            {"id": "s2", "question": "Where was ${s1.answer} born?", "depends_on": ["s1"],
             "answer_type": "place"}]
    proofs = {"s1": {"doc_id": 7, "answer": "Alpha",
                     "evidence": "Work X was written by Alpha.", "confidence": .95, "quality": .94},
              "s2": {"doc_id": 8, "answer": "Rome",
                     "evidence": "Alpha was born in Rome.", "confidence": .95, "quality": .94}}
    bindings = {"s1": "Alpha", "s2": "Rome"}
    candidates = {i: {"doc_id": i, "base_score": .9, "sources": [], "verified": []}
                  for i in range(12)}
    for nid, proof in proofs.items():
        candidates[proof["doc_id"]]["verified"] = [dict(proof, goal="dag:" + nid)]
    candidates[6]["sources"] = [{"goal": "static:wrong", "score": .99, "rank": 1,
                                 "question": "Who directed Unrelated Movie?", "requirements": {}}]
    return {"query": "Where was the writer of Work X born?", "evidence_candidates": candidates,
            "_evidence_plan": plan, "_evidence_winning_bindings": bindings,
            "_evidence_beams": [{"bindings": bindings, "proofs": proofs}],
            "evidence_trace": {"plan": plan, "bindings": bindings}}


class DependencyScoringTests(unittest.TestCase):
    def test_default_legacy_ranking_and_scores_remain_exact(self):
        rag, state = FakeRAG("legacy"), fixture()
        engine = base.EvidenceRetrieval(rag)
        ids, scores = np.arange(205), np.linspace(1., .01, 205)
        expected = engine._finalize_legacy(state["query"], ids, scores, {}, copy.deepcopy(state))
        actual = engine.finalize(state["query"], ids, scores, {}, copy.deepcopy(state))
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2]["selected_prefix"], expected[2]["selected_prefix"])
        self.assertEqual(len(actual[2]["finalizer_input_hash"]), 64)

    def test_frozen_control_top200_and_tail_preserved(self):
        ids, scores, state = np.arange(205), np.linspace(1., .01, 205), fixture()
        engine = base.EvidenceRetrieval(FakeRAG())
        legacy = engine._finalize_legacy(state["query"], ids, scores, {}, copy.deepcopy(state))
        actual = engine.finalize(state["query"], ids, scores, {}, copy.deepcopy(state))
        self.assertEqual(set(actual[0][:200]), set(legacy[0][:200]))
        selected = set(actual[0][:5])
        self.assertEqual(actual[0][5:].tolist(), [int(d) for d in legacy[0] if d not in selected])
        self.assertTrue(np.all(actual[1][:-1] >= actual[1][1:]))
        self.assertTrue(actual[2]["dependency_scoring"]["top200_set_preserved"])

    def test_same_upstream_inputs_have_same_fingerprint_in_both_modes(self):
        state, ids, scores = fixture(), np.arange(205), np.linspace(1., .01, 205)
        outputs = [base.EvidenceRetrieval(FakeRAG(mode)).finalize(
            state["query"], ids, scores, {}, copy.deepcopy(state)) for mode in ["legacy", "dependency"]]
        self.assertEqual(outputs[0][2]["finalizer_input_hash"], outputs[1][2]["finalizer_input_hash"])

    def test_actual_parent_support_controls_child_coverage(self):
        rag, state = FakeRAG(), fixture()
        ids, scores, details, diag = scorer.dependency_scored_prefix(
            state["query"], np.arange(205), np.linspace(1., .01, 205), state,
            rag.pcrag_config, lambda d: rag.documents[d])
        self.assertEqual(ids[:2].tolist(), [7, 8])
        self.assertEqual(details[0]["new_supported_nodes"], ["s1"])
        self.assertEqual(details[1]["new_supported_nodes"], ["s2"])
        self.assertEqual(details[1]["redundancy"], 0.0)
        self.assertEqual(diag["covered_nodes"], ["s1", "s2"])
        self.assertNotIn(6, ids[:5])

    def test_overlapping_routes_do_not_penalize_parent_child_proof_documents(self):
        rag, state = FakeRAG(), fixture()
        for doc_id in [7, 8]:
            state["evidence_candidates"][doc_id]["sources"] = [
                {"goal": "dag:s1", "score": .93, "rank": 1,
                 "question": "Who wrote Work X?", "requirements": {}},
                {"goal": "dag:s2", "score": .93, "rank": 1,
                 "question": "Where was Alpha born?", "requirements": {"s1": "Alpha"}},
            ]
        ids, _, details, _ = scorer.dependency_scored_prefix(
            state["query"], np.arange(205), np.linspace(1., .01, 205), state,
            rag.pcrag_config, lambda d: rag.documents[d])
        self.assertEqual(ids[:2].tolist(), [7, 8])
        self.assertGreater(len(set(base.normalized_text(rag.documents[7]).split())
                               & set(base.normalized_text(rag.documents[8]).split())), 0)
        self.assertEqual(details[1]["redundancy"], 0.0)

    def test_malformed_plan_questions_and_winning_proofs_fall_back(self):
        ids, scores = np.arange(205), np.linspace(1., .01, 205)
        mutations = [
            ("missing_question", "invalid_plan_question"),
            ("null_question", "invalid_plan_question"),
            ("empty_question", "invalid_plan_question"),
            ("missing_proofs", "invalid_winning_proofs"),
            ("null_proofs", "invalid_winning_proofs"),
        ]
        for kind, expected_reason in mutations:
            with self.subTest(kind=kind):
                state = fixture()
                if kind == "missing_question":
                    state["_evidence_plan"][0].pop("question")
                elif kind == "null_question":
                    state["_evidence_plan"][0]["question"] = None
                elif kind == "empty_question":
                    state["_evidence_plan"][0]["question"] = " "
                elif kind == "missing_proofs":
                    state["_evidence_beams"][0].pop("proofs")
                else:
                    state["_evidence_beams"][0]["proofs"] = None
                rag = FakeRAG()
                engine = base.EvidenceRetrieval(rag)
                old = engine._finalize_legacy(state["query"], ids, scores, {}, copy.deepcopy(state))
                new = engine.finalize(state["query"], ids, scores, {}, copy.deepcopy(state))
                np.testing.assert_array_equal(new[0], old[0])
                np.testing.assert_array_equal(new[1], old[1])
                self.assertEqual(new[2]["dependency_scoring"]["plan_error"], expected_reason)
                self.assertEqual(new[2]["dependency_scoring"]["fallback"],
                                 "no_locally_supported_dependency_goals")

    def test_wrong_relation_and_missing_ancestor_fall_back_to_legacy(self):
        rag, state = FakeRAG(), fixture()
        rag.documents[7] = "Work X\nWork X was mentioned by Alpha."
        wrong = "Work X was mentioned by Alpha."
        state["_evidence_beams"][0]["proofs"]["s1"]["evidence"] = wrong
        state["evidence_candidates"][7]["verified"][0]["evidence"] = wrong
        ids, scores = np.arange(205), np.linspace(1., .01, 205)
        engine = base.EvidenceRetrieval(rag)
        expected = engine._finalize_legacy(state["query"], ids, scores, {}, copy.deepcopy(state))
        actual = engine.finalize(state["query"], ids, scores, {}, copy.deepcopy(state))
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[2]["dependency_scoring"]["fallback"], "no_locally_supported_dependency_goals")
        reasons = [row["reason"] for row in actual[2]["dependency_scoring"]["rejected_proofs"]]
        self.assertIn("missing_reliable_ancestor", reasons)

    def test_confidence_does_not_change_rewards_and_off_topic_routes_cannot_cover(self):
        rag, state = FakeRAG(), fixture()
        original = copy.deepcopy(state)
        for proof in state["_evidence_beams"][0]["proofs"].values():
            proof["confidence"] = .1
        for candidate in state["evidence_candidates"].values():
            for proof in candidate["verified"]:
                proof["confidence"] = .1
        args = (np.arange(205), np.linspace(1., .01, 205))
        old = scorer.dependency_scored_prefix(original["query"], *args, original,
                                              rag.pcrag_config, lambda d: rag.documents[d])
        new = scorer.dependency_scored_prefix(state["query"], *args, state,
                                              rag.pcrag_config, lambda d: rag.documents[d])
        np.testing.assert_array_equal(new[0], old[0])
        self.assertEqual(new[3]["merged_goal_count"], 2)
        self.assertEqual(new[3]["weak_route_documents"], 0)

    def test_same_subject_relation_routes_merge_without_merging_distinct_subjects(self):
        self.assertEqual(scorer._goal("Who wrote Work X?"), scorer._goal("Who is the author of Work X?"))
        self.assertNotEqual(scorer._goal("Who wrote Work X?"), scorer._goal("Who wrote Work Y?"))
        self.assertNotEqual(scorer._goal("Who wrote Work X?"), scorer._goal("Who directed Work X?"))

    def test_verified_proofs_are_not_forced_into_top5(self):
        rag, state = FakeRAG(), fixture()
        for nid, old_id, new_id in [("s1", 7, 198), ("s2", 8, 199)]:
            rag.documents[new_id] = rag.documents[old_id]
            candidate = state["evidence_candidates"].pop(old_id)
            candidate["doc_id"] = new_id
            candidate["verified"][0]["doc_id"] = new_id
            state["evidence_candidates"][new_id] = candidate
            state["_evidence_beams"][0]["proofs"][nid]["doc_id"] = new_id
        ids, _, _, diag = scorer.dependency_scored_prefix(
            state["query"], np.arange(205), np.linspace(1., .01, 205), state,
            rag.pcrag_config, lambda d: rag.documents[d])
        self.assertEqual(ids[:5].tolist(), list(range(5)))
        self.assertEqual(diag["covered_nodes"], [])

    def test_gold_fields_are_not_fingerprinted_or_read(self):
        state, ids, scores = fixture(), np.arange(205), np.linspace(1., .01, 205)
        old = scorer.scoring_input_sha256(state["query"], ids, scores, state)
        state["gold_answers"] = object()
        state["question_decomposition"] = object()
        self.assertEqual(old, scorer.scoring_input_sha256(state["query"], ids, scores, state))
        rag = FakeRAG()
        scorer.dependency_scored_prefix(state["query"], ids, scores, state,
                                        rag.pcrag_config, lambda d: rag.documents[d])

    def test_config_rejects_unsupported_modes_and_stages(self):
        with self.assertRaises(ValueError):
            config_module.PCRAGConfig(evidence_scoring_mode="unknown")
        with self.assertRaises(ValueError):
            config_module.PCRAGConfig(evidence_scoring_mode="dependency", improvement_stage=3)
        cfg = config_module.PCRAGConfig(evidence_scoring_mode="dependency", improvement_stage=4)
        self.assertEqual(cfg.evidence_scoring_mode, "dependency")


if __name__ == "__main__":
    unittest.main()
