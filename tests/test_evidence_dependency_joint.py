"""CPU safety and set-selection tests for the opt-in joint objective."""
import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_dependency_joint_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package
joint = importlib.import_module(PACKAGE + ".evidence_dependency_joint")
dag = importlib.import_module(PACKAGE + ".evidence_dag_package")


def fixture():
    documents = {doc_id: f"Page {doc_id}\nOrdinary unrelated text." for doc_id in range(220)}
    nodes = [
        {"id": "a", "question": "Who wrote Work X?", "depends_on": [], "answer_type": "person"},
        {"id": "b", "question": "Where was ${a.answer} born?", "depends_on": ["a"], "answer_type": "place"},
    ]
    bindings = {"a": "Alpha", "b": "Rome"}
    proofs = {
        "a": {"doc_id": 0, "answer": "Alpha", "evidence": "Work X was written by Alpha.", "confidence": .95},
        "b": {"doc_id": 12, "answer": "Rome", "evidence": "Alpha was born in Rome.", "confidence": .95},
    }
    for nid, proof in proofs.items():
        title = "Work X" if nid == "a" else "Alpha"
        documents[proof["doc_id"]] = title + "\n" + proof["evidence"]
    candidates = {doc_id: {"verified": []} for doc_id in documents}
    for nid, proof in proofs.items():
        candidates[proof["doc_id"]]["verified"].append(dict(proof, goal="dag:" + nid))
    state = {"_evidence_plan": nodes, "_evidence_winning_bindings": bindings,
             "_evidence_beams": [{"bindings": bindings, "proofs": proofs}],
             "evidence_candidates": candidates, "evidence_trace": {}}
    return "Where was the writer of Work X born?", documents, state


def add_proof(documents, state, nid, doc_id, quote, title=None):
    title = title or ("Work X" if nid == "a" else "Alpha")
    documents[doc_id] = title + "\n" + quote
    state["evidence_candidates"][doc_id]["verified"].append({
        "doc_id": doc_id, "answer": state["_evidence_winning_bindings"][nid],
        "evidence": quote, "confidence": .95, "goal": "dag:" + nid})


def run(query, documents, state, checker=None, order=None):
    return joint.rerank_dependency_joint(
        query, list(range(220)) if order is None else order,
        state, documents.__getitem__, proof_check=checker,
        signal_ids=list(range(220)), signal_scores=np.linspace(1., 0., 220))


class DependencyJointTests(unittest.TestCase):
    def test_no_proofs_exactly_reproduces_legacy_dag_without_state_mutation(self):
        query, documents, state = fixture()
        state["_evidence_beams"][0]["proofs"] = {}
        for candidate in state["evidence_candidates"].values():
            candidate["verified"] = []
        before = copy.deepcopy(state)
        expected, _ = dag.rerank_dag_packages(list(range(220)), state, documents.__getitem__)
        actual, _, diagnostic = run(query, documents, state)
        self.assertEqual(actual, expected)
        self.assertEqual(state, before)
        self.assertEqual(diagnostic["fallback"], "no_locally_supported_dependency_goals")
        self.assertEqual(diagnostic["extra_llm_requests"], 0)
        self.assertEqual(diagnostic["extra_embedding_calls"], 0)

    def test_root_only_cannot_displace_strong_top2_or_complete_whole_plan(self):
        query, documents, state = fixture()
        state["_evidence_beams"][0]["proofs"].pop("b")
        state["evidence_candidates"][12]["verified"] = []
        # Put the supporting root second, analogous to a correct intermediate
        # relation competing with the page that answers the final question.
        order = [1, 0] + list(range(2, 220))
        actual, _, diagnostic = run(query, documents, state, order=order)
        self.assertEqual(actual, order)
        self.assertEqual(diagnostic["node_denominator"], 2)
        self.assertEqual(diagnostic["terminal_denominator"], 1)
        top5 = diagnostic["prefix_optimization"][0]
        self.assertEqual(top5["node_coverage_after"], .5)
        self.assertEqual(top5["terminal_coverage_after"], 0.)
        self.assertEqual(diagnostic["protected_top5_doc_ids"], [0])

    def test_missing_parent_cannot_give_existing_child_terminal_credit(self):
        query, documents, state = fixture()
        state["_evidence_beams"][0]["proofs"].pop("a")
        state["evidence_candidates"][0]["verified"] = []
        actual, _, diagnostic = run(query, documents, state)
        self.assertEqual(actual, list(range(220)))
        self.assertEqual(diagnostic["prefix_optimization"][0]["coverage_after"], [])

    def test_unknown_final_relation_can_promote_only_when_explicit_checker_supports_it(self):
        query, documents, state = fixture()
        state["_evidence_plan"][1]["question"] = "What nickname was used by ${a.answer}?"
        state["_evidence_plan"][1]["answer_type"] = "name"
        state["_evidence_winning_bindings"]["b"] = "Blue Bird"
        quote = "Alpha used the nickname Blue Bird."
        proof = {"doc_id": 12, "answer": "Blue Bird", "evidence": quote, "confidence": .95}
        state["_evidence_beams"][0]["proofs"]["b"] = proof
        state["evidence_candidates"][12]["verified"] = [dict(proof, goal="dag:b")]
        documents[12] = "Alpha\n" + quote

        def explicit_nickname(node, question, proof, source, bindings):
            if node["id"] == "b" and proof["evidence"] == quote:
                return "explicit_owned_nickname", None
            return dag._proof_relation(node, question, proof, source, bindings)

        expected, _ = dag.rerank_dag_packages(list(range(220)), state, documents.__getitem__)
        actual, _, diagnostic = run(query, documents, state, explicit_nickname)
        self.assertNotIn(12, expected[:5])
        self.assertIn(12, actual[:5])
        self.assertEqual(actual[:2], expected[:2])
        self.assertEqual(diagnostic["prefix_optimization"][0]["complete_terminals_after"], ["b"])
        self.assertEqual(diagnostic["protected_top5_doc_ids"], [0, 12])
        self.assertEqual(set(actual[:200]), set(expected[:200]))
        self.assertEqual(actual[10:], [d for d in expected if d not in actual[:10]])

    def test_all_source_alternatives_retained_after_first_three(self):
        query, documents, state = fixture()
        for doc_id in [6, 7, 8, 9, 15, 30]:
            add_proof(documents, state, "b", doc_id, "Alpha was born in Rome.")
        _, _, diagnostic = run(query, documents, state)
        self.assertEqual({p["doc_id"] for p in diagnostic["eligible_proofs"]["b"]},
                         {6, 7, 8, 9, 12, 15, 30})

    def test_root_not_anchored_to_original_question_cannot_support_new_package(self):
        query, documents, state = fixture()
        actual, _, diagnostic = run("Where was the writer of Unrelated Work born?", documents, state)
        expected, _ = dag.rerank_dag_packages(list(range(220)), state, documents.__getitem__)
        self.assertEqual(actual, expected)
        self.assertNotIn("a", diagnostic["eligible_proofs"])
        self.assertIn("root_subject_not_in_original_question",
                      [item["reason"] for item in diagnostic["rejected_proofs"]])

    def test_cross_branch_requirement_cannot_complete_chain(self):
        query, documents, state = fixture()
        for proof in [state["_evidence_beams"][0]["proofs"]["b"],
                      state["evidence_candidates"][12]["verified"][0]]:
            proof["requirements"] = {"a": "Other Person"}
        actual, _, diagnostic = run(query, documents, state)
        self.assertEqual(actual, list(range(220)))
        self.assertNotIn("b", diagnostic["eligible_proofs"])

    def test_outside_top200_proof_cannot_expand_the_candidate_pool(self):
        query, documents, state = fixture()
        state["_evidence_beams"][0]["proofs"].pop("b")
        state["evidence_candidates"][12]["verified"] = []
        add_proof(documents, state, "b", 205, "Alpha was born in Rome.")
        actual, _, diagnostic = run(query, documents, state)
        self.assertEqual(actual, list(range(220)))
        self.assertNotIn("b", diagnostic["eligible_proofs"])
        self.assertTrue(diagnostic["top200_set_preserved"])

    def test_exact_mask_dp_uses_shared_document_when_prefix_capacity_is_tight(self):
        node_order = ["rootA", "rootB", "leafA", "leafB"]
        ancestors = {"rootA": set(), "rootB": set(), "leafA": {"rootA"}, "leafB": {"rootB"}}
        choices = {"rootA": [{"doc_id": 2}, {"doc_id": 12}],
                   "rootB": [{"doc_id": 12}], "leafA": [{"doc_id": 13}], "leafB": [{"doc_id": 14}]}
        bit, by_doc, closed, mask_for = joint._node_masks(node_order, ancestors, choices)
        actual, diagnostic = joint._optimize_set(
            list(range(30)), 5, 2, node_order, {"rootA", "rootB"}, {"leafA", "leafB"},
            choices, bit, by_doc, closed, mask_for, {})
        # Existing rootA's page is protected, leaving only two free slots. A
        # shared parent page makes one terminal closure possible in those slots.
        self.assertEqual(set(actual[:5]), {0, 1, 2, 12, 13})
        self.assertEqual(diagnostic["complete_terminals_after"], ["leafA"])
        self.assertIn("rootA", diagnostic["coverage_after"])

    def test_top10_optimizes_without_changing_selected_top5(self):
        node_order = ["a", "b", "c"]
        ancestors = {"a": set(), "b": set(), "c": set()}
        choices = {"a": [{"doc_id": 2}], "b": [{"doc_id": 4}], "c": [{"doc_id": 18}]}
        bit, by_doc, closed, mask_for = joint._node_masks(node_order, ancestors, choices)
        actual, diagnostic = joint._optimize_set(
            list(range(30)), 10, 5, node_order, set(node_order), set(node_order),
            choices, bit, by_doc, closed, mask_for, {})
        self.assertEqual(actual[:5], list(range(5)))
        self.assertIn(18, actual[:10])
        self.assertEqual(diagnostic["terminal_coverage_before"], 2 / 3)
        self.assertEqual(diagnostic["terminal_coverage_after"], 1.)

    def test_invalid_branch_and_candidate_collections_fall_back_without_mutation(self):
        for key, value in [("_evidence_beams", 12), ("evidence_candidates", None)]:
            query, documents, state = fixture()
            state[key] = value
            before = copy.deepcopy(state)
            actual, _, diagnostic = run(query, documents, state)
            self.assertEqual(actual, list(range(220)))
            self.assertEqual(state, before)
            self.assertTrue(diagnostic["fallback"].startswith("invalid_"))

    def test_mixin_legacy_mode_unchanged_and_joint_output_scores_monotone(self):
        class Parent:
            def finalize(self, query, ids, scores, context, state):
                return ids, scores, state["evidence_trace"]

        class Engine(dag.DAGPackageMixin, Parent):
            def __init__(self, documents, mode):
                self.improvements, self.stage = {"dag_package"}, 4
                self.cfg = SimpleNamespace(evidence_scoring_mode=mode)
                self._document = documents.__getitem__

        query, documents, state = fixture()
        ids, scores = np.arange(220), np.linspace(1., 0., 220)
        old = Engine(documents, "legacy").finalize(query, ids, scores, {}, copy.deepcopy(state))
        new = Engine(documents, "dependency_joint").finalize(query, ids, scores, {}, copy.deepcopy(state))
        np.testing.assert_array_equal(old[0], new[0])
        np.testing.assert_array_equal(old[1], new[1])
        self.assertTrue(np.all(new[1][:-1] > new[1][1:]))
        self.assertEqual(len(new[0]), len(set(new[0])))
        self.assertTrue(new[2]["dependency_joint_selection"]["top200_set_preserved"])
        self.assertNotIn("dependency_joint_selection", old[2])


if __name__ == "__main__":
    unittest.main()
