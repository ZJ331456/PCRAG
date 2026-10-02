"""CPU checks for binding controls and proof-closed prefix selection."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("selection_ablation_test_package")
PACKAGE.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + ".ablation_selection", ROOT / "src/pathcondrag/ablation_selection.py")
selection = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selection
SPEC.loader.exec_module(selection)


def candidate(doc_id, score=0.0, verified=()):
    return {"doc_id": doc_id, "base_score": score, "sources": [], "verified": list(verified)}


def proof(doc_id, answer, goal=None):
    value = {"doc_id": doc_id, "answer": answer, "quality": 0.9,
             "confidence": 0.9, "evidence": "An exact supporting quotation."}
    if goal is not None:
        value["goal"] = "dag:" + goal
    return value


class BindingControlsTest(unittest.TestCase):
    def setUp(self):
        self.documents = {1: "Alpha wrote Work X in 1990.",
                          2: "Beta was born in Paris in 1958."}
        self.hypothesis = {"doc_id": 1, "answer": "Alpha", "confidence": 0.9,
                           "evidence": "Alpha wrote Work X in 1990."}

    def test_string_control_accepts_ungrounded_answer_without_quote(self):
        payload = {"hypotheses": [{"doc_id": "D1", "answer": "Invented Person", "confidence": 0.8}]}
        accepted, rejected = selection.verify_string_hypotheses(payload, self.documents)
        self.assertEqual(accepted[0]["answer"], "Invented Person")
        self.assertEqual(accepted[0]["evidence"], "")
        self.assertFalse(rejected)
        literal, _ = selection.verify_hypotheses(payload, self.documents)
        self.assertFalse(literal)

    def test_string_control_rejects_missing_document_empty_answer_and_nonfinite(self):
        payload = {"hypotheses": [
            {"doc_id": "D99", "answer": "Alpha"},
            {"doc_id": 1, "answer": "   "},
            {"doc_id": 1, "answer": "Alpha", "confidence": float("nan")},
            {"doc_id": 1, "answer": "Alpha", "confidence": float("inf")},
            {"doc_id": True, "answer": "Alpha"}]}
        accepted, rejected = selection.verify_string_hypotheses(payload, self.documents)
        self.assertFalse(accepted)
        self.assertEqual(len(rejected), 5)

    def test_string_and_literal_controls_share_answer_syntax_checks(self):
        for answer, confidence in [
            ("x" * 201, 0.9), ("unknown", 0.9), ("${s1.answer}", 0.9),
            ("entity-" + "a" * 32, 0.9), ("Alpha", 0.1), ("Alpha", -0.1)]:
            with self.subTest(answer=answer, confidence=confidence):
                payload = {"hypotheses": [dict(self.hypothesis, answer=answer, confidence=confidence)]}
                string, rejected = selection.verify_string_hypotheses(payload, self.documents)
                literal, _ = selection.verify_hypotheses(payload, self.documents)
                self.assertFalse(string)
                self.assertFalse(literal)
                self.assertTrue(rejected)

    def verdict(self, **updates):
        return dict(self.hypothesis, label="entailed", relation_supported=True, **updates)

    def test_relation_requires_matching_literal_quote_and_explicit_entailment(self):
        accepted, rejected = selection.verify_relation_verdicts(
            {"verdicts": [self.verdict()]}, [self.hypothesis], self.documents)
        self.assertEqual([item["answer"] for item in accepted], ["Alpha"])
        self.assertTrue(accepted[0]["relation_supported"])
        self.assertFalse(rejected)

    def test_relation_rejects_missing_changed_and_conflicting_claims(self):
        changed_doc = self.verdict()
        changed_doc["doc_id"] = 2
        changed_answer = self.verdict()
        changed_answer["answer"] = "Beta"
        changed_quote = self.verdict()
        changed_quote["evidence"] = "Alpha wrote Work X"
        denied = self.verdict()
        denied["label"] = "contradicted"
        string_boolean = self.verdict()
        string_boolean["relation_supported"] = "true"
        cases = [[], [changed_doc], [changed_answer], [changed_quote], [denied],
                 [self.verdict(), denied], [string_boolean]]
        for verdicts in cases:
            with self.subTest(verdicts=verdicts):
                accepted, rejected = selection.verify_relation_verdicts(
                    {"verdicts": verdicts}, [self.hypothesis], self.documents)
                self.assertFalse(accepted)
                self.assertTrue(rejected)

    def test_relation_cannot_rescue_nonliteral_evidence(self):
        bad = dict(self.hypothesis, evidence="Alpha wrote some other imaginary book.")
        accepted, rejected = selection.verify_relation_verdicts(
            {"verdicts": [dict(bad, label="entailed", relation_supported=True)]},
            [bad], self.documents)
        self.assertFalse(accepted)
        self.assertIn("evidence_not_exact_substring", [item["reason"] for item in rejected])


class ProofSelectionTest(unittest.TestCase):
    def setUp(self):
        self.plan = [{"id": "s1", "depends_on": []}, {"id": "s2", "depends_on": ["s1"]}]
        self.proofs = {"s1": proof(1, "Alpha"), "s2": proof(2, "Rome")}
        self.bindings = {"s1": "Alpha", "s2": "Rome"}
        self.candidates = {
            1: candidate(1, 0.1, [proof(1, "Alpha", "s1")]),
            2: candidate(2, 1.0, [proof(2, "Rome", "s2")]),
            3: candidate(3, 0.5), 4: candidate(4, 0.02)}

    def select(self, **updates):
        arguments = dict(candidates=self.candidates, plan=self.plan, proofs=self.proofs,
                         goal_scores={}, document_tokens={}, bindings=self.bindings,
                         budget=2, policy="closure", base_weight=1.0,
                         relation_weight=0.0, coverage_weight=0.0, redundancy_weight=0.0)
        arguments.update(updates)
        return selection.select_evidence_prefix(**arguments)

    def test_ancestor_support_is_selected_before_successor(self):
        selected, diagnostic = self.select()
        self.assertEqual(selected, [1, 2])
        self.assertTrue(diagnostic["proof_closed"])
        self.assertEqual(diagnostic["selected_prefix"][1]["protected_nodes"], ["s2"])

    def test_tight_budget_excludes_child_without_parent(self):
        selected, diagnostic = self.select(budget=1)
        self.assertEqual(selected, [3])
        self.assertTrue(diagnostic["proof_closed"])
        ordinary, _ = self.select(budget=1, policy="coverage")
        self.assertEqual(ordinary, [2])

    def test_missing_ancestor_proof_and_conflicting_binding_are_blocked(self):
        selected, diagnostic = self.select(proofs={"s2": self.proofs["s2"]})
        self.assertNotIn(2, selected)
        self.assertIn(2, diagnostic["blocked_proof_document_ids"])
        conflicting = dict(self.bindings, s1="Someone Else")
        selected, diagnostic = self.select(bindings=conflicting)
        self.assertTrue({1, 2}.isdisjoint(selected))
        self.assertIn("proof_answer_conflicts_with_binding",
                      [item["reason"] for item in diagnostic["rejected_proofs"]])

    def test_incompatible_candidate_proof_cannot_bypass_constraint(self):
        candidates = dict(self.candidates)
        candidates[2] = candidate(2, 1.0, [proof(2, "Paris", "s2")])
        selected, diagnostic = self.select(candidates=candidates)
        self.assertNotIn(2, selected)
        self.assertIn("candidate_proof_incompatible_with_winning_branch",
                      [item["reason"] for item in diagnostic["rejected_proofs"]])

    def test_two_parents_must_fit_budget_together(self):
        plan = [{"id": "s1", "depends_on": []}, {"id": "s2", "depends_on": []},
                {"id": "s3", "depends_on": ["s1", "s2"]}]
        proofs = {"s1": proof(1, "A"), "s2": proof(2, "B"), "s3": proof(3, "C")}
        candidates = {1: candidate(1, 0.1), 2: candidate(2, 0.1),
                      3: candidate(3, 1.0), 4: candidate(4, 0.3)}
        selected, _ = self.select(plan=plan, proofs=proofs, candidates=candidates,
                                  bindings={"s1": "A", "s2": "B", "s3": "C"}, budget=2)
        self.assertNotIn(3, selected)
        selected, _ = self.select(plan=plan, proofs=proofs, candidates=candidates,
                                  bindings={"s1": "A", "s2": "B", "s3": "C"}, budget=3)
        self.assertEqual(selected, [1, 2, 3])

    def test_one_document_can_support_parent_and_child_without_double_cost(self):
        proofs = {"s1": proof(1, "Alpha"), "s2": proof(1, "Rome")}
        selected, diagnostic = self.select(proofs=proofs, candidates={1: candidate(1, 1.0)}, budget=1)
        self.assertEqual(selected, [1])
        self.assertTrue(diagnostic["proof_closed"])

    def test_cyclic_or_unknown_plan_is_explicitly_rejected(self):
        invalid_plans = [
            [{"id": "s1", "depends_on": ["s2"]}, {"id": "s2", "depends_on": ["s1"]}],
            [{"id": "s1", "depends_on": ["missing"]}]]
        for plan in invalid_plans:
            with self.subTest(plan=plan):
                selected, diagnostic = self.select(plan=plan)
                self.assertFalse(selected)
                self.assertEqual(diagnostic["status"], "invalid_plan")

    def test_joint_replacement_improves_same_coverage_objective(self):
        candidates = {doc_id: candidate(doc_id) for doc_id in range(3)}
        goals = {0: {"A": 0.8, "B": 0.8}, 1: {"A": 1.0, "C": 0.5},
                 2: {"B": 1.0, "C": 0.5}}
        ordinary, baseline = self.select(candidates=candidates, plan=[], proofs={}, bindings={},
                                         goal_scores=goals, base_weight=0.0, coverage_weight=1.0)
        improved, diagnostic = self.select(candidates=candidates, plan=[], proofs={}, bindings={},
                                           goal_scores=goals, base_weight=0.0,
                                           coverage_weight=1.0, policy="joint")
        self.assertEqual(ordinary, [0, 1])
        self.assertEqual(set(improved), {1, 2})
        self.assertGreater(diagnostic["objective_value"], baseline["objective_value"])
        self.assertFalse(diagnostic["global_optimum_guaranteed"])
        self.assertTrue(diagnostic["bindings_fixed"])
        self.assertFalse(diagnostic["binding_optimization"])

    def test_joint_is_closed_and_retains_baseline_for_small_budgets(self):
        for budget in range(6):
            with self.subTest(budget=budget):
                _, baseline = self.select(budget=budget)
                result, joint = self.select(budget=budget, policy="joint")
                self.assertLessEqual(len(result), budget)
                self.assertEqual(len(set(result)), len(result))
                self.assertTrue(joint["proof_closed"])
                self.assertGreaterEqual(joint["objective_value"] + 1e-12, baseline["objective_value"])
                self.assertLessEqual(joint["proof_mask_count"], 16)

    def test_score_override_and_empty_candidates_work(self):
        selected, _ = self.select(policy="coverage", base_scores={1: 2.0, 2: 0.0, 3: 0.1, 4: 0.0})
        self.assertEqual(selected[0], 1)
        selected, diagnostic = self.select(candidates={}, plan=[], proofs={}, bindings={}, policy="joint")
        self.assertFalse(selected)
        self.assertEqual(diagnostic["objective_value"], 0.0)

    def test_fixed_goal_denominator_is_shared_across_branch_scores(self):
        candidates = {1: candidate(1), 2: candidate(2)}
        first_goals = {1: {"A": 1.0}, 2: {}}
        second_goals = {1: {"A": 1.0}, 2: {"B": 1.0}}
        _, first = self.select(candidates=candidates, plan=[], proofs={}, bindings={},
                               goal_scores=first_goals, base_weight=0.0, coverage_weight=1.0,
                               budget=1, goal_total=2)
        _, second = self.select(candidates=candidates, plan=[], proofs={}, bindings={},
                                goal_scores=second_goals, base_weight=0.0, coverage_weight=1.0,
                                budget=1, goal_total=2)
        self.assertEqual(first["objective_value"], second["objective_value"])
        self.assertEqual(first["objective_value"], 0.5)

    def test_each_branch_view_can_use_different_answers_from_same_document(self):
        plan = [{"id": "s1", "depends_on": []}]
        for answer in ["Alpha", "Beta"]:
            with self.subTest(answer=answer):
                winning = {"s1": proof(1, answer)}
                candidates = {1: candidate(1, 1.0, [proof(1, answer, "s1")])}
                selected, diagnostic = self.select(candidates=candidates, plan=plan, proofs=winning,
                                                    bindings={"s1": answer}, budget=1)
                self.assertEqual(selected, [1])
                self.assertFalse(diagnostic["blocked_proof_document_ids"])


if __name__ == "__main__":
    unittest.main()
