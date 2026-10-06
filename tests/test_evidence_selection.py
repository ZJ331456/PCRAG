"""CPU checks for continuous prefixes and passage-grounded proof closure."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("improved_evidence_selection_test_package")
PACKAGE.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + ".evidence_selection", ROOT / "src/pathcondrag/evidence_selection.py")
selection = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selection
SPEC.loader.exec_module(selection)
base_module = sys.modules[PACKAGE.__name__ + ".evidence_retrieval"]


class Improved(selection.SelectionImprovementMixin, base_module.EvidenceRetrieval):
    def __init__(self, rag, flags=("selection",)):
        super().__init__(rag)
        self.improvements = frozenset(flags)


class Rag:
    def __init__(self, budget=10):
        self.pcrag_config = SimpleNamespace(improvement_stage=4, evidence_budget=5,
                                           evidence_selection_top_k=budget)
        self.documents = {d: f"Passage {d}\nAn unrelated factual statement about item {d}." for d in range(101)}
        self.documents[0] = "Writer biography\nAlpha wrote Work X in 1990."
        self.documents[1] = "Birth biography\nAlpha was born in Paris in 1960."
        self.passage_node_keys = list(range(101))
        self.chunk_embedding_store = SimpleNamespace(get_row=lambda key: {"content": self.documents[key]})


def candidate(doc_id, goal=None, score=1., verified=()):
    return {"doc_id": doc_id, "base_score": 0., "verified": list(verified),
            "sources": [] if goal is None else [{"goal": goal, "score": score, "requirements": {}}]}


def proof(doc_id, answer, quote, node, relation=True, confidence=.95):
    result = {"doc_id": doc_id, "answer": answer, "evidence": quote, "quality": .95,
              "confidence": confidence, "goal": "dag:" + node}
    if relation:
        result["relation_supported"] = True
    return result


def closure_state(rag, root=True, relation=True):
    p0 = proof(0, "Alpha", "Alpha wrote Work X in 1990.", "s1", relation)
    p1 = proof(1, "Paris", "Alpha was born in Paris in 1960.", "s2", relation)
    plan = [{"id": "s1", "question": "Who wrote Work X?", "depends_on": [], "answer_type": "person"},
            {"id": "s2", "question": "Where was ${s1.answer} born?", "depends_on": ["s1"], "answer_type": "place"}]
    candidates = {1: candidate(1, "dag:s2", verified=[p1])}
    proofs = {"s2": p1}
    if root:
        candidates[0] = candidate(0, "dag:s1", verified=[p0])
        proofs["s1"] = p0
    bindings = {"s1": "Alpha", "s2": "Paris"}
    return {"evidence_candidates": candidates, "evidence_trace": {"plan": plan},
            "_evidence_plan": plan, "_evidence_winning_bindings": bindings,
            "_evidence_beams": [{"bindings": bindings, "proofs": proofs}]}


class EvidenceSelectionTests(unittest.TestCase):
    def test_disabled_flags_delegate_to_original_exp4(self):
        rag = Rag()
        state = {"evidence_candidates": {3: candidate(3, "g1")}, "evidence_trace": {}}
        ids, scores = np.arange(12), np.linspace(1., .1, 12)
        expected = base_module.EvidenceRetrieval(rag).finalize("q", ids, scores, {}, state)
        actual = Improved(rag, ()).finalize("q", ids, scores, {}, state)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])

    def test_continuous_top10_has_same_first_five_as_top5_selection(self):
        state = {"evidence_candidates": {d: candidate(d, "g" + str(d % 3)) for d in [7, 8, 9, 10]},
                 "evidence_trace": {}}
        ids, scores = np.arange(15), np.linspace(1., .1, 15)
        five = Improved(Rag(5)).finalize("q", ids, scores, {}, dict(state, evidence_trace={}))[0]
        ten = Improved(Rag(10)).finalize("q", ids, scores, {}, dict(state, evidence_trace={}))[0]
        self.assertEqual(five[:5].tolist(), ten[:5].tolist())

    def test_closure_only_without_reliable_proofs_preserves_original_candidate_greedy(self):
        rag = Rag()
        # Original exp4 forcibly selects five candidates even without new goal
        # gains. Closure alone must not silently activate the selection guard.
        ids, scores = np.arange(15), np.linspace(1., .1, 15)
        candidates = {d: candidate(d) for d in range(8, 13)}
        expected = base_module.EvidenceRetrieval(rag).finalize(
            "q", ids, scores, {}, {"evidence_candidates": candidates, "evidence_trace": {}})[0]
        actual, _, trace = Improved(rag, ("closure",)).finalize(
            "q", ids, scores, {}, {"evidence_candidates": candidates, "evidence_trace": {}})
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(actual[:5].tolist(), [8, 9, 10, 11, 12])
        self.assertFalse(trace["improvement_selection"]["enabled"])

    def test_once_goal_is_covered_tail_keeps_base_channel(self):
        state = {"evidence_candidates": {d: candidate(d, "same", 1. - .01 * (d - 10))
                                          for d in range(10, 15)}, "evidence_trace": {}}
        ids = np.arange(15)
        actual, scores, trace = Improved(Rag()).finalize("q", ids, np.linspace(1., .1, 15), {}, state)
        self.assertEqual(actual[0], 10)
        self.assertEqual(actual[1:10].tolist(), list(range(9)))
        self.assertEqual(actual[10:].tolist(), [9, 11, 12, 13, 14])
        self.assertEqual(trace["improvement_selection"]["coverage_promotions"], 1)
        self.assertTrue(np.all(scores[:-1] > scores[1:]))

    def test_zero_gain_does_not_force_candidate_prefix(self):
        ids, scores = np.arange(15), np.linspace(1., .1, 15)
        state = {"evidence_candidates": {d: candidate(d) for d in range(10, 15)}, "evidence_trace": {}}
        actual, _, trace = Improved(Rag()).finalize("q", ids, scores, {}, state)
        self.assertEqual(actual.tolist(), ids.tolist())
        self.assertEqual(trace["improvement_selection"]["coverage_promotions"], 0)
        self.assertEqual(trace["improvement_selection"]["base_fills"], 10)

    def test_extra_candidate_passage_is_retained_and_duplicates_are_not_added(self):
        ids, scores = np.arange(15), np.linspace(1., .1, 15)
        state = {"evidence_candidates": {100: candidate(100, "new")}, "evidence_trace": {}}
        actual, _, _ = Improved(Rag()).finalize("q", ids, scores, {}, state)
        self.assertEqual(set(actual.tolist()), set(ids.tolist()) | {100})
        self.assertEqual(len(actual), len(set(actual.tolist())))

    def test_reliable_relation_child_includes_ancestor_before_it(self):
        rag = Rag()
        state = closure_state(rag)
        ids = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 10, 11])
        actual, _, trace = Improved(rag, ("selection", "closure")).finalize("q", ids, np.linspace(1., .1, 12), {}, state)
        selected = actual[:10].tolist()
        self.assertIn(0, selected)
        self.assertIn(1, selected)
        self.assertLess(selected.index(0), selected.index(1))
        self.assertEqual(trace["improvement_closure"]["reliable_proofs"]["s2"]["check"], "relation_checked")

    def test_closure_only_can_use_conservative_literal_proofs(self):
        rag = Rag()
        state = closure_state(rag, relation=False)
        ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11])
        actual, _, trace = Improved(rag, ("closure",)).finalize("q", ids, np.linspace(1., .1, 12), {}, state)
        self.assertLess(actual.tolist().index(0), actual.tolist().index(1))
        self.assertEqual(trace["improvement_closure"]["reliable_proofs"]["s1"]["check"], "conservative_literal")
        self.assertFalse(trace["improvement_selection"]["enabled"])

    def test_missing_ancestor_retains_ordinary_coverage_competition(self):
        rag = Rag(2)
        state = closure_state(rag, root=False)
        ids = np.array([2, 3, 4, 5, 0, 1])
        actual, _, trace = Improved(rag, ("selection", "closure")).finalize("q", ids, np.linspace(1., .1, 6), {}, state)
        self.assertEqual(actual[:2].tolist(), [1, 2])
        self.assertEqual(trace["selected_prefix"][0]["closure_status"], "ordinary_document")
        self.assertEqual(trace["selected_prefix"][0]["selection_source"], "coverage")
        self.assertEqual(trace["improvement_closure"]["selected_bundle_count"], 0)
        self.assertIn(1, trace["improvement_closure"]["soft_fallback_docs"])
        self.assertTrue(any(r["reason"] == "missing_reliable_ancestor" for r in trace["improvement_closure"]["rejected_proofs"]))

    def test_rejected_literal_proofs_do_not_push_relevant_passages_behind_unrelated_candidates(self):
        rag = Rag()
        rag.documents[0] = "The work was written by Alpha in 1990."
        rag.documents[1] = "He was born in Paris in 1960."
        state = closure_state(rag, relation=False)
        for nid, doc_id in (("s1", 0), ("s2", 1)):
            state["_evidence_beams"][0]["proofs"][nid]["evidence"] = rag.documents[doc_id]
        state["evidence_candidates"].update({d: candidate(d, "noise:" + str(d), .2)
                                              for d in range(6, 11)})
        ids, scores = np.arange(12), np.linspace(1., .1, 12)
        expected, _, _ = base_module.EvidenceRetrieval(rag).finalize("q", ids, scores, {}, dict(state, evidence_trace={}))
        actual, _, trace = Improved(rag, ("closure",)).finalize("q", ids, scores, {}, state)
        # Literal subject/coreference checks reject these proofs. Their passages
        # still compete using exactly the original exp4 retrieval objective.
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(set(actual[:2]), {0, 1})
        self.assertEqual(trace["improvement_closure"]["soft_fallback_docs"], [0, 1])
        self.assertEqual(trace["improvement_closure"]["ancestor_document_bundles"], {})
        self.assertTrue(all(d["closure_status"] == "ordinary_document"
                            for d in trace["selected_prefix"]))

    def test_reliable_ancestor_chain_fits_budget_and_keeps_parent_order(self):
        rag = Rag(2)
        state = closure_state(rag)
        ids = np.array([1, 0, 2, 3, 4, 5])
        actual, _, trace = Improved(rag, ("selection", "closure")).finalize("q", ids, np.linspace(1., .1, 6), {}, state)
        self.assertEqual(actual[:2].tolist(), [0, 1])
        self.assertEqual(trace["improvement_closure"]["selected_bundle_count"], 1)
        self.assertTrue(all(d["closure_status"] == "complete_bundle"
                            for d in trace["selected_prefix"]))

    def test_over_budget_bundle_keeps_ordinary_document_and_does_not_claim_closure(self):
        rag = Rag(1)
        state = closure_state(rag)
        # Only the child is in the dense ranking. The ancestor remains available
        # as a candidate, but both proof documents cannot fit the one-slot prefix.
        ids = np.array([1, 2, 3, 4, 0])
        actual, _, trace = Improved(rag, ("selection", "closure")).finalize("q", ids, np.linspace(1., .1, 5), {}, state)
        self.assertEqual(actual[0], 1)
        self.assertEqual(trace["selected_prefix"][0]["selection_source"], "base_fallback")
        self.assertEqual(trace["selected_prefix"][0]["closure_status"], "ordinary_document")
        self.assertEqual(trace["improvement_closure"]["selected_bundle_count"], 0)
        self.assertEqual(trace["improvement_closure"]["deferred_bundle_documents"]["1"],
                         "bundle_exceeds_remaining_prefix_budget")

    def test_one_document_proving_two_nodes_keeps_both_nodes_ancestors(self):
        rag = Rag(3)
        rag.documents[2] = "Alpha was born in Paris. Alpha worked in London."
        state = closure_state(rag)
        state["_evidence_plan"].append({"id": "s3", "question": "Where did ${s1.answer} work?",
                                          "depends_on": ["s1"], "answer_type": "place"})
        state["_evidence_plan"].append({"id": "s4", "question": "Which city links ${s2.answer}?",
                                          "depends_on": ["s2"], "answer_type": "place"})
        p3 = proof(2, "London", "Alpha worked in London.", "s3")
        p4 = proof(2, "Paris", "Alpha was born in Paris.", "s4")
        state["_evidence_winning_bindings"].update(s3="London", s4="Paris")
        state["_evidence_beams"][0]["proofs"].update(s3=p3, s4=p4)
        state["evidence_candidates"][2] = candidate(2, "dag:s3", verified=[p3, p4])
        bundles, _, diagnostics = Improved(rag, ("closure",))._reliable_proof_bundles(
            state, state["evidence_candidates"], state["_evidence_winning_bindings"])
        self.assertEqual(bundles[2], {0, 1, 2})
        self.assertEqual(diagnostics["ancestor_document_bundles"]["2"], [0, 1, 2])
        actual, _, trace = Improved(rag, ("selection", "closure")).finalize(
            "q", np.array([2, 3, 4, 0, 1]), np.linspace(1., .1, 5), {}, state)
        self.assertEqual(actual[:3].tolist(), [0, 1, 2])
        self.assertTrue(trace["improvement_closure"]["reliable_proof_closed_at_5"])

    def test_wrong_literal_subject_does_not_force_ancestor_chain(self):
        rag = Rag(2)
        rag.documents[0] = "Alpha wrote Work Y in 1990."
        state = closure_state(rag, relation=False)
        state["_evidence_beams"][0]["proofs"]["s1"]["evidence"] = rag.documents[0]
        ids = np.array([2, 3, 4, 0, 1])
        _, _, trace = Improved(rag, ("selection", "closure")).finalize("q", ids, np.linspace(1., .1, 5), {}, state)
        self.assertEqual(trace["improvement_closure"]["soft_fallback_docs"], [0, 1])
        self.assertTrue(any(r["reason"] == "named_subject_not_in_quote" for r in trace["improvement_closure"]["rejected_proofs"]))

    def test_binding_conflict_is_not_used_as_reliable_proof(self):
        rag = Rag(2)
        state = closure_state(rag)
        state["_evidence_beams"][0]["proofs"]["s1"]["answer"] = "Beta"
        ids = np.array([2, 3, 4, 0, 1])
        _, _, trace = Improved(rag, ("selection", "closure")).finalize("q", ids, np.linspace(1., .1, 5), {}, state)
        self.assertTrue(any(r["reason"] == "proof_conflicts_with_winning_binding"
                            for r in trace["improvement_closure"]["rejected_proofs"]))

    def test_invalid_plan_falls_back_without_failing_retrieval(self):
        rag = Rag()
        state = closure_state(rag)
        state["_evidence_plan"][0]["depends_on"] = ["s2"]
        actual, _, trace = Improved(rag, ("selection", "closure")).finalize("q", np.arange(12), np.linspace(1., .1, 12), {}, state)
        self.assertEqual(len(actual), 12)
        self.assertEqual(trace["improvement_closure"]["plan_error"], "cyclic_dependencies")

    def test_no_benchmark_support_or_decomposition_fields_are_read(self):
        class GuardedState(dict):
            def get(self, key, default=None):
                if key in {"gold_answers", "supporting_facts", "question_decomposition", "gold_docs"}:
                    raise AssertionError("Benchmark labels were read")
                return super().get(key, default)
        state = GuardedState(evidence_candidates={3: candidate(3, "g1")}, evidence_trace={})
        Improved(Rag()).finalize("q", np.arange(12), np.linspace(1., .1, 12), {}, state)


if __name__ == "__main__":
    unittest.main()
