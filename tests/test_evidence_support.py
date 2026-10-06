"""CPU checks for bounded winning-ancestor citation preservation."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("ancestor_support_test_package")
PACKAGE.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + ".evidence_support", ROOT / "src/pathcondrag/evidence_support.py")
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


class Parent:
    def finalize(self, query, ids, scores, ctx, state):
        self.calls += 1
        return ids, scores, state["evidence_trace"]


class Engine(support.AncestorSupportMixin, Parent):
    def __init__(self, mode="tail_only", enabled=True):
        self.calls = 0
        self.improvements = frozenset({"support"} if enabled else {})
        self.cfg = SimpleNamespace(evidence_support_mode=mode)
        self.documents = {d: f"Article {d}\nA factual passage number {d}." for d in range(20)}

    def _document(self, doc_id):
        return self.documents[doc_id]

    @staticmethod
    def _goal_scores(candidate, bindings):
        return dict(candidate.get("goals", {}))


def state_for(engine, ancestor_ids=(11,), child_id=2):
    nodes, proofs, bindings = [], {}, {}
    for i, doc_id in enumerate(ancestor_ids):
        nid, answer = "a" + str(i), "Answer " + str(i)
        quote = answer + " is directly cited in this passage."
        engine.documents[doc_id] = f"Ancestor {i}\n" + quote
        nodes.append({"id": nid, "question": "Which answer?", "depends_on": [] if i == 0 else ["a" + str(i - 1)]})
        proofs[nid] = {"doc_id": doc_id, "answer": answer, "evidence": quote, "confidence": .95}
        bindings[nid] = answer
    quote = "Terminal answer is directly cited in this passage."
    engine.documents[child_id] = "Terminal article\n" + quote
    nodes.append({"id": "leaf", "question": "Which terminal answer?", "depends_on": ["a" + str(len(ancestor_ids) - 1)]})
    proofs["leaf"] = {"doc_id": child_id, "answer": "Terminal answer", "evidence": quote, "confidence": .95}
    bindings["leaf"] = "Terminal answer"
    candidates = {d: {"verified": [], "sources": [], "goals": {}} for d in range(20)}
    for nid, proof in proofs.items():
        candidates[proof["doc_id"]]["verified"].append(dict(proof, goal="dag:" + nid))
    return {"_evidence_plan": nodes, "_evidence_winning_bindings": bindings,
            "_evidence_beams": [{"bindings": dict(bindings), "proofs": proofs}],
            "evidence_candidates": candidates,
            "evidence_trace": {"selected_prefix": [{"doc_id": d} for d in range(5)]}}


def run(engine, state, ids=None, scores=None):
    ids = np.arange(20) if ids is None else ids
    scores = np.linspace(1., .1, len(ids)) if scores is None else scores
    return engine.finalize("Question?", ids, scores, {}, state)


def duplicates(engine, state, victim=4, partner=3):
    text = "Shared article\nThe article describes a person and their career in detail."
    engine.documents[victim] = text
    engine.documents[partner] = text
    state["evidence_candidates"][victim]["goals"] = {"shared": .8}
    state["evidence_candidates"][partner]["goals"] = {"shared": .9}


class EvidenceSupportTests(unittest.TestCase):
    def test_disabled_is_exact_delegate_including_scores_and_trace(self):
        engine = Engine(enabled=False)
        state = state_for(engine)
        ids, scores = np.arange(20), np.ones(20)
        actual = run(engine, state, ids, scores)
        self.assertIs(actual[0], ids)
        self.assertIs(actual[1], scores)
        self.assertIs(actual[2], state["evidence_trace"])
        self.assertEqual(engine.calls, 1)

    def test_tail_keeps_all_first_five_and_promotes_grounded_ancestor(self):
        engine = Engine()
        state = state_for(engine)
        original = copy.deepcopy(state)
        ids, scores, trace = run(engine, state)
        self.assertEqual(ids[:5].tolist(), list(range(5)))
        self.assertEqual(ids[5], 11)
        self.assertEqual(ids[6:].tolist(), [d for d in range(5, 20) if d != 11])
        self.assertEqual(set(ids), set(range(20)))
        self.assertTrue(np.all(scores[:-1] > scores[1:]))
        self.assertTrue(trace["improvement_support"]["top5_preserved"])
        self.assertFalse(trace["improvement_support"]["semantic_relation_upgrade"])
        self.assertEqual(state, original)

    def test_two_ancestors_fit_and_keep_topological_order(self):
        engine = Engine()
        ids, _, trace = run(engine, state_for(engine, (12, 11)))
        self.assertEqual(ids[5:7].tolist(), [12, 11])
        self.assertEqual(len(trace["improvement_support"]["promotions"]), 2)

    def test_overbudget_bundle_is_deferred_without_losing_documents(self):
        engine = Engine()
        ids, _, trace = run(engine, state_for(engine, (11, 12, 13)))
        self.assertEqual(ids.tolist(), list(range(20)))
        self.assertEqual(trace["improvement_support"]["deferred_chains"][0]["reason"], "ancestor_bundle_exceeds_budget")

    def test_missing_proof_leaves_original_order(self):
        engine = Engine()
        state = state_for(engine)
        del state["_evidence_beams"][0]["proofs"]["a0"]
        ids, _, trace = run(engine, state)
        self.assertEqual(ids.tolist(), list(range(20)))
        self.assertEqual(trace["improvement_support"]["deferred_chains"][0]["reason"], "missing_valid_ancestor")

    def test_only_exact_matching_winning_branch_may_supply_proofs(self):
        engine = Engine()
        state = state_for(engine)
        losing = copy.deepcopy(state["_evidence_beams"][0])
        losing["bindings"]["a0"] = "Other answer"
        losing["proofs"]["a0"].update(doc_id=12, answer="Other answer", evidence="Other answer is stated in this losing branch.")
        engine.documents[12] = "Losing document\nOther answer is stated in this losing branch."
        state["_evidence_beams"].insert(0, losing)
        ids, _, _ = run(engine, state)
        self.assertEqual(ids[5], 11)
        self.assertNotEqual(ids[5], 12)
        del state["_evidence_beams"][1]["proofs"]["a0"]
        self.assertEqual(run(engine, state)[0].tolist(), list(range(20)))

    def test_invalid_quote_answer_confidence_and_pool_do_not_promote(self):
        mutations = [({"evidence": "An invented quotation contains Answer 0."}, None),
                     ({"answer": "Different answer"}, None),
                     ({"confidence": float("nan")}, None),
                     ({"confidence": float("inf")}, None),
                     ({"confidence": .84}, None),
                     ({"requirements": {"a0": "A conflicting answer"}}, None),
                     ({}, "remove_pool")]
        for change, extra in mutations:
            with self.subTest(change=change, extra=extra):
                engine = Engine()
                state = state_for(engine)
                state["_evidence_beams"][0]["proofs"]["a0"].update(change)
                if extra:
                    del state["evidence_candidates"][11]
                ids, _, trace = run(engine, state)
                self.assertEqual(ids.tolist(), list(range(20)))
                self.assertTrue(trace["improvement_support"]["rejected_proofs"])

    def test_cycle_is_reported_and_all_candidates_preserved(self):
        engine = Engine()
        state = state_for(engine)
        state["_evidence_plan"][0]["depends_on"] = ["leaf"]
        ids, _, trace = run(engine, state)
        self.assertEqual(ids.tolist(), list(range(20)))
        self.assertEqual(trace["improvement_support"]["plan_error"], "cyclic_dependencies")

    def test_child_outside_first_five_cannot_anchor_promotion(self):
        engine = Engine()
        ids, _, trace = run(engine, state_for(engine, child_id=8))
        self.assertEqual(ids.tolist(), list(range(20)))
        self.assertFalse(trace["improvement_support"]["promotions"])

    def test_already_closed_top10_is_untouched_in_tail_mode(self):
        engine = Engine()
        ids, _, trace = run(engine, state_for(engine, (8,)))
        self.assertEqual(ids.tolist(), list(range(20)))
        self.assertFalse(trace["improvement_support"]["promotions"])

    def test_bounded_swap_only_displaces_an_unverified_near_duplicate(self):
        engine = Engine("bounded_swap")
        state = state_for(engine)
        duplicates(engine, state)
        ids, scores, trace = run(engine, state)
        self.assertEqual(ids[:5].tolist(), [0, 1, 2, 3, 11])
        self.assertEqual(ids[5], 4)
        self.assertEqual(set(ids), set(range(20)))
        self.assertTrue(np.all(scores[:-1] > scores[1:]))
        promotion = trace["improvement_support"]["promotions"][0]
        self.assertEqual(promotion["victim"], 4)
        self.assertEqual(promotion["duplicate_partner"], 3)
        self.assertFalse(trace["improvement_support"]["top5_preserved"])
        self.assertTrue(trace["improvement_support"]["r1_preserved"])
        self.assertEqual([d["doc_id"] for d in trace["selected_prefix"]], ids[:5].tolist())

    def test_verified_unique_goal_or_distinct_fact_is_never_a_victim(self):
        for reason in ("verified", "unique_goal", "numbers", "negation", "different_body"):
            with self.subTest(reason=reason):
                engine = Engine("bounded_swap")
                state = state_for(engine)
                duplicates(engine, state)
                # Make both duplicate documents ineligible when necessary.
                if reason == "verified":
                    for d in (3, 4):
                        state["evidence_candidates"][d]["verified"] = [{"goal": "dag:unrelated"}]
                elif reason == "unique_goal":
                    for d in (3, 4):
                        state["evidence_candidates"][d]["goals"] = {"unique_" + str(d): .9}
                elif reason == "numbers":
                    engine.documents[3] += " In 1990."
                    engine.documents[4] += " In 1991."
                elif reason == "negation":
                    engine.documents[4] += " Never."
                else:
                    engine.documents[4] = "Shared article\nAn entirely different event with unrelated facts."
                ids, _, trace = run(engine, state)
                self.assertEqual(ids[:5].tolist(), list(range(5)))
                self.assertEqual(ids[5], 11)
                self.assertTrue(trace["improvement_support"]["top5_preserved"])

    def test_one_swap_and_two_tail_citations_fit_a_four_node_chain(self):
        engine = Engine("bounded_swap")
        state = state_for(engine, (11, 12, 13))
        duplicates(engine, state)
        ids, _, trace = run(engine, state)
        self.assertEqual(ids[:5].tolist(), [0, 1, 2, 3, 11])
        self.assertEqual(ids[5:8].tolist(), [12, 13, 4])
        self.assertEqual(sum(p["scope"] == "top5_swap" for p in trace["improvement_support"]["promotions"]), 1)
        self.assertEqual(sum(p["scope"] == "top10_tail" for p in trace["improvement_support"]["promotions"]), 2)

    def test_same_title_is_not_global_deduplication(self):
        engine = Engine()
        state = state_for(engine)
        duplicates(engine, state)
        ids, _, _ = run(engine, state)
        self.assertIn(3, ids[:5])
        self.assertIn(4, ids[:5])

    def test_ancestor_already_in_tail_can_swap_without_duplicate_ids(self):
        engine = Engine("bounded_swap")
        state = state_for(engine, (8,))
        duplicates(engine, state)
        ids, _, trace = run(engine, state)
        self.assertEqual(ids[:5].tolist(), [0, 1, 2, 3, 8])
        self.assertEqual(ids[5], 4)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(range(20)))
        self.assertEqual(len(trace["improvement_support"]["promotions"]), 1)

    def test_tail_promotion_limit_is_shared_between_all_children(self):
        engine = Engine()
        state = state_for(engine)
        branch = state["_evidence_beams"][0]
        for index, child_id in enumerate((3, 4), 1):
            root_id, node_id, doc_id = "root" + str(index), "leaf" + str(index), 11 + index
            for nid, d in ((root_id, doc_id), (node_id, child_id)):
                answer = "Cited answer " + nid
                quote = answer + " is directly supported by this document."
                engine.documents[d] = "Article " + nid + "\n" + quote
                proof = {"doc_id": d, "answer": answer, "evidence": quote, "confidence": .95}
                branch["proofs"][nid] = proof
                branch["bindings"][nid] = answer
                state["_evidence_winning_bindings"][nid] = answer
                state["evidence_candidates"][d]["verified"].append(dict(proof, goal="dag:" + nid))
                state["_evidence_plan"].append({"id": nid, "question": "Which answer?",
                                               "depends_on": [] if nid == root_id else [root_id]})
        ids, _, trace = run(engine, state)
        self.assertEqual(ids[:5].tolist(), list(range(5)))
        self.assertEqual(ids[5:7].tolist(), [11, 12])
        self.assertEqual(len(trace["improvement_support"]["promotions"]), 2)
        self.assertTrue(any(d["reason"] == "ancestor_bundle_exceeds_budget"
                            for d in trace["improvement_support"]["deferred_chains"]))
        self.assertEqual(set(ids), set(range(20)))


if __name__ == "__main__":
    unittest.main()
