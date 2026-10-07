"""CPU tests for finite DAG closure selection and strict source support."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("dag_package_evidence_test_package")
PACKAGE.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + ".evidence_dag_package", ROOT / "src/pathcondrag/evidence_dag_package.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class Parent:
    def finalize(self, query, ids, scores, ctx, state):
        self.parent_calls += 1
        return ids, scores, state["evidence_trace"]


class Engine(module.DAGPackageMixin, Parent):
    def __init__(self, enabled=True, stage=4):
        self.improvements = {"dag_package"} if enabled else {"planning", "plan_prune"}
        self.stage, self.parent_calls = stage, 0
        self.documents = {d: f"Other {d}\nAn unrelated article {d}." for d in range(220)}

    def _document(self, doc_id):
        return self.documents[doc_id]


def state_for(engine, nodes, bindings, proof_rows):
    proofs, candidates = {}, {d: {"verified": []} for d in engine.documents}
    for nid, doc_id, quote in proof_rows:
        proof = {"doc_id": doc_id, "answer": bindings[nid], "evidence": quote, "confidence": .95}
        proofs[nid] = proof
        candidates[doc_id]["verified"].append(dict(proof, goal="dag:" + nid))
    return {"_evidence_plan": nodes, "_evidence_winning_bindings": dict(bindings),
            "_evidence_beams": [{"bindings": dict(bindings), "proofs": proofs}],
            "evidence_candidates": candidates,
            "evidence_trace": {"selected_prefix": [{"doc_id": d} for d in range(5)]}}


def parallel(engine, terminal=5):
    rows = [("filmA", 0, "Airheads is a 1994 American film directed by Michael Lehmann."),
            ("filmB", 2, "It was directed by Po-Chih Leong."),
            ("bornA", terminal, "Michael Stephen Lehmann (born March 30, 1957) is an American director."),
            ("bornB", 3, "Leong Po-Chih (b. 31 December 1939) is a British-Chinese director.")]
    engine.documents[0] = "Airheads\n" + rows[0][2]
    engine.documents[2] = "Return to Cabin by the Lake\nReturn to Cabin by the Lake is an American film. " + rows[1][2]
    engine.documents[terminal] = "Michael Lehmann\n" + rows[2][2]
    engine.documents[3] = "Po-Chih Leong\n" + rows[3][2]
    nodes = [{"id": "filmA", "question": "Who directed Airheads?", "depends_on": [], "answer_type": "person"},
             {"id": "filmB", "question": "Who directed Return To Cabin By The Lake?", "depends_on": [], "answer_type": "person"},
             {"id": "bornA", "question": "When was ${filmA.answer} born?", "depends_on": ["filmA"], "answer_type": "date"},
             {"id": "bornB", "question": "When was ${filmB.answer} born?", "depends_on": ["filmB"], "answer_type": "date"}]
    return state_for(engine, nodes, {"filmA": "Michael Lehmann", "filmB": "Po-Chih Leong",
                                   "bornA": "March 30, 1957", "bornB": "31 December 1939"}, rows)


def chain(engine, count=4, terminal=15, shared=False):
    quotes = ["Sample Film was directed by Alice Example.",
              "Alice Example was born in Sample Town.",
              "Sample Town is located in Sample County.",
              "Sample County is located in Sample Region."]
    titles = ["Sample Film", "Alice Example", "Sample Town", "Sample County"]
    answers = ["Alice Example", "Sample Town", "Sample County", "Sample Region"]
    ids = [0, 2, 5, terminal]
    if shared:
        ids = [0, 2, 2, terminal]
    for i in range(count):
        d = ids[i]
        if shared and i == 2:
            engine.documents[d] += " " + quotes[i]
        else:
            engine.documents[d] = titles[i] + "\n" + quotes[i]
    nodes = [{"id": "a", "question": "Who directed Sample Film?", "depends_on": [], "answer_type": "person"},
             {"id": "b", "question": "Where was ${a.answer} born?", "depends_on": ["a"], "answer_type": "place"},
             {"id": "c", "question": "Where is ${b.answer} located?", "depends_on": ["b"], "answer_type": "place"},
             {"id": "d", "question": "Where is ${c.answer} located?", "depends_on": ["c"], "answer_type": "place"}]
    return state_for(engine, nodes[:count], dict(zip("abcd"[:count], answers)),
                     [("abcd"[i], ids[i], quotes[i]) for i in range(count)])


def run(engine, state, ids=None, scores=None):
    ids = np.arange(220) if ids is None else ids
    scores = np.arange(len(ids), 0, -1, dtype=float) if scores is None else scores
    return engine.finalize("A question without benchmark labels", ids, scores, {}, state)


class DAGPackageTests(unittest.TestCase):
    def test_disabled_or_earlier_stage_returns_exact_parent_objects(self):
        for e in [Engine(False), Engine(stage=3)]:
            s = parallel(e); ids, scores = np.arange(220), np.ones(220)
            actual = run(e, s, ids, scores)
            self.assertIs(actual[0], ids); self.assertIs(actual[1], scores)
            self.assertIs(actual[2], s["evidence_trace"]); self.assertEqual(e.parent_calls, 1)

    def test_parallel_branches_complete_prefix_with_full_name_and_reordered_lead(self):
        e = Engine(); s = parallel(e); before = copy.deepcopy(s)
        ids, scores, trace = run(e, s)
        self.assertEqual(ids[:5].tolist(), [0, 1, 2, 3, 5])
        self.assertEqual(s, before)
        diag = trace["improvement_dag_package"]
        self.assertEqual(len(diag["eligible_proofs"]), 4)
        self.assertEqual(diag["prefix_optimization"][0]["complete_terminals_after"], ["bornA", "bornB"])
        self.assertTrue(np.all(scores[:-1] > scores[1:]))
        self.assertEqual(diag["extra_requests"], 0)

    def test_promote_top10_external_doc_and_preserve_top200(self):
        e = Engine(); s = parallel(e, terminal=25)
        ids, _, trace = run(e, s)
        self.assertIn(25, ids[:5]); self.assertIn(25, ids[:10])
        self.assertEqual(ids[:2].tolist(), [0, 1])
        self.assertEqual(set(ids[:200]), set(range(200)))
        self.assertFalse(trace["improvement_dag_package"]["top10_set_preserved"])

    def test_four_node_chain_jointly_fits_top5(self):
        e = Engine(); s = chain(e)
        ids, _, trace = run(e, s)
        self.assertEqual(set(ids[:5]), {0, 1, 2, 5, 15})
        self.assertEqual(trace["improvement_dag_package"]["prefix_optimization"][0]["complete_terminals_after"], ["d"])

    def test_three_node_chain_has_ancestor_closure(self):
        e = Engine(); s = chain(e, count=3)
        ids, _, trace = run(e, s)
        self.assertIn(5, ids[:5])
        self.assertEqual(trace["improvement_dag_package"]["prefix_optimization"][0]["coverage_after"], ["a", "b", "c"])

    def test_shared_document_counts_once_in_joint_capacity(self):
        e = Engine(); s = chain(e, shared=True)
        ids, _, trace = run(e, s)
        self.assertIn(15, ids[:5]); self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(trace["improvement_dag_package"]["prefix_optimization"][0]["coverage_after"], list("abcd"))

    def test_six_independent_nodes_protect_old_proofs_and_finish_top10(self):
        e = Engine(); nodes, bindings, rows = [], {}, []
        for i, d in enumerate([0, 2, 3, 5, 11, 18]):
            nid, title, answer = f"n{i}", f"Work {i}", f"Creator {i}"
            quote = f"{title} was directed by {answer}."
            e.documents[d] = title + "\n" + quote
            nodes.append({"id": nid, "question": f"Who directed {title}?", "depends_on": [], "answer_type": "person"})
            bindings[nid] = answer; rows.append((nid, d, quote))
        s = state_for(e, nodes, bindings, rows); ids, _, trace = run(e, s)
        self.assertTrue({0, 2, 3, 5} <= set(ids[:5]))
        self.assertTrue({0, 2, 3, 5, 11, 18} <= set(ids[:10]))
        self.assertEqual(len(trace["improvement_dag_package"]["eligible_proofs"]), 6)

    def test_cross_branch_binding_cannot_supply_support(self):
        e = Engine(); s = parallel(e)
        s["_evidence_beams"][0]["bindings"]["filmA"] = "Other Person"
        self.assertEqual(run(e, s)[0].tolist(), list(range(220)))

    def test_alternative_quote_with_conflicting_requirements_is_rejected(self):
        e = Engine(); s = parallel(e)
        s["_evidence_beams"][0]["proofs"].pop("bornA")
        s["evidence_candidates"][5]["verified"][0]["requirements"] = {"filmA": "Other Person"}
        self.assertEqual(run(e, s)[0].tolist(), list(range(220)))

    def test_unquoted_high_confidence_is_not_evidence(self):
        e = Engine(); s = parallel(e)
        bad = "Michael Lehmann was born March 30, 1957 in another invented place."
        for p in [s["_evidence_beams"][0]["proofs"]["bornA"], s["evidence_candidates"][5]["verified"][0]]:
            p.update(evidence=bad, confidence=1.)
        self.assertEqual(run(e, s)[0].tolist(), list(range(220)))

    def test_same_page_wrong_owner_not_promoted(self):
        e = Engine(); s = parallel(e)
        bad = "Other Person was born March 30, 1957. Michael Lehmann is mentioned elsewhere."
        e.documents[5] = "Michael Lehmann\n" + bad
        for p in [s["_evidence_beams"][0]["proofs"]["bornA"], s["evidence_candidates"][5]["verified"][0]]:
            p["evidence"] = bad
        self.assertEqual(run(e, s)[0].tolist(), list(range(220)))

    def test_parent_quote_not_grounded_blocks_descendant_package(self):
        e = Engine(); s = chain(e)
        for p in [s["_evidence_beams"][0]["proofs"]["b"], s["evidence_candidates"][2]["verified"][0]]:
            p["evidence"] = "Alice Example was born in an invented Other Town."
        ids, _, trace = run(e, s)
        self.assertNotIn(15, ids[:10])
        self.assertIn("b", [d["node"] for d in trace["improvement_dag_package"]["deferred"]])

    def test_ambiguous_topic_pronoun_rejected(self):
        e = Engine(); s = parallel(e)
        e.documents[2] = ("Return to Cabin by the Lake\nReturn to Cabin by the Lake is a film. "
                          "Another Film was released later. It was directed by Po-Chih Leong.")
        ids, _, trace = run(e, s)
        self.assertNotIn("filmB", trace["improvement_dag_package"]["eligible_proofs"])
        self.assertIn("ambiguous_document_topic_pronoun", [d["reason"] for d in trace["improvement_dag_package"]["rejected_proofs"]])

    def test_wrong_named_biography_lead_does_not_become_title_alias(self):
        e = Engine(); s = parallel(e)
        bad = "Leong Other Chih (born 31 December 1939) is a director."
        e.documents[3] = "Po-Chih Leong\n" + bad
        for p in [s["_evidence_beams"][0]["proofs"]["bornB"], s["evidence_candidates"][3]["verified"][0]]:
            p["evidence"] = bad
        _, _, trace = run(e, s)
        self.assertNotIn("bornB", trace["improvement_dag_package"]["eligible_proofs"])

    def test_distant_named_subject_breaks_pronoun_inside_long_quote(self):
        e = Engine(); s = parallel(e)
        bad = "Michael Lehmann is a director. Other Person moved abroad. He was born March 30, 1957."
        e.documents[5] = "Michael Lehmann\n" + bad
        for p in [s["_evidence_beams"][0]["proofs"]["bornA"], s["evidence_candidates"][5]["verified"][0]]:
            p["evidence"] = bad
        _, _, trace = run(e, s)
        self.assertNotIn("bornA", trace["improvement_dag_package"]["eligible_proofs"])

    def test_top10_external_complete_branch_preserves_all_supported_top5(self):
        e = Engine(); nodes, bindings, rows = [], {}, []
        for i, d in enumerate([0, 1, 2, 3, 4, 25]):
            nid, title, answer = f"n{i}", f"Work {i}", f"Creator {i}"
            quote = f"{title} was directed by {answer}."
            e.documents[d] = title + "\n" + quote
            nodes.append({"id": nid, "question": f"Who directed {title}?", "depends_on": [], "answer_type": "person"})
            bindings[nid] = answer; rows.append((nid, d, quote))
        s = state_for(e, nodes, bindings, rows)
        ids, _, trace = run(e, s)
        self.assertEqual(ids[:5].tolist(), list(range(5)))
        self.assertIn(25, ids[:10])
        self.assertTrue(any(p["top_k"] == 10 and p["doc_id"] == 25 for p in trace["improvement_dag_package"]["promotions"]))
        self.assertEqual(set(ids[:200]), set(range(200)))

    def test_relative_birth_date_does_not_transfer_identity(self):
        e = Engine(); s = parallel(e)
        bad = "Michael Lehmann's father was born March 30, 1957."
        e.documents[5] = "Michael Lehmann\n" + bad
        for p in [s["_evidence_beams"][0]["proofs"]["bornA"], s["evidence_candidates"][5]["verified"][0]]:
            p["evidence"] = bad
        self.assertEqual(run(e, s)[0].tolist(), list(range(220)))

    def test_unknown_relation_cooccurrence_cannot_count(self):
        e = Engine(); s = chain(e)
        s["_evidence_plan"][1]["question"] = "Which town did ${a.answer} mention?"
        self.assertNotIn(15, run(e, s)[0][:10])

    def test_over_depth_or_node_budget_and_cycle_return_original(self):
        for extra in [3, 1]:
            e = Engine(); s = chain(e)
            for i in range(extra):
                previous = "d" if i == 0 else f"z{i-1}"
                s["_evidence_plan"].append({"id": f"z{i}", "question": f"Where is ${{{previous}.answer}}?", "depends_on": [previous]})
            self.assertEqual(run(e, s)[0].tolist(), list(range(220)))
        e = Engine(); s = parallel(e)
        s["_evidence_plan"][0]["depends_on"] = ["bornA"]
        self.assertEqual(run(e, s)[0].tolist(), list(range(220)))

    def test_candidate_outside_top200_not_promoted(self):
        e = Engine(); s = parallel(e, terminal=210)
        self.assertEqual(run(e, s)[0].tolist(), list(range(220)))

    def test_unchanged_scores_and_ids_when_already_complete(self):
        e = Engine(); s = parallel(e)
        ids = np.array([0, 1, 2, 3, 5, 4] + list(range(6, 220)))
        scores = np.arange(220, 0, -1, dtype=float)
        actual = run(e, s, ids, scores)
        self.assertIs(actual[0], ids); self.assertIs(actual[1], scores)

    def test_no_benchmark_keys_consulted(self):
        class GuardedState(dict):
            def get(self, key, default=None):
                if key in {"gold_docs", "gold_answers", "dataset", "type", "question_decomposition", "benchmark_hops"}:
                    raise AssertionError("annotation accessed")
                return super().get(key, default)
        e = Engine(); s = GuardedState(parallel(e))
        self.assertIn(5, run(e, s)[0][:5])

    def test_duplicate_order_and_invalid_structure_return_original(self):
        e = Engine(); s = parallel(e)
        ids = np.array([0, 0] + list(range(2, 220)))
        self.assertIs(run(e, s, ids)[0], ids)
        s["_evidence_plan"] = "not a plan"
        self.assertEqual(run(e, s)[0].tolist(), list(range(220)))


if __name__ == "__main__":
    unittest.main()
