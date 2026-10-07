"""CPU tests for bounded local terminal evidence selection and actual failures."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("terminal_evidence_test_package")
PACKAGE.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + ".evidence_terminal", ROOT / "src/pathcondrag/evidence_terminal.py")
terminal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = terminal
SPEC.loader.exec_module(terminal)


class Parent:
    def finalize(self, query, ids, scores, ctx, state):
        self.calls += 1
        return ids, scores, state["evidence_trace"]


class Engine(terminal.TerminalEvidenceMixin, Parent):
    def __init__(self, mode="prefix", enabled=True):
        self.calls = 0
        self.improvements = frozenset({"terminal"} if enabled else {})
        self.cfg = SimpleNamespace(evidence_terminal_mode=mode)
        self.documents = {d: f"Article {d}\nAn unrelated existing article {d}." for d in range(25)}

    def _document(self, doc_id):
        return self.documents[doc_id]


def fixture(engine, root_doc=0, leaf_doc=7):
    first = "Movie One is a film directed by Alice Example."
    second = "Alice Example was born on 12 June 1929."
    engine.documents[root_doc] = "Movie One\n" + first
    engine.documents[leaf_doc] = "Alice Example\n" + second
    plan = [{"id": "root", "question": "Who directed Movie One?", "depends_on": [], "answer_type": "person"},
            {"id": "leaf", "question": "When was ${root.answer} born?", "depends_on": ["root"], "answer_type": "time"}]
    bindings = {"root": "Alice Example", "leaf": "12 June 1929"}
    proofs = {"root": {"doc_id": root_doc, "answer": bindings["root"], "evidence": first, "confidence": .95},
              "leaf": {"doc_id": leaf_doc, "answer": bindings["leaf"], "evidence": second, "confidence": .95}}
    candidates = {d: {"verified": [], "sources": []} for d in range(25)}
    for nid, p in proofs.items():
        candidates[p["doc_id"]]["verified"].append(dict(p, goal="dag:" + nid))
    return {"_evidence_plan": plan, "_evidence_winning_bindings": bindings,
            "_evidence_beams": [{"bindings": dict(bindings), "proofs": proofs}],
            "evidence_candidates": candidates, "evidence_trace": {"selected_prefix": [{"doc_id": d} for d in range(5)]}}


def run(engine, state, ids=None, scores=None):
    ids = np.arange(25) if ids is None else ids
    scores = np.arange(len(ids), 0, -1, dtype=float) if scores is None else scores
    return engine.finalize("Question?", ids, scores, {}, state)


class EvidenceTerminalTests(unittest.TestCase):
    def test_disabled_delegates_exact_parent_objects(self):
        engine = Engine(enabled=False)
        state = fixture(engine)
        ids, scores = np.arange(25), np.ones(25)
        actual = run(engine, state, ids, scores)
        self.assertIs(actual[0], ids)
        self.assertIs(actual[1], scores)
        self.assertIs(actual[2], state["evidence_trace"])
        self.assertEqual(engine.calls, 1)

    def test_prefix_promotes_missing_terminal_and_preserves_top2_top10_top200(self):
        engine = Engine()
        state = fixture(engine)
        before = copy.deepcopy(state)
        ids, scores, trace = run(engine, state)
        self.assertEqual(ids[:2].tolist(), [0, 1])
        self.assertEqual(ids[:5].tolist(), [0, 1, 2, 3, 7])
        self.assertEqual(ids[7], 4)
        self.assertEqual(set(ids[:10]), set(range(10)))
        self.assertEqual(set(ids), set(range(25)))
        self.assertTrue(np.all(scores[:-1] > scores[1:]))
        self.assertEqual(state, before)
        diag = trace["improvement_terminal"]
        self.assertEqual(diag["prefix_promotions"], 1)
        self.assertTrue(diag["top2_preserved"])
        self.assertTrue(diag["top10_set_preserved"])
        self.assertTrue(diag["top200_set_preserved"])
        self.assertEqual(diag["extra_requests"], 0)

    def test_tail_keeps_top5_and_promotes_terminal_to_six(self):
        engine = Engine("tail_only")
        ids, _, trace = run(engine, fixture(engine))
        self.assertEqual(ids[:5].tolist(), list(range(5)))
        self.assertEqual(ids[5], 7)
        self.assertTrue(trace["improvement_terminal"]["top10_set_preserved"])

    def test_prefix_does_not_promote_far_terminal(self):
        engine = Engine()
        ids, _, trace = run(engine, fixture(engine, leaf_doc=12))
        self.assertEqual(ids.tolist(), list(range(25)))
        self.assertEqual(trace["improvement_terminal"]["deferred_chains"][0]["reason"], "prefix_preserves_existing_top10")

    def test_tail_admits_mechanical_terminal_from_top20_and_retains_displaced(self):
        engine = Engine("tail_only")
        ids, _, trace = run(engine, fixture(engine, leaf_doc=12))
        self.assertEqual(ids[:5].tolist(), list(range(5)))
        self.assertEqual(ids[5], 12)
        self.assertIn(9, ids)
        self.assertFalse(trace["improvement_terminal"]["top10_set_preserved"])

    def test_farther_than_twenty_stays_untouched(self):
        engine = Engine("tail_only")
        ids, _, _ = run(engine, fixture(engine, leaf_doc=22))
        self.assertEqual(ids.tolist(), list(range(25)))

    def test_missing_or_unsupported_ancestor_disables_terminal(self):
        for corruption in ("delete", "wrong_subject"):
            with self.subTest(corruption=corruption):
                engine = Engine()
                state = fixture(engine)
                if corruption == "delete":
                    state["_evidence_beams"][0]["proofs"].pop("root")
                    state["evidence_candidates"][0]["verified"] = []
                else:
                    quote = "Movie Two is a film directed by Alice Example. Movie One is mentioned elsewhere."
                    engine.documents[0] = "Movie One\n" + quote
                    state["_evidence_beams"][0]["proofs"]["root"]["evidence"] = quote
                    state["evidence_candidates"][0]["verified"] = []
                ids, _, trace = run(engine, state)
                self.assertEqual(ids.tolist(), list(range(25)))
                self.assertEqual(trace["improvement_terminal"]["deferred_chains"][0]["reason"], "missing_locally_supported_chain")

    def test_wrong_subject_person_date_is_rejected_despite_literal_answer(self):
        engine = Engine()
        state = fixture(engine)
        quote = "Alice Example's father was born on 12 June 1929."
        engine.documents[7] = "Alice Example\n" + quote
        state["_evidence_beams"][0]["proofs"]["leaf"]["evidence"] = quote
        state["evidence_candidates"][7]["verified"] = []
        self.assertEqual(run(engine, state)[0].tolist(), list(range(25)))

    def test_invalid_literal_proof_fields_do_not_promote(self):
        mutations = [{"confidence": float("nan")}, {"confidence": float("inf")}, {"confidence": .84},
                     {"answer": "31 December 1800"}, {"evidence": "Alice Example was born on 12 June 1929 in an invented place."},
                     {"requirements": {"root": "Different person"}}]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                engine = Engine()
                state = fixture(engine)
                state["_evidence_beams"][0]["proofs"]["leaf"].update(mutation)
                state["evidence_candidates"][7]["verified"] = []
                self.assertEqual(run(engine, state)[0].tolist(), list(range(25)))

    def test_only_exact_winning_branch_is_used(self):
        engine = Engine()
        state = fixture(engine)
        losing = copy.deepcopy(state["_evidence_beams"][0])
        losing["bindings"]["root"] = "Wrong Director"
        state["_evidence_beams"].insert(0, losing)
        self.assertEqual(run(engine, state)[0][4], 7)
        state["_evidence_beams"].pop()
        self.assertEqual(run(engine, state)[0].tolist(), list(range(25)))

    def test_alternative_quote_must_match_winning_answer(self):
        engine = Engine()
        state = fixture(engine)
        bad = dict(state["_evidence_beams"][0]["proofs"]["leaf"], answer="1900", evidence="Alice Example was born in 1900.", doc_id=6, goal="dag:leaf")
        engine.documents[6] = "Other person\n" + bad["evidence"]
        state["evidence_candidates"][6]["verified"].append(bad)
        ids, _, trace = run(engine, state)
        self.assertEqual(ids[4], 7)
        self.assertNotEqual(ids[4], 6)
        self.assertTrue(any(p["reason"] == "proof_conflicts_with_winning_binding" for p in trace["improvement_terminal"]["rejected_proofs"]))

    def test_literal_boilerplate_without_relation_is_not_promoted(self):
        engine = Engine()
        state = fixture(engine)
        state["_evidence_plan"][1]["question"] = "What relates to ${root.answer}?"
        self.assertEqual(run(engine, state)[0].tolist(), list(range(25)))

    def test_top5_locally_supported_documents_are_not_victims(self):
        engine = Engine()
        state = fixture(engine, root_doc=4)
        ids, _, _ = run(engine, state)
        self.assertEqual(ids[4], 4)
        self.assertEqual(ids[3], 7)

    def test_terminal_without_dependency_is_not_used_for_global_selection(self):
        engine = Engine()
        state = fixture(engine)
        state["_evidence_plan"][1]["depends_on"] = []
        self.assertEqual(run(engine, state)[0].tolist(), list(range(25)))

    def test_cycle_and_duplicate_ranking_fall_back(self):
        engine = Engine()
        state = fixture(engine)
        state["_evidence_plan"][0]["depends_on"] = ["leaf"]
        self.assertEqual(run(engine, state)[0].tolist(), list(range(25)))
        state = fixture(engine)
        ids = np.array([0, 0, 1, 2, 3, 7])
        self.assertEqual(run(engine, state, ids)[0].tolist(), ids.tolist())

    def test_real_rudolph_location_construction_is_not_zero_trigger(self):
        engine = Engine()
        state = fixture(engine)
        first = "A now older Rudolph, still unable to find a place in the world, returns home to the North Pole"
        second = "The North Pole is at the center of the Northern Hemisphere."
        engine.documents[0] = "Rudolph the Red-Nosed Reindeer\n" + first
        engine.documents[7] = "North Pole\n" + second
        state["_evidence_plan"] = [{"id": "root", "question": "Where does Rudolph the red-nosed reindeer live?", "depends_on": [], "answer_type": "place"},
                                   {"id": "leaf", "question": "Where is ${root.answer} on the world map?", "depends_on": ["root"], "answer_type": "place"}]
        state["_evidence_winning_bindings"] = {"root": "the North Pole", "leaf": "At the center of the Northern Hemisphere"}
        branch = state["_evidence_beams"][0]
        branch["bindings"] = dict(state["_evidence_winning_bindings"])
        for nid, quote in [("root", first), ("leaf", second)]:
            branch["proofs"][nid].update(answer=branch["bindings"][nid], evidence=quote)
        state["evidence_candidates"][0]["verified"] = []
        state["evidence_candidates"][7]["verified"] = []
        ids, _, trace = run(engine, state)
        self.assertEqual(ids[4], 7)
        self.assertEqual(trace["improvement_terminal"]["prefix_promotions"], 1)

    def test_unknown_relative_fact_does_not_transfer_to_subject(self):
        node = {"question": "Where does Alice Example live?", "answer_type": "place", "depends_on": []}
        proof = {"answer": "London", "evidence": "Alice Example's father lives in London."}
        check, reason = terminal._relation_check(node, node["question"], proof, "Alice Example\n" + proof["evidence"], {})
        self.assertIsNone(check)
        self.assertIsNotNone(reason)

    def test_unknown_edict_subject_cannot_protect_ecditius_chain(self):
        node = {"question": "Who was the person to whom the edict was addressed?", "answer_type": "person", "depends_on": []}
        proof = {"answer": "Ecdicius", "evidence": "Julian addressed an order to Ecdicius, the Prefect of Egypt."}
        check, reason = terminal._relation_check(node, node["question"], proof, "Athanasius of Alexandria\n" + proof["evidence"], {})
        self.assertIsNone(check)
        self.assertEqual(reason, "unresolved_quote_subject")

    def test_two_independent_terminal_goals_fit_without_evicting_ancestors(self):
        engine = Engine()
        state = fixture(engine)
        root_quote = "Movie Two is a film directed by Bob Example."
        leaf_quote = "Bob Example was born on 1 January 1900."
        engine.documents[2] = "Movie Two\n" + root_quote
        engine.documents[8] = "Bob Example\n" + leaf_quote
        state["_evidence_plan"].extend([
            {"id": "root2", "question": "Who directed Movie Two?", "depends_on": [], "answer_type": "person"},
            {"id": "leaf2", "question": "When was ${root2.answer} born?", "depends_on": ["root2"], "answer_type": "date"}])
        state["_evidence_winning_bindings"].update(root2="Bob Example", leaf2="1 January 1900")
        branch = state["_evidence_beams"][0]
        branch["bindings"] = dict(state["_evidence_winning_bindings"])
        branch["proofs"].update(root2={"doc_id": 2, "answer": "Bob Example", "evidence": root_quote, "confidence": .95},
                               leaf2={"doc_id": 8, "answer": "1 January 1900", "evidence": leaf_quote, "confidence": .95})
        ids, _, trace = run(engine, state)
        self.assertEqual(ids[:5].tolist(), [0, 1, 2, 8, 7])
        self.assertEqual(trace["improvement_terminal"]["prefix_promotions"], 2)
        self.assertEqual(set(ids[:10]), set(range(10)))

    def test_three_valid_leaves_cannot_exceed_two_prefix_promotions(self):
        engine = Engine()
        state = fixture(engine)
        extra = [("death", 8, "When did ${root.answer} die?", "1 January 1990", "Alice Example died on 1 January 1990.", "date"),
                 ("nation", 9, "What nationality is ${root.answer}?", "Canadian", "Alice Example is a Canadian actress.", "nationality")]
        branch = state["_evidence_beams"][0]
        for nid, doc_id, question, answer, quote, answer_type in extra:
            engine.documents[doc_id] = "Alice Example\n" + quote
            state["_evidence_plan"].append({"id": nid, "question": question, "depends_on": ["root"], "answer_type": answer_type})
            state["_evidence_winning_bindings"][nid] = answer
            branch["proofs"][nid] = {"doc_id": doc_id, "answer": answer, "evidence": quote, "confidence": .95}
        branch["bindings"] = dict(state["_evidence_winning_bindings"])
        ids, _, trace = run(engine, state)
        self.assertEqual(ids[:2].tolist(), [0, 1])
        self.assertEqual(trace["improvement_terminal"]["prefix_promotions"], 2)
        self.assertTrue(any(p["reason"] == "prefix_promotion_budget" for p in trace["improvement_terminal"]["deferred_chains"]))

    def test_whole_chain_must_fit_existing_top10_for_prefix(self):
        engine = Engine()
        state = fixture(engine, root_doc=15, leaf_doc=7)
        ids, _, trace = run(engine, state)
        self.assertEqual(ids.tolist(), list(range(25)))
        self.assertEqual(trace["improvement_terminal"]["deferred_chains"][0]["reason"], "prefix_preserves_existing_top10")

    def test_tail_does_not_expand_top10_for_unknown_literal_relation(self):
        engine = Engine("tail_only")
        state = fixture(engine, leaf_doc=12)
        quote = "Alice Example lives in London."
        engine.documents[12] = "Alice Example\n" + quote
        state["_evidence_plan"][1].update(question="Where does ${root.answer} live?", answer_type="place")
        state["_evidence_winning_bindings"]["leaf"] = "London"
        branch = state["_evidence_beams"][0]
        branch["bindings"] = dict(state["_evidence_winning_bindings"])
        branch["proofs"]["leaf"].update(answer="London", evidence=quote)
        state["evidence_candidates"][12]["verified"] = []
        ids, _, trace = run(engine, state)
        self.assertEqual(ids.tolist(), list(range(25)))
        self.assertEqual(trace["improvement_terminal"]["deferred_chains"][0]["reason"], "outside_top10_requires_mechanical_relation_and_free_slot")


if __name__ == "__main__":
    unittest.main()
