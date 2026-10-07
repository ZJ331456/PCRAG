"""CPU checks for source ownership and bounded evidence-package selection."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("package_evidence_test_package")
PACKAGE.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + ".evidence_package", ROOT / "src/pathcondrag/evidence_package.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class Parent:
    def finalize(self, query, ids, scores, ctx, state):
        self.parent_calls += 1
        return ids, scores, state["evidence_trace"]


class Engine(module.EvidencePackageMixin, Parent):
    def __init__(self, enabled=True):
        self.improvements = {"package"} if enabled else set()
        self.parent_calls = 0
        self.documents = {d: f"Other {d}\nAn unrelated article {d}." for d in range(220)}

    def _document(self, doc_id):
        return self.documents[doc_id]


def state_for(engine, nodes, bindings, proof_rows):
    proofs = {}
    candidates = {d: {"sources": [], "verified": []} for d in engine.documents}
    for nid, doc_id, quote in proof_rows:
        proof = {"doc_id": doc_id, "answer": bindings[nid], "evidence": quote, "confidence": .95}
        proofs[nid] = proof
        candidates[doc_id]["verified"].append(dict(proof, goal="dag:" + nid))
    return {"_evidence_plan": nodes, "_evidence_winning_bindings": bindings,
            "_evidence_beams": [{"bindings": dict(bindings), "proofs": proofs}],
            "evidence_candidates": candidates,
            "evidence_trace": {"selected_prefix": [{"doc_id": d} for d in range(5)]}}


def geographic(engine):
    first = "Tessa grew up in Umina Beach, New South Wales."
    second = "Umina Beach is a suburb within the Central Coast Council local government area."
    engine.documents[1] = "Microwave Jenny\n" + first
    engine.documents[5] = "Umina Beach, New South Wales\n" + second
    nodes = [{"id": "root", "question": "Where did Tessa grow up?", "depends_on": [], "answer_type": "place"},
             {"id": "local", "question": "Which local government area does ${root.answer} belong to?",
              "depends_on": ["root"], "answer_type": "region"}]
    return state_for(engine, nodes, {"root": "Umina Beach", "local": "Central Coast Council"},
                     [("root", 1, first), ("local", 5, second)])


QUERY = "Who co-founded View Askew Productions and produced numerous movies starring Jason Lee?"


def founder(engine):
    source = "View Askew Productions is a production company founded by Kevin Smith and Scott Mosier in 1994."
    engine.documents[0] = "View Askew Productions\n" + source
    engine.documents[5] = ("Kevin Smith\nKevin Patrick Smith is an American filmmaker. "
                           "The films form a canon named after his company View Askew Productions, "
                           "which he co-founded with Scott Mosier. He co-produced Clerks.")
    engine.documents[3] = ("Jason Lee (actor)\nJason Michael Lee is an actor. "
                           "He is also known for his roles in Kevin Smith films such as Mallrats and Dogma.")
    nodes = [{"id": "founder", "question": QUERY, "depends_on": [], "answer_type": "person"}]
    return state_for(engine, nodes, {"founder": "Kevin Smith"}, [("founder", 0, source)])


def run(engine, state, query=QUERY, ids=None, scores=None):
    ids = np.arange(220) if ids is None else ids
    scores = np.arange(len(ids), 0, -1, dtype=float) if scores is None else scores
    return engine.finalize(query, ids, scores, {}, state)


class EvidencePackageTests(unittest.TestCase):
    def test_no_flag_returns_exact_parent_objects(self):
        engine = Engine(False)
        state = geographic(engine)
        ids, scores = np.arange(220), np.ones(220)
        actual = run(engine, state, ids=ids, scores=scores)
        self.assertIs(actual[0], ids)
        self.assertIs(actual[1], scores)
        self.assertIs(actual[2], state["evidence_trace"])
        self.assertEqual(engine.parent_calls, 1)

    def test_two_node_geographic_package_changes_only_three_to_ten(self):
        engine = Engine()
        state = geographic(engine)
        before = copy.deepcopy(state)
        ids, scores, trace = run(engine, state)
        self.assertEqual(ids[:5].tolist(), [0, 1, 2, 3, 5])
        self.assertEqual(ids[5], 4)
        self.assertEqual(ids[:2].tolist(), [0, 1])
        self.assertEqual(set(ids[:10]), set(range(10)))
        self.assertEqual(set(ids[:200]), set(range(200)))
        self.assertEqual(state, before)
        self.assertTrue(np.all(scores[:-1] > scores[1:]))
        diag = trace["improvement_package"]
        self.assertEqual(diag["extra_requests"], 0)
        self.assertEqual(diag["promotions"][0]["check"], "owned_geographic_membership")

    def test_single_node_founder_identity_and_selected_actor_condition(self):
        engine = Engine()
        ids, _, trace = run(engine, founder(engine))
        self.assertEqual(ids[4], 5)
        self.assertEqual(trace["improvement_package"]["promotions"][0]["check"],
                         "founder_identity_and_owned_actor_films")
        self.assertIn(3, trace["improvement_package"]["promotions"][0]["bundle_doc_ids"])

    def test_single_identity_needs_second_condition_not_just_same_name(self):
        engine = Engine()
        state = founder(engine)
        engine.documents[3] = "Jason Lee (actor)\nJason Lee is an actor. Kevin Smith is mentioned in a footnote."
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_creator_identity_requires_same_owned_work_on_both_documents(self):
        for role, verb in [("director", "directed"), ("author", "wrote"), ("composer", "composed")]:
            with self.subTest(role=role):
                engine = Engine()
                question = f"Who {verb} Sample Work?"
                quote = {"director": "Sample Work is a film directed by Alice Example.",
                         "author": "Sample Work is a novel written by Alice Example.",
                         "composer": "Sample Work was composed by Alice Example."}[role]
                engine.documents[0] = "Sample Work\n" + quote
                engine.documents[5] = f"Alice Example\nAlice Example is a {role}. She {verb} Sample Work."
                node = {"id": "creator", "question": question, "depends_on": [], "answer_type": "person"}
                state = state_for(engine, [node], {"creator": "Alice Example"}, [("creator", 0, quote)])
                ids, _, trace = run(engine, state, question)
                self.assertEqual(ids[4], 5)
                self.assertEqual(trace["improvement_package"]["promotions"][0]["check"], "creator_identity_and_owned_same_work")
                engine.documents[5] = f"Alice Example\nAlice Example is a {role}. Other Person {verb} Sample Work."
                self.assertEqual(run(engine, state, question)[0].tolist(), list(range(220)))

    def test_explicit_identity_type_conflict_is_rejected(self):
        engine = Engine()
        state = founder(engine)
        engine.documents[5] = "Kevin Smith\nKevin Smith is a comedy film. He co-founded View Askew Productions. He co-produced Clerks."
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_wrong_geographic_owner_not_promoted(self):
        engine = Engine()
        state = geographic(engine)
        bad = "Nearby Beach is within Central Coast Council. Umina Beach is mentioned elsewhere."
        engine.documents[5] = "Umina Beach, New South Wales\n" + bad
        state["_evidence_beams"][0]["proofs"]["local"]["evidence"] = bad
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_wrong_predecessor_subject_does_not_license_a_package(self):
        engine = Engine()
        state = geographic(engine)
        quote = "Tessa's father grew up in Umina Beach, New South Wales."
        engine.documents[1] = "Microwave Jenny\n" + quote
        state["_evidence_beams"][0]["proofs"]["root"]["evidence"] = quote
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_same_surname_wrong_person_identity_rejected(self):
        engine = Engine()
        state = founder(engine)
        engine.documents[5] = "John Smith\nJohn Smith is a filmmaker. He co-founded View Askew Productions."
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_wrong_subject_founder_does_not_license_identity(self):
        engine = Engine()
        state = founder(engine)
        bad = "Other Company was founded by Kevin Smith. View Askew Productions is mentioned elsewhere."
        engine.documents[0] = "View Askew Productions\n" + bad
        state["_evidence_beams"][0]["proofs"]["founder"]["evidence"] = bad
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_another_person_owns_founder_predicate_on_identity_page(self):
        engine = Engine()
        state = founder(engine)
        engine.documents[5] = ("Kevin Smith\nKevin Smith is a filmmaker. "
                               "John Brown was his colleague, and he co-founded View Askew Productions.")
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_actor_relative_does_not_transfer_work_relationship(self):
        engine = Engine()
        state = founder(engine)
        engine.documents[3] = ("Jason Lee (actor)\nJason Lee is an actor. "
                               "Jason Lee's father starred in Kevin Smith films.")
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_relation_date_conflict_rejected(self):
        engine = Engine()
        state = founder(engine)
        query = QUERY.replace("co-founded", "co-founded in 1994")
        engine.documents[5] = ("Kevin Smith\nKevin Smith is a filmmaker. "
                               "He co-founded View Askew Productions in 2005. He co-produced Clerks.")
        self.assertEqual(run(engine, state, query)[0].tolist(), list(range(220)))

    def test_confidence_is_not_a_replacement_for_source(self):
        for bad in ({"confidence": float("nan")}, {"confidence": .84},
                    {"evidence": "Invented non-document evidence for Central Coast Council."},
                    {"requirements": {"root": "Wrong Place"}}):
            with self.subTest(bad=bad):
                engine = Engine()
                state = geographic(engine)
                state["_evidence_beams"][0]["proofs"]["local"].update(bad)
                self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_cross_branch_cannot_supply_support(self):
        engine = Engine()
        state = geographic(engine)
        state["_evidence_beams"][0]["bindings"]["root"] = "Other Place"
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_complete_parallel_route_and_longer_chains_left_unchanged(self):
        engine = Engine()
        state = geographic(engine)
        state["evidence_trace"]["planning_outputs"] = {"routing": {"kind": "bridge_comparison", "expand": True}}
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))
        state = geographic(engine)
        state["_evidence_plan"].append({"id": "third", "question": "Where is ${local.answer}?",
                                         "depends_on": ["local"], "answer_type": "place"})
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_identity_candidate_must_be_within_existing_six_to_ten(self):
        engine = Engine()
        state = founder(engine)
        engine.documents[12], engine.documents[5] = engine.documents[5], engine.documents[12]
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_zero_budget_and_protected_prefix_disable_exchange(self):
        engine = Engine()
        state = geographic(engine)
        actual, diag = module.rerank_evidence_packages(QUERY, range(220), state, engine._document, max_promotions=0)
        self.assertEqual(actual, list(range(220)))
        self.assertEqual(diag["deferred"][-1]["reason"], "promotion_budget")
        for d in [2, 3, 4]:
            state["evidence_candidates"][d]["verified"] = [{"goal": "existing:protected"}]
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))

    def test_adaptation_creator_pronoun_is_grounded_by_same_document_identity(self):
        engine = Engine()
        first = 'Part of the rock opera Quadrophenia, the song was also released as a single.'
        second = "It was directed by Franc Roddam."
        engine.documents[0] = '5:15\n5:15 is a song written by Pete Townshend. ' + first
        engine.documents[5] = "Quadrophenia (film)\nQuadrophenia is a film based on the rock opera of the same name. " + second
        nodes = [{"id": "work", "question": "What rock opera includes 5:15?", "depends_on": [], "answer_type": "work"},
                 {"id": "director", "question": "Who directed the film based on ${work.answer}?",
                  "depends_on": ["work"], "answer_type": "person"}]
        state = state_for(engine, nodes, {"work": "Quadrophenia", "director": "Franc Roddam"},
                          [("work", 0, first), ("director", 5, second)])
        ids, _, trace = run(engine, state, "Who directed the film based on the rock opera 5:15 appeared in?")
        self.assertEqual(ids[4], 5)
        self.assertEqual(trace["improvement_package"]["promotions"][0]["check"], "owned_adaptation_creator")

    def test_adaptation_other_named_subject_not_transferred(self):
        engine = Engine()
        state = geographic(engine)
        state["_evidence_plan"][1]["question"] = "Who directed the film based on ${root.answer}?"
        state["_evidence_plan"][1]["answer_type"] = "person"
        state["_evidence_winning_bindings"]["local"] = "Franc Roddam"
        state["_evidence_beams"][0]["bindings"]["local"] = "Franc Roddam"
        quote = "Another Film was directed by Franc Roddam."
        engine.documents[5] = "Umina Beach (film)\nUmina Beach is a film based on a book. " + quote
        state["_evidence_beams"][0]["proofs"]["local"].update(answer="Franc Roddam", evidence=quote)
        self.assertEqual(run(engine, state)[0].tolist(), list(range(220)))


if __name__ == "__main__":
    unittest.main()
