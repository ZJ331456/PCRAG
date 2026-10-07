"""CPU checks for bounded missing-bridge and identity recovery."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_evidence_bridge_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".evidence_bridge", ROOT / "src/pathcondrag/evidence_bridge.py")
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)
base = sys.modules[PACKAGE + ".evidence_retrieval"]


ROLE_QUERY = "On which station did the 2014 South Korean television series, starring the main rapper of boyband SS501, air?"
PRODUCT_QUERY = 'Which internationl football team has had a player endorse the "Nike Hypervenom" football boot?'
ROLE_PLAN = {"nodes": [
    {"id": "r1", "question": "Who is the main rapper of boyband SS501?", "depends_on": [], "answer_type": "person"},
    {"id": "r2", "question": "Which 2014 South Korean television series starred ${r1.answer}?", "depends_on": ["r1"], "answer_type": "work"},
    {"id": "r3", "question": "On which station did ${r2.answer} air?", "depends_on": ["r2"], "answer_type": "organization"},
]}
PRODUCT_PLAN = {"nodes": [
    {"id": "r1", "question": "Which player endorses Nike Hypervenom football boots?", "depends_on": [], "answer_type": "person"},
    {"id": "r2", "question": "Which national team does ${r1.answer} represent?", "depends_on": ["r1"], "answer_type": "team"},
]}


def hypothesis(answer, quote):
    return {"answer": answer, "evidence": quote, "doc_id": "D0", "confidence": .9}


def state(query=ROLE_QUERY):
    old_plan = [{"id": "s1", "question": "Which series starred the main rapper of boyband SS501?", "depends_on": [], "answer_type": "work"}]
    return {"query": query, "static_sub_questions": ["Who is the main rapper of boyband SS501?"],
            "_evidence_plan": old_plan, "_evidence_search_cache": {}, "_evidence_winning_bindings": {},
            "evidence_candidates": {0: {"sources": [{"source": "base"}, {"source": "dag:s1"}],
                                          "verified": [{"goal": "dag:s1"}]}},
            "evidence_trace": {"plan": copy.deepcopy(old_plan), "search_count": 4,
                               "llm_plan_calls": 2, "llm_verification_calls": 2,
                               "routes": [{"source": "base"}, {"source": "dag:s1"}], "planning_outputs": {},
                               "bindings": {}, "verification_outputs": []}}


class Parent:
    _collect_jobs = staticmethod(base.EvidenceRetrieval._collect_jobs)

    def _dependency_search(self, states, executor):
        self.batches.append([s["query"] for s in states])
        for s in states:
            if s["_evidence_plan"] and s["_evidence_plan"][0]["id"] == "r1":
                s["evidence_trace"]["search_count"] = min(self.max_searches, s["evidence_trace"]["search_count"] + 2)
                s["evidence_trace"]["llm_verification_calls"] += 2
                s["_evidence_search_cache"]["repair"] = ([1], [.9])
                bindings = ({n["id"]: f"Answer {i}" for i, n in enumerate(s["_evidence_plan"])}
                            if self.succeed else {"r1": "Answer 1"})
                s["_evidence_winning_bindings"] = bindings
                s["evidence_trace"]["bindings"] = bindings
                s["evidence_trace"]["verification_outputs"].append({"attempts": 2, "outputs": []})

    def _verify(self, question, answer_type, docs):
        key = question + "::" + ",".join(map(str, docs))
        self._verification_diagnostics[key] = {"attempts": 2, "outputs": self.outputs}
        self.verify_calls += 1
        return self.original_verification


class Engine(bridge.BridgeRecoveryMixin, Parent):
    def __init__(self, enabled=True, payload=ROLE_PLAN, succeed=True):
        self.improvements = {"bridge_recovery"} if enabled else set()
        self.payload = copy.deepcopy(payload)
        self.succeed = succeed
        self.batches = []
        self.max_searches = 20
        self.infer_calls = []
        self._verification_diagnostics = {}
        self.outputs = []
        self.original_verification = ([], [{"reason": "answer_not_supported_in_quote"}], None)
        self.verify_calls = 0

    def _infer_object(self, messages):
        self.infer_calls.append(copy.deepcopy(messages))
        return self.payload, None


class IdentityTests(unittest.TestCase):
    def test_adjacent_pronoun_recovers_exact_contiguous_quote(self):
        doc = "Mr. Tumnus\nMr. Tumnus is a fictional character. He is a close friend of Lucy Pevensie."
        item = hypothesis("Mr. Tumnus", "He is a close friend of Lucy Pevensie.")
        recovered, reason = bridge.recover_source_identity_quote(item, doc, "Which character is a friend of Lucy Pevensie?", "character")
        self.assertEqual(reason, "contiguous_document_identity")
        self.assertEqual(recovered["evidence"], doc)

    def test_document_anchored_surname_recovers(self):
        doc = "Pierre Cuypers\nPierre Cuypers was an architect. Cuypers designed the Rijksmuseum."
        recovered, _ = bridge.recover_source_identity_quote(hypothesis("Pierre Cuypers", "Cuypers designed the Rijksmuseum."), doc, "Which architect designed the Rijksmuseum?", "person")
        self.assertIsNotNone(recovered)
        self.assertIn("Pierre Cuypers", recovered["evidence"])

    def test_short_firstname_requires_same_document_topic(self):
        doc = "Donald Duck\nDonald Duck is a cartoon character. Donald was voiced by Clarence Nash."
        recovered, _ = bridge.recover_source_identity_quote(hypothesis("Donald Duck", "Donald was voiced by Clarence Nash."), doc, "Which cartoon character was voiced by Clarence Nash?", "character")
        self.assertIsNotNone(recovered)

    def test_relation_target_must_match_question(self):
        doc = "Pierre Cuypers\nPierre Cuypers was an architect. Cuypers designed the Rotterdam bridge."
        recovered, reason = bridge.recover_source_identity_quote(
            hypothesis("Pierre Cuypers", "Cuypers designed the Rotterdam bridge."),
            doc, "Which architect designed the Rijksmuseum?", "person")
        self.assertIsNone(recovered)
        self.assertEqual(reason, "relationship_not_locally_supported")

    def test_surname_after_different_full_named_antecedent_is_not_resolved(self):
        doc = "Pierre Cuypers\nPierre Cuypers was an architect. Joseph Cuypers was his relative. Cuypers designed the church."
        recovered, reason = bridge.recover_source_identity_quote(
            hypothesis("Pierre Cuypers", "Cuypers designed the church."),
            doc, "Which architect designed the church?", "person")
        self.assertIsNone(recovered)
        self.assertEqual(reason, "different_named_antecedent")

    def test_different_person_same_surname_rejected(self):
        doc = "Pierre Cuypers\nPierre Cuypers was an architect. Joseph Cuypers designed the church."
        recovered, reason = bridge.recover_source_identity_quote(hypothesis("Pierre Cuypers", "Joseph Cuypers designed the church."), doc, "Which architect designed the church?", "person")
        self.assertIsNone(recovered)
        self.assertEqual(reason, "different_full_name_in_quote")

    def test_different_person_same_firstname_rejected(self):
        doc = "Donald Duck\nDonald Duck is a cartoon character. Donald Trump was voiced by John Smith."
        recovered, _ = bridge.recover_source_identity_quote(hypothesis("Donald Duck", "Donald Trump was voiced by John Smith."), doc, "Which character was voiced by John Smith?")
        self.assertIsNone(recovered)

    def test_pronoun_after_different_subject_rejected(self):
        doc = "Alice Smith\nAlice Smith is an actor. Mary Jones was her friend. She starred in Film X."
        recovered, reason = bridge.recover_source_identity_quote(hypothesis("Alice Smith", "She starred in Film X."), doc, "Who starred in Film X?", "person")
        self.assertIsNone(recovered)
        self.assertEqual(reason, "pronoun_not_adjacent_to_identity")

    def test_family_property_not_transferred(self):
        doc = "Alice Smith\nAlice Smith is an actor. Her father was born in Paris."
        recovered, _ = bridge.recover_source_identity_quote(hypothesis("Alice Smith", "Her father was born in Paris."), doc, "Where was Alice Smith born?", "person")
        self.assertIsNone(recovered)

    def test_paraphrase_never_recovers(self):
        doc = "Mr. Tumnus\nMr. Tumnus is a character. He is a close friend of Lucy Pevensie."
        recovered, reason = bridge.recover_source_identity_quote(hypothesis("Mr. Tumnus", "He befriended Lucy Pevensie."), doc, "Who befriended Lucy Pevensie?")
        self.assertIsNone(recovered)
        self.assertEqual(reason, "original_quote_not_exact")

    def test_unrecognized_relation_keeps_original_unknown(self):
        doc = "Alice Smith\nAlice Smith is an actor. She traveled near Paris."
        recovered, reason = bridge.recover_source_identity_quote(hypothesis("Alice Smith", "She traveled near Paris."), doc, "Which county near Paris did Alice Smith visit?")
        self.assertIsNone(recovered)
        self.assertEqual(reason, "relationship_not_locally_supported")

    def test_long_span_and_bad_reference_rejected(self):
        doc = "Alice Smith\nAlice Smith is an actor. " + "padding " * 200 + "She starred in Film X."
        self.assertIsNone(bridge.recover_source_identity_quote(hypothesis("Alice Smith", "She starred in Film X."), doc, "Who starred in Film X?")[0])
        doc = "Alice Smith\nAlice Smith is an actor. She starred in Film X."
        item = dict(hypothesis("Alice Smith", "She starred in Film X."), doc_id="bad")
        self.assertIsNone(bridge.recover_source_identity_quote(item, doc, "Who starred in Film X?")[0])


class BridgePlanTests(unittest.TestCase):
    def test_two_patterns_and_parallel_exclusion(self):
        self.assertEqual(bridge.bridge_recovery_kind(ROLE_QUERY), "role_work_station")
        self.assertEqual(bridge.bridge_recovery_kind(PRODUCT_QUERY), "product_player_team")
        self.assertIsNone(bridge.bridge_recovery_kind("Which film director of Film A and Film B was born earlier?"))
        self.assertIsNone(bridge.bridge_recovery_kind("Where was the writer of Work X born?"))

    def test_valid_bounded_repair_plan(self):
        nodes, reason = bridge._repair_plan_reason(ROLE_PLAN, ROLE_QUERY, "role_work_station")
        self.assertIsNone(reason)
        self.assertEqual(len(nodes), 3)
        self.assertIsNone(bridge._repair_plan_reason(PRODUCT_PLAN, PRODUCT_QUERY, "product_player_team")[1])

    def test_numeric_id_missing_placeholder_and_changed_year_rejected(self):
        for kind in ("id", "placeholder", "year", "subject", "country"):
            bad = copy.deepcopy(ROLE_PLAN)
            if kind == "id":
                bad["nodes"][0]["id"] = "1"
            elif kind == "placeholder":
                bad["nodes"][1]["question"] = "Which 2014 South Korean television series starred the rapper?"
            elif kind == "year":
                bad["nodes"][1]["question"] = bad["nodes"][1]["question"].replace("2014", "2015")
            elif kind == "subject":
                bad["nodes"][0]["question"] = "Who is the main rapper of boyband EXO?"
            else:
                bad["nodes"][1]["question"] = bad["nodes"][1]["question"].replace("South Korean", "American")
            with self.subTest(kind=kind):
                self.assertIsNotNone(bridge._repair_plan_reason(bad, ROLE_QUERY, "role_work_station")[1])

    def test_disabled_is_exact_delegate(self):
        engine = Engine(enabled=False)
        original = state()
        before = copy.deepcopy(original)
        engine._dependency_search([original], None)
        self.assertEqual(original, before)
        self.assertEqual(len(engine.infer_calls), 0)
        result = engine._verify("Unknown?", "unknown", {0: "Document"})
        self.assertIs(result, engine.original_verification)

    def test_complete_chain_no_new_request_or_replanning(self):
        engine = Engine()
        original = state()
        original["_evidence_winning_bindings"] = {"s1": "Drama X"}
        plan = copy.deepcopy(original["_evidence_plan"])
        engine._dependency_search([original], None)
        self.assertEqual(original["_evidence_plan"], plan)
        self.assertFalse(engine.infer_calls)
        self.assertEqual(original["evidence_trace"]["improvement_bridge"]["deferred_reason"], "existing_complete_chain")

    def test_invalid_repair_preserves_original_dag_once(self):
        engine = Engine(payload={"nodes": [{"id": "1"}]})
        original = state()
        plan = copy.deepcopy(original["_evidence_plan"])
        candidates = copy.deepcopy(original["evidence_candidates"])
        engine._dependency_search([original], None)
        self.assertEqual(original["_evidence_plan"], plan)
        self.assertEqual(original["evidence_candidates"], candidates)
        self.assertEqual(len(engine.infer_calls), 1)
        self.assertEqual(original["evidence_trace"]["llm_plan_calls"], 3)
        self.assertIn("never numeric", engine.infer_calls[0][0]["content"])

    def test_failed_repair_preserves_dag_and_records_spent_budget(self):
        engine = Engine(succeed=False)
        original = state()
        plan = copy.deepcopy(original["_evidence_plan"])
        candidates = copy.deepcopy(original["evidence_candidates"])
        engine._dependency_search([original], None)
        self.assertEqual(original["_evidence_plan"], plan)
        self.assertEqual(original["evidence_candidates"], candidates)
        self.assertEqual(original["evidence_trace"]["search_count"], 6)
        self.assertEqual(original["evidence_trace"]["llm_verification_calls"], 4)
        self.assertEqual(original["_evidence_search_cache"]["repair"], ([1], [.9]))

    def test_success_replaces_semantic_ids_without_stale_dag_proofs(self):
        engine = Engine()
        original = state()
        engine._dependency_search([original], None)
        self.assertEqual([n["id"] for n in original["_evidence_plan"]], ["r1", "r2", "r3"])
        self.assertEqual(original["evidence_candidates"][0]["verified"], [])
        self.assertEqual(original["evidence_candidates"][0]["sources"], [{"source": "base"}])
        self.assertTrue(original["evidence_trace"]["improvement_bridge"]["applied"])

    def test_repairs_batched_and_original_search_budget_preserved(self):
        engine = Engine()
        a, b = state(), state()
        a["evidence_trace"]["search_count"] = 19
        engine._dependency_search([a, b], None)
        self.assertEqual(len(engine.infer_calls), 2)
        self.assertEqual([len(batch) for batch in engine.batches], [2, 2])
        self.assertEqual(a["evidence_trace"]["search_count"], 20)

    def test_source_identity_uses_original_two_request_outputs_without_new_calls(self):
        doc = "Mr. Tumnus\nMr. Tumnus is a fictional character. He is a close friend of Lucy Pevensie."
        engine = Engine()
        engine.outputs = [{"hypotheses": [hypothesis("Mr. Tumnus", "He is a close friend of Lucy Pevensie.")]}]
        accepted, _rejected, reason = engine._verify("Which character is a friend of Lucy Pevensie?", "character", {0: doc})
        self.assertIsNone(reason)
        self.assertEqual(accepted[0]["answer"], "Mr. Tumnus")
        self.assertEqual(engine.verify_calls, 1)
        self.assertEqual(engine.infer_calls, [])
        self.assertEqual(engine._verification_diagnostics["Which character is a friend of Lucy Pevensie?::0"]["attempts"], 2)

    def test_existing_accepted_hypothesis_not_modified(self):
        engine = Engine()
        engine.original_verification = ([{"answer": "Existing"}], [], None)
        result = engine._verify("Unknown?", "unknown", {0: "Existing"})
        self.assertIs(result, engine.original_verification)


if __name__ == "__main__":
    unittest.main()
