"""CPU tests for failure classification, semantic slots and strict fallback."""
import copy
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_failure_recovery_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".evidence_failure_recovery", ROOT / "src/pathcondrag/evidence_failure_recovery.py")
failure = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = failure
spec.loader.exec_module(failure)
base = sys.modules[PACKAGE + ".evidence_retrieval"]


QUERY = "James Fowle Baldwin helped design a historic railroad that operated in Massachusetts and whose line later operated as part of what?"
PAYLOAD = {"nodes": [
    {"id": "r1", "question_template": "Which historic railroad operating in Massachusetts did James Fowle Baldwin help design?",
     "inputs": {}, "answer_type": "railroad"},
    {"id": "r2", "question_template": "The line of {{railroad}} later operated as part of what?",
     "inputs": {"railroad": "r1"}, "answer_type": "railroad"},
]}


def state(query=QUERY, plan=None):
    plan = copy.deepcopy(plan or [])
    return {"query": query, "static_sub_questions": [], "base": (None, None, {"hops": 2}),
            "_evidence_plan": plan, "_evidence_winning_bindings": {}, "_evidence_search_cache": {},
            "evidence_candidates": {0: {"base_score": .8, "sources": [{"source": "base"}, {"source": "dag:s1"}],
                                         "verified": [{"goal": "dag:s1", "answer": "Old answer"}]},
                                    9: {"base_score": 0, "sources": [{"source": "dag:s1"}], "verified": []}},
            "evidence_trace": {"plan": copy.deepcopy(plan), "bindings": {}, "branch_scores": [],
                               "routes": [{"source": "base"}, {"source": "dag:s1"}], "search_count": 4,
                               "llm_plan_calls": 2, "llm_verification_calls": 2, "verification_outputs": [],
                               "planning_outputs": {"validation_errors": ["dependency_reference_mismatch"]},
                               "semantic_failures": [], "rejected_hypotheses": []}}


class Parent:
    _collect_jobs = staticmethod(base.EvidenceRetrieval._collect_jobs)

    def _dependency_search(self, states, executor):
        self.batches.append(len(states))
        for sample in states:
            plan = sample["_evidence_plan"]
            if not plan or not plan[0]["id"].startswith("r"):
                continue
            self.clone_before_search.append(copy.deepcopy(sample))
            trace = sample["evidence_trace"]
            trace["search_count"] = min(self.max_searches, trace["search_count"] + 2)
            trace["llm_verification_calls"] += 2
            sample["_evidence_search_cache"]["repair search"] = ([1], [.9])
            bindings = {node["id"]: f"Answer {index}" for index, node in enumerate(plan)}
            if not self.complete:
                bindings.pop(plan[-1]["id"], None)
            sample["_evidence_winning_bindings"], trace["bindings"] = bindings, bindings
            trace["verification_outputs"].append({"attempts": 2, "outputs": [], "node": plan[0]["id"]})
            for index, node in enumerate(plan):
                if node["id"] not in bindings:
                    continue
                sample["evidence_candidates"][index + 1] = {
                    "base_score": 0, "sources": [{"source": f"dag:{node['id']}"}],
                    "verified": [{"goal": f"dag:{node['id']}", "answer": bindings[node["id"]],
                                  "doc_id": index + 1, "evidence": f"Answer {index} is the requested entity.",
                                  "confidence": .9, "quality": .9}]}

    def _verify(self, question, answer_type, docs):
        key = question + "::" + ",".join(map(str, docs))
        self._verification_diagnostics[key] = {"attempts": 2, "outputs": [{"hypotheses": []}]}
        self.verifies += 1
        return self.original_result


class Engine(failure.FailureRecoveryMixin, Parent):
    def __init__(self, enabled=True, payload=PAYLOAD, complete=True, supported=True):
        self.improvements = {"failure_recovery"} if enabled else set()
        self.payload = copy.deepcopy(payload)
        self.complete, self.supported = complete, supported
        self.batches, self.clone_before_search, self.requests = [], [], []
        self.max_searches = 20
        self._verification_diagnostics = {}
        self.verifies = 0
        self.original_result = ([], [], None)

    def _infer_object(self, messages):
        self.requests.append(copy.deepcopy(messages))
        return self.payload, None

    def _document(self, doc_id):
        return f"Answer {doc_id - 1} is the requested entity."

    def _failure_task_support(self, state, plan):
        return [] if self.supported else [{"node": plan[0]["id"], "reason": "unknown_source_relation"}]


class TemplateTests(unittest.TestCase):
    def test_slots_program_render_literal_references_and_dependencies(self):
        rendered, reason = failure.render_repair_templates(PAYLOAD)
        self.assertIsNone(reason)
        self.assertEqual(rendered["nodes"][1]["question"], "The line of ${r1.answer} later operated as part of what?")
        self.assertEqual(rendered["nodes"][1]["depends_on"], ["r1"])

    def test_missing_slot_does_not_get_dependency_appended(self):
        bad = copy.deepcopy(PAYLOAD)
        bad["nodes"][1]["question_template"] = "Whose line later operated as part of what?"
        self.assertEqual(failure.render_repair_templates(bad)[1], "slot_input_mismatch")

    def test_declared_dependency_but_no_inputs_is_rejected(self):
        bad = copy.deepcopy(PAYLOAD)
        bad["nodes"][1].update(question_template="Whose line later operated as part of what?", inputs={}, depends_on=["r1"])
        self.assertEqual(failure.render_repair_templates(bad)[1], "declared_dependency_without_semantic_slot")

    def test_unknown_slots_numeric_ids_and_model_written_literals_rejected(self):
        for kind in ("slot", "id", "literal", "role", "regarding"):
            bad = copy.deepcopy(PAYLOAD)
            if kind == "slot":
                bad["nodes"][1]["question_template"] = "What is {{country}}?"
            elif kind == "id":
                bad["nodes"][0]["id"] = "1"
            elif kind == "literal":
                bad["nodes"][1]["question_template"] = "What is ${r1.answer}?"
            elif kind == "role":
                bad["nodes"][1].update(question_template="What is {{foo}}?", inputs={"foo": "r1"})
            else:
                bad["nodes"][1]["question_template"] = "Whose line later operated as part of what regarding {{railroad}}?"
            self.assertIsNotNone(failure.render_repair_templates(bad)[1], kind)

    def test_unknown_or_cycle_dependency_is_rejected(self):
        bad = copy.deepcopy(PAYLOAD)
        bad["nodes"][1]["inputs"]["railroad"] = "r3"
        self.assertEqual(failure.render_repair_templates(bad)[1], "unknown_or_self_dependency")
        bad = copy.deepcopy(PAYLOAD)
        bad["nodes"][0].update(question_template="Who designed {{railroad}}?", inputs={"railroad": "r2"})
        self.assertEqual(failure.validate_failure_repair(bad, QUERY)[1], "cyclic_dependencies")

    def test_valid_historic_question_and_possessive_identity_normalization(self):
        self.assertIsNone(failure.validate_failure_repair(PAYLOAD, QUERY)[1])
        query = "Where is Pfrang Association's headquarters?"
        payload = {"nodes": [{"id": "r1", "question_template": "Where is the headquarters of Pfrang Association?", "inputs": {}, "answer_type": "place"}]}
        self.assertIsNone(failure.validate_failure_repair(payload, query)[1])

    def test_year_country_named_work_and_role_drops_rejected(self):
        query = 'Who wrote the main 2014 American novel "Work X"?'
        payload = {"nodes": [{"id": "r1", "question_template": query, "inputs": {}, "answer_type": "person"}]}
        self.assertIsNone(failure.validate_failure_repair(payload, query)[1])
        for value in ("main", "2014", "American", "Work X"):
            bad = copy.deepcopy(payload)
            bad["nodes"][0]["question_template"] = query.replace(value, "")
            self.assertIsNotNone(failure.validate_failure_repair(bad, query)[1], value)

    def test_pure_comparison_pruned_without_dropping_evidence_nodes(self):
        payload = {"nodes": [
            {"id": "r1", "question_template": "Who directed Film X?", "inputs": {}, "answer_type": "person"},
            {"id": "r2", "question_template": "Who directed Film Y?", "inputs": {}, "answer_type": "person"},
            {"id": "r3", "question_template": "Are {{first}} and {{second}} the same person?", "inputs": {"first": "r1", "second": "r2"}, "answer_type": "comparison"}]}
        nodes, reason, detail = failure.validate_failure_repair(payload, "Do Film X and Film Y have the same director?")
        self.assertIsNone(reason)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(detail["pruned_nodes"]), 1)

    def test_six_nodes_four_depth_capacity_unchanged(self):
        payload = {"nodes": [{"id": f"r{i}", "question_template": "Who directed Film X?" if i == 1 else "Who directed {{person}}?",
                              "inputs": {} if i == 1 else {"person": f"r{i-1}"}, "answer_type": "person"} for i in range(1, 6)]}
        self.assertEqual(failure.validate_failure_repair(payload, "Who directed Film X?")[1], "dependency_depth_exceeded")


class ClassificationTests(unittest.TestCase):
    def test_pure_quote_failure_atomic_performer_plan_does_not_replan(self):
        plan = [{"id": "s1", "question": "Who is the performer of song Changed It?", "depends_on": [], "answer_type": "person"},
                {"id": "s2", "question": "Where was ${s1.answer} born?", "depends_on": ["s1"], "answer_type": "place"}]
        original = state("What is the place of birth of the performer of song Changed It?", plan)
        original["evidence_trace"]["rejected_hypotheses"] = [{"node": "s1", "reason": "evidence_not_exact_substring"}]
        trigger, reason = failure.classify_recovery_failure(original)
        self.assertIsNone(trigger)
        self.assertEqual(reason, "citation_only_failure_keep_structure")

    def test_compound_attribute_lookup_is_real_unbound_structure(self):
        node = {"id": "s1", "question": "Where was the author of Work X born?", "depends_on": [], "answer_type": "place"}
        self.assertEqual(failure.classify_recovery_failure(state(node["question"], [node]))[0]["kind"], "compound_unknown_relation")

    def test_legacy_star_attribute_and_nested_actor_qualifiers_are_structural(self):
        questions = [
            "What is one of the stars of The Newcomers known for?",
            'Who is the actor in the CW show developed by Greg Berlanti who also appears in "Gods and Generals"?',
        ]
        for question in questions:
            node = {"id": "s1", "question": question, "depends_on": [], "answer_type": "attribute"}
            self.assertEqual(failure.classify_recovery_failure(state(question, [node]))[0]["kind"], "compound_unknown_relation")

    def test_existing_complete_but_state_used_as_town_triggers(self):
        plan = [{"id": "s1", "question": "Which state is WLUJ licensed in?", "depends_on": [], "answer_type": "place"},
                {"id": "s2", "question": "When did the town ${s1.answer} become capital?", "depends_on": ["s1"], "answer_type": "year"}]
        original = state("When did the town WLUJ is licensed in become capital?", plan)
        original["_evidence_winning_bindings"] = {"s1": "Illinois", "s2": "1839"}
        self.assertEqual(failure.classify_recovery_failure(original)[0]["kind"], "variable_type_conflict")

    def test_unjoined_terminal_constraint_triggers_even_when_fully_bound(self):
        plan = [{"id": "s1", "question": "Which city is WLUJ licensed in?", "depends_on": [], "answer_type": "city"},
                {"id": "s2", "question": "Where is the statue?", "depends_on": [], "answer_type": "place"},
                {"id": "s3", "question": "When did ${s1.answer} become capital?", "depends_on": ["s1"], "answer_type": "year"}]
        original = state("When did WLUJ's town become capital of the state containing the statue?", plan)
        original["_evidence_winning_bindings"] = {"s1": "Town", "s2": "State", "s3": "1839"}
        trigger = failure.classify_recovery_failure(original)[0]
        self.assertEqual(trigger["kind"], "unjoined_constraint_branch")
        self.assertEqual(trigger["node_ids"], ["s2"])

    def test_explicit_political_body_child_in_law_and_release_direction_conflicts(self):
        fixtures = [
            ("What is the political party holding the majority?", "U.S. Senate", "The U.S. Senate passed a bill.", "political_body_bound_as_party"),
            ("Who is Person X's child-in-law?", "Jane Smith", "Their daughter was Jane Smith.", "child_bound_as_child_in_law"),
            ("Which team was Jim Wilson released by?", "Seattle Mariners", "He was released by the Indians, then signed with the Seattle Mariners.", "released_by_object_conflict"),
        ]
        for question, answer, quote, expected in fixtures:
            node = {"id": "s1", "question": question, "depends_on": [], "answer_type": "organization"}
            original = state(question, [node])
            original["_evidence_winning_bindings"] = {"s1": answer}
            original["evidence_candidates"][0]["verified"] = [{"goal": "dag:s1", "answer": answer, "evidence": quote}]
            self.assertEqual(failure.classify_recovery_failure(original)[0]["reason"], expected)

    def test_clear_witness_contradiction_triggers_unknown_does_not(self):
        node = {"id": "s1", "question": "Who directed Film X?", "depends_on": [], "answer_type": "person"}
        original = state(node["question"], [node])
        original["evidence_trace"]["rejected_hypotheses"] = [{"node": "s1", "reason": "contradicted_owner:someone_else"}]
        self.assertEqual(failure.classify_recovery_failure(original)[0]["kind"], "source_relation_contradiction")
        original["evidence_trace"]["rejected_hypotheses"][0]["reason"] = "unknown_relation"
        self.assertIsNone(failure.classify_recovery_failure(original)[0])

    def test_complete_safe_plan_and_unrelated_rejected_candidate_stay_original(self):
        node = {"id": "s1", "question": "Who directed Film X?", "depends_on": [], "answer_type": "person"}
        original = state(node["question"], [node])
        original["_evidence_winning_bindings"] = {"s1": "Jane Smith"}
        original["evidence_trace"]["rejected_hypotheses"] = [{"node": "s1", "reason": "contradicted_owner:wrong_other_candidate"}]
        self.assertIsNone(failure.classify_recovery_failure(original)[0])


class ExecutionTests(unittest.TestCase):
    def test_disabled_exact_delegate_no_cache_or_diagnostic_side_effect(self):
        engine = Engine(enabled=False)
        original = state()
        before = copy.deepcopy(original)
        engine._dependency_search([original], None)
        self.assertEqual(original, before)
        self.assertFalse(engine.requests)
        self.assertFalse(hasattr(engine, "_failure_verification_cache"))
        self.assertIs(engine._verify("Q", "person", {1: "D"}), engine.original_result)

    def test_snapshot_is_deep_copy_of_parent_before_additional_spend(self):
        original, engine = state(), Engine()
        before = copy.deepcopy(original["evidence_trace"])
        engine._dependency_search([original], None)
        snapshot = original["evidence_trace"]["improvement_failure_recovery"]["original_parent_fields"]
        self.assertEqual(snapshot, {key: before.get(key) for key in failure._SNAPSHOT_FIELDS})
        self.assertEqual(snapshot["llm_plan_calls"], 2)
        self.assertEqual(original["evidence_trace"]["llm_plan_calls"], 3)
        original["evidence_trace"]["routes"].append({"source": "later"})
        self.assertEqual(snapshot["routes"], before["routes"])

    def test_valid_repair_applies_only_complete_supported_structure(self):
        original, engine = state(), Engine()
        engine._dependency_search([original], None)
        diagnostic = original["evidence_trace"]["improvement_failure_recovery"]
        self.assertTrue(diagnostic["applied"])
        self.assertEqual(diagnostic["extra_plan_requests"], 1)
        self.assertEqual(diagnostic["extra_searches"], 2)
        self.assertEqual(diagnostic["extra_verification_requests"], 2)
        self.assertIn("${r1.answer}", original["_evidence_plan"][1]["question"])
        self.assertNotIn(9, original["evidence_candidates"])
        self.assertEqual(original["evidence_candidates"][0]["verified"], [])

    def test_incomplete_or_unknown_semantic_support_rolls_back_inputs_and_records_spend(self):
        for options in ({"complete": False}, {"supported": False}):
            engine, original = Engine(**options), state()
            before = copy.deepcopy(original)
            engine._dependency_search([original], None)
            for key in ("_evidence_plan", "_evidence_winning_bindings", "evidence_candidates"):
                self.assertEqual(original[key], before[key])
            self.assertEqual(original["evidence_trace"]["search_count"], 6)
            self.assertEqual(original["evidence_trace"]["llm_verification_calls"], 4)
            self.assertFalse(original["evidence_trace"]["improvement_failure_recovery"]["applied"])

    def test_illegal_schema_does_not_search_and_counts_one_new_planning(self):
        bad = copy.deepcopy(PAYLOAD)
        bad["nodes"][1]["question_template"] = "Whose line later operated as part of what?"
        engine, original = Engine(payload=bad), state()
        engine._dependency_search([original], None)
        self.assertEqual(len(engine.requests), 1)
        self.assertEqual(original["evidence_trace"]["search_count"], 4)
        self.assertEqual(original["evidence_trace"]["llm_verification_calls"], 2)

    def test_batched_repair_requests_preserve_existing_search_cap(self):
        engine = Engine()
        a, b, exhausted = state(), state(), state()
        a["evidence_trace"]["search_count"] = 19
        exhausted["evidence_trace"]["search_count"] = 20
        with ThreadPoolExecutor(max_workers=8) as executor:
            engine._dependency_search([a, b, exhausted], executor)
        self.assertEqual(engine.batches, [3, 2])
        self.assertEqual(len(engine.requests), 2)
        self.assertEqual(a["evidence_trace"]["search_count"], 20)
        self.assertEqual(exhausted["evidence_trace"]["improvement_failure_recovery"]["result"], "existing_search_budget_exhausted")

    def test_cached_verification_exact_docs_and_type_only_during_repair(self):
        engine = Engine()
        engine._failure_verification_cache, engine._failure_repair_active = {}, False
        engine._verify("Q", "person", {1: "Document"})
        engine._failure_repair_active = True
        engine._verify("Q", "person", {1: "Document"})
        self.assertEqual(engine.verifies, 1)
        self.assertEqual(engine._verification_diagnostics["Q::1"]["attempts"], 0)
        engine._verify("Q", "person", {1: "Other document"})
        engine._verify("Q", "organization", {1: "Document"})
        self.assertEqual(engine.verifies, 3)

    def test_person_list_splitting_is_literal_bounded_and_only_in_repair(self):
        engine = Engine()
        quote = "The film starred Christopher McCoy, Kate Bosworth, Paul Dano and Chris Evans."
        engine.original_result = ([{"answer": "Christopher McCoy, Kate Bosworth, Paul Dano and Chris Evans", "evidence": quote, "doc_id": 1, "confidence": .9}], [], None)
        engine._failure_verification_cache, engine._failure_repair_active = {}, False
        original = engine._verify("Who starred?", "person", {1: quote})
        self.assertIs(original, engine.original_result)
        engine._failure_repair_active = True
        repaired, _rejected, _reason = engine._verify("Who starred?", "person", {1: quote})
        self.assertEqual(len(repaired), 3)
        self.assertIn("Chris Evans", [proof["answer"] for proof in repaired])
        self.assertFalse(any("," in proof["answer"] for proof in repaired))
        self.assertEqual(engine.verifies, 1)
        self.assertFalse(failure._person_list("Charles Willoughby, 10th Baron Willoughby", "Charles Willoughby, 10th Baron Willoughby was a person."))
        self.assertFalse(failure._person_list("Jane Smith and John Jones", "Only Jane Smith is documented."))


if __name__ == "__main__":
    unittest.main()
