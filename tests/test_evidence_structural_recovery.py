"""CPU tests for bounded, opt-in structural repair and strict fallback."""
import copy
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_evidence_structural_recovery_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    PACKAGE + ".evidence_structural_recovery", ROOT / "src/pathcondrag/evidence_structural_recovery.py")
recovery = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recovery
spec.loader.exec_module(recovery)
base = sys.modules[PACKAGE + ".evidence_retrieval"]


QUERY = "James Fowle Baldwin helped design a historic railroad that operated in Massachusetts and whose line later operated as part of what?"
PLAN = {"nodes": [
    {"id": "r1", "question": "Which historic railroad operating in Massachusetts did James Fowle Baldwin help design?",
     "depends_on": [], "answer_type": "railroad"},
    {"id": "r2", "question": "The line of ${r1.answer} later operated as part of what?",
     "depends_on": ["r1"], "answer_type": "railroad"},
]}


def state(query=QUERY, plan=None, errors=None):
    if plan is None:
        plan = []
    return {
        "query": query, "static_sub_questions": ["Which railroad did James Fowle Baldwin design?"],
        "base": (None, None, {"hops": 2}), "_evidence_plan": copy.deepcopy(plan),
        "_evidence_search_cache": {"old search": ([0], [.8])}, "_evidence_winning_bindings": {},
        "evidence_candidates": {0: {"base_score": .8, "sources": [{"source": "base"}, {"source": "dag:s1"}],
                                     "verified": [{"goal": "dag:s1", "answer": "Old"}]},
                                7: {"base_score": 0, "sources": [{"source": "dag:s1"}], "verified": []}},
        "evidence_trace": {"plan": copy.deepcopy(plan), "search_count": 4, "llm_plan_calls": 2,
                           "llm_verification_calls": 2, "routes": [{"source": "base"}, {"source": "dag:s1"}],
                           "planning_outputs": {"validation_errors": errors or [], "outputs": []},
                           "bindings": {}, "verification_outputs": [], "semantic_failures": [],
                           "rejected_hypotheses": [], "branch_scores": []},
    }


class Parent:
    _collect_jobs = staticmethod(base.EvidenceRetrieval._collect_jobs)

    def _dependency_search(self, states, executor):
        self.batches.append([s["query"] for s in states])
        for s in states:
            if not s["_evidence_plan"] or s["_evidence_plan"][0]["id"] != "r1":
                continue
            self.clone_before_search.append(copy.deepcopy(s))
            s["evidence_trace"]["search_count"] = min(self.max_searches, s["evidence_trace"]["search_count"] + 2)
            s["evidence_trace"]["llm_verification_calls"] += 2
            s["_evidence_search_cache"]["repair search"] = ([1], [.9])
            bindings = {node["id"]: f"Answer {index}" for index, node in enumerate(s["_evidence_plan"])}
            if not self.succeed:
                bindings.pop(s["_evidence_plan"][-1]["id"], None)
            s["_evidence_winning_bindings"] = bindings
            s["evidence_trace"]["bindings"] = bindings
            s["evidence_trace"]["verification_outputs"].append({"attempts": 2, "outputs": []})
            for index, node in enumerate(s["_evidence_plan"]):
                if node["id"] not in bindings or not self.cited:
                    continue
                s["evidence_candidates"][index + 1] = {
                    "base_score": 0, "sources": [{"source": f"dag:{node['id']}"}],
                    "verified": [{"goal": f"dag:{node['id']}", "answer": f"Answer {index}",
                                  "doc_id": index + 1, "evidence": f"Answer {index} is the requested entity.",
                                  "confidence": .9, "quality": .9}],
                }

    def _verify(self, question, answer_type, docs):
        key = question + "::" + ",".join(map(str, docs))
        self._verification_diagnostics[key] = {"attempts": 2, "outputs": [{"hypotheses": []}]}
        self.verify_calls += 1
        return self.original_verification


class Engine(recovery.StructuralRecoveryMixin, Parent):
    def __init__(self, enabled=True, payload=PLAN, succeed=True, cited=True):
        self.improvements = {"structural_recovery"} if enabled else set()
        self.payload = copy.deepcopy(payload)
        self.succeed, self.cited = succeed, cited
        self.batches, self.clone_before_search, self.infer_calls = [], [], []
        self.max_searches = 20
        self._verification_diagnostics = {}
        self.verify_calls = 0
        self.original_verification = ([], [], None)

    def _infer_object(self, messages):
        self.infer_calls.append(copy.deepcopy(messages))
        return self.payload, None

    def _document(self, doc_id):
        return f"Answer {doc_id - 1} is the requested entity."


class TriggerTests(unittest.TestCase):
    def test_invalid_placeholder_diagnosed_from_original_empty_plan(self):
        original = state(errors=["dependency_reference_mismatch", "declared_dependency_without_placeholder"])
        trigger, reason = recovery.structural_recovery_trigger(original)
        self.assertIsNone(reason)
        self.assertEqual(trigger["kind"], "invalid_plan")
        self.assertEqual(trigger["reason"], "declared_dependency_without_placeholder")

    def test_empty_plan_without_validation_errors_has_bounded_retry_signal(self):
        self.assertEqual(recovery.structural_recovery_trigger(state())[0]["kind"], "empty_plan")

    def test_missing_intermediate_is_diagnostic_but_failed_terminal_alone_is_not(self):
        original = state(plan=PLAN["nodes"])
        self.assertEqual(recovery.structural_recovery_trigger(original)[0]["kind"], "unbound_intermediate")
        original["_evidence_winning_bindings"] = {"r1": "Railroad X"}
        self.assertEqual(recovery.structural_recovery_trigger(original)[1], "no_diagnostic_structure_failure")

    def test_compound_root_trigger_is_generic_and_simple_unknown_root_is_kept(self):
        node = {"id": "s1", "question": "Where was the author of Novel X born?", "depends_on": [], "answer_type": "place"}
        original = state(query=node["question"], plan=[node])
        self.assertEqual(recovery.structural_recovery_trigger(original)[0]["kind"], "compound_unknown_root")
        original["_evidence_plan"][0]["question"] = "Where was Person X born?"
        self.assertEqual(recovery.structural_recovery_trigger(original)[1], "no_diagnostic_structure_failure")

    def test_complete_dag_preserved_even_if_previous_planner_attempt_failed(self):
        original = state(plan=PLAN["nodes"], errors=["dependency_reference_mismatch"])
        original["_evidence_winning_bindings"] = {"r1": "Railroad X", "r2": "Railroad Y"}
        self.assertEqual(recovery.structural_recovery_trigger(original)[1], "existing_complete_dag")


class ValidationTests(unittest.TestCase):
    def test_valid_generic_repair_preserves_question_constraints(self):
        nodes, reason, _detail = recovery.validate_structural_repair(PLAN, QUERY)
        self.assertIsNone(reason)
        self.assertEqual([node["id"] for node in nodes], ["r1", "r2"])

    def test_illegal_placeholder_and_numeric_ids_rejected(self):
        for kind in ("placeholder", "numeric_id"):
            payload = copy.deepcopy(PLAN)
            if kind == "placeholder":
                payload["nodes"][1]["question"] = "The line of r1.answer later operated as part of what?"
            else:
                payload["nodes"][0]["id"] = "1"
            self.assertIsNotNone(recovery.validate_structural_repair(payload, QUERY)[1])

    def test_declared_missing_reference_does_not_become_root(self):
        payload = copy.deepcopy(PLAN)
        payload["nodes"][1]["question"] = "Whose line later operated as part of what?"
        self.assertEqual(recovery.validate_structural_repair(payload, QUERY)[1], "declared_dependency_without_placeholder")

    def test_canonicalization_uses_existing_placeholder_only(self):
        payload = copy.deepcopy(PLAN)
        payload["nodes"][1]["depends_on"] = []
        nodes, reason, detail = recovery.validate_structural_repair(payload, QUERY)
        self.assertIsNone(reason)
        self.assertEqual(nodes[1]["depends_on"], ["r1"])
        self.assertTrue(detail["canonicalization_changes"])

    def test_named_work_person_country_year_and_role_qualifiers_cannot_disappear(self):
        payload = {"nodes": [
            {"id": "r1", "question": 'Who is the main writer of the 2014 American novel "Work X"?', "depends_on": [], "answer_type": "person"},
            {"id": "r2", "question": "Where was ${r1.answer} born?", "depends_on": ["r1"], "answer_type": "place"},
        ]}
        query = 'Where was the main writer of the 2014 American novel "Work X" born?'
        self.assertIsNone(recovery.validate_structural_repair(payload, query)[1])
        for token in ("main", "2014", "American", "Work X"):
            bad = copy.deepcopy(payload)
            bad["nodes"][0]["question"] = bad["nodes"][0]["question"].replace(token, "")
            self.assertIsNotNone(recovery.validate_structural_repair(bad, query)[1], token)

    def test_invented_proper_name_and_terminal_relation_swap_rejected(self):
        payload = copy.deepcopy(PLAN)
        payload["nodes"][0]["question"] += " in London"
        self.assertEqual(recovery.validate_structural_repair(payload, QUERY)[1], "repair_introduced_named_identity")
        payload = {"nodes": [{"id": "r1", "question": "Where was Person X born?", "depends_on": [], "answer_type": "place"}]}
        self.assertEqual(recovery.validate_structural_repair(payload, "When did Person X die?")[1], "relation_attribute_mismatch_death_to_birth")

    def test_only_pure_derived_comparison_is_pruned(self):
        payload = {"nodes": [
            {"id": "r1", "question": "Who directed Film X?", "depends_on": [], "answer_type": "person"},
            {"id": "r2", "question": "Who directed Film Y?", "depends_on": [], "answer_type": "person"},
            {"id": "r3", "question": "Are ${r1.answer} and ${r2.answer} the same person?", "depends_on": ["r1", "r2"], "answer_type": "comparison"},
        ]}
        nodes, reason, detail = recovery.validate_structural_repair(payload, "Do Film X and Film Y have the same director?")
        self.assertIsNone(reason)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(detail["pruned_nodes"]), 1)

    def test_node_and_depth_caps_remain_six_and_four(self):
        payload = {"nodes": [{"id": f"r{i}", "question": "Who directed Film X?" if i == 1 else f"Who directed ${{r{i - 1}.answer}}?",
                              "depends_on": [] if i == 1 else [f"r{i - 1}"], "answer_type": "person"} for i in range(1, 6)]}
        self.assertEqual(recovery.validate_structural_repair(payload, "Who directed Film X?")[1], "dependency_depth_exceeded")
        payload = {"nodes": [{"id": f"r{i}", "question": "Who directed Film X?", "depends_on": [], "answer_type": "person"} for i in range(1, 8)]}
        self.assertEqual(recovery.validate_structural_repair(payload, "Who directed Film X?")[1], "invalid_node_count")


class RecoveryTests(unittest.TestCase):
    def test_flag_off_exact_delegation_without_recovery_cache_or_diagnostics(self):
        engine = Engine(enabled=False)
        original = state()
        before = copy.deepcopy(original)
        engine._dependency_search([original], None)
        self.assertEqual(original, before)
        self.assertFalse(engine.infer_calls)
        self.assertFalse(hasattr(engine, "_structural_verification_cache"))
        self.assertIs(engine._verify("Q", "person", {0: "Passage"}), engine.original_verification)

    def test_complete_plan_and_nonstructural_failure_make_no_request(self):
        engine = Engine()
        complete = state(plan=PLAN["nodes"])
        complete["_evidence_winning_bindings"] = {"r1": "Railroad X", "r2": "Railroad Y"}
        simple = state(plan=[{"id": "s1", "question": "Who directed Film X?", "depends_on": [], "answer_type": "person"}])
        engine._dependency_search([complete, simple], None)
        self.assertFalse(engine.infer_calls)
        self.assertEqual(complete["_evidence_plan"], PLAN["nodes"])

    def test_invalid_plan_repair_only_one_planning_request_and_keeps_candidate_pool(self):
        engine = Engine(payload={"nodes": [{"id": "1"}]})
        original = state(errors=["dependency_reference_mismatch"])
        candidates = copy.deepcopy(original["evidence_candidates"])
        engine._dependency_search([original], None)
        self.assertEqual(len(engine.infer_calls), 1)
        self.assertEqual(original["_evidence_plan"], [])
        self.assertEqual(original["evidence_candidates"], candidates)
        self.assertEqual(original["evidence_trace"]["llm_plan_calls"], 3)
        self.assertIn("dependency_reference_mismatch", engine.infer_calls[0][1]["content"])

    def test_failed_repair_preserves_plan_candidates_bindings_and_records_spend(self):
        engine = Engine(succeed=False)
        original = state(errors=["dependency_reference_mismatch"])
        before = copy.deepcopy(original)
        engine._dependency_search([original], None)
        for key in ("_evidence_plan", "evidence_candidates", "_evidence_winning_bindings"):
            self.assertEqual(original[key], before[key])
        self.assertEqual(original["evidence_trace"]["search_count"], 6)
        self.assertEqual(original["evidence_trace"]["llm_verification_calls"], 4)
        self.assertEqual(original["_evidence_search_cache"]["repair search"], ([1], [.9]))
        self.assertEqual(original["evidence_trace"]["improvement_structural_recovery"]["result"], "incomplete_repair_preserved_original_dag")

    def test_complete_bindings_without_citations_cannot_replace_original(self):
        engine = Engine(cited=False)
        original = state()
        engine._dependency_search([original], None)
        self.assertEqual(original["_evidence_plan"], [])
        self.assertFalse(original["evidence_trace"]["improvement_structural_recovery"]["applied"])

    def test_grounded_success_removes_old_dag_proof_source_and_dag_only_candidate(self):
        engine = Engine()
        original = state(errors=["dependency_reference_mismatch"])
        engine._dependency_search([original], None)
        self.assertEqual(original["_evidence_plan"], PLAN["nodes"])
        self.assertEqual(original["evidence_candidates"][0]["sources"], [{"source": "base"}])
        self.assertEqual(original["evidence_candidates"][0]["verified"], [])
        self.assertNotIn(7, original["evidence_candidates"])
        self.assertTrue(original["evidence_trace"]["improvement_structural_recovery"]["applied"])

    def test_equivalent_renamed_plan_does_not_repeat_search_or_verification(self):
        engine = Engine()
        old = copy.deepcopy(PLAN["nodes"])
        for node in old:
            node["id"] = node["id"].replace("r", "s")
            node["question"] = node["question"].replace("${r", "${s")
            node["depends_on"] = [dep.replace("r", "s") for dep in node["depends_on"]]
        original = state(plan=old)
        engine._dependency_search([original], None)
        self.assertEqual(len(engine.batches), 2)
        self.assertEqual(engine.batches[1], [])
        self.assertEqual(original["evidence_trace"]["llm_verification_calls"], 2)
        self.assertEqual(original["evidence_trace"]["improvement_structural_recovery"]["result"], "unchanged_plan_skipped")

    def test_original_search_budget_applies_to_repairs_and_exhausted_questions_skip(self):
        engine = Engine()
        a, b = state(), state()
        a["evidence_trace"]["search_count"] = 19
        b["evidence_trace"]["search_count"] = 20
        engine._dependency_search([a, b], None)
        self.assertEqual(len(engine.infer_calls), 1)
        self.assertEqual(a["evidence_trace"]["search_count"], 20)
        self.assertEqual(b["evidence_trace"]["improvement_structural_recovery"]["result"], "existing_search_budget_exhausted")

    def test_repair_plans_and_dependency_search_are_batched_in_original_executor(self):
        engine = Engine()
        originals = [state(), state(), state()]
        with ThreadPoolExecutor(max_workers=8) as executor:
            engine._dependency_search(originals, executor)
        self.assertEqual([len(batch) for batch in engine.batches], [3, 3])
        self.assertEqual(len(engine.infer_calls), 3)
        self.assertTrue(all(item["evidence_trace"]["improvement_structural_recovery"]["extra_plan_requests"] == 1 for item in originals))

    def test_exact_existing_verification_cached_only_during_repair(self):
        engine = Engine()
        engine._structural_verification_cache = {}
        engine._structural_recovery_reusing = False
        docs = {1: "Answer 0 is the requested entity."}
        engine.original_verification = ([{"answer": "Answer 0"}], [], None)
        engine._verify("Atomic question?", "person", docs)
        engine._structural_recovery_reusing = True
        reused = engine._verify("Atomic question?", "person", docs)
        self.assertEqual(reused, engine.original_verification)
        self.assertEqual(engine.verify_calls, 1)
        self.assertEqual(engine._verification_diagnostics["Atomic question?::1"]["attempts"], 0)
        engine._verify("Atomic question?", "person", {1: "Different passage"})
        engine._verify("Atomic question?", "work", docs)
        self.assertEqual(engine.verify_calls, 3)

    def test_grounding_gate_rejects_fabricated_quote_even_with_full_bindings(self):
        original = state(plan=PLAN["nodes"])
        original["_evidence_winning_bindings"] = {"r1": "Railroad X", "r2": "Railroad Y"}
        original["evidence_candidates"] = {
            i: {"verified": [{"goal": f"dag:r{i}", "answer": f"Railroad {'X' if i == 1 else 'Y'}",
                             "doc_id": i, "evidence": "Invented Railroad X statement.", "confidence": .9}]}
            for i in (1, 2)
        }
        self.assertFalse(recovery._complete_cited_repair(original, PLAN["nodes"], lambda _id: "Actual source text."))

    def test_real_dependency_search_binds_repaired_reference_and_uses_original_verifier(self):
        class Rag:
            pcrag_config = SimpleNamespace(improvement_stage=4, evidence_max_searches=20)

            def dense_passage_retrieval(self, question):
                doc_id = 1 if "line of Railroad X" in question else 0
                return np.array([doc_id]), np.array([.9])

        class ActualEngine(recovery.StructuralRecoveryMixin, base.EvidenceRetrieval):
            def __init__(self):
                self.improvements = {"structural_recovery"}
                super().__init__(Rag())
                self._documents = {
                    0: "James Fowle Baldwin helped design historic Railroad X, which operated in Massachusetts.",
                    1: "The line of Railroad X later operated as part of Railroad Y.",
                }
                self.plan_calls, self.verify_calls = 0, 0

            def _infer_object(self, messages):
                if "Repair a diagnosed" in messages[0]["content"]:
                    self.plan_calls += 1
                    return copy.deepcopy(PLAN), None
                self.verify_calls += 1
                doc_id = 1 if "[D1]" in messages[1]["content"] else 0
                return {"hypotheses": [{"answer": "Railroad Y" if doc_id == 1 else "Railroad X",
                                         "evidence": self._documents[doc_id], "doc_id": f"D{doc_id}",
                                         "confidence": .95}]}, None

        engine = ActualEngine()
        original = state(errors=["dependency_reference_mismatch"])
        with ThreadPoolExecutor(max_workers=8) as executor:
            engine._dependency_search([original], executor)
        self.assertEqual(engine.plan_calls, 1)
        self.assertEqual(engine.verify_calls, 2)
        self.assertEqual(original["_evidence_winning_bindings"], {"r1": "Railroad X", "r2": "Railroad Y"})
        self.assertTrue(original["evidence_trace"]["improvement_structural_recovery"]["applied"])
        repaired_routes = original["evidence_trace"]["routes"]
        self.assertIn("The line of Railroad X later operated as part of what?", [route.get("query") for route in repaired_routes])


if __name__ == "__main__":
    unittest.main()
