import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src/pathcondrag"
PACKAGE = "_planning_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("evidence_retrieval")
planning = load_module("evidence_planning")


class Engine(planning.PlannerImprovementMixin, planning.AdaptiveSearchMixin, base.EvidenceRetrieval):
    def __init__(self, rag, improvements):
        self.improvements = frozenset(improvements)
        super().__init__(rag)


class FakeRAG:
    def __init__(self, responses):
        self.pcrag_config = SimpleNamespace(improvement_stage=4)
        self.responses = iter(responses)
        self.calls = []
        self.passage_node_keys = list(range(8))
        self.documents = {i: "No requested fact is stated in this document." for i in range(8)}
        self.documents[0] = "Alpha directed Film A in 1990."
        self.documents[4] = "Beta directed Film B in 1991."
        self.chunk_embedding_store = SimpleNamespace(get_row=lambda key: {"content": self.documents[key]})
        self.llm_model = SimpleNamespace(infer=self.infer)

    def infer(self, messages, temperature):
        self.calls.append(messages)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value) if isinstance(value, dict) else value, {"finish_reason": "stop"}

    @staticmethod
    def _extract_llm_text(value):
        return str(value)

    def dense_passage_retrieval(self, query):
        return np.arange(8), np.linspace(1.0, 0.1, 8)


def comparison_plan():
    return {"nodes": [
        {"id": "a1", "question": "Who directed Film A?", "depends_on": [], "answer_type": "person"},
        {"id": "a2", "question": "When was ${a1.answer} born?", "depends_on": ["a1"], "answer_type": "date"},
        {"id": "b1", "question": "Who directed Film B?", "depends_on": [], "answer_type": "person"},
        {"id": "b2", "question": "When was ${b1.answer} born?", "depends_on": ["b1"], "answer_type": "date"},
    ]}


def answer(name, doc_id, quote):
    return {"hypotheses": [{"answer": name, "doc_id": f"D{doc_id}",
                            "evidence": quote, "confidence": 0.9}]}


class PlanningImprovementTests(unittest.TestCase):
    def test_two_depth_hint_accepts_four_atomic_parallel_nodes(self):
        rag = FakeRAG([comparison_plan()])
        engine = Engine(rag, {"planning"})
        nodes, reason = engine._plan("Whose director was born earlier, Film A or Film B?", 2)
        self.assertIsNone(reason)
        self.assertEqual(len(nodes), 4)
        diagnostics = next(iter(engine._plan_diagnostics.values()))
        self.assertEqual(diagnostics["depth_hint"], 2)
        self.assertEqual(diagnostics["actual_depth"], 2)
        self.assertEqual(diagnostics["attempts"], 1)
        prompt = str(rag.calls[0])
        self.assertIn("single passage", prompt)
        self.assertIn("Do not compress", prompt)
        self.assertIn("NOT the total number", prompt)
        self.assertNotIn("gold", prompt)

    def test_disabling_planning_preserves_original_node_limit(self):
        rag = FakeRAG([comparison_plan(), comparison_plan()])
        nodes, reason = Engine(rag, set())._plan("Compare directors", 2)
        self.assertEqual(nodes, [])
        self.assertEqual(reason, "invalid_node_count")
        self.assertEqual(len(rag.calls), 2)

    def test_depth_limit_is_independent_from_total_node_capacity(self):
        payload = {"nodes": []}
        for index in range(5):
            deps = [f"s{index}"] if index else []
            question = f"What is ${{s{index}.answer}}?" if index else "Who wrote Work X?"
            payload["nodes"].append({"id": f"s{index+1}", "question": question, "depends_on": deps})
        self.assertEqual(planning.validate_retrieval_plan(payload, 6, 4)[1], "dependency_depth_exceeded")
        self.assertIsNone(planning.validate_retrieval_plan(payload, 6, 5)[1])

    def test_comparison_is_not_an_extra_retrieval_node(self):
        payload = comparison_plan()
        payload["nodes"].append({"id": "final", "question": "Compare ${a2.answer} and ${b2.answer}",
                                 "depends_on": ["a2", "b2"], "answer_type": "comparison"})
        self.assertEqual(planning.validate_retrieval_plan(payload, 6, 4)[1],
                         "derived_comparison_is_not_retrieval")

    def test_planning_repair_and_http_failure_are_bounded(self):
        rag = FakeRAG([{"nodes": []}, comparison_plan()])
        engine = Engine(rag, {"planning"})
        self.assertEqual(len(engine._plan("Compare", 2)[0]), 4)
        self.assertEqual(engine._plan_diagnostics["Compare"]["attempts"], 2)
        rag = FakeRAG([ConnectionError("HTTP unavailable")])
        with self.assertRaisesRegex(ConnectionError, "HTTP unavailable"):
            Engine(rag, {"planning"})._plan("Compare", 2)
        self.assertEqual(len(rag.calls), 1)


class CanonicalPlanValidationTests(unittest.TestCase):
    def engine(self, responses, mode="canonical_refs"):
        rag = FakeRAG(responses)
        rag.pcrag_config.evidence_plan_validation = mode
        return Engine(rag, {"planning"}), rag

    def test_existing_refs_repair_missing_and_redundant_metadata_without_rewriting_questions(self):
        payload = {"nodes": [
            {"id": "s1", "question": "Which saint is a cathedral dedicated to?", "depends_on": []},
            {"id": "s2", "question": "Which basilica is named after ${s1.answer}?", "depends_on": []},
            {"id": "s3", "question": "Who governs the city containing ${s2.answer}?",
             "depends_on": ["s1", "s2", "s2"]},
        ]}
        original = json.dumps(payload)
        engine, rag = self.engine([payload])
        nodes, reason = engine._plan("Which governor oversees that basilica's city?", 3)
        self.assertIsNone(reason)
        self.assertEqual([n["depends_on"] for n in nodes], [[], ["s1"], ["s2"]])
        self.assertEqual([n["question"] for n in nodes], [n["question"] for n in payload["nodes"]])
        self.assertEqual(json.dumps(payload), original)
        self.assertEqual(len(rag.calls), 1)
        changes = next(iter(engine._plan_diagnostics.values()))["canonicalization_changes"][0]["changes"]
        self.assertEqual(changes[0]["added"], ["s1"])
        self.assertEqual(changes[1]["removed"], ["s1"])

    def test_actual_missing_placeholder_requires_llm_repair_instead_of_becoming_root(self):
        bad = {"nodes": [
            {"id": "s1", "question": "Which saint is Mantua Cathedral dedicated to?", "depends_on": []},
            {"id": "s2", "question": "Which basilica is named after the saint Mantua Cathedral is dedicated to?",
             "depends_on": ["s1"]},
        ]}
        corrected = {"nodes": [bad["nodes"][0], dict(bad["nodes"][1],
                    question="Which basilica is named after ${s1.answer}?")]}
        engine, rag = self.engine([bad, corrected])
        nodes, reason = engine._plan("Which basilica shares the cathedral's saint?", 2)
        self.assertIsNone(reason)
        self.assertEqual(nodes[1]["depends_on"], ["s1"])
        self.assertIn("${s1.answer}", nodes[1]["question"])
        self.assertEqual(len(rag.calls), 2)
        diagnostics = next(iter(engine._plan_diagnostics.values()))
        self.assertEqual(diagnostics["validation_errors"], ["declared_dependency_without_placeholder"])

    def test_birth_instead_of_requested_death_is_rejected_then_repaired_in_second_call(self):
        bad = comparison_plan()
        corrected = comparison_plan()
        for node in corrected["nodes"]:
            if node["depends_on"]:
                node["question"] = node["question"].replace("When was", "When did").replace(" born?", " die?")
        query = "Which film's director died earlier, Film A or Film B?"
        engine, rag = self.engine([bad, corrected])
        nodes, reason = engine._plan(query, 2)
        self.assertIsNone(reason)
        self.assertEqual(len(nodes), 4)
        self.assertEqual(len(rag.calls), 2)
        self.assertEqual(engine._plan_diagnostics[query]["validation_errors"],
                         ["relation_attribute_mismatch_death_to_birth"])
        self.assertIn("death, birth and tenure end are different facts", str(rag.calls[-1]))
        self.assertTrue(all(" die?" in n["question"] for n in nodes if n["depends_on"]))

    def test_two_attribute_failures_never_trigger_third_request(self):
        engine, rag = self.engine([comparison_plan(), comparison_plan()])
        nodes, reason = engine._plan("Which director died earlier?", 2)
        self.assertEqual(nodes, [])
        self.assertEqual(reason, "relation_attribute_mismatch_death_to_birth")
        self.assertEqual(len(rag.calls), 2)

    def test_unknown_self_and_cyclic_references_remain_invalid(self):
        for ref, deps, reason in (("unknown", [], "unknown_or_self_dependency"),
                                  ("s2", [], "unknown_or_self_dependency"),
                                  ("s1", ["unknown"], "unknown_or_self_dependency")):
            payload = {"nodes": [
                {"id": "s1", "question": "Who wrote Work X?", "depends_on": []},
                {"id": "s2", "question": "Where was ${" + ref + ".answer} born?", "depends_on": deps},
            ]}
            self.assertEqual(planning.canonicalize_plan_dependencies(payload)[2], reason)
        payload = {"nodes": [
            {"id": "s1", "question": "Who is ${s2.answer}?", "depends_on": []},
            {"id": "s2", "question": "Who is ${s1.answer}?", "depends_on": []},
        ]}
        normalized, _, reason = planning.canonicalize_plan_dependencies(payload)
        self.assertIsNone(reason)
        self.assertEqual(planning.validate_retrieval_plan(normalized, 6, 4)[1], "cyclic_dependencies")

    def test_fake_or_incomplete_placeholder_cannot_be_promoted_to_root(self):
        for ref in ("${s1}", "${s1.value}", "${s1.answer", "{s1.answer}", "s1.answer", "${1.answer}"):
            payload = {"nodes": [
                {"id": "s1", "question": "Who wrote Work X?", "depends_on": []},
                {"id": "s2", "question": f"Where was {ref} born?", "depends_on": []},
            ]}
            with self.subTest(ref=ref):
                self.assertEqual(planning.canonicalize_plan_dependencies(payload)[2],
                                 "invalid_dependency_placeholder")

    def test_wrong_attribute_guard_allows_required_death_node_and_intermediate_birth(self):
        nodes = [{"question": "Where was Person A born?"}, {"question": "When did Person A die?"}]
        self.assertIsNone(planning.relation_faithfulness_error("When did Person A die?", nodes))
        self.assertEqual(planning.relation_faithfulness_error("When was Person A born?", [nodes[1]]),
                         "relation_attribute_mismatch_birth_to_death")

    def test_requested_tenure_endpoint_cannot_be_replaced_by_birth_date(self):
        query = "What date did the Governor of the city containing that basilica end?"
        bad = [{"question": "Who governed that city?"}, {"question": "What date was the governor born?"}]
        good = [{"question": "Who governed that city?"}, {"question": "What date did the governor's tenure end?"}]
        self.assertEqual(planning.relation_faithfulness_error(query, bad),
                         "relation_attribute_mismatch_end_to_birth")
        self.assertIsNone(planning.relation_faithfulness_error(query, good))
        self.assertIsNone(planning.relation_faithfulness_error("What date was a person in West End born?", bad))

    def test_default_and_explicit_strict_keep_original_prompt_and_validation(self):
        query = "Which film's director died earlier, Film A or Film B?"
        default_rag = FakeRAG([comparison_plan()])
        default = Engine(default_rag, {"planning"})
        strict, strict_rag = self.engine([comparison_plan()], "strict")
        self.assertEqual(default._plan(query, 2), strict._plan(query, 2))
        self.assertEqual(default_rag.calls, strict_rag.calls)
        self.assertEqual(default._plan_diagnostics, strict._plan_diagnostics)
        self.assertNotIn("Relation faithfulness:", str(strict_rag.calls))
        self.assertNotIn("canonicalization_changes", strict._plan_diagnostics[query])

    def test_new_mode_preserves_capacity_and_depth_limits(self):
        payload = {"nodes": []}
        for index in range(5):
            payload["nodes"].append({"id": f"s{index + 1}",
                                     "question": f"Who is ${{s{index}.answer}}?" if index else "Who wrote Work X?",
                                     "depends_on": []})
        engine, rag = self.engine([payload, payload])
        self.assertEqual(engine._plan("Follow an evidence chain", 3)[1], "dependency_depth_exceeded")
        self.assertEqual(len(rag.calls), 2)
        payload = {"nodes": [{"id": f"s{i}", "question": "Who wrote Work X?", "depends_on": []}
                             for i in range(7)]}
        engine, rag = self.engine([payload, payload])
        self.assertEqual(engine._plan("Retrieve evidence", 2)[1], "invalid_node_count")
        self.assertEqual(len(rag.calls), 2)


class AdaptiveSearchTests(unittest.TestCase):
    def docs(self, engine):
        return engine._verification_documents(list(range(8)))

    def test_successful_first_batch_does_not_spend_extra_request(self):
        rag = FakeRAG([answer("Alpha", 0, "Alpha directed Film A in 1990.")])
        engine = Engine(rag, {"adaptive"})
        accepted, rejected, reason = engine._verify("Who directed Film A?", "person", self.docs(engine))
        self.assertEqual(accepted[0]["answer"], "Alpha")
        self.assertEqual(len(rag.calls), 1)
        diagnostics = next(v for k, v in engine._verification_diagnostics.items() if k.endswith("0,1,2,3,4,5"))
        self.assertEqual(diagnostics["adaptive_extra_batches"], 0)

    def test_missing_fact_checks_only_the_next_three_documents(self):
        rag = FakeRAG([{"hypotheses": []}, answer("Beta", 4, "Beta directed Film B in 1991.")])
        engine = Engine(rag, {"adaptive"})
        accepted, rejected, reason = engine._verify("Who directed Film B?", "person", self.docs(engine))
        self.assertIsNone(reason)
        self.assertEqual(accepted[0]["doc_id"], 4)
        self.assertEqual(len(rag.calls), 2)
        self.assertNotIn("[D4]", str(rag.calls[0]))
        self.assertIn("[D4]", str(rag.calls[1]))
        diagnostic = engine._verification_diagnostics["Who directed Film B?::0,1,2,3,4,5"]
        self.assertEqual(diagnostic["attempts"], 2)
        self.assertEqual(diagnostic["adaptive_extra_batches"], 1)
        self.assertEqual(diagnostic["verified_document_batches"], [[0, 1, 2], [3, 4, 5]])

    def test_empty_fallback_is_not_recursive(self):
        rag = FakeRAG([{"hypotheses": []}, {"hypotheses": []}])
        engine = Engine(rag, {"adaptive"})
        accepted, rejected, reason = engine._verify("Who wrote Missing?", "person", self.docs(engine))
        self.assertEqual(accepted, [])
        self.assertIsNone(reason)
        self.assertEqual(len(rag.calls), 2)

    def test_protocol_or_http_failure_never_triggers_evidence_fallback(self):
        rag = FakeRAG(["invalid JSON", "invalid JSON"])
        engine = Engine(rag, {"adaptive"})
        self.assertEqual(engine._verify("Who?", "person", self.docs(engine))[2], "invalid_json_object")
        self.assertEqual(len(rag.calls), 2)
        rag = FakeRAG([ConnectionError("HTTP failed")])
        engine = Engine(rag, {"adaptive"})
        with self.assertRaisesRegex(ConnectionError, "HTTP failed"):
            engine._verify("Who?", "person", self.docs(engine))
        self.assertEqual(len(rag.calls), 1)

    def test_two_beams_only_when_distinct_verified_scores_are_close(self):
        engine = Engine(FakeRAG([]), {"adaptive"})
        beams = [
            {"bindings": {"s1": "Alpha"}, "proofs": {}, "qualities": [.9]},
            {"bindings": {"s1": "Beta"}, "proofs": {}, "qualities": [.88]},
            {"bindings": {"s1": "Gamma"}, "proofs": {}, "qualities": [.4]},
        ]
        self.assertEqual(len(engine._prune_beams(beams, 2)), 2)
        self.assertEqual(len(engine._prune_beams([beams[0], beams[2]], 2)), 1)
        duplicate = dict(beams[0])
        self.assertEqual(len(engine._prune_beams([beams[0], duplicate], 2)), 1)
        self.assertEqual(engine.max_searches, 20)

    def test_disabling_adaptive_keeps_three_documents_and_one_beam(self):
        engine = Engine(FakeRAG([]), {"planning"})
        self.assertEqual(list(self.docs(engine)), [0, 1, 2])
        self.assertEqual(engine.beam_width, 1)

    def test_verify_only_expands_missing_evidence_without_retaining_second_beam(self):
        rag = FakeRAG([{"hypotheses": []}, answer("Beta", 4, "Beta directed Film B in 1991.")])
        rag.pcrag_config.evidence_adaptive_mode = "verify_only"
        engine = Engine(rag, {"adaptive"})
        accepted, _, reason = engine._verify("Who directed Film B?", "person", self.docs(engine))
        self.assertIsNone(reason)
        self.assertEqual(accepted[0]["doc_id"], 4)
        self.assertEqual(len(rag.calls), 2)
        self.assertEqual(engine.beam_width, 1)
        beams = [{"bindings": {"s1": "Alpha"}, "proofs": {}, "qualities": [.9]},
                 {"bindings": {"s1": "Beta"}, "proofs": {}, "qualities": [.88]}]
        self.assertEqual(len(engine._prune_beams(beams, 2)), 1)

    def test_beam_only_retains_close_alternative_without_expanding_verification(self):
        rag = FakeRAG([{"hypotheses": []}])
        rag.pcrag_config.evidence_adaptive_mode = "beam_only"
        engine = Engine(rag, {"adaptive"})
        self.assertEqual(list(self.docs(engine)), [0, 1, 2])
        accepted, _, reason = engine._verify("Who directed Film B?", "person", self.docs(engine))
        self.assertIsNone(reason)
        self.assertFalse(accepted)
        self.assertEqual(len(rag.calls), 1)
        beams = [{"bindings": {"s1": "Alpha"}, "proofs": {}, "qualities": [.9]},
                 {"bindings": {"s1": "Beta"}, "proofs": {}, "qualities": [.88]}]
        self.assertEqual(len(engine._prune_beams(beams, 2)), 2)

    def test_explicit_both_replays_the_default_adaptive_requests(self):
        responses = [{"hypotheses": []}, answer("Beta", 4, "Beta directed Film B in 1991.")]
        default_rag, explicit_rag = FakeRAG(responses), FakeRAG(responses)
        explicit_rag.pcrag_config.evidence_adaptive_mode = "both"
        left, right = Engine(default_rag, {"adaptive"}), Engine(explicit_rag, {"adaptive"})
        self.assertEqual(left._verify("Who directed Film B?", "person", self.docs(left)),
                         right._verify("Who directed Film B?", "person", self.docs(right)))
        self.assertEqual(default_rag.calls, explicit_rag.calls)
        self.assertEqual(left._verification_diagnostics, right._verification_diagnostics)


if __name__ == "__main__":
    unittest.main()
