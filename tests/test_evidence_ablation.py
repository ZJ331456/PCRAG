"""CPU checks for pool isolation, productive call budgets and evidence gates."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_evidence_ablation_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package


def load_module(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name,
                                                  ROOT / "src/pathcondrag" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("evidence_retrieval")
runtime = load_module("evidence_ablation")

PLAN = {"nodes": [
    {"id": "s1", "question": "Who wrote Work X?", "depends_on": [], "answer_type": "person"},
    {"id": "s2", "question": "Where was ${s1.answer} born?", "depends_on": ["s1"], "answer_type": "place"},
]}
DOCS = ["Biography\nAlpha wrote Work X in 1990.",
        "Biography\nAlpha was born in Rome in 1960.",
        "Biography\nBeta wrote Work X in 1989.",
        "Biography\nBeta was born in Paris in 1958."]
ALPHA = {"hypotheses": [{"answer": "Alpha", "doc_id": "D0",
                         "evidence": "Alpha wrote Work X in 1990.", "confidence": .9}]}
ROME = {"hypotheses": [{"answer": "Rome", "doc_id": "D1",
                        "evidence": "Alpha was born in Rome in 1960.", "confidence": .9}]}


class FakeLLM:
    def __init__(self, responses=(), responder=None):
        self.responses = iter(responses)
        self.responder = responder
        self.lock = threading.Lock()
        self.calls = []

    def infer(self, messages, temperature):
        with self.lock:
            self.calls.append(messages)
            result = self.responder(messages) if self.responder else next(self.responses)
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, tuple):
            return result
        return json.dumps(result) if isinstance(result, dict) else result, {"finish_reason": "stop"}


class FakeRAG:
    def __init__(self, mode="validation", binding="literal", stage=4, responses=(), responder=None, count=4):
        self.pcrag_config = SimpleNamespace(
            improvement_stage=stage, evidence_budget=5, evidence_pool_size=80,
            evidence_candidate_top_k=20, evidence_ablation_mode=mode,
            evidence_binding_mode=binding, evidence_selection_mode="coverage",
            evidence_ablation_inputs_file="unused" if mode.startswith("budget_") or mode == "fixed_pool" else "")
        self.documents = DOCS + [f"Passage {i}\nUnique extra evidence document {i}." for i in range(4, count)]
        self.passage_node_keys = ["chunk-" + hashlib.md5(x.encode()).hexdigest() for x in self.documents]
        rows = dict(zip(self.passage_node_keys, self.documents))
        self.chunk_embedding_store = SimpleNamespace(get_row=lambda key: {"content": rows[key]})
        self.llm_model = FakeLLM(responses, responder)
        self.searches = []
        self.main_thread = threading.get_ident()
        self.query_to_embedding = {"passage": {}}
        self.passage_embeddings = np.tile(np.asarray([[1., 0.]]), (count, 1))
        self.encode_calls = []
        self.embedding_model = SimpleNamespace(batch_encode=self.encode)

    def encode(self, query, instruction, norm):
        assert threading.get_ident() == self.main_thread
        self.encode_calls.append((query, instruction, norm))
        return np.asarray([1., 0.])

    def dense_passage_retrieval(self, query):
        assert threading.get_ident() == self.main_thread
        self.searches.append(query)
        ids = [1, 0, 2, 3] if "Alpha" in query else [3, 2, 0, 1] if "Beta" in query else [0, 2, 1, 3]
        return np.asarray(ids), np.asarray([.9, .8, .6, .4])

    @staticmethod
    def _extract_llm_text(response):
        return str(response)


def state(question="Where was the writer of Work X born?", query_idx=0):
    return {"query": question, "query_idx": query_idx,
            "base": (np.arange(4), np.asarray([.9, .8, .7, .6]), {"hops": 2}),
            "static_sub_questions": [], "pcqd_sub_questions": []}


def inputs(rag, budgets, pool=None):
    return {"schema_version": 1, "records": [
        {"question": question, "query_index": i, "sample_id": str(i), "evidence_call_budget": budget,
         "pool": pool if pool is not None else []}
        for i, (question, budget) in enumerate(budgets.items())]}


def engine_with_inputs(rag, payload):
    # The production constructor reads the input file. Tests provide the same
    # validated schema directly without leaving local artifacts.
    mode = rag.pcrag_config.evidence_ablation_mode
    rag.pcrag_config.evidence_ablation_mode = "normal"
    rag.pcrag_config.evidence_ablation_inputs_file = ""
    engine = runtime.EvidenceAblation(rag)
    rag.pcrag_config.evidence_ablation_mode = mode
    engine.mode = mode
    engine.set_query_inputs(payload)
    return engine


class EvidenceAblationTests(unittest.TestCase):
    def test_zero_budget_skips_plan_and_verification(self):
        rag = FakeRAG(mode="budget_dag")
        query = state()
        engine = engine_with_inputs(rag, inputs(rag, {query["query"]: 0}))
        engine.process_window([query])
        self.assertEqual(rag.llm_model.calls, [])
        self.assertEqual(query["evidence_trace"]["ablation"]["llm_calls"], 0)
        self.assertEqual(query["evidence_trace"]["plan"], [])

    def test_logical_tokens_and_cached_calls_are_reported_separately(self):
        plan = {"nodes": [PLAN["nodes"][0]]}
        rag = FakeRAG(responses=[
            (json.dumps(plan), {"finish_reason": "stop", "prompt_tokens": 15, "completion_tokens": 8}, True),
            (json.dumps(ALPHA), {"finish_reason": "stop", "prompt_tokens": 12, "completion_tokens": 3}, False)])
        query = state()
        runtime.EvidenceAblation(rag).process_window([query])
        trace = query["evidence_trace"]["ablation"]
        self.assertEqual(trace["llm_calls"], 2)
        self.assertEqual(trace["logical_prompt_tokens"], 27)
        self.assertEqual(trace["logical_completion_tokens"], 11)
        self.assertEqual(trace["cache_hits"], 1)
        self.assertEqual(trace["logical_network_calls"], 1)

    def test_same_prompt_single_flight_preserves_two_logical_calls_one_network_call(self):
        class DelayedCacheLLM:
            def __init__(self):
                self.cache = {}
                self.lock = threading.Lock()
                self.network_calls = 0

            def infer(self, messages, temperature):
                key = json.dumps(messages, sort_keys=True)
                with self.lock:
                    cached = self.cache.get(key)
                    if cached is not None:
                        return cached, {"finish_reason": "stop", "prompt_tokens": 9,
                                        "completion_tokens": 4}, True
                    self.network_calls += 1
                    sequence = self.network_calls
                # Exactly the production cache race: network happens outside
                # the DB lock, then the response is committed before returning.
                time.sleep(.05)
                response = json.dumps({"sequence": sequence})
                with self.lock:
                    self.cache[key] = response
                return response, {"finish_reason": "stop", "prompt_tokens": 9,
                                  "completion_tokens": 4}, False
        rag = FakeRAG()
        rag.llm_model = DelayedCacheLLM()
        engine = runtime.EvidenceAblation(rag)
        states = [state("Question A?"), state("Question B?")]
        for query in states:
            engine._prepare_state(query)
        messages = [{"role": "user", "content": "Same shared subquestion"}]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(engine._owned_job, query, engine._infer_object, messages)
                       for query in states]
            outputs = [future.result() for future in futures]
        self.assertEqual(outputs, [({"sequence": 1}, None)] * 2)
        self.assertEqual(rag.llm_model.network_calls, 1)
        self.assertEqual(sum(query["_ablation_call_count"] for query in states), 2)
        self.assertEqual(sum(query["_ablation_usage"]["cache_hits"] for query in states), 1)
        self.assertEqual(sum(query["_ablation_usage"]["logical_prompt_tokens"] for query in states), 18)

    def test_distinct_prompts_are_not_globally_serialized(self):
        barrier = threading.Barrier(2)
        class ConcurrentLLM:
            def infer(self, messages, temperature):
                # A global inference lock would make the first worker time out.
                barrier.wait(timeout=2)
                return '{"ok":true}', {"finish_reason": "stop"}, False
        rag = FakeRAG()
        rag.llm_model = ConcurrentLLM()
        engine = runtime.EvidenceAblation(rag)
        states = [state("Question A?"), state("Question B?")]
        for query in states:
            engine._prepare_state(query)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(engine._owned_job, query, engine._infer_object,
                                       [{"role": "user", "content": query["query"]}])
                       for query in states]
            self.assertEqual([future.result() for future in futures], [({"ok": True}, None)] * 2)

    def test_flat_budgets_are_exact_and_thread_isolated(self):
        for mode in ("budget_qd", "budget_iterative"):
            with self.subTest(mode=mode):
                rag = FakeRAG(mode=mode, responder=lambda _: {"queries": ["Who wrote Work X?"]})
                states = [state("Question A?"), state("Question B?", query_idx=1)]
                engine = engine_with_inputs(rag, inputs(rag, {"Question A?": 1, "Question B?": 3}))
                with ThreadPoolExecutor(max_workers=2) as executor:
                    engine.process_window(states, executor)
                self.assertEqual(len(rag.llm_model.calls), 4)
                self.assertEqual([s["evidence_trace"]["ablation"]["llm_calls"] for s in states], [1, 3])
                self.assertEqual([s["evidence_trace"]["llm_plan_calls"] for s in states], [1, 3])
                if mode == "budget_iterative":
                    self.assertTrue(any("Retrieved passages" in str(call) for call in rag.llm_model.calls))
                else:
                    self.assertFalse(any("Retrieved passages" in str(call) for call in rag.llm_model.calls))

    def test_duplicate_question_samples_keep_independent_call_budgets(self):
        question = "Identical benchmark question?"
        rag = FakeRAG(mode="budget_qd", responder=lambda _: {"queries": ["Who wrote Work X?"]})
        payload = {"schema_version": 1, "records": [
            {"question": question, "sample_id": "row176", "query_index": 176,
             "pool": [], "evidence_call_budget": 1},
            {"question": question, "sample_id": "row525", "query_index": 525,
             "pool": [], "evidence_call_budget": 3}]}
        engine = engine_with_inputs(rag, payload)
        states = [state(question, query_idx=0), state(question, query_idx=1)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            engine.process_window(states, executor)
        self.assertEqual([s["evidence_trace"]["ablation"]["llm_calls"] for s in states], [1, 3])
        self.assertEqual([s["evidence_trace"]["ablation"]["global_query_index"] for s in states], [176, 525])
        ambiguous = state(question)
        del ambiguous["query_idx"]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            engine._record(ambiguous)
        with self.assertRaisesRegex(ValueError, "text does not match"):
            engine._record(state("Another question?", query_idx=0))

    def test_duplicate_question_dag_diagnostics_do_not_overwrite_other_sample(self):
        rag = FakeRAG(mode="budget_dag", responses=[PLAN, ALPHA, ROME])
        question = state()["query"]
        payload = {"schema_version": 1, "records": [
            {"question": question, "sample_id": "row176", "query_index": 176,
             "pool": [], "evidence_call_budget": 0},
            {"question": question, "sample_id": "row525", "query_index": 525,
             "pool": [], "evidence_call_budget": 3}]}
        engine = engine_with_inputs(rag, payload)
        states = [state(question, query_idx=0), state(question, query_idx=1)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            engine.process_window(states, executor)
        self.assertEqual([s["evidence_trace"]["llm_plan_calls"] for s in states], [0, 1])
        self.assertEqual(states[0]["evidence_trace"]["planning_outputs"]["outputs"], [])
        self.assertEqual(states[1]["evidence_trace"]["plan"], PLAN["nodes"])
        self.assertEqual(states[1]["evidence_trace"]["bindings"], {"s1": "Alpha", "s2": "Rome"})
        self.assertIsNot(states[0]["_ablation_plan_diagnostic"], states[1]["_ablation_plan_diagnostic"])

    def test_duplicate_question_frozen_pools_are_keyed_by_global_sample_identity(self):
        rag = FakeRAG(mode="fixed_pool", stage=3, count=201)
        question = state()["query"]
        payload = {"schema_version": 1, "records": [
            {"question": question, "sample_id": str(global_index), "query_index": global_index,
             "evidence_call_budget": 1, "pool": [
                 {"doc_hash": key, "score": 1. - i / 200.}
                 for i, key in enumerate(rag.passage_node_keys[start:start + 200])]}
            for start, global_index in [(0, 176), (1, 525)]]}
        engine = engine_with_inputs(rag, payload)
        states = [state(question, query_idx=0), state(question, query_idx=1)]
        engine.process_window(states)
        outputs = [engine.finalize(question, np.arange(201), np.ones(201), {}, s)[0] for s in states]
        self.assertEqual(set(outputs[0].tolist()), set(range(200)))
        self.assertEqual(set(outputs[1].tolist()), set(range(1, 201)))
        self.assertEqual(set(engine._frozen_pools), {176, 525})

    def test_invalid_json_calls_are_productive_retries_within_budget(self):
        rag = FakeRAG(mode="budget_qd", responses=["bad json", {"queries": ["Find the writer"]}])
        query = state()
        engine = engine_with_inputs(rag, inputs(rag, {query["query"]: 2}))
        engine.process_window([query])
        self.assertEqual(len(rag.llm_model.calls), 2)
        self.assertEqual(query["evidence_trace"]["search_count"], 1)
        self.assertIn("invalid_json_object", [x["reason"] for x in query["evidence_trace"]["semantic_failures"]])

    def test_dag_repair_does_not_exceed_budget(self):
        rag = FakeRAG(mode="budget_dag", responses=[{"nodes": []}])
        query = state()
        engine = engine_with_inputs(rag, inputs(rag, {query["query"]: 1}))
        engine.process_window([query])
        self.assertEqual(len(rag.llm_model.calls), 1)
        self.assertEqual(query["evidence_trace"]["llm_plan_calls"], 1)
        self.assertIn("llm_budget_exhausted", [x["reason"] for x in query["evidence_trace"]["semantic_failures"]])

    def test_dag_partial_answer_survives_budget_exhaustion(self):
        rag = FakeRAG(mode="budget_dag", responses=[PLAN, ALPHA])
        query = state()
        engine = engine_with_inputs(rag, inputs(rag, {query["query"]: 2}))
        engine.process_window([query])
        self.assertEqual(len(rag.llm_model.calls), 2)
        self.assertEqual(query["evidence_trace"]["bindings"], {"s1": "Alpha"})
        self.assertEqual(query["evidence_trace"]["llm_verification_calls"], 1)
        self.assertEqual(query["evidence_trace"]["ablation"]["llm_calls"], 2)

    def test_independent_nodes_have_correct_parallel_call_accounting(self):
        independent = {"nodes": [dict(x) for x in PLAN["nodes"]]}
        independent["nodes"][1]["question"] = "Where was Alpha born?"
        independent["nodes"][1]["depends_on"] = []
        def respond(messages):
            prompt = str(messages)
            return independent if "Plan evidence retrieval" in prompt else ROME if "Where was Alpha" in prompt else ALPHA
        rag = FakeRAG(mode="budget_dag", responder=respond)
        query = state()
        engine = engine_with_inputs(rag, inputs(rag, {query["query"]: 3}))
        with ThreadPoolExecutor(max_workers=3) as executor:
            engine.process_window([query], executor)
        trace = query["evidence_trace"]
        self.assertEqual(trace["llm_plan_calls"], 1)
        self.assertEqual(trace["llm_verification_calls"], 2)
        self.assertEqual(trace["actual_evidence_llm_calls"], 3)
        self.assertEqual(trace["bindings"], {"s1": "Alpha", "s2": "Rome"})

    def test_terminal_http_error_propagates(self):
        rag = FakeRAG(mode="budget_qd", responses=[RuntimeError("HTTP 503")])
        query = state()
        engine = engine_with_inputs(rag, inputs(rag, {query["query"]: 1}))
        with ThreadPoolExecutor(max_workers=2) as executor:
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                engine.process_window([query], executor)

    def test_literal_control_matches_original_parent(self):
        rag, original_rag = FakeRAG(responses=[PLAN, ALPHA, ROME]), FakeRAG(responses=[PLAN, ALPHA, ROME])
        query, original_query = state(), state()
        engine = runtime.EvidenceAblation(rag)
        original = base.EvidenceRetrieval(original_rag)
        engine.process_window([query])
        original.process_window([original_query])
        ids, scores, _ = engine.finalize(query["query"], *query["base"][:2], {}, query)
        old_ids, old_scores, _ = original.finalize(original_query["query"], *original_query["base"][:2], {}, original_query)
        self.assertEqual(ids.tolist(), old_ids.tolist())
        np.testing.assert_allclose(scores, old_scores)
        self.assertEqual(rag.llm_model.calls, original_rag.llm_model.calls)
        self.assertEqual(query["evidence_trace"]["bindings"], original_query["evidence_trace"]["bindings"])

    def test_frozen_routes_and_final_results_never_escape_top200(self):
        rag = FakeRAG(mode="fixed_pool", stage=3, count=201)
        query = state()
        query["static_sub_questions"] = ["Find evidence"]
        pool = [{"doc_hash": key, "score": 1. - i / 200.} for i, key in enumerate(rag.passage_node_keys[:200])]
        engine = engine_with_inputs(rag, inputs(rag, {query["query"]: 3}, pool))
        engine.process_window([query])
        outside_ids = np.asarray([200] + list(range(200)))
        outside_scores = np.linspace(2., 0., 201)
        ids, _, trace = engine.finalize(query["query"], outside_ids, outside_scores, {}, query)
        self.assertEqual(len(ids), 200)
        self.assertEqual(set(ids.tolist()), set(range(200)))
        self.assertEqual(len(query["evidence_candidates"]), 200)
        self.assertEqual(rag.searches, [])
        self.assertTrue(trace["fixed_pool_set_preserved"])
        self.assertTrue(all(200 not in route["doc_ids"] for route in trace["routes"]))
        self.assertEqual(rag.encode_calls[0][1], "Given a question, retrieve relevant documents that best answer the question.")
        self.assertTrue(rag.encode_calls[0][2])

    def test_frozen_pool_requires_exactly_200_and_matching_content(self):
        rag = FakeRAG(mode="fixed_pool", stage=3, count=201)
        query = state()
        pool = [{"doc_hash": key, "score": 1.} for key in rag.passage_node_keys[:199]]
        with self.assertRaisesRegex(ValueError, "exactly 200"):
            engine_with_inputs(rag, inputs(rag, {query["query"]: 3}, pool))
        pool.append({"doc_hash": rag.passage_node_keys[199], "score": 1.})
        engine = engine_with_inputs(rag, inputs(rag, {query["query"]: 3}, pool))
        rag.chunk_embedding_store = SimpleNamespace(get_row=lambda _: {"content": "Wrong indexed content"})
        with self.assertRaisesRegex(ValueError, "content"):
            engine.process_window([query])

    def test_runtime_can_be_instantiated_before_passage_keys_are_prepared(self):
        rag = FakeRAG(mode="normal")
        keys = rag.passage_node_keys
        del rag.passage_node_keys
        engine = runtime.EvidenceAblation(rag)
        self.assertIsNone(engine._chunk_to_id)
        rag.passage_node_keys = keys
        query = state()
        # Normal mode does not require a frozen pool or eager key mapping.
        engine._prepare_state(query)
        self.assertEqual(len(query["evidence_candidates"]), 4)

    def test_ablation_inputs_reject_gold_and_bound_answers(self):
        rag = FakeRAG(mode="budget_dag")
        query = state()
        for field in ("gold_docs", "gold_answers", "plan", "bindings", "answer"):
            payload = inputs(rag, {query["query"]: 2})
            payload["records"][0][field] = "leak"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "only question"):
                engine_with_inputs(rag, payload)

    def test_relation_gate_rejects_named_entity_without_supported_relation(self):
        literal_claim = ALPHA["hypotheses"][0]
        judgement = {"verdicts": [dict(literal_claim, label="insufficient", relation_supported=False)]}
        rag = FakeRAG(binding="relation", responses=[PLAN, ALPHA, judgement])
        query = state()
        engine = runtime.EvidenceAblation(rag)
        engine.process_window([query])
        self.assertEqual(query["evidence_trace"]["bindings"], {})
        self.assertEqual(query["evidence_trace"]["actual_evidence_llm_calls"], 3)
        self.assertTrue(any(x.get("gate") == "relation" for x in query["evidence_trace"]["rejected_hypotheses"]))

    def test_string_and_literal_have_same_extraction_prompt_only_gate_differs(self):
        plan = {"nodes": [PLAN["nodes"][0]]}
        bad_quote = {"hypotheses": [dict(ALPHA["hypotheses"][0], evidence="Alpha invented fabricated text.")]}
        string_rag = FakeRAG(binding="string", responses=[plan, bad_quote])
        literal_rag = FakeRAG(binding="literal", responses=[plan, bad_quote, {"hypotheses": []}])
        string_state, literal_state = state(), state()
        runtime.EvidenceAblation(string_rag).process_window([string_state])
        runtime.EvidenceAblation(literal_rag).process_window([literal_state])
        self.assertEqual(string_rag.llm_model.calls, literal_rag.llm_model.calls[:2])
        self.assertEqual(string_state["evidence_trace"]["bindings"], {"s1": "Alpha"})
        self.assertEqual(literal_state["evidence_trace"]["bindings"], {})

    def test_selection_policies_share_search_and_can_jointly_change_binding(self):
        root_answers = {"hypotheses": [
            dict(ALPHA["hypotheses"][0], confidence=.95),
            {"answer": "Beta", "doc_id": "D2", "evidence": "Beta wrote Work X in 1989.", "confidence": .8}]}
        paris = {"hypotheses": [{"answer": "Paris", "doc_id": "D3",
                                "evidence": "Beta was born in Paris in 1958.", "confidence": .8}]}
        def respond(messages):
            prompt = str(messages)
            return PLAN if "Plan evidence retrieval" in prompt else ROME if "Where was Alpha" in prompt else paris if "Where was Beta" in prompt else root_answers
        searches, calls = [], []
        for policy in ("coverage", "ancestor", "joint"):
            rag = FakeRAG(mode="selection", responder=respond)
            rag.pcrag_config.evidence_selection_mode = policy
            rag.pcrag_config.evidence_budget = 2
            engine = runtime.EvidenceAblation(rag)
            query = state()
            with ThreadPoolExecutor(max_workers=3) as executor:
                engine.process_window([query], executor)
            self.assertEqual(len(query["_evidence_beams"]), 2)
            self.assertTrue(query["evidence_trace"]["ablation"]["selection_shared_branches"])
            searches.append(rag.searches)
            calls.append(rag.llm_model.calls)
            ids, scores, trace = engine.finalize(query["query"], np.arange(4), np.asarray([.1, .1, .9, 1.]), {}, query)
            self.assertEqual(set(ids.tolist()), set(range(4)))
            self.assertTrue(np.all(scores[:-1] >= scores[1:]))
            diagnostic = trace["selection_diagnostics"]
            if policy == "joint":
                self.assertEqual(len(diagnostic["evaluated_binding_branches"]), 2)
                self.assertTrue(diagnostic["binding_optimization"])
                objectives = [b["objective_value"] for b in diagnostic["evaluated_binding_branches"]]
                self.assertEqual(diagnostic["objective_value"], max(objectives))
                self.assertEqual(trace["bindings"]["s1"], "Beta")
            else:
                self.assertEqual(len(diagnostic["evaluated_binding_branches"]), 1)
                self.assertEqual(trace["bindings"]["s1"], "Alpha")
        self.assertEqual(searches[0], searches[1])
        self.assertEqual(searches[1], searches[2])
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[1], calls[2])


if __name__ == "__main__":
    unittest.main()
