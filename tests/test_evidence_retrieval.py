import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/pathcondrag/evidence_retrieval.py"
spec = importlib.util.spec_from_file_location("evidence_retrieval_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def infer(self, messages, temperature):
        self.calls.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, tuple) else (response, {"finish_reason": "stop"})


class FakeRAG:
    def __init__(self, stage=3, responses=()):
        self.pcrag_config = SimpleNamespace(improvement_stage=stage, evidence_budget=2,
                                           evidence_pool_size=80, evidence_candidate_top_k=20)
        self.passage_node_keys = ["d0", "d1", "d2", "d3"]
        self.documents = {
            "d0": "Biography\nAlpha wrote Work X in 1990.",
            "d1": "Biography\nAlpha was born in Rome in 1960.",
            "d2": "Biography\nBeta wrote Work X in 1989.",
            "d3": "Biography\nBeta was born in Paris in 1958.",
        }
        self.chunk_embedding_store = SimpleNamespace(get_row=lambda key: {"content": self.documents[key]})
        self.llm_model = FakeLLM(responses)
        self.searches = []
        self.main_thread = threading.get_ident()

    def dense_passage_retrieval(self, question):
        assert threading.get_ident() == self.main_thread, "Embedding call escaped caller thread"
        self.searches.append(question)
        ids = [1, 0, 2, 3] if "Alpha" in question else [3, 2, 0, 1] if "Beta" in question else [0, 2, 1, 3]
        return np.asarray(ids), np.asarray([0.9, 0.8, 0.6, 0.4])

    @staticmethod
    def _extract_llm_text(response):
        return str(response)


def state_fixture():
    return {"query_idx": 0, "query": "Where was the writer of Work X born?",
            "base": (np.arange(4), np.asarray([.9, .8, .7, .6]), {"hops": 2}),
            "static_sub_questions": [], "pcqd_sub_questions": []}


PLAN = ('{"nodes":[{"id":"s1","question":"Who wrote Work X?",'
        '"depends_on":[],"answer_type":"person"},{"id":"s2",'
        '"question":"Where was ${s1.answer} born?","depends_on":["s1"],'
        '"answer_type":"place"}]}')
ANSWER_ALPHA = ('{"hypotheses":[{"answer":"Alpha","doc_id":"D0",'
                '"evidence":"Alpha wrote Work X in 1990.","confidence":0.9}]}')
ANSWER_ROME = ('{"hypotheses":[{"answer":"Rome","doc_id":"D1",'
               '"evidence":"Alpha was born in Rome in 1960.","confidence":0.9}]}')


class EvidenceRetrievalTests(unittest.TestCase):
    def test_valid_dag_reorders_and_binds_answers(self):
        payload, reason = module.parse_object(PLAN)
        nodes, reason = module.validate_plan(payload, 2)
        self.assertIsNone(reason)
        self.assertEqual([[n["id"] for n in layer] for layer in module.plan_layers(nodes)], [["s1"], ["s2"]])
        self.assertIsNone(module.bind_question(nodes[1]["question"], {}))
        self.assertEqual(module.bind_question(nodes[1]["question"], {"s1": "Alpha"}), "Where was Alpha born?")

    def test_rejects_cycles_invalid_references_and_hashes(self):
        cycle = {"nodes": [
            {"id": "s1", "question": "Who is ${s2.answer}?", "depends_on": ["s2"]},
            {"id": "s2", "question": "Who is ${s1.answer}?", "depends_on": ["s1"]},
        ]}
        self.assertEqual(module.validate_plan(cycle, 2)[1], "cyclic_dependencies")
        self.assertEqual(module.validate_plan({"nodes": [{"id": "s1", "question": "${s2.answer}", "depends_on": []}]}, 2)[1],
                         "dependency_reference_mismatch")
        self.assertIsNone(module.bind_question("Who is ${s1.answer}?", {"s1": "0702e6dee64edf543eecf3ac4986fba0"}))

    def test_verification_requires_known_doc_exact_span_and_literal_answer(self):
        docs = {0: "Alpha wrote Work X in 1990."}
        payload = {"hypotheses": [
            {"answer": "Alpha", "doc_id": "D0", "evidence": docs[0], "confidence": .9},
            {"answer": "Alpha", "doc_id": "D3", "evidence": docs[0]},
            {"answer": "Alpha", "doc_id": "D0", "evidence": "Alpha invented Work X."},
            {"answer": "Beta", "doc_id": "D0", "evidence": docs[0]},
        ]}
        accepted, rejected = module.verify_hypotheses(payload, docs)
        self.assertEqual([x["answer"] for x in accepted], ["Alpha"])
        self.assertEqual({x["reason"] for x in rejected}, {
            "unknown_document_reference", "evidence_not_exact_substring", "answer_not_supported_in_quote"})

    def test_subquestion_local_rank_priors_restart(self):
        a = module.local_rank_scores(np.asarray([.9, .8, .7]), 10)
        b = module.local_rank_scores(np.asarray([.6, .5, .4]), 10)
        self.assertAlmostEqual(a[0], b[0])
        self.assertGreater(a[0], a[1])

    def test_coverage_prefix_preserves_distinct_evidence_with_same_title(self):
        rag = FakeRAG(stage=3)
        rag.pcrag_config.evidence_coverage_weight = 1.0
        engine = module.EvidenceRetrieval(rag)
        state = state_fixture()
        engine._prepare_state(state)
        state["evidence_candidates"] = {
            0: {"doc_id": 0, "base_score": .9, "verified": [], "sources": [
                {"goal": "s1", "score": 1., "requirements": {}}]},
            1: {"doc_id": 1, "base_score": .8, "verified": [], "sources": [
                {"goal": "s2", "score": 1., "requirements": {}}]},
            2: {"doc_id": 2, "base_score": .85, "verified": [], "sources": [
                {"goal": "s1", "score": 1., "requirements": {}}]},
        }
        ids, scores, trace = engine.finalize(state["query"], np.arange(4), np.asarray([.9, .8, .7, .6]), {}, state)
        self.assertEqual(ids[:2].tolist(), [0, 1])
        self.assertEqual(len(set(ids.tolist())), 4)
        self.assertTrue(np.all(scores[:-1] >= scores[1:]))
        self.assertGreater(trace["selected_prefix"][1]["coverage_gain"], 0.)

    def test_dependency_search_binds_verified_answer_and_does_not_read_gold(self):
        rag = FakeRAG(stage=4, responses=[PLAN, ANSWER_ALPHA, ANSWER_ROME])
        engine = module.EvidenceRetrieval(rag)
        state = state_fixture()
        # These fields are a trap: experimental benchmark answers must never be read.
        state["gold_answers"] = {"s1": "forbidden"}
        state["question_decomposition"] = [{"answer": "forbidden"}]
        engine.process_window([state])
        self.assertEqual(rag.searches, ["Who wrote Work X?", "Where was Alpha born?"])
        self.assertEqual(state["evidence_trace"]["bindings"], {"s1": "Alpha", "s2": "Rome"})
        self.assertEqual(state["evidence_trace"]["llm_verification_calls"], 2)
        self.assertNotIn("forbidden", str(rag.llm_model.calls))

    def test_semantic_failure_is_recorded_and_http_error_propagates(self):
        rag = FakeRAG(stage=4, responses=["not JSON"])
        state = state_fixture()
        module.EvidenceRetrieval(rag).process_window([state])
        self.assertEqual(state["evidence_trace"]["semantic_failures"][0]["reason"], "invalid_json_object")
        rag = FakeRAG(stage=4, responses=[ConnectionError("HTTP failed")])
        with self.assertRaisesRegex(ConnectionError, "HTTP failed"):
            module.EvidenceRetrieval(rag).process_window([state_fixture()])

    def test_beam_alternatives_preserve_ancestor_bindings(self):
        alternatives = ('{"hypotheses":[{"answer":"Alpha","doc_id":"D0",'
                        '"evidence":"Alpha wrote Work X in 1990.","confidence":0.8},'
                        '{"answer":"Beta","doc_id":"D2",'
                        '"evidence":"Beta wrote Work X in 1989.","confidence":0.9}]}')
        paris = ('{"hypotheses":[{"answer":"Paris","doc_id":"D3",'
                 '"evidence":"Beta was born in Paris in 1958.","confidence":0.95}]}')
        rag = FakeRAG(stage=5, responses=[PLAN, alternatives, ANSWER_ROME, paris])
        state = state_fixture()
        module.EvidenceRetrieval(rag).process_window([state])
        branches = state["evidence_trace"]["branch_scores"]
        self.assertEqual(len(branches), 2)
        self.assertEqual({tuple(sorted(x["bindings"].items())) for x in branches}, {
            (("s1", "Alpha"), ("s2", "Rome")), (("s1", "Beta"), ("s2", "Paris"))})
        self.assertFalse(module.branch_matches({"s1": "Alpha"}, {"s1": "Beta"}))

    def test_search_budget_is_bounded(self):
        rag = FakeRAG(stage=2)
        rag.pcrag_config.evidence_max_searches = 2
        state = state_fixture()
        state["static_sub_questions"] = ["q1", "q2", "q3", "q4"]
        module.EvidenceRetrieval(rag).process_window([state])
        self.assertEqual(len(rag.searches), 2)
        self.assertEqual(state["evidence_trace"]["search_count"], 2)

    def test_truncated_response_records_semantic_fallback(self):
        rag = FakeRAG(stage=4, responses=[(PLAN, {"finish_reason": "length"})])
        state = state_fixture()
        module.EvidenceRetrieval(rag).process_window([state])
        self.assertEqual(state["evidence_trace"]["semantic_failures"][0]["reason"], "truncated_response")
        self.assertEqual(state["evidence_trace"]["llm_verification_calls"], 0)

    def test_embeddings_remain_on_main_thread_with_llm_executor(self):
        rag = FakeRAG(stage=4, responses=[PLAN, ANSWER_ALPHA, ANSWER_ROME])
        state = state_fixture()
        with ThreadPoolExecutor(max_workers=2) as executor:
            module.EvidenceRetrieval(rag).process_window([state], executor)
        self.assertEqual(state["evidence_trace"]["bindings"]["s2"], "Rome")

    def test_wrong_beam_candidate_cannot_cover_bound_successor(self):
        candidate = {"sources": [{"goal": "s2", "score": .99,
                                   "requirements": {"s1": "Beta"}}], "verified": []}
        self.assertEqual(module.EvidenceRetrieval._goal_scores(candidate, {"s1": "Alpha"}), {})
        self.assertEqual(module.EvidenceRetrieval._goal_scores(candidate, {"s1": "Beta"}), {"s2": .99})

    def test_missing_verified_parent_never_searches_unbound_child(self):
        rag = FakeRAG(stage=4, responses=[PLAN, '{"hypotheses":[]}'])
        state = state_fixture()
        module.EvidenceRetrieval(rag).process_window([state])
        self.assertEqual(rag.searches, ["Who wrote Work X?"])
        reasons = {x["reason"] for x in state["evidence_trace"]["semantic_failures"]}
        self.assertIn("unbound_dependency", reasons)


if __name__ == "__main__":
    unittest.main()
