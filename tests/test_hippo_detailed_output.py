"""Detailed Hippo baseline export and bounded prefetch, without model/API calls."""

import importlib.util
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HIPPO_ROOT = Path("/root/baseline/HippoRAG")
sys.path.insert(0, str(HIPPO_ROOT / "src"))

from hipporag.HippoRAG import HippoRAG
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import QuerySolution

spec = importlib.util.spec_from_file_location("hippo_detailed_main", HIPPO_ROOT / "main.py")
hippo_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hippo_main)


class HippoDetailedOutputTest(unittest.TestCase):
    def test_top10_export_keeps_full_candidate_recall_and_original_index(self):
        docs = [f"Document {index}" for index in range(200)]
        solution = QuerySolution("question", docs, np.linspace(1.0, 0.0, 200),
                                 retrieval_diagnostics={"facts_before_rerank": [("a", "r", "b")]})
        row = hippo_main.detailed_result(solution, {"id": "sample", "question_decomposition": [{}] * 3},
                                         41, [docs[0], docs[8], docs[19]], ["answer"])
        self.assertEqual(len(row["docs"]), 10)
        self.assertEqual(len(row["doc_scores"]), 10)
        self.assertEqual(len(row["candidate_docs"]), 200)
        self.assertEqual(row["query_index"], 41)
        self.assertEqual(row["benchmark_hops"], 3)
        self.assertEqual(row["retrieval_metrics"]["Recall@5"], 1 / 3)
        self.assertEqual(row["retrieval_metrics"]["Recall@10"], 2 / 3)
        self.assertEqual(row["retrieval_metrics"]["Recall@200"], 1)
        self.assertEqual([entry["rank"] for entry in row["gold_document_ranks"]], [1, 9, 20])
        self.assertFalse(row["all_gold_in_top10"])
        self.assertEqual(solution.docs, docs)

    def test_explicit_indices_preserve_original_order_and_reject_duplicates(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.json"
            path.write_text(json.dumps({"selected_indices": [8, 2]}))
            self.assertEqual(hippo_main.select_sample_indices(10, 2, 42, path), [8, 2])
            path.write_text("[2, 2]")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                hippo_main.select_sample_indices(10, 2, 42, path)
            path.write_text("[12]")
            with self.assertRaisesRegex(ValueError, "invalid"):
                hippo_main.select_sample_indices(10, 1, 42, path)

    def test_only_llm_filters_run_in_workers_and_results_remain_ordered(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.global_config = SimpleNamespace(llm_prefetch_workers=8)
        rag.rerank_time = 0.0
        main_thread = threading.get_ident()
        observed = []
        barrier = threading.Barrier(8)

        def scores(query):
            self.assertEqual(threading.get_ident(), main_thread)
            return np.asarray([float(query)])

        def prepare(values):
            self.assertEqual(threading.get_ident(), main_thread)
            return [0], [("a", "r", "b")]

        def filter_facts(query, indices, facts):
            self.assertNotEqual(threading.get_ident(), main_thread)
            observed.append(query)
            barrier.wait(timeout=10)
            return indices, facts, {"query": query}

        rag.get_fact_scores = scores
        rag._prepare_fact_candidates = prepare
        rag._filter_fact_candidates = filter_facts
        result = list(rag._iter_prefetched_fact_reranks([str(index) for index in range(8)]))
        self.assertEqual([entry[0] for entry in result], [str(index) for index in range(8)])
        self.assertEqual(len(observed), 8)
        self.assertEqual(result[7][-1]["candidate_fact_scores"], [7.0])

    def test_prefetch_failure_is_propagated(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.global_config = SimpleNamespace(llm_prefetch_workers=2)
        rag.rerank_time = 0.0
        rag.get_fact_scores = lambda query: np.asarray([1.0])
        rag._prepare_fact_candidates = lambda scores: ([0], [("a", "r", "b")])
        rag._filter_fact_candidates = lambda *args: (_ for _ in ()).throw(RuntimeError("HTTP unavailable"))
        with self.assertRaisesRegex(RuntimeError, "HTTP unavailable"):
            list(rag._iter_prefetched_fact_reranks(["a", "b"]))

    def test_worker_config_rejects_unbounded_requests(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            BaseConfig(llm_prefetch_workers=9)


class HippoHttpStatsTest(unittest.TestCase):
    def make_client(self, directory, create):
        client = CacheOpenAI.__new__(CacheOpenAI)
        client.global_config = SimpleNamespace(llm_supports_max_completion_tokens=False, azure_endpoint=None,
                                              llm_base_url="http://localhost/v1", azure_api_version=None,
                                              azure_chat_deployment=None)
        client.llm_name = "qwen3-8b"
        client.request_model_name = client.llm_name
        client.llm_config = SimpleNamespace(generate_params={"model": client.llm_name, "max_completion_tokens": 2048})
        client.cache_file_name = str(Path(directory) / "cache.sqlite")
        client._llm_stats_lock = threading.Lock()
        client._llm_stats = dict(cache_hits=0, cache_misses=0, http_attempts=0, retries=0, failures=0)
        client.max_retries = 2
        client.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return client

    def response(self):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12))

    def test_actual_http_cap_stats_cache_and_forced_no_thinking(self):
        state = {"active": 0, "maximum": 0}
        lock = threading.Lock()

        def create(**params):
            self.assertFalse(params["extra_body"]["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(params["max_tokens"], 2048)
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.02)
            with lock:
                state["active"] -= 1
            return self.response()

        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, create)
            def request(index):
                return client.infer([{"role": "user", "content": str(index)}],
                                    extra_body={"chat_template_kwargs": {"enable_thinking": True}})
            with ThreadPoolExecutor(max_workers=16) as executor:
                outputs = list(executor.map(request, range(16)))
            self.assertTrue(all(not output[2] for output in outputs))
            self.assertTrue(request(0)[2])
            stats = client.get_request_stats()
            self.assertEqual(stats["http_attempts"], 16)
            self.assertEqual(stats["cache_misses"], 16)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["failures"], 0)
            self.assertLessEqual(state["maximum"], 8)
            self.assertGreater(state["maximum"], 1)

    def test_http_failure_is_counted_and_raised_without_cache_entry(self):
        def create(**params):
            raise ValueError("invalid HTTP response")
        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, create)
            with self.assertRaisesRegex(ValueError, "invalid HTTP response"):
                client.infer([{"role": "user", "content": "q"}])
            stats = client.get_request_stats()
            self.assertEqual(stats["http_attempts"], 1)
            self.assertEqual(stats["failures"], 1)

    def test_connection_retry_has_single_bounded_layer(self):
        import httpx
        import openai
        attempts = []
        def create(**params):
            attempts.append(params)
            if len(attempts) == 1:
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://localhost/v1"))
            return self.response()
        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, create)
            with patch("hipporag.llm.openai_gpt.time.sleep"):
                client.infer([{"role": "user", "content": "q"}])
            stats = client.get_request_stats()
            self.assertEqual(stats["http_attempts"], 2)
            self.assertEqual(stats["retries"], 1)
            self.assertEqual(stats["failures"], 0)


if __name__ == "__main__":
    unittest.main()
