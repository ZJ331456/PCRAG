"""Recognition-memory transport failures are distinct from empty semantic output."""

import json
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathcondrag.BaseRAG import BaseRAG
from pathcondrag.rerank import DSPyFilter


class FactFilterSafetyTest(unittest.TestCase):
    def make_filter(self, infer):
        runtime = SimpleNamespace(global_config=SimpleNamespace(rerank_dspy_file_path=None, llm_name="qwen3-8b"),
                                  llm_model=SimpleNamespace(infer=infer))
        return DSPyFilter(runtime)

    def make_rag(self, rerank_filter):
        rag = BaseRAG.__new__(BaseRAG)
        rag.global_config = SimpleNamespace(linking_top_k=5)
        rag.fact_node_keys = ["fact-a"]
        rag.fact_embedding_store = SimpleNamespace(get_rows=lambda ids: {"fact-a": {"content": "('a', 'r', 'b')"}})
        rag.rerank_filter = rerank_filter
        return rag

    def test_shared_defaults_are_immutable_for_concurrent_calls(self):
        barrier = threading.Barrier(8)
        def infer(**kwargs):
            self.assertEqual(kwargs["max_completion_tokens"], 512)
            kwargs["extra_body"]["unchanged"] = "worker-local mutation"
            barrier.wait(timeout=10)
            return "[[ ## fact_after_filter ## ]]\n" + json.dumps({"fact": [["a", "r", "b"]]}), {
                "finish_reason": "stop", "prompt_tokens": 5, "completion_tokens": 3,
            }, True
        filter_model = self.make_filter(infer)
        filter_model.default_gen_kwargs = {"temperature": 0.0, "extra_body": {"unchanged": "original"}}
        original = deepcopy(filter_model.default_gen_kwargs)
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda index: filter_model.rerank(str(index), [("a", "r", "b")], [7], 5), range(8)))
        self.assertEqual(filter_model.default_gen_kwargs, original)
        for indices, facts, trace in results:
            self.assertEqual(indices, [7])
            self.assertEqual(facts, [("a", "r", "b")])
            self.assertTrue(trace["cache_hit"])
            self.assertIn("fact_after_filter", trace["response"])
            self.assertEqual(trace["llm_metadata"]["finish_reason"], "stop")

    def test_transport_failure_propagates_through_filter_and_base_rag(self):
        def infer(**kwargs):
            raise RuntimeError("terminal connection failure")
        rag = self.make_rag(self.make_filter(infer))
        with self.assertRaisesRegex(RuntimeError, "terminal connection failure"):
            rag.rerank_facts("question", np.asarray([0.9]))

    def test_malformed_semantic_output_keeps_dense_fallback_diagnostics(self):
        filter_model = self.make_filter(lambda **kwargs: ("unstructured response", {"finish_reason": "stop"}, False))
        rag = self.make_rag(filter_model)
        indices, facts, trace = rag.rerank_facts("question", np.asarray([0.9]))
        self.assertEqual(indices, [])
        self.assertEqual(facts, [])
        self.assertEqual(trace["no_facts_reason"], "missing_fact_after_filter_field")
        self.assertEqual(trace["facts_before_rerank"], [("a", "r", "b")])
        self.assertEqual(trace["reranker_info"]["response"], "unstructured response")

    def test_explicit_empty_fact_selection_is_valid_semantic_output(self):
        filter_model = self.make_filter(lambda **kwargs: (
            '[[ ## fact_after_filter ## ]]\n{"fact": []}', {"finish_reason": "stop"}, False))
        rag = self.make_rag(filter_model)
        indices, facts, trace = rag.rerank_facts("question", np.asarray([0.9]))
        self.assertEqual((indices, facts), ([], []))
        self.assertEqual(trace["no_facts_reason"], "explicit_empty_fact_list")


if __name__ == "__main__":
    unittest.main()
