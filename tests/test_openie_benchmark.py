"""CPU-only checks for the isolated cold-cache OpenIE benchmark driver."""

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import benchmark_openie_concurrency as bench  # noqa: E402
from pathcondrag.utils.misc_utils import NerRawOutput, TripleRawOutput  # noqa: E402


class BenchmarkDriverTests(unittest.TestCase):
    def test_seeded_sample_preserves_corpus_order_and_exact_passage(self):
        corpus = [{"title": f"title-{i}", "text": f"text-{i}"} for i in range(20)]
        chunks1, selected1 = bench.select_corpus(corpus, 6, 42)
        chunks2, selected2 = bench.select_corpus(corpus, 6, 42)
        self.assertEqual(selected1, selected2)
        self.assertEqual(list(chunks1), list(chunks2))
        self.assertEqual(
            [item["corpus_index"] for item in selected1],
            sorted(item["corpus_index"] for item in selected1),
        )
        for item in selected1:
            idx = item["corpus_index"]
            self.assertEqual(chunks1[item["chunk_id"]]["content"], f"title-{idx}\ntext-{idx}")
        with self.assertRaisesRegex(ValueError, "duplicate passages"):
            bench.select_corpus([corpus[0], corpus[0]], 2, 42)

    def test_results_remain_ordered_and_all_observable_errors_are_reported(self):
        selected = [{"corpus_index": 3, "chunk_id": "c", "title": "C", "passage_sha256": "x"},
                    {"corpus_index": 7, "chunk_id": "a", "title": "A", "passage_sha256": "y"}]
        extractor = SimpleNamespace(
            ner_results={
                "a": NerRawOutput("a", "raw", ["A"], {"ner_max_tokens_used": 1024}),
                "c": NerRawOutput("c", "raw", ["C"], {"ner_max_tokens_used": 512}),
            },
            triple_results={"a": TripleRawOutput("a", "raw", [["A", "rel", "B"]], {})},
            ner_seconds={"a": 0.2, "c": 0.1}, triple_seconds={"a": 0.3},
        )
        rows = bench.collect_rows(selected, extractor)
        self.assertEqual([row["chunk_id"] for row in rows], ["c", "a"])
        problems, parse_retries = bench.check_run(rows, {
            "cache_hits": 1, "retries": 2, "failures": 1,
        })
        self.assertEqual(parse_retries, 1)
        self.assertTrue(any("missing triple" in problem for problem in problems))
        self.assertTrue(any("cold cache" in problem for problem in problems))
        self.assertTrue(any("HTTP/network retry" in problem for problem in problems))
        self.assertTrue(any("terminal request failure" in problem for problem in problems))

    def test_driver_writes_complete_case_without_calling_model(self):
        class FakeLLM:
            llm_config = SimpleNamespace(generate_params={
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
            })

            def get_request_stats(self):
                return {"cache_hits": 0, "cache_misses": 4, "http_attempts": 4,
                        "retries": 0, "failures": 0,
                        "max_in_flight": bench.LLM_MAX_IN_FLIGHT}

        class FakeOpenIE:
            def __init__(self, llm, max_workers):
                self.ner_results = {}
                self.triple_results = {}
                self.ner_seconds = {}
                self.triple_seconds = {}

            def batch_openie(self, chunks):
                for chunk_id in reversed(list(chunks)):
                    self.ner_results[chunk_id] = NerRawOutput(
                        chunk_id, "raw ner", ["entity"], {"ner_max_tokens_used": 512}
                    )
                    self.triple_results[chunk_id] = TripleRawOutput(
                        chunk_id, "raw triple", [["entity", "rel", "value"]], {}
                    )
                    self.ner_seconds[chunk_id] = 0.1
                    self.triple_seconds[chunk_id] = 0.2

        with tempfile.TemporaryDirectory() as tmp:
            corpus_path = Path(tmp) / "corpus.json"
            corpus_path.write_text(json.dumps([
                {"title": "A", "text": "first"},
                {"title": "B", "text": "second"},
            ]), encoding="utf-8")
            output_dir = Path(tmp) / "case"
            args = argparse.Namespace(
                corpus_path=str(corpus_path), sample_size=2, seed=42,
                llm_name="qwen3-8b", llm_base_url="http://127.0.0.1:8035/v1",
                openie_workers=8, output_dir=str(output_dir),
            )
            with patch.dict(os.environ, {
                "PATHCONDRAG_LLM_MAX_IN_FLIGHT": str(bench.LLM_MAX_IN_FLIGHT),
            }):
                with patch.object(bench.CacheOpenAI, "from_experiment_config", return_value=FakeLLM()):
                    with patch.object(bench, "TimedOpenIE", FakeOpenIE):
                        with patch("builtins.print"):
                            self.assertEqual(bench.run(args), 0)

            rows = json.loads((output_dir / "results.json").read_text())
            report = json.loads((output_dir / "report.json").read_text())
            self.assertEqual([row["corpus_index"] for row in rows], [0, 1])
            self.assertEqual(report["sample_indices"], [0, 1])
            self.assertEqual(report["request_stats"]["http_attempts"], 4)
            self.assertIn("src/pathcondrag/llm/openai_gpt.py", report["code_sha256"])
            self.assertEqual(report["problems"], [])
            self.assertTrue(report["complete"])
            with self.assertRaises(FileExistsError):
                with patch.dict(os.environ, {
                    "PATHCONDRAG_LLM_MAX_IN_FLIGHT": str(bench.LLM_MAX_IN_FLIGHT),
                }):
                    bench.run(args)

    def test_endpoint_preflight_rejects_wrong_served_model(self):
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"id": "different-model"}]},
        )
        with patch.object(bench.httpx, "get", return_value=response) as get:
            with self.assertRaisesRegex(RuntimeError, "not served"):
                bench.check_llm_endpoint("http://127.0.0.1:8035/v1", "qwen3-8b")
        get.assert_called_once_with("http://127.0.0.1:8035/v1/models", timeout=10)


if __name__ == "__main__":
    unittest.main()
