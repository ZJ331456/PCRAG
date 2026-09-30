"""Offline checks for stable OpenIE input and result ordering."""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathcondrag.BaseRAG import BaseRAG  # noqa: E402
from pathcondrag.information_extraction import openie_openai  # noqa: E402
from pathcondrag.utils.misc_utils import (  # noqa: E402
    NerRawOutput, TripleRawOutput, compute_mdhash_id,
)


class QuietProgress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, value):
        pass


class OpenIEOrderTests(unittest.TestCase):
    def test_missing_chunk_ids_follow_corpus_order_with_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            contents = ["third passage", "already indexed", "first passage"]
            chunk_ids = [compute_mdhash_id(content, "chunk-") for content in contents]
            source = dict(zip(chunk_ids, contents))
            output_path = Path(tmp) / "openie.json"
            output_path.write_text(json.dumps({
                "docs": [{"idx": "old-id", "passage": "already indexed"}]
            }))
            rag = BaseRAG.__new__(BaseRAG)
            rag.global_config = SimpleNamespace(force_openie_from_scratch=False)
            rag.openie_results_path = str(output_path)

            existing, missing = rag.load_existing_openie(source.keys())
            self.assertEqual(missing, [chunk_ids[0], chunk_ids[2]])
            self.assertEqual(existing[0]["idx"], chunk_ids[1])

            rag.global_config.force_openie_from_scratch = True
            existing, missing = rag.load_existing_openie(source.keys())
            self.assertEqual(existing, [])
            self.assertEqual(missing, chunk_ids)

    def test_out_of_order_futures_return_input_order_and_correct_dependencies(self):
        source = {
            "a": {"content": "first"},
            "b": {"content": "second"},
            "c": {"content": "third"},
        }
        ner_c_finished = threading.Event()
        triple_c_finished = threading.Event()
        lock = threading.Lock()
        ner_completion = []
        triple_completion = []
        triples_started_before_ner_done = []

        def ner(chunk_key, passage):
            if chunk_key == "a":
                self.assertTrue(ner_c_finished.wait(2))
            if chunk_key == "c":
                ner_c_finished.set()
            with lock:
                ner_completion.append(chunk_key)
            return NerRawOutput(chunk_key, "{}", [f"entity-{chunk_key}"], {})

        def triple(chunk_key, passage, named_entities):
            with lock:
                if len(ner_completion) != len(source):
                    triples_started_before_ner_done.append(chunk_key)
            self.assertEqual(named_entities, [f"entity-{chunk_key}"])
            if chunk_key == "a":
                self.assertTrue(triple_c_finished.wait(2))
            if chunk_key == "c":
                triple_c_finished.set()
            with lock:
                triple_completion.append(chunk_key)
            return TripleRawOutput(chunk_key, "{}", [[chunk_key, "rel", "value"]], {})

        extractor = openie_openai.OpenIE.__new__(openie_openai.OpenIE)
        extractor.max_workers = 3
        extractor.ner = ner
        extractor.triple_extraction = triple

        with patch.object(openie_openai, "tqdm", QuietProgress):
            ner_results, triple_results = extractor.batch_openie(source)

        self.assertNotEqual(ner_completion, list(source))
        self.assertNotEqual(triple_completion, list(source))
        self.assertEqual(triples_started_before_ner_done, [])
        self.assertEqual(list(ner_results), list(source))
        self.assertEqual(list(triple_results), list(source))
        self.assertEqual(triple_results["a"].triples, [["a", "rel", "value"]])

    def test_empty_message_exception_cannot_bypass_failure_gate(self):
        class FailingLLM:
            def infer(self, **kwargs):
                raise TimeoutError()

        extractor = openie_openai.OpenIE(FailingLLM(), max_workers=1)
        with patch.object(openie_openai, "tqdm", QuietProgress):
            with self.assertRaisesRegex(RuntimeError, "NER failed for 1 chunk"):
                extractor.batch_openie({"a": {"content": "first"}})

    def test_ner_discards_parseable_length_at_both_budgets(self):
        class FakeLLM:
            llm_config = SimpleNamespace(generate_params={"seed": None})

            def __init__(self):
                self.calls = []
                self.responses = iter([
                    ('{"named_entities":["partial-512"]}', "length", True),
                    ('{"named_entities":["partial-1024"]}', "length", False),
                    ('{"named_entities":["complete","complete"]}', "stop", False),
                ])

            def infer(self, **kwargs):
                self.calls.append(kwargs)
                text, finish_reason, cache_hit = next(self.responses)
                return text, {"finish_reason": finish_reason}, cache_hit

        llm = FakeLLM()
        result = openie_openai.OpenIE(llm, max_workers=1).ner("a", "passage")

        self.assertEqual(result.unique_entities, ["complete"])
        self.assertEqual([call["max_completion_tokens"] for call in llm.calls], [512, 1024, 1024])
        self.assertNotIn("seed", llm.calls[0])
        self.assertNotIn("seed", llm.calls[1])
        self.assertEqual(llm.calls[2]["seed"], 1)
        self.assertEqual(result.metadata["finish_reason"], "stop")
        self.assertEqual(result.metadata["ner_max_tokens_used"], 1024)
        self.assertEqual(result.metadata["length_observed_count"], 2)
        self.assertEqual(result.metadata["length_retry_count"], 2)
        self.assertEqual(result.metadata["openie_attempt_count"], 3)

    def test_triples_discard_parseable_length_and_retry_at_same_cap(self):
        class FakeLLM:
            llm_config = SimpleNamespace(generate_params={"seed": None})

            def __init__(self):
                self.calls = []
                self.responses = iter([
                    ('{"triples":[["partial","r","x"]]}', "length", True),
                    ('{"triples":[["complete","r","y"],["other","r","z"]]}', "stop", False),
                ])

            def infer(self, **kwargs):
                self.calls.append(kwargs)
                text, finish_reason, cache_hit = next(self.responses)
                return text, {"finish_reason": finish_reason}, cache_hit

        llm = FakeLLM()
        result = openie_openai.OpenIE(llm, max_workers=1).triple_extraction("a", "passage", ["x"])

        self.assertEqual(result.triples, [["complete", "r", "y"], ["other", "r", "z"]])
        self.assertEqual([call["max_completion_tokens"] for call in llm.calls], [2048, 2048])
        self.assertNotIn("seed", llm.calls[0])
        self.assertEqual(llm.calls[1]["seed"], 1)
        self.assertEqual(result.metadata["finish_reason"], "stop")
        self.assertEqual(result.metadata["length_observed_count"], 1)
        self.assertEqual(result.metadata["length_retry_count"], 1)
        self.assertEqual(result.metadata["openie_attempt_count"], 2)

    def test_ner_that_remains_truncated_fails_the_batch(self):
        class AlwaysLengthLLM:
            llm_config = SimpleNamespace(generate_params={"seed": None})

            def __init__(self):
                self.calls = []

            def infer(self, **kwargs):
                self.calls.append(kwargs)
                return '{"named_entities":["partial"]}', {"finish_reason": "length"}, False

        llm = AlwaysLengthLLM()
        extractor = openie_openai.OpenIE(llm, max_workers=1)
        result = extractor.ner("a", "passage")
        self.assertEqual(result.unique_entities, [])
        self.assertEqual(result.metadata["finish_reason"], "length")
        self.assertIn("finish_reason=length", result.metadata["error"])
        self.assertEqual(result.metadata["length_observed_count"], 4)
        self.assertEqual(result.metadata["length_retry_count"], 3)
        self.assertEqual([call["max_completion_tokens"] for call in llm.calls],
                         [512, 1024, 1024, 1024])
        self.assertEqual([call.get("seed") for call in llm.calls], [None, None, 1, 2])

        with patch.object(openie_openai, "tqdm", QuietProgress):
            with self.assertRaisesRegex(RuntimeError, "NER failed for 1 chunk"):
                extractor.batch_openie({"a": {"content": "passage"}})

    def test_triples_that_remain_truncated_fail_the_batch(self):
        class AlwaysLengthLLM:
            llm_config = SimpleNamespace(generate_params={"seed": 7})

            def __init__(self):
                self.calls = []

            def infer(self, **kwargs):
                self.calls.append(kwargs)
                return '{"triples":[["partial","r","x"]]}', {"finish_reason": "length"}, False

        llm = AlwaysLengthLLM()
        extractor = openie_openai.OpenIE(llm, max_workers=1)
        result = extractor.triple_extraction("a", "passage", ["x"])
        self.assertEqual(result.triples, [])
        self.assertEqual(result.metadata["finish_reason"], "length")
        self.assertIn("finish_reason=length", result.metadata["error"])
        self.assertEqual(result.metadata["length_observed_count"], 3)
        self.assertEqual(result.metadata["length_retry_count"], 2)
        self.assertEqual([call.get("seed") for call in llm.calls], [None, 8, 9])

        extractor.ner = lambda chunk_key, passage: NerRawOutput(chunk_key, "{}", ["x"], {})
        with patch.object(openie_openai, "tqdm", QuietProgress):
            with self.assertRaisesRegex(RuntimeError, "Triple extraction failed for 1 chunk"):
                extractor.batch_openie({"a": {"content": "passage"}})


if __name__ == "__main__":
    unittest.main()
