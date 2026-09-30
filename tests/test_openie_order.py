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


if __name__ == "__main__":
    unittest.main()
