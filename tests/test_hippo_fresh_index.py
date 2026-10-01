"""Fresh baseline indexing gates and complete OpenIE responses, with no models."""

import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

HIPPO_ROOT = Path("/root/baseline/HippoRAG")
sys.path.insert(0, str(HIPPO_ROOT / "src"))

from hipporag.information_extraction.openie_openai import OpenIE
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import NerRawOutput, TripleRawOutput

spec = importlib.util.spec_from_file_location("hippo_fresh_main", HIPPO_ROOT / "main.py")
hippo_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hippo_main)


class SequenceLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.settings = []
        self.llm_config = SimpleNamespace(generate_params={"seed": None})
        self.global_config = SimpleNamespace(response_format=None)

    def infer(self, **kwargs):
        self.settings.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        response, finish = reply
        return response, {"finish_reason": finish, "prompt_tokens": 1, "completion_tokens": 1}, False


class HippoFreshIndexTest(unittest.TestCase):
    def test_fresh_directory_rejects_any_old_cache_or_index(self):
        config = BaseConfig(force_index_from_scratch=True, force_openie_from_scratch=True)
        with tempfile.TemporaryDirectory() as directory:
            hippo_main.require_fresh_index_directory(directory, config)
            (Path(directory) / "old_cache.sqlite").write_text("old")
            with self.assertRaisesRegex(ValueError, "new empty"):
                hippo_main.require_fresh_index_directory(directory, config)
        with self.assertRaisesRegex(ValueError, "cannot reuse"):
            hippo_main.require_fresh_index_directory("unused", config, True)

    def test_repaired_but_truncated_ner_is_never_accepted(self):
        llm = SequenceLLM([('{"named_entities": ["partial"]}', "length"),
                           ('{"named_entities": ["complete", "entities"]}', "stop")])
        output = OpenIE(llm).ner("chunk", "passage")
        self.assertEqual(output.unique_entities, ["complete", "entities"])
        self.assertEqual([call["max_new_tokens"] for call in llm.settings], [512, 1024])
        self.assertEqual(output.metadata["length_observed_count"], 1)
        self.assertEqual(output.metadata["finish_reason"], "stop")

    def test_persistent_length_is_error_after_finite_unchanged_budgets(self):
        llm = SequenceLLM([('{"triples": [["a", "r", "b"]]}', "length")] * 4)
        output = OpenIE(llm).triple_extraction("chunk", "passage", ["a"])
        self.assertEqual(output.triples, [])
        self.assertIn("remains truncated", output.metadata["error"])
        self.assertEqual([call["max_new_tokens"] for call in llm.settings], [2048] * 4)
        self.assertEqual(llm.settings[1]["seed"], 1)
        self.assertEqual(llm.settings[2]["frequency_penalty"], 0.2)
        self.assertEqual(llm.settings[3]["frequency_penalty"], 0.5)

    def test_transport_and_all_invalid_triples_are_recorded_as_failures(self):
        llm = SequenceLLM([RuntimeError("HTTP unavailable")])
        output = OpenIE(llm).ner("chunk", "passage")
        self.assertIn("HTTP unavailable", output.metadata["error"])
        llm = SequenceLLM([('{"triples": [["invalid"]]}', "stop")])
        output = OpenIE(llm).triple_extraction("chunk", "passage", ["a"])
        self.assertIn("no valid triples", output.metadata["error"])

    def test_batch_has_eight_workers_at_each_stage_and_blocks_failed_ner(self):
        openie = OpenIE(SequenceLLM([]), max_workers=8)
        barrier = threading.Barrier(8)
        observed = {"ner": [], "triples": []}
        def ner(chunk, passage):
            observed["ner"].append(chunk)
            barrier.wait(timeout=10)
            return NerRawOutput(chunk, "response", ["a"], {"finish_reason": "stop"})
        def triples(chunk, passage, entities):
            observed["triples"].append(chunk)
            barrier.wait(timeout=10)
            return TripleRawOutput(chunk, "response", [["a", "r", "b"]], {"finish_reason": "stop"})
        openie.ner = ner
        openie.triple_extraction = triples
        chunks = {str(index): {"content": str(index)} for index in range(8)}
        ner_results, triple_results = openie.batch_openie(chunks)
        self.assertEqual(set(ner_results), set(chunks))
        self.assertEqual(set(triple_results), set(chunks))
        self.assertEqual(len(observed["ner"]), 8)
        self.assertEqual(len(observed["triples"]), 8)
        openie.ner = lambda chunk, passage: NerRawOutput(chunk, "", [], {"error": "HTTP failed"})
        openie.triple_extraction = lambda *args: self.fail("Triples should not run after a failed NER stage")
        with self.assertRaisesRegex(RuntimeError, "NER failed"):
            openie.batch_openie({"bad": {"content": "p"}})

    def test_completion_report_uses_actual_openie_coverage_and_http_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "graph.pickle"
            manifest_path = Path(directory) / "index_manifest.json"
            graph_path.write_text("graph")
            manifest_path.write_text("manifest")
            rag = SimpleNamespace(
                chunk_embedding_store=SimpleNamespace(get_all_ids=lambda: ["chunk"], get_all_texts=lambda: ["passage"]),
                entity_embedding_store=SimpleNamespace(get_all_ids=lambda: ["entity"]),
                graph=SimpleNamespace(vcount=lambda: 2),
                _graph_pickle_filename=str(graph_path), index_manifest_path=str(manifest_path),
                llm_model=SimpleNamespace(get_request_stats=lambda: {"http_attempts": 2, "failures": 0, "max_in_flight": 8}),
                _openie_info=[{"idx": "chunk", "passage": "passage", "extracted_entities": ["a"],
                               "extracted_triples": [["a", "r", "b"]],
                               "openie_metadata": {"ner": {"finish_reason": "stop"}, "triples": {"finish_reason": "stop"}}}],
            )
            config = BaseConfig(dataset="musique", save_dir=directory, embedding_batch_size=4,
                                force_index_from_scratch=True, force_openie_from_scratch=True)
            report = hippo_main.index_build_report(rag, ["passage"], config, 2.5)
            self.assertEqual(report["openie_document_count"], 1)
            self.assertEqual(report["indexed_docs"], 1)
            self.assertEqual(report["openie_failure_count"], 0)
            self.assertTrue(report["index_build_complete"])
            self.assertEqual(report["runtime_config"]["embedding_batch_size"], 4)
            rag._openie_info[0]["openie_metadata"]["triples"]["finish_reason"] = "length"
            with self.assertRaisesRegex(RuntimeError, "Incomplete or failed"):
                hippo_main.index_build_report(rag, ["passage"], config, 2.5)


if __name__ == "__main__":
    unittest.main()
