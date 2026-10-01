"""Fixture checks for embedding shell helpers, without model requests."""

import argparse
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from utils import embedding


class EmbeddingScriptUtilsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def put_json(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def invoke(self, handler, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            handler(argparse.Namespace(**kwargs))

    def read_output(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def test_smoke_sample_indices_and_stable_corpus_deduplication(self):
        samples = [
            {"question": "a", "paragraphs": [{"title": "中文", "text": "shared"}]},
            {"question": "b", "paragraphs": [{"title": "中文", "paragraph_text": "shared"},
                                               {"paragraph_text": "fallback"}]},
            {"question": "c", "paragraphs": None},
        ]
        self.put_json("source/musique.json", samples)
        self.invoke(embedding.prepare_smoke_data, datasets_dir=self.root / "source",
                    output_dir=self.root, sample_size=9, sample_seed=42)
        self.assertEqual(self.read_output("musique.json"), samples)
        self.assertEqual(self.read_output("selected_indices.json"), [0, 1, 2])
        self.assertEqual(self.read_output("musique_corpus.json"), [
            {"title": "中文", "text": "shared"}, {"title": "", "text": "fallback"},
        ])
        self.invoke(embedding.prepare_smoke_data, datasets_dir=self.root / "source",
                    output_dir=self.root, sample_size=2, sample_seed=7)
        expected = sorted(random.Random(7).sample(range(3), k=2))
        self.assertEqual(self.read_output("selected_indices.json"), expected)
        self.assertEqual(self.read_output("musique.json"), [samples[index] for index in expected])

    def test_pair_summary_preserves_fields_and_optional_metrics(self):
        self.put_json("hipporag2_musique/metrics.json", {"retrieval_metrics": {"Recall@5": 0.7},
                                                       "n_docs": 12, "n_samples": 2})
        self.put_json("pathcondrag/result.json", {"retrieval_metrics": {"Recall@5": 0.8},
                                                "qa_metrics": {"F1": 0.5}})
        self.invoke(embedding.pair_summary, output_dir=self.root, mode="smoke", tag="8b",
                    embedding="model", llm="llm", embedding_batch_size=4)
        self.assertEqual(self.read_output("pair_summary.json"), {
            "mode": "smoke", "tag": "8b", "embedding": "model", "llm": "llm",
            "embedding_batch_size": 4, "embedding_max_seq_len": 2048,
            "hipporag2": {"retrieval": {"Recall@5": 0.7}, "qa": None, "n_docs": 12, "n_samples": 2},
            "pathcondrag_pc3": {"retrieval": {"Recall@5": 0.8}, "qa": {"F1": 0.5}},
        })

    def test_instruction_summary_optional_prior_baseline(self):
        self.put_json("hipporag2_musique/metrics_retrieve.json", {"retrieval_metrics": {"Recall@5": 0.7}})
        self.put_json("pathcondrag/result.json", {"retrieval_metrics": {"Recall@5": 0.8}})
        kwargs = {"output_dir": self.root, "source_hippo": self.root / "old/hipporag2_musique",
                  "embedding": "model", "llm": "llm"}
        self.invoke(embedding.instruction_fix_summary, **kwargs)
        self.assertEqual(self.read_output("pair_summary_retrieve.json")["baseline_before_fix"],
                         {"hipporag2": None, "pathcondrag_pc3": None})
        self.put_json("old/pair_summary.json", {"hipporag2": {"retrieval": {"Recall@5": 0.6}},
                                              "pathcondrag_pc3": None})
        self.invoke(embedding.instruction_fix_summary, **kwargs)
        summary = self.read_output("pair_summary_retrieve.json")
        self.assertEqual(summary["hipporag2_retrieve"], {"Recall@5": 0.7})
        self.assertEqual(summary["baseline_before_fix"],
                         {"hipporag2": {"Recall@5": 0.6}, "pathcondrag_pc3": None})

    def test_distinct_summary_requires_metrics(self):
        self.put_json("hipporag2_musique/metrics.json", {"retrieval_metrics": {}})
        self.put_json("pathcondrag/result.json", {"retrieval_metrics": {}, "qa_metrics": {}})
        with self.assertRaises(KeyError):
            self.invoke(embedding.distinct_summary, output_dir=self.root)
        self.assertFalse((self.root / "pair_summary.json").exists())

    def test_instruction_evaluation_keeps_index_reuse_and_generation_config(self):
        samples = [{"question": "Where?"}]
        self.put_json("source/musique.json", samples)
        self.put_json("source/musique_corpus.json", [{"title": "Title", "text": "Text"}])
        config_factory = mock.Mock(side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs))
        rag = mock.MagicMock()
        rag.__enter__.return_value = rag
        rag.retrieve.return_value = ([], {"Recall@5": 0.9})
        rag_factory = mock.Mock(return_value=rag)
        gold_docs = mock.Mock(return_value=[["Title\nText"]])
        main_module = types.ModuleType("main")
        main_module.get_gold_docs = gold_docs
        rag_module = types.ModuleType("hipporag.HippoRAG")
        rag_module.HippoRAG = rag_factory
        config_module = types.ModuleType("hipporag.utils.config_utils")
        config_module.BaseConfig = config_factory
        modules = {"main": main_module, "hipporag": types.ModuleType("hipporag"),
                   "hipporag.HippoRAG": rag_module,
                   "hipporag.utils": types.ModuleType("hipporag.utils"),
                   "hipporag.utils.config_utils": config_module}
        old_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {}):
            self.invoke(embedding.evaluate_hippo_instruction_fix, hippo_root=self.root / "hippo",
                        datasets_dir=self.root / "source", save_dir=str(self.root),
                        llm="llm", llm_base_url="http://local/v1", embedding="embedding-model",
                        embedding_batch_size=4)
        config = rag_factory.call_args.kwargs["global_config"]
        self.assertFalse(config.force_index_from_scratch)
        self.assertFalse(config.force_openie_from_scratch)
        self.assertEqual(config.max_new_tokens, 2048)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.embedding_batch_size, 4)
        self.assertEqual(config.retrieval_top_k, 200)
        rag.index.assert_called_once_with(["Title\nText"])
        rag.retrieve.assert_called_once_with(queries=["Where?"], gold_docs=[["Title\nText"]])
        gold_docs.assert_called_once_with(samples, "musique")
        metrics = self.read_output("metrics_retrieve.json")
        self.assertEqual(metrics["retrieval_metrics"], {"Recall@5": 0.9})
        self.assertEqual((metrics["n_samples"], metrics["n_docs"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
