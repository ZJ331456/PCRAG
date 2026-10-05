"""CPU checks for index/retrieval concurrency separation and env precedence."""

import contextlib
import importlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_dataset import build_parser  # noqa: E402
from pathcondrag.config import PCRAGConfig  # noqa: E402
from pathcondrag.index.openie.openie_openai import OpenIE  # noqa: E402
from pathcondrag.utils.config_utils import BaseConfig  # noqa: E402


class OpenIEConcurrencyConfigTests(unittest.TestCase):
    def test_defaults_preserve_openie_environment_and_use_retrieval_eight(self):
        self.assertIsNone(BaseConfig().openie_max_workers)
        self.assertIsNone(build_parser().parse_args(["--dataset", "musique"]).openie_max_workers)
        config = PCRAGConfig(openie_max_workers=8, llm_prefetch_workers=4)
        self.assertEqual(config.openie_max_workers, 8)
        self.assertEqual(config.llm_prefetch_workers, 4)
        self.assertEqual(PCRAGConfig().llm_prefetch_workers, 8)
        self.assertEqual(build_parser().parse_args(["--dataset", "musique"]).llm_prefetch_workers, 8)
        self.assertEqual(PCRAGConfig(llm_prefetch_workers=1).llm_prefetch_workers, 1)

    def test_invalid_explicit_limits_fail_without_clamping(self):
        for value in (0, -1, 9, True, 1.5, "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BaseConfig(openie_max_workers=value)
        for value in ("0", "9", "1.5"):
            with self.subTest(cli=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(["--dataset", "musique", "--openie_max_workers", value])

    def test_explicit_limit_overrides_all_legacy_worker_environment(self):
        legacy_env = {
            "HIPPO_OPENIE_MAX_WORKERS": "3",
            "HIPPO_OPENIE_NER_WORKERS": "2",
            "HIPPO_OPENIE_TRIPLE_WORKERS": "1",
        }
        with patch.dict(os.environ, legacy_env):
            self.assertEqual(OpenIE(object()).worker_limits(), (2, 1))
            self.assertEqual(
                OpenIE(object(), max_workers=8, respect_env_workers=False).worker_limits(),
                (8, 8),
            )
        with patch.dict(os.environ, {key: "" for key in legacy_env}):
            self.assertEqual(OpenIE(object()).worker_limits(), (8, 8))

    def test_base_rag_passes_configured_workers_before_loading_embeddings(self):
        module = importlib.import_module("pathcondrag.BaseRAG")

        class StopBeforeEmbedding(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            for workers, prompt_version in ((None, 'optimized'), (4, 'optimized'),
                                            (8, 'optimized'), (8, 'origin')):
                with self.subTest(workers=workers, prompt_version=prompt_version):
                    config = BaseConfig(save_dir=temporary, openie_max_workers=workers,
                                        openie_prompt_version=prompt_version)
                    llm = object()
                    with patch.object(module, "_get_llm_class", return_value=llm):
                        with patch.object(module, "OpenIE", side_effect=StopBeforeEmbedding) as constructor:
                            with self.assertRaises(StopBeforeEmbedding):
                                module.BaseRAG(global_config=config)
                    constructor.assert_called_once_with(
                        llm_model=llm,
                        max_workers=8 if workers is None else workers,
                        respect_env_workers=workers is None,
                        prompt_version=prompt_version,
                    )


if __name__ == "__main__":
    unittest.main()
