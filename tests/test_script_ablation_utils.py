"""Fixture checks for source guards and ablation result/report behavior."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import ablations


class AblationToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parser = argparse.ArgumentParser()
        ablations.register_commands(self.parser.add_subparsers(required=True))

    def tearDown(self):
        self.temp.cleanup()

    def run_command(self, command, **options):
        argv = [command]
        for key, value in options.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        args = self.parser.parse_args(argv)
        with redirect_stdout(io.StringIO()) as output:
            code = args.handler(args)
        return code, output.getvalue()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def complete_qwen_result(self):
        return {
            "runtime_config": {
                "embedding_model_name": ablations.QWEN_MODEL,
                "embedding_batch_size": 4,
                "retrieval_top_k": 200,
                "hop_force_max": 2,
                "hop_multi_min_signals": 2,
                **{key: True for key in ablations.ACTIVE_COMPONENTS},
                "use_bridge_cache_index": True,
            },
            "sample_size_effective": 1000,
            "indexed_docs": 11656,
            "eval_mode": "retrieve",
            "retrieval_metrics": {key: 0.5 for key in ablations.RECALL_KEYS},
        }

    def test_source_validation_preserves_full_run_and_model_guards(self):
        result = self.complete_qwen_result()
        source = self.write("source.json", result)
        index = self.root / "index"
        self.write(
            f"index/{ablations.QWEN_INDEX_DIR}/index_manifest.json",
            {"embedding": {"model_name": ablations.QWEN_MODEL}},
        )
        options = {
            "source_result": source, "source_index": index,
            "embedding_model": ablations.QWEN_MODEL,
        }
        code, output = self.run_command("ablation-qwen-validate-source", **options)
        self.assertEqual(code, 0)
        self.assertIn("[verified]", output)

        result["runtime_config"]["embedding_batch_size"] = 8
        self.write("source.json", result)
        with self.assertRaisesRegex(SystemExit, "embedding_batch_size=8"):
            self.run_command("ablation-qwen-validate-source", **options)
        result["runtime_config"]["embedding_batch_size"] = 4
        result["sample_size_effective"] = 999
        self.write("source.json", result)
        with self.assertRaisesRegex(SystemExit, "completed 1000-question"):
            self.run_command("ablation-qwen-validate-source", **options)

        result["sample_size_effective"] = 1000
        self.write("source.json", result)
        self.write(
            f"index/{ablations.QWEN_INDEX_DIR}/index_manifest.json",
            {"embedding": {"model_name": "wrong-model"}},
        )
        with self.assertRaisesRegex(SystemExit, "Source index embedding"):
            self.run_command("ablation-qwen-validate-source", **options)

    def test_completion_checks_zero_recall_is_valid_and_missing_is_incomplete(self):
        result = self.complete_qwen_result()
        result["retrieval_metrics"]["Recall@20"] = 0
        path = self.write("qwen.json", result)
        code, _ = self.run_command("ablation-qwen-result-complete", result=path)
        self.assertEqual(code, 0)
        result["runtime_config"]["embedding_batch_size"] = 8
        self.write("qwen.json", result)
        code, _ = self.run_command("ablation-qwen-result-complete", result=path)
        self.assertEqual(code, 1)

        path = self.write("pc3.json", {"retrieval_metrics": {"Recall@5": 0}})
        code, _ = self.run_command("ablation-pc3-result-complete", result=path)
        self.assertEqual(code, 0)
        self.write("pc3.json", {"retrieval_metrics": None})
        code, _ = self.run_command("ablation-pc3-result-complete", result=path)
        self.assertEqual(code, 1)

    def test_qwen_summary_preserves_case_order_delta_precision_and_module_usage(self):
        source = self.write("source.json", self.complete_qwen_result())
        output_root = self.root / "pathcondrag-ablation"
        for index, name in enumerate(ablations.COMPONENT_CASES):
            self.write(f"pathcondrag-ablation/results/{name}.json", {
                "retrieval_metrics": {
                    key: 0.543217 + index * 0.01 for key in ablations.RECALL_KEYS
                },
                "retrieval_diagnostics": {"module_usage": {"中文模块": index}},
            })
        code, output = self.run_command(
            "ablation-qwen-summary", out_root=output_root, source_result=source,
        )
        self.assertEqual(code, 0)
        self.assertIn("[summary]", output)
        summary = json.loads((output_root / "summary.json").read_text())
        self.assertEqual(list(summary["ablations"]), list(ablations.COMPONENT_CASES))
        self.assertEqual(summary["source_index"], str(self.root / "pathcondrag" / "index"))
        first = summary["ablations"]["wo_qcappr"]
        self.assertEqual(first["delta_vs_existing_full"]["Recall@5"], 0.0432)
        self.assertEqual(first["module_usage"], {"中文模块": 0})

    def test_pc3_summary_sorts_files_and_retains_missing_metrics(self):
        self.write("results/z_case.json", {"retrieval_metrics": {"Recall@5": 0.25}})
        self.write("results/a_case.json", {})
        code, output = self.run_command("ablation-pc3-summary", result_dir=self.root / "results")
        self.assertEqual(code, 0)
        self.assertIn("nan", output)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual([row["case"] for row in summary], ["a_case", "z_case"])
        self.assertIsNone(summary[0]["retrieval"]["Recall@5"])
        self.assertEqual(summary[1]["retrieval"]["Recall@5"], 0.25)


if __name__ == "__main__":
    unittest.main()
