"""Exercise shell-facing reporting commands without loading model dependencies."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ENTRYPOINT = Path(__file__).resolve().parents[1] / "scripts" / "experiment_tools.py"


class PrefetchCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "带空格 reports"
        self.root.mkdir()
        self.case = self.root / "pathcondrag_bs8"
        self.case.mkdir()
        self.result_path = self.case / "result.json"
        self.log = self.root / "vllm.log"
        self.prefix = 'POST /v1/chat/completions HTTP/1.1" 503\n'
        self.log.write_text(self.prefix + 'POST /v1/chat/completions HTTP/1.1" 200\n')
        self.result = {
            "sample_size_effective": 2,
            "hop_source": "benchmark",
            "retrieval_seconds": 2.5,
            "retrieval_metrics": {"Recall@5": 0.75},
            "hop_distribution": {"2": 1, "4": 1},
            "retrieval_diagnostics": {
                "hop_counter": {"2": 1, "4": 1},
                "qd_used_count": 2,
                "pcqd_used_count": 2,
            },
            "llm_request_stats": {"failures": 0},
        }
        self.write_result()

    def write_result(self):
        self.result_path.write_text(json.dumps(self.result))

    def run_command(self, *args):
        return subprocess.run(
            [sys.executable, str(ENTRYPOINT), *map(str, args)],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_report(self, *, sample_size=2, log_start=None):
        return self.run_command(
            "prefetch-report", "--result", self.result_path,
            "--case-dir", self.case, "--name", self.case.name,
            "--workers", 8, "--hop-mode", "benchmark", "--elapsed", 3,
            "--sample-size", sample_size, "--vllm-log", self.log,
            "--log-start", len(self.prefix.encode()) if log_start is None else log_start,
            "--log-end", self.log.stat().st_size if self.log.exists() else 0,
        )

    def test_report_from_arbitrary_cwd_preserves_fields_and_log_slice(self):
        process = self.run_report()
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((self.case / "report.json").read_text())
        self.assertEqual(report, {
            "name": "pathcondrag_bs8", "llm_prefetch_workers": 8,
            "hop_mode": "benchmark", "hop_source": "benchmark",
            "seconds": 3, "retrieval_seconds": 2.5, "n_samples": 2,
            "retrieval_metrics": {"Recall@5": 0.75},
            "hop_distribution": {"2": 1, "4": 1},
            "hop_counter": {"2": 1, "4": 1}, "qd_used": 2, "pcqd_used": 2,
            "llm_request_stats": {"failures": 0}, "http_status_in_log": {"200": 1},
        })
        self.assertIn("[done] pathcondrag_bs8 sec=3 R@5=0.75", process.stdout)

    def test_rejects_sample_and_hop_mismatch_before_writing_report(self):
        for sample_size, hop_source, message in (
            (3, "benchmark", "sample mismatch: 2 != 3"),
            (0, "benchmark", "expected full (~1000), got 2"),
            (2, "estimated", "hop_source=estimated want=benchmark"),
        ):
            with self.subTest(message=message):
                self.result["hop_source"] = hop_source
                self.write_result()
                process = self.run_report(sample_size=sample_size)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn(message, process.stderr)
                self.assertFalse((self.case / "report.json").exists())

    def test_http_or_llm_failure_keeps_report_and_missing_log_remains_optional(self):
        process = self.run_report(log_start=0)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("non-200 in vLLM log", process.stderr)
        self.assertTrue((self.case / "report.json").exists())
        self.result["llm_request_stats"]["failures"] = 1
        self.write_result()
        process = self.run_report()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("LLM failures=1", process.stderr)
        self.result["llm_request_stats"]["failures"] = 0
        self.write_result()
        self.log.unlink()
        process = self.run_report(log_start=0)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((self.case / "report.json").read_text())
        self.assertEqual(report["http_status_in_log"], {})

    def test_summary_uses_case_order_and_skips_absent_reports(self):
        for name in ("pathcondrag_bs2", "pathcondrag_bs8_fixedhop", "pathcondrag_bs8"):
            case = self.root / name
            case.mkdir(exist_ok=True)
            (case / "report.json").write_text(json.dumps({"name": name}))
        process = self.run_command("prefetch-summary", "--out-root", self.root)
        self.assertEqual(process.returncode, 0, process.stderr)
        raw = (self.root / "comparison.json").read_text()
        summary = json.loads(raw)
        self.assertEqual(summary["out_root"], str(self.root))
        self.assertEqual(list(summary["cases"]), [
            "pathcondrag_bs8", "pathcondrag_bs8_fixedhop", "pathcondrag_bs2",
        ])
        self.assertIn("带空格", raw)


if __name__ == "__main__":
    unittest.main()
