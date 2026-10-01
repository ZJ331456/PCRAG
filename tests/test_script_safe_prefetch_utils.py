"""CPU fixtures for benchmark validation and ordered result comparisons."""

import argparse
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from utils import safe_prefetch


class SafePrefetchUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case_dir = self.root / "workers_1"
        self.case_dir.mkdir()
        self.result_path = self.case_dir / "result.json"
        self.log_path = self.root / "vllm.log"
        prefix = 'POST /v1/chat/completions HTTP/1.1" 503\n'
        self.log_path.write_text(prefix + 'POST /v1/chat/completions HTTP/1.1" 200\n')
        self.args = argparse.Namespace(
            result=self.result_path,
            case_dir=self.case_dir,
            elapsed=10,
            sample_size=2,
            vllm_log=self.log_path,
            log_start=len(prefix.encode()),
            log_end=self.log_path.stat().st_size,
        )
        self.result = {
            "results": [
                {"docs": ["A\na", "B\nb", "other"], "doc_scores": [0.8, 0.6, 0.3]},
                {"docs": ["C\nc", "other"], "doc_scores": [0.7, 0.2]},
            ],
            "selected_indices": [0, 1],
            "sample_size_effective": 2,
            "hop_source": "benchmark",
            "retrieval_seconds": 9.5,
            "retrieval_metrics": {"Recall@1": 0.5, "Recall@2": 0.75, "Recall@5": 1.0, "Recall@10": 1.0},
            "retrieval_diagnostics": {"hop_counter": {"2": 2}, "qd_used_count": 2, "pcqd_used_count": 2},
            "hop_distribution": {"2": 2},
            "llm_request_stats": {"http_attempts": 1, "failures": 0},
        }
        self.write_result()

    def write_result(self):
        self.result_path.write_text(json.dumps(self.result))

    def quiet(self, handler, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return handler(args)

    def test_report_counts_only_requested_log_slice(self):
        self.result["llm_request_stats"]["label"] = "中文"
        self.write_result()
        self.quiet(safe_prefetch.report, self.args)
        report_text = (self.case_dir / "report.json").read_text()
        report = json.loads(report_text)
        self.assertIn("\\u4e2d\\u6587", report_text)
        self.assertEqual(report["http_status_in_log"], {"200": 1})
        self.assertEqual(report["retrieval_metrics"], self.result["retrieval_metrics"])
        self.assertEqual(report["hop_counter"], {"2": 2})
        self.assertEqual(report["seconds"], 10)

    def test_report_preserves_negative_log_read_and_missing_file_errors(self):
        # Preserve read(-1) and invalid negative lengths from the old runner.
        self.args.log_end = self.args.log_start - 1
        self.quiet(safe_prefetch.report, self.args)
        report = json.loads((self.case_dir / "report.json").read_text())
        self.assertEqual(report["http_status_in_log"], {"200": 1})
        self.args.log_end = 0
        with self.assertRaises(ValueError):
            self.quiet(safe_prefetch.report, self.args)
        self.args.vllm_log = self.root / "missing.log"
        with self.assertRaises(FileNotFoundError):
            self.quiet(safe_prefetch.report, self.args)

    def test_report_rejects_incomplete_samples_and_wrong_hop_source(self):
        self.result["selected_indices"] = [0]
        self.write_result()
        with self.assertRaisesRegex(SystemExit, "Unexpected completed sample count"):
            self.quiet(safe_prefetch.report, self.args)
        self.result["selected_indices"] = [0, 1]
        self.result["hop_source"] = "estimated"
        self.write_result()
        with self.assertRaisesRegex(SystemExit, "Benchmark hop source"):
            self.quiet(safe_prefetch.report, self.args)

    def test_report_rejects_http_and_terminal_errors(self):
        self.args.log_start = 0
        with self.assertRaisesRegex(SystemExit, "Non-200"):
            self.quiet(safe_prefetch.report, self.args)
        self.args.log_start = self.args.log_end
        with self.assertRaisesRegex(SystemExit, "no HTTP statuses"):
            self.quiet(safe_prefetch.report, self.args)
        self.result["llm_request_stats"]["failures"] = 1
        self.write_result()
        with self.assertRaisesRegex(SystemExit, "terminal LLM request failure"):
            self.quiet(safe_prefetch.report, self.args)

    def test_compare_uses_gold_fallback_and_preserves_order_and_score_counts(self):
        self.quiet(safe_prefetch.report, self.args)
        other_dir = self.root / "workers_4"
        other_dir.mkdir()
        other = copy.deepcopy(self.result)
        other["results"][0]["docs"] = ["B\nb", "A\na", "other"]
        other["results"][1]["doc_scores"] = [0.6, 0.2]
        (other_dir / "result.json").write_text(json.dumps(other))
        (other_dir / "report.json").write_text(json.dumps({"seconds": 5}))
        samples = self.root / "samples.json"
        samples.write_text(json.dumps([
            {"paragraphs": [
                {"title": "A", "text": "a", "is_supporting": True},
                {"title": "B", "paragraph_text": "b"},
                {"title": "ignored", "text": "ignored", "is_supporting": False},
            ]},
            {"paragraphs": [
                {"title": "C", "text": "c", "is_supporting": True},
                {"title": "D", "text": "missing", "is_supporting": True},
            ]},
        ]))
        args = argparse.Namespace(run_root=self.root, samples=samples)
        self.quiet(safe_prefetch.compare, args)
        summary = json.loads((self.root / "comparison.json").read_text())
        self.assertEqual(summary["workers_1"]["same_ordered_top5_vs_workers_1"], 2)
        case = summary["workers_4"]
        self.assertEqual(case["all_gold_top5"], 0.5)
        self.assertEqual(case["same_ordered_top5_vs_workers_1"], 1)
        self.assertEqual(case["same_top5_scores_vs_workers_1"], 1)
        self.assertEqual(case["different_question_indices_vs_workers_1"], [0])


if __name__ == "__main__":
    unittest.main()
