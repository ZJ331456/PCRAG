"""CPU checks for completion gates and conditional proposal deletion."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from utils import openie_repair_finish as finish


class FinishRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / "outputs/repair"
        self.out.mkdir(parents=True)
        self.proposal = self.root / "docs/proposal.md"
        self.proposal.parent.mkdir()
        self.proposal.write_text("Requested proposal", encoding="utf-8")
        self.args = SimpleNamespace(out_root=str(self.out), hippo_root=str(self.root / "baseline"),
                                    wait_pid=0, python="/rag/python", datasets_dir="/datasets",
                                    llm_base_url="http://local/v1", remove_proposal=True)
        self.validation = {"repaired_index": str(self.out / "repaired_index"), "asset_sha256": {"asset": "sha"}}
        self.smoke = {"cases": {"hipporag2": {}, "exp4_dependency_binding": {}}}

    def invoke(self, *, retrieval_error=None):
        with patch.object(finish, "ROOT", self.root), patch.object(finish, "PROPOSAL", self.proposal), \
                patch.object(finish, "require_validation", return_value=self.validation), \
                patch.object(finish, "require_retrieval", return_value=self.smoke, side_effect=retrieval_error), \
                patch.object(finish.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="commit\n")):
            return finish.run(self.args)

    def test_failed_retrieval_preserves_proposal_and_reports_failure(self):
        with self.assertRaisesRegex(ValueError, "invalid actual result"):
            self.invoke(retrieval_error=ValueError("invalid actual result"))
        self.assertTrue(self.proposal.exists())
        report = json.loads((self.out / "completion_report.json").read_text())
        self.assertIs(report["complete"], False)
        self.assertIs(report["proposal_deleted"], False)

    def test_proposal_is_removed_only_after_successful_gates(self):
        report = self.invoke()
        self.assertIs(report["complete"], True)
        self.assertIs(report["proposal_deleted"], True)
        self.assertFalse(self.proposal.exists())

    def test_default_opt_out_preserves_proposal(self):
        self.args.remove_proposal = False
        self.assertIs(self.invoke()["complete"], True)
        self.assertTrue(self.proposal.exists())

    def test_unknown_waited_pid_is_an_error(self):
        with patch.object(finish, "process_identity", return_value=None):
            with self.assertRaisesRegex(ValueError, "already absent"):
                finish.wait_for_repair(99999, self.out)

    def test_unrelated_pid_is_rejected(self):
        with patch.object(finish, "process_identity", return_value={"command": ["vllm"], "start_tick": 1}):
            with self.assertRaisesRegex(ValueError, "does not belong"):
                finish.wait_for_repair(1, self.out)

    def test_runtime_dependency_overlay_reaches_actual_smoke_command(self):
        (self.out / "runtime_deps").mkdir()
        self.args.remove_proposal = False
        with patch.object(finish, "ROOT", self.root), patch.object(finish, "PROPOSAL", self.proposal), \
                patch.object(finish, "require_validation", return_value=self.validation), \
                patch.object(finish, "require_retrieval", return_value=self.smoke), \
                patch.object(finish.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="head")) as run:
            finish.run(self.args)
        command = run.call_args_list[0]
        self.assertEqual(command.args[0][:3], ["/rag/python", "-B", "-u"])
        self.assertEqual(command.kwargs["env"]["PYTHONPATH"].split(":" )[0], str(self.out / "runtime_deps"))

    def test_success_report_alone_cannot_replace_required_case_artifacts(self):
        report = {"complete": True, "graph_vectors_openie_unchanged": True,
                  "repaired_index": self.validation["repaired_index"], "asset_sha256": self.validation["asset_sha256"],
                  "cases": {name: {"complete": True, "n_samples": 3,
                                   "result_path": str(self.out / "retrieval_smoke" / name / "result.json")}
                            for name in ("hipporag2", "exp4_dependency_binding")}}
        finish.write_json(self.out / "retrieval_smoke_report.json", report)
        with self.assertRaisesRegex(ValueError, "artifact"):
            finish.require_retrieval(self.out, self.validation)


if __name__ == "__main__":
    unittest.main()
