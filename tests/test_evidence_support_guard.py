"""Keep newly closed dependency evidence through the existing semantic swap gate."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest


PACKAGE = "_dependency_support_guard_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/pathcondrag")]
sys.modules[PACKAGE] = package
guard = importlib.import_module(PACKAGE + ".evidence_support_guard")


class ClosedDependencySupportTests(unittest.TestCase):
    def setUp(self):
        self.documents = {i: f"Page {i}\nThis source has an ordinary unrelated description."
                          for i in range(12)}
        self.order = list(range(12))
        self.proposal = {"promotions": [{"doc_id": 6, "victim": 2, "root_doc_id": 0}]}
        self.state = {"evidence_trace": {"plan": [], "bindings": {}, "branch_scores": []}}

    def run_guard(self):
        return guard.support_guard("Who wrote Example Work?", self.order, self.state,
                                   self.documents.__getitem__, self.proposal)

    def test_legacy_swap_still_allowed_without_joint_certificate(self):
        allowed, diagnostic = self.run_guard()
        self.assertTrue(allowed)
        self.assertNotIn("protected_joint_doc_ids", diagnostic)

    def test_closed_joint_support_cannot_be_swapped_out(self):
        self.state["evidence_trace"]["dependency_joint_selection"] = {
            "enabled": True, "mode": "dependency_joint", "protected_top5_doc_ids": [2]}
        allowed, diagnostic = self.run_guard()
        self.assertFalse(allowed)
        self.assertEqual(diagnostic["veto_reason"], "victim_is_closed_dependency_support")
        self.assertIn(2, diagnostic["protected_doc_ids"])

    def test_certificate_cannot_protect_outside_prefix_or_other_modes(self):
        for certificate in ({"enabled": True, "mode": "dependency", "protected_top5_doc_ids": [2]},
                            {"enabled": True, "mode": "dependency_joint", "protected_top5_doc_ids": [8, True, "2"]}):
            with self.subTest(certificate=certificate):
                self.state["evidence_trace"]["dependency_joint_selection"] = certificate
                allowed, diagnostic = self.run_guard()
                self.assertTrue(allowed)
                self.assertNotIn(2, diagnostic["protected_doc_ids"])


if __name__ == "__main__":
    unittest.main()
