"""Control-integrity checks use fake indices and outputs, without model calls."""
import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import exp4_ablations as controls
from utils import improvement_experiments as shared
import test_improvement_experiment_tools as fixtures


class Exp4AblationToolsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ImprovementExperimentTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.reference = f.out
        manifest = shared.manifest_at(self.reference)
        manifest["cases"] = ["exp3_prefix_coverage", "exp4_dependency_binding"]
        f.write(self.reference / "manifest.json", manifest)
        for name, stage in zip(manifest["cases"], (3, 4)):
            shared.initialize_case(SimpleNamespace(out_root=str(self.reference), name=name))
            result = f.result(stage)
            for row in result["results"]:
                row["retrieval_trace"]["evidence"] = {"llm_plan_calls": 1, "llm_verification_calls": 2,
                    "plan": [{"question": "DO NOT EXPOSE HISTORICAL ANSWER"}], "bindings": {"s1": "GOLD LEAK"}}
            f.write(self.reference / "cases" / name / "result.json", result)
            f.report_case(name)
        self.out = f.root / "new controls"
        self.args = argparse.Namespace(out_root=str(self.out), reference_root=str(self.reference),
            source_index=str(f.source), sample_size=0, sample_seed=42, sample_indices_file="", cases="")
        controls.prepare(self.args)

    def result(self, name):
        mode, binding, selection, stage = controls.CASES[name]
        result = self.fixture.result(stage)
        result["runtime_config"].update(evidence_ablation_mode=mode,
            evidence_binding_mode=binding, evidence_selection_mode=selection)
        for row in result["results"]:
            row["retrieval_trace"] = {"evidence": {"ablation": {
                "mode": mode, "binding_mode": binding, "selection_mode": selection,
                "llm_calls": 3, "call_budget": 3,
                "selection_shared_branches": mode == "selection", "shared_beam_width": 3}}}
            if selection in ("ancestor", "joint"):
                row["retrieval_trace"]["evidence"]["selection_diagnostics"] = {
                    "closure_valid": True, "policy": selection}
        return result

    def test_runtime_inputs_exclude_labels_and_historical_answers(self):
        inputs = controls.read_json(self.out / "ablation_inputs.json")
        for record in inputs["records"]:
            self.assertEqual(set(record), {"question", "sample_id", "query_index", "pool", "evidence_call_budget"})
            self.assertEqual(record["evidence_call_budget"], 3)
            self.assertEqual(len(record["pool"]), 200)
        self.assertNotIn("GOLD LEAK", (self.out / "ablation_inputs.json").read_text())

    def test_hardlinks_only_immutable_assets_and_cache_isolation(self):
        name = "exp4_abla3_literal"
        controls.initialize_case(SimpleNamespace(out_root=str(self.out), name=name))
        source, target = self.fixture.model, self.out / "cases" / name / "index" / shared.MODEL_DIR
        self.assertEqual((source / "graph.pickle").stat().st_ino, (target / "graph.pickle").stat().st_ino)
        self.assertNotEqual((source / "index_manifest.json").stat().st_ino, (target / "index_manifest.json").stat().st_ino)
        cache = self.out / "cases" / name / "index" / "llm_cache" / "qwen.sqlite"
        cache.write_bytes(b"new independent cache")
        self.assertEqual((self.reference / "initial_llm_cache" / "qwen.sqlite").read_bytes(), b"initial-cache")
        with self.assertRaisesRegex(ValueError, "Incomplete existing"):
            controls.initialize_case(SimpleNamespace(out_root=str(self.out), name=name))

    def test_fixed_pool_rejects_an_outside_candidate_even_if_corpus_valid(self):
        name = "exp4_abla2_fixed_binding"
        result = self.result(name)
        inputs = controls.read_json(self.out / "ablation_inputs.json")
        controls.validate_controls(result, controls.manifest_at(self.out), name, inputs)
        inputs["records"][0]["pool"][0]["doc_hash"] = "chunk-" + "0" * 32
        with self.assertRaisesRegex(ValueError, "Fixed Top200"):
            controls.validate_controls(result, controls.manifest_at(self.out), name, inputs)

    def test_budget_overrun_and_unspent_qd_allowance_are_rejected(self):
        name = "exp4_abla1_budget_qd"
        result = self.result(name)
        inputs, manifest = controls.read_json(self.out / "ablation_inputs.json"), controls.manifest_at(self.out)
        controls.validate_controls(result, manifest, name, inputs)
        control = result["results"][0]["retrieval_trace"]["evidence"]["ablation"]
        control["llm_calls"] = 4
        with self.assertRaisesRegex(ValueError, "exceeded"):
            controls.validate_controls(result, manifest, name, inputs)
        control["llm_calls"] = 2
        with self.assertRaisesRegex(ValueError, "query-generation"):
            controls.validate_controls(result, manifest, name, inputs)

    def test_missing_ancestor_check_and_wrong_variant_cannot_validate(self):
        name = "exp4_abla4_joint"
        result = self.result(name)
        inputs, manifest = controls.read_json(self.out / "ablation_inputs.json"), controls.manifest_at(self.out)
        result["results"][0]["retrieval_trace"]["evidence"]["selection_diagnostics"]["closure_valid"] = False
        with self.assertRaisesRegex(ValueError, "ancestor support"):
            controls.validate_controls(result, manifest, name, inputs)
        result["runtime_config"]["evidence_selection_mode"] = "coverage"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            controls.validate_controls(result, manifest, name, inputs)

    def test_success_marker_rechecks_results_and_non200_blocks_it(self):
        name = "exp4_abla3_literal"
        controls.initialize_case(SimpleNamespace(out_root=str(self.out), name=name))
        result = self.result(name)
        path = self.out / "cases" / name / "result.json"
        self.fixture.write(path, result)
        report_args = argparse.Namespace(out_root=str(self.out), name=name, elapsed=3,
            vllm_log=str(self.fixture.log), log_start=0, log_end=self.fixture.log.stat().st_size)
        self.fixture.log.write_text('POST /v1/chat/completions HTTP/1.1" 503\n')
        report_args.log_end = self.fixture.log.stat().st_size
        with self.assertRaisesRegex(ValueError, "Non-200"):
            controls.report(report_args)
        self.assertFalse((path.parent / "validated.ok").exists())
        self.fixture.log.write_text('POST /v1/chat/completions HTTP/1.1" 200\n')
        report_args.log_end = self.fixture.log.stat().st_size
        controls.report(report_args)
        self.assertEqual(controls.ready(SimpleNamespace(out_root=str(self.out), name=name)), 0)
        result["retrieval_metrics"]["Recall@5"] = 0
        self.fixture.write(path, result)
        with self.assertRaisesRegex(ValueError, "files changed"):
            controls.ready(SimpleNamespace(out_root=str(self.out), name=name))

    def test_prepare_does_not_change_old_results_and_checks_inputs_on_resume(self):
        old = (self.reference / "manifest.json").read_bytes()
        controls.prepare(self.args)
        self.assertEqual((self.reference / "manifest.json").read_bytes(), old)
        inputs = controls.read_json(self.out / "ablation_inputs.json")
        inputs["records"][0]["evidence_call_budget"] += 1
        self.fixture.write(self.out / "ablation_inputs.json", inputs)
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            controls.prepare(self.args)

    def test_selection_replays_same_responses_and_rejects_different_branches(self):
        coverage = "exp4_abla4_coverage"
        controls.initialize_case(SimpleNamespace(out_root=str(self.out), name=coverage))
        case = self.out / "cases" / coverage
        (case / "index" / "llm_cache" / "qwen.sqlite").write_bytes(b"shared exploration responses")
        self.fixture.write(case / "result.json", self.result(coverage))
        args = argparse.Namespace(out_root=str(self.out), name=coverage, elapsed=3,
            vllm_log=str(self.fixture.log), log_start=0, log_end=self.fixture.log.stat().st_size)
        controls.report(args)
        self.assertEqual(controls.ready(SimpleNamespace(out_root=str(self.out), name=coverage)), 0)
        ancestor = "exp4_abla4_ancestor"
        controls.initialize_case(SimpleNamespace(out_root=str(self.out), name=ancestor))
        replay_case = self.out / "cases" / ancestor
        self.assertEqual((replay_case / "index" / "llm_cache" / "qwen.sqlite").read_bytes(),
                         b"shared exploration responses")
        result = self.result(ancestor)
        result["results"][0]["retrieval_trace"]["evidence"]["routes"] = [{"query": "changed branch"}]
        self.fixture.write(replay_case / "result.json", result)
        args.name = ancestor
        with self.assertRaisesRegex(ValueError, "different branches"):
            controls.report(args)
        self.assertFalse((replay_case / "validated.ok").exists())
        self.fixture.write(replay_case / "result.json", self.result(ancestor))
        controls.report(args)
        report = controls.read_json(replay_case / "report.json")
        self.assertTrue(report["control_diagnostics"]["shared_exploration_replayed"])

    def test_selection_dependency_is_added_and_failure_creates_no_case(self):
        self.assertEqual(controls.selected_cases("exp4_abla4_joint"),
                         ["exp4_abla4_coverage", "exp4_abla4_joint"])
        name = "exp4_abla4_ancestor"
        with self.assertRaisesRegex(ValueError, "shared selection exploration"):
            controls.initialize_case(SimpleNamespace(out_root=str(self.out), name=name))
        self.assertFalse((self.out / "cases" / name).exists())


if __name__ == "__main__":
    unittest.main()
