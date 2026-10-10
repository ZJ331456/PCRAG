"""CPU checks for the full joint-only retrieval workflow controls."""
from pathlib import Path
import ast
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from utils import multi_dataset_retrieval_dependency_joint as runner


class JointWorkflowTests(unittest.TestCase):
    def test_smoke_work_is_separate_from_shared_index_and_cases(self):
        out = ROOT / 'outputs/example_nv2'
        self.assertEqual(runner.work_directory(out, True), out / runner.SMOKE_NAME)
        explicit = ROOT / 'outputs/example_nv2_joint_smoke'
        self.assertEqual(runner.work_directory(out, True, explicit), explicit)
        for invalid in (out, out / 'shared_indexes/x', out / 'cases/x', out / 'metadata/x', out.parent):
            with self.assertRaises(ValueError):
                runner.work_directory(out, True, invalid)

    def test_smoke_out_root_requires_smoke(self):
        with self.assertRaisesRegex(ValueError, 'requires --smoke'):
            runner.work_directory(ROOT / 'outputs/example_nv2', False, ROOT / 'outputs/test_smoke')

    def test_full_indices_cover_the_entire_dataset(self):
        manifest = {'selected_indices': list(range(10))}
        self.assertEqual(runner.selected_indices('musique', [{}] * 10, [2] * 5 + [4] * 5, manifest), list(range(10)))
        with self.assertRaisesRegex(ValueError, 'every question'):
            runner.selected_indices('musique', [{}] * 10, [2] * 10, {'selected_indices': [0, 1]})

    def test_tiny_smoke_selects_two_and_four_hops_without_truncating_corpus(self):
        indices = runner.selected_indices('musique', [{}] * 10, [2] * 5 + [4] * 5,
                                          {'selected_indices': list(range(10))}, True)
        self.assertEqual(indices, [0, 5])

    def test_joint_command_is_normal_eval_not_replay_and_has_one_scoring_switch(self):
        fake_round2 = SimpleNamespace(command=lambda *args: ['python', 'eval_dataset.py', '--reuse_index'])
        with patch.object(runner, 'round2', fake_round2, create=True):
            command = runner.command(None, None, None, None, None)
        self.assertEqual(command[-2:], ['--evidence_scoring_mode', 'dependency_joint'])
        self.assertNotIn('--frozen_upstream_json', command)
        self.assertNotIn('--frozen', command)

    def test_only_one_method_and_existing_final_flags(self):
        self.assertEqual(runner.CASE_NAME, 'legacy_dependency_joint')
        self.assertEqual(runner.FLAGS, ('planning', 'plan_prune', 'dag_package', 'support_semantic_veto'))

    def test_existing_nv2_entry_dispatches_joint_before_any_index_workflow(self):
        source = (ROOT / 'scripts/utils/multi_dataset_retrieval_nv2.py').read_text()
        definition = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        isolated = ast.Module(body=[definition], type_ignores=[])
        # Execute the real dispatch function without importing GPU-connected
        # entry module dependencies. Old workflow globals are intentionally
        # absent: accidentally reaching them would make this test fail.
        namespace = {'__package__': 'utils', 'JOINT_CASE_NAME': runner.CASE_NAME}
        exec(compile(isolated, 'nv2_entry_dispatch', 'exec'), namespace)
        args = SimpleNamespace(methods=[runner.CASE_NAME])
        with patch.object(runner, 'run', return_value=17) as joint:
            self.assertEqual(namespace['run'](args), 17)
            joint.assert_called_once_with(args)

    def test_runtime_validation_keeps_normal_retrieval_and_both_diagnostic_checks(self):
        config = {'evidence_scoring_mode': 'dependency_joint', 'evidence_plan_validation': 'canonical_refs',
                  'evidence_plan_routing': 'question_structure', 'embedding_batch_size': 4,
                  'openie_max_workers': 8, 'llm_prefetch_workers': 8, 'max_new_tokens': 2048,
                  'embedding_model_name': '/root/models/NV-Embed-v2'}
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            joint = lambda result: {'validated': True, 'per_question': []}
            semantic = SimpleNamespace(validate_finalizer=lambda case, result: {'validated': True})
            with patch.object(runner, 'validate_joint_selection', joint, create=True), patch.object(runner, 'semantic', semantic, create=True):
                report = runner.validate_case(case, {}, {'runtime_config': config})
                self.assertEqual(report['execution_mode'], 'normal_retrieval_no_frozen_replay')
                self.assertTrue((case / 'dependency_joint_selection_validation.json').is_file())
                self.assertTrue((case / 'support_semantic_veto_validation.json').is_file())
                config['enable_thinking'] = True
                with self.assertRaisesRegex(ValueError, 'disabled-thinking'):
                    runner.validate_case(case, {}, {'runtime_config': config})


if __name__ == '__main__':
    unittest.main()
