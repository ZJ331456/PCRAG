from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from utils import exp4_round2 as runner


def reports(r5=.7, r10=.8):
    return {dataset: {'retrieval_metrics': {'Recall@5': r5, 'Recall@10': r10},
                      'all_gold_top5': .4} for dataset in runner.DATASETS}


class Round2RunnerTests(unittest.TestCase):
    def test_eight_configurations_separate_adaptive_mechanisms_and_strict_binding(self):
        by_name = {item.name: item for item in runner.VARIANTS}
        self.assertEqual(len(by_name), 8)
        self.assertEqual(by_name['adaptive_verify'].adaptive_mode, 'verify_only')
        self.assertEqual(by_name['adaptive_beam'].adaptive_mode, 'beam_only')
        self.assertEqual(by_name['adaptive_both'].adaptive_mode, 'both')
        self.assertEqual(by_name['binding_strict'].binding_validation, 'strict_relation')
        args = SimpleNamespace(python='python', llm_base_url='http://local/v1')
        context = {'dataset': 'musique', 'manifest': {'data_path': '/data', 'corpus_path': '/corpus'}}
        command = runner.command(args, context, Path('/case'), [7, 9], by_name['binding_strict_adaptive_verify'])
        expected = {'--embedding_batch_size': '4', '--llm_prefetch_workers': '8',
                    '--openie_max_workers': '8', '--max_new_tokens': '2048',
                    '--evidence_adaptive_mode': 'verify_only',
                    '--evidence_binding_validation': 'strict_relation', '--hop_source': 'benchmark',
                    '--candidate_output_top_k': '200', '--result_top_k': '10'}
        for key, value in expected.items():
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn('--reuse_index', command)
        self.assertNotIn('--force_index_from_scratch', command)

    def test_four_of_six_target_requires_each_dataset_and_limits_regression(self):
        baseline = reports()
        candidate = reports(.74, .795)
        candidate['2wikimultihopqa']['retrieval_metrics']['Recall@10'] = .84
        judged = runner.target_assessment(candidate, baseline)
        self.assertTrue(judged['strong_target_met'])
        self.assertFalse(judged['strict_target_met'])
        self.assertEqual(judged['qualified_metrics'], 4)
        candidate['musique']['retrieval_metrics']['Recall@10'] = .79
        self.assertTrue(runner.target_assessment(candidate, baseline)['strong_target_met'])
        candidate['musique']['retrieval_metrics']['Recall@10'] = .789
        self.assertFalse(runner.target_assessment(candidate, baseline)['strong_target_met'])
        candidate = reports(.74, .84)
        candidate['hotpotqa'] = baseline['hotpotqa']
        judged = runner.target_assessment(candidate, baseline)
        self.assertEqual(judged['qualified_metrics'], 4)
        self.assertFalse(judged['every_dataset_has_target_gain'])
        self.assertFalse(judged['strong_target_met'])

    def test_ceiling_is_reported_without_lowering_the_target(self):
        baseline, candidate = reports(), reports(.74, .84)
        baseline['hotpotqa']['retrieval_metrics'] = {'Recall@5': .98, 'Recall@10': 1.0}
        candidate['hotpotqa']['retrieval_metrics'] = {'Recall@5': 1.0, 'Recall@10': 1.0}
        judged = runner.target_assessment(candidate, baseline)
        self.assertEqual(judged['qualified_metrics'], 4)
        self.assertEqual(judged['unreachable_metrics'], ['hotpotqa/Recall@5', 'hotpotqa/Recall@10'])
        self.assertAlmostEqual(judged['metric_details']['hotpotqa']['Recall@5']['maximum_possible_gain'], 2.0)
        self.assertEqual(judged['metric_details']['hotpotqa']['Recall@5']['target_gain'], 4.0)
        self.assertFalse(judged['strong_target_met'])

    def test_absolute_relative_and_error_reduction_units_are_distinct(self):
        baseline, candidate = reports(.8, .8), reports(.84, .84)
        for unit, gain in [('absolute_pp', 4.0), ('relative_percent', 5.0), ('error_reduction', 20.0)]:
            judged = runner.target_assessment(candidate, baseline, gain_unit=unit)
            self.assertAlmostEqual(judged['metric_details']['musique']['Recall@5']['gain'], gain)
        weak = runner.target_assessment(reports(.71, .801), reports())
        self.assertTrue(weak['weak_signal'])
        self.assertFalse(weak['strong_target_met'])

    def test_new_screening_and_confirmation_exclude_prior_72_and_each_other(self):
        args = SimpleNamespace(screen_size=48, confirm_size=24, screen_seed=142, confirm_seed=242)
        contexts = {dataset: {'data': [{'n_gold': 2}] * 750 + [{'n_gold': 4}] * 250,
                              'hops': [2] * 1000} for dataset in runner.DATASETS}
        exclusions = {'excluded_indices': {dataset: list(range(72)) for dataset in runner.DATASETS}}
        with patch.object(runner.previous.experiments, 'gold_docs',
                          side_effect=lambda sample, dataset: set(range(sample['n_gold']))):
            indices = runner.development_indices(args, contexts, exclusions)
        for dataset in runner.DATASETS:
            screened, confirmed = indices['screen'][dataset], indices['confirmation'][dataset]
            self.assertEqual(len(screened), 48)
            self.assertEqual(len(confirmed), 24)
            self.assertTrue(set(screened).isdisjoint(range(72)))
            self.assertTrue(set(confirmed).isdisjoint(range(72)))
            self.assertTrue(set(confirmed).isdisjoint(screened))
            self.assertTrue(set(confirmed).isdisjoint(indices['smoke'][dataset]))

    def test_combination_has_measured_support_and_keeps_verified_adaptive_mode(self):
        entries = {item.name: {'config': runner.asdict(item), 'weak_signal': False,
                              'strong_target_met': False, 'score': 0.0} for item in runner.VARIANTS}
        self.assertEqual(runner.combined_candidate(entries), (None, []))
        entries['planning_selection']['weak_signal'] = True
        entries['adaptive_verify'].update(weak_signal=True, score=1.0)
        candidate, basis = runner.combined_candidate(entries)
        self.assertEqual(candidate.flags, ('planning', 'selection', 'adaptive'))
        self.assertEqual(candidate.adaptive_mode, 'verify_only')
        self.assertEqual(set(basis), {'planning_selection', 'adaptive_verify'})
        self.assertNotIn('binding', candidate.flags)
        entries[candidate.name] = {'config': runner.asdict(candidate), 'weak_signal': True,
                                    'strong_target_met': False, 'score': 2.0}
        self.assertEqual(runner.combined_candidate(entries)[0], candidate)

    def test_private_index_cleanup_preserves_exports_public_hardlinks_and_resume(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            public, case = out / 'public', out / 'case'
            public.mkdir()
            (case / 'index').mkdir(parents=True)
            (public / 'graph').write_text('frozen public asset')
            os.link(public / 'graph', case / 'index/graph')
            variant = runner.VARIANTS[1]
            manifest = {'model_dir': 'model', 'selected_indices': [7], 'source_asset_sha256': {'graph': 'same'}}
            context = {'source': public, 'manifest': manifest, 'cache_hashes': {'cache': 'same'}}
            runner.write_json(case / 'result.json', {'results': [{'query_index': 7}]})
            runner.write_json(case / 'report.json', {'validated': True, 'n_samples': 1,
                                                    'per_question': [{'query_index': 7}]})
            runner.write_json(case / 'manifest.json', manifest)
            runner.write_json(case / 'round2_config.json', runner.asdict(variant))
            runner.write_json(case / 'before.json', {'asset_sha256': {'graph': 'same'},
                                                     'initial_cache_sha256': {'cache': 'same'}})
            marker = {f'{kind}_sha256': runner.previous.experiments.sha256(case / f'{kind}.json')
                      for kind in ('result', 'report', 'manifest')}
            marker['round2_config_sha256'] = runner.previous.experiments.sha256(case / 'round2_config.json')
            runner.write_json(case / 'validated.ok', marker)
            with patch.object(runner.previous.experiments, 'asset_hashes', return_value={'graph': 'same'}):
                runner.cleanup_private_index(context, case)
                self.assertFalse((case / 'index').exists())
                self.assertEqual((public / 'graph').read_text(), 'frozen public asset')
                self.assertTrue(runner.verify_saved_case(context, case, manifest, variant)['validated'])
                runner.cleanup_private_index(context, case)
            (case / 'private_index_cleanup.json').write_text('{}')
            with patch.object(runner.previous.experiments, 'asset_hashes', return_value={'graph': 'same'}):
                with self.assertRaisesRegex(ValueError, 'cleanup proof changed'):
                    runner.verify_saved_case(context, case, manifest, variant)

    def test_protocol_includes_new_drivers_and_has_stable_json_roundtrip(self):
        args = SimpleNamespace(screen_seed=142, confirm_seed=242, gain_unit='absolute_pp', target_gain=4,
                               max_regression=1, allowed_exceptions=2, reference='matched_exp4',
                               previous_round_exclusions={'source_sha256': 'old-round', 'excluded_indices': {}})
        with patch.object(runner.previous, 'protocol_for', return_value={'algorithm_code_sha256': {}}):
            protocol = runner.protocol_for(args, {})
        for name in ('scripts/exp4_round2.py', 'scripts/utils/exp4_round2.py', 'scripts/run_exp4_round2.sh'):
            self.assertIn(name, protocol['algorithm_code_sha256'])
        self.assertEqual(protocol, json.loads(json.dumps(protocol)))
        self.assertEqual(protocol['target']['gain'], 4)
        self.assertEqual(protocol['target']['max_regression_absolute_pp'], 1)

    def test_mode_all_runs_only_small_stages_and_preserves_prior_directory(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / 'outputs') as name:
            out = Path(name)
            old = out / 'metadata/exp4_improvement_selection'
            old.mkdir(parents=True)
            (old / 'sentinel').write_text('original round unchanged')
            args = SimpleNamespace(out_root=str(out), mode='all', screen_size=48, confirm_size=24,
                                   screen_seed=142, confirm_seed=242, llm_base_url='http://local/v1',
                                   hippo_root='/hippo', vllm_log=str(out / 'vllm.log'))
            Path(args.vllm_log).write_text('')
            contexts = {dataset: {'source': out / 'public', 'manifest': {
                'model_dir': 'model', 'source_asset_sha256': {}}} for dataset in runner.DATASETS}
            response = MagicMock()
            response.__enter__.return_value.status = 200
            indices = {phase: {dataset: [] for dataset in runner.DATASETS}
                       for phase in ('smoke', 'screen', 'confirmation')}

            def phase_smoke(args, contexts, temp, env, summary, file):
                summary['smoke'] = {'complete': True}

            def phase_screen(args, contexts, temp, env, summary, file):
                summary['screen'] = {'complete': True}

            with patch.object(runner.urllib.request, 'urlopen', return_value=response), \
                    patch.object(runner.previous, 'code_hashes', return_value={}), \
                    patch.object(runner.previous, 'context_for', side_effect=lambda out, dataset, temp: contexts[dataset]), \
                    patch.object(runner.previous, 'environment', return_value={}), \
                    patch.object(runner, 'prior_exclusions', return_value={}), \
                    patch.object(runner, 'development_indices', return_value=indices), \
                    patch.object(runner, 'protocol_for', return_value={'round': 2}), \
                    patch.object(runner.previous.experiments, 'asset_hashes', return_value={}), \
                    patch.object(runner, 'smoke', side_effect=phase_smoke) as smoke, \
                    patch.object(runner, 'screen', side_effect=phase_screen) as screen, \
                    patch.object(runner, 'confirm') as confirm, \
                    patch.object(runner.previous, 'full') as full, \
                    patch.object(runner.previous, 'cleanup_trials') as cleanup:
                self.assertEqual(runner.run(args), 0)
            smoke.assert_called_once()
            screen.assert_called_once()
            confirm.assert_called_once()
            full.assert_not_called()
            cleanup.assert_not_called()
            saved = runner.read_json(out / 'metadata/exp4_round2_selection/selection.json')
            self.assertEqual(saved['progress']['planned_groups'], 36)
            self.assertEqual(saved['progress']['maximum_possible_groups'], 48)
            self.assertFalse(saved['automatic_full_run_enabled'])
            self.assertEqual((old / 'sentinel').read_text(), 'original round unchanged')


if __name__ == '__main__':
    unittest.main()
