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
                    '--evidence_plan_validation': 'strict',
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
                original_summary = out / 'metadata/exp4_round2_selection/selection.json'
                saved_original = original_summary.read_bytes()
                args.run_tag = 'relation_plan_fix'
                args.variants = ','.join(item.name for item in runner.FOLLOWUP_VARIANTS)
                self.assertEqual(runner.run(args), 0)
                self.assertEqual(original_summary.read_bytes(), saved_original)
            self.assertEqual(smoke.call_count, 2)
            self.assertEqual(screen.call_count, 2)
            self.assertEqual(confirm.call_count, 2)
            full.assert_not_called()
            cleanup.assert_not_called()
            saved = runner.read_json(out / 'metadata/exp4_round2_selection/selection.json')
            self.assertEqual(saved['progress']['planned_groups'], 36)
            self.assertEqual(saved['progress']['maximum_possible_groups'], 48)
            self.assertFalse(saved['automatic_full_run_enabled'])
            followup = runner.read_json(out / 'metadata/exp4_round2_selection_relation_plan_fix/selection.json')
            self.assertEqual(followup['progress']['planned_groups'], 24)
            self.assertEqual(followup['progress']['maximum_possible_groups'], 36)
            self.assertEqual(followup['run_tag'], 'relation_plan_fix')
            self.assertTrue((out / '_exp4_round2_trials_relation_plan_fix').is_dir())
            self.assertEqual((old / 'sentinel').read_text(), 'original round unchanged')

    def test_tagged_namespace_and_filtered_profiles_do_not_reuse_original_output_paths(self):
        metadata, temp = runner.namespace('relation_plan_fix')
        self.assertEqual(metadata, 'exp4_round2_selection_relation_plan_fix')
        self.assertEqual(temp, '_exp4_round2_trials_relation_plan_fix')
        for tag in ('', '../other', 'a/b', 'a-b', 'x' * 81):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                runner.namespace(tag)
        self.assertEqual(runner.selected_variants(SimpleNamespace()), runner.VARIANTS)
        requested = ','.join(item.name for item in runner.FOLLOWUP_VARIANTS)
        chosen = runner.selected_variants(SimpleNamespace(variants=requested))
        self.assertEqual(chosen, runner.FOLLOWUP_VARIANTS)
        self.assertEqual(len(chosen), 4)
        self.assertFalse(any('closure' in item.flags for item in chosen))
        for names in ('unknown', 'binding_conservative,binding_conservative', ''):
            with self.assertRaises(ValueError):
                runner.selected_variants(SimpleNamespace(variants=names))

    def test_followup_command_and_smoke_forward_plan_and_relation_validation(self):
        args = SimpleNamespace(python='python', llm_base_url='http://local/v1',
                               variants=','.join(item.name for item in runner.FOLLOWUP_VARIANTS))
        context = {'dataset': 'musique', 'manifest': {'data_path': '/data', 'corpus_path': '/corpus'}}
        planned = runner.FOLLOWUP_VARIANTS[0]
        command = runner.command(args, context, Path('/case'), [7, 9], planned)
        self.assertEqual(command[command.index('--evidence_plan_validation') + 1], 'canonical_refs')
        bound = runner.FOLLOWUP_VARIANTS[2]
        command = runner.command(args, context, Path('/case'), [7, 9], bound)
        self.assertEqual(command[command.index('--evidence_binding_validation') + 1], 'conservative_relation')
        self.assertEqual(command[command.index('--evidence_adaptive_mode') + 1], 'verify_only')
        smoke = runner.smoke_combination(args)
        self.assertEqual(smoke.plan_validation, 'canonical_refs')
        self.assertEqual(smoke.binding_validation, 'conservative_relation')
        self.assertEqual(smoke.flags, runner.previous.FLAGS)

    def test_tagged_exclusions_union_both_previous_rounds_and_preserve_their_files(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            old = out / 'metadata/exp4_improvement_selection/selection.json'
            old.parent.mkdir(parents=True)
            old_data = {'screen': {'screen_indices': {d: list(range(48)) for d in runner.DATASETS},
                                   'confirmation_indices': {d: list(range(48, 72)) for d in runner.DATASETS}}}
            runner.write_json(old, old_data)
            second = out / 'metadata/exp4_round2_selection/selection.json'
            second.parent.mkdir(parents=True)
            second_data = {'indices': {'screen': {d: list(range(72, 120)) for d in runner.DATASETS},
                                      'confirmation': {d: list(range(120, 144)) for d in runner.DATASETS}}}
            runner.write_json(second, second_data)
            original_bytes = old.read_bytes(), second.read_bytes()
            original = runner.prior_exclusions(out)
            self.assertEqual(len(original['excluded_indices']['musique']), 72)
            tagged = runner.prior_exclusions(out, 'relation_plan_fix')
            self.assertEqual(len(tagged['sources']), 2)
            for dataset in runner.DATASETS:
                self.assertEqual(tagged['excluded_indices'][dataset], list(range(144)))
            self.assertEqual((old.read_bytes(), second.read_bytes()), original_bytes)

    def test_filtered_combination_uses_only_available_positive_profiles(self):
        entries = {item.name: {'config': runner.asdict(item), 'weak_signal': False,
                              'strong_target_met': False, 'score': 0.0} for item in runner.FOLLOWUP_VARIANTS}
        self.assertEqual(runner.combined_candidate(entries), (None, []))
        entries['planning_selection_refs'].update(weak_signal=True, score=1.0)
        entries['binding_conservative_adaptive_verify'].update(weak_signal=True, score=2.0)
        combo, basis = runner.combined_candidate(entries)
        self.assertEqual(combo.flags, ('planning', 'selection', 'binding', 'adaptive'))
        self.assertEqual(combo.plan_validation, 'canonical_refs')
        self.assertEqual(combo.binding_validation, 'conservative_relation')
        self.assertEqual(combo.adaptive_mode, 'verify_only')
        self.assertEqual(set(basis), {'planning_selection_refs', 'binding_conservative_adaptive_verify'})

    def test_tagged_protocol_contains_subset_namespace_and_canonical_fields(self):
        args = SimpleNamespace(screen_seed=142, confirm_seed=242, gain_unit='absolute_pp', target_gain=4,
                               max_regression=1, allowed_exceptions=2, reference='matched_exp4',
                               previous_round_exclusions={}, run_tag='relation_plan_fix',
                               variants=','.join(item.name for item in runner.FOLLOWUP_VARIANTS))
        with patch.object(runner.previous, 'protocol_for', return_value={'algorithm_code_sha256': {}}):
            protocol = runner.protocol_for(args, {})
        self.assertEqual(protocol['run_tag'], 'relation_plan_fix')
        self.assertEqual(len(protocol['variant_configs']), 4)
        self.assertEqual(protocol['variant_configs'][0]['plan_validation'], 'canonical_refs')
        self.assertEqual(protocol['smoke_combination']['binding_validation'], 'conservative_relation')
        self.assertEqual(protocol, json.loads(json.dumps(protocol)))

    def test_representative_profiles_forward_support_without_global_selection(self):
        args = SimpleNamespace(python='python', llm_base_url='http://local/v1',
            variants=','.join(item.name for item in runner.REPRESENTATIVE_VARIANTS))
        context = {'dataset': 'musique', 'manifest': {'data_path': '/data', 'corpus_path': '/corpus'}}
        self.assertEqual(len(runner.selected_variants(args)), 4)
        for variant in runner.selected_variants(args):
            command = runner.command(args, context, Path('/case'), [1, 5], variant)
            self.assertEqual(command[command.index('--evidence_improvements') + 1], ','.join(variant.flags))
            self.assertEqual(command[command.index('--evidence_plan_routing') + 1], variant.plan_routing)
            self.assertEqual(command[command.index('--evidence_support_mode') + 1], variant.support_mode)
            self.assertNotIn('selection', variant.flags)
            self.assertNotIn('binding', variant.flags)
        self.assertEqual(runner.smoke_combination(args).flags, ('planning', 'support'))

    def test_failure_profiles_forward_atomic_routing_terminal_modes_and_original_baseline(self):
        args = SimpleNamespace(python='python', llm_base_url='http://local/v1',
            variants=','.join(item.name for item in runner.FAILURE_VARIANTS))
        context = {'dataset': 'musique', 'manifest': {'data_path': '/data', 'corpus_path': '/corpus'}}
        self.assertEqual(len(runner.selected_variants(args)), 5)
        for variant in runner.selected_variants(args):
            command = runner.command(args, context, Path('/case'), [7, 9], variant)
            self.assertEqual(command[command.index('--evidence_improvements') + 1], ','.join(variant.flags))
            self.assertEqual(command[command.index('--evidence_terminal_mode') + 1], variant.terminal_mode)
        self.assertEqual(runner.FAILURE_VARIANTS[0].plan_routing, 'all')
        self.assertEqual(runner.FAILURE_VARIANTS[-1].plan_routing, 'dependency_depth')
        self.assertEqual(runner.smoke_combination(args).terminal_mode, 'prefix')
        self.assertEqual(runner.smoke_combination(args).flags, ('planning', 'plan_prune', 'terminal'))
        self.assertEqual(runner.variant_from_config(runner.asdict(runner.FAILURE_VARIANTS[-1])),
                         runner.FAILURE_VARIANTS[-1])

    def test_all_prefix_metrics_are_guarded_even_when_r5_and_r10_improve(self):
        baseline, candidate = reports(), reports(.75, .85)
        for dataset in runner.DATASETS:
            baseline[dataset]['retrieval_metrics'].update({'Recall@1': .3, 'Recall@2': .5})
            candidate[dataset]['retrieval_metrics'].update({'Recall@1': .3, 'Recall@2': .5})
        candidate['hotpotqa']['retrieval_metrics']['Recall@2'] = .489
        metrics = ('Recall@1', 'Recall@2', 'Recall@5', 'Recall@10')
        judged = runner.target_assessment(candidate, baseline, protected_metrics=metrics)
        self.assertEqual(judged['qualified_metrics'], 6)
        self.assertFalse(judged['strong_target_met'])
        self.assertFalse(judged['weak_signal'])
        self.assertFalse(judged['protection_metric_details']['hotpotqa']['Recall@2']['regression_limit_met'])
        candidate['hotpotqa']['retrieval_metrics']['Recall@2'] = .49
        self.assertTrue(runner.target_assessment(candidate, baseline, protected_metrics=metrics)['strict_target_met'])

    def test_raw_guard_cannot_be_hidden_by_population_calibration(self):
        args = SimpleNamespace(sampling_policy='representative', target_gain=4,
            gain_unit='absolute_pp', max_regression=1, allowed_exceptions=2, require_raw_guard=True,
            protected_metrics='Recall@1,Recall@2,Recall@5,Recall@10')
        baseline, candidate = reports(), reports(.75, .85)
        for dataset in runner.DATASETS:
            baseline[dataset]['retrieval_metrics'].update({'Recall@1': .3, 'Recall@2': .5})
            candidate[dataset]['retrieval_metrics'].update({'Recall@1': .3, 'Recall@2': .5})
            baseline[dataset]['representative_evaluation'] = {'full_population': {
                'metrics': dict(baseline[dataset]['retrieval_metrics'])}}
            candidate[dataset]['representative_evaluation'] = {'full_population': {
                'metrics': dict(candidate[dataset]['retrieval_metrics'])}}
        candidate['musique']['retrieval_metrics']['Recall@1'] = .28
        judged = runner.assess(args, candidate, baseline)
        self.assertTrue(judged['weighted_regression_limit_met'])
        self.assertFalse(judged['raw_regression_limit_met'])
        self.assertFalse(judged['regression_limit_met'])
        self.assertFalse(judged['strong_target_met'])

    def test_six_metric_guard_rejects_r200_drop_without_discarding_measured_r5_gain(self):
        baseline, candidate = reports(), reports(.75, .85)
        metrics = ('Recall@1', 'Recall@2', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@200')
        for dataset in runner.DATASETS:
            baseline[dataset]['retrieval_metrics'].update({'Recall@1': .3, 'Recall@2': .5,
                                                          'Recall@20': .9, 'Recall@200': .98})
            candidate[dataset]['retrieval_metrics'].update({'Recall@1': .3, 'Recall@2': .5,
                                                          'Recall@20': .9, 'Recall@200': .98})
        candidate['musique']['retrieval_metrics']['Recall@200'] = .969
        judged = runner.target_assessment(candidate, baseline, protected_metrics=metrics)
        self.assertEqual(judged['qualified_metrics'], 6)
        self.assertAlmostEqual(judged['metric_details']['musique']['Recall@5']['delta_pp'], 5.0)
        self.assertAlmostEqual(judged['protection_metric_details']['musique']['Recall@200']['delta_pp'], -1.1)
        self.assertFalse(judged['regression_limit_met'])
        self.assertFalse(judged['strong_target_met'])
        args = SimpleNamespace(require_raw_guard=True)
        with tempfile.TemporaryDirectory() as name, patch.object(runner, 'phase_baselines') as run:
            summary = {'screen': {'complete': True, 'variants': {'bad': judged}}}
            runner.confirm(args, {}, Path(name), {}, summary, Path(name) / 'selection.json')
            run.assert_not_called()
            self.assertEqual(summary['selection_status'], 'small_sample_no_guarded_candidate')

    def test_explicit_prior_tags_are_validated_and_saved_in_protocol(self):
        args = SimpleNamespace(screen_seed=542, confirm_seed=642, gain_unit='absolute_pp', target_gain=4,
            max_regression=1, allowed_exceptions=2, reference='matched_exp4', previous_round_exclusions={},
            run_tag='failure_focused', variants=','.join(item.name for item in runner.FAILURE_VARIANTS),
            exclude_run_tags='structure_representative', protected_metrics='Recall@1,Recall@2,Recall@5,Recall@10',
            require_raw_guard=True)
        with patch.object(runner.previous, 'protocol_for', return_value={'algorithm_code_sha256': {}}):
            protocol = runner.protocol_for(args, {})
        self.assertEqual(protocol['exclude_run_tags'], ['structure_representative'])
        self.assertEqual(protocol['target']['protected_metrics'], ['Recall@1', 'Recall@2', 'Recall@5', 'Recall@10'])
        self.assertTrue(protocol['target']['require_raw_guard'])
        args.exclude_run_tags = 'failure_focused'
        with self.assertRaisesRegex(ValueError, 'own namespace'):
            runner.exclusion_tags(args)
        for value in ('Recall@3', 'Recall@5,Recall@5', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.protected_metrics(SimpleNamespace(protected_metrics=value))

    def test_confirmation_deduplicates_rankings_not_equal_aggregate_scores(self):
        with tempfile.TemporaryDirectory() as name:
            temp = Path(name)
            entries = []
            for variant, docs in [('first', ['A', 'B']), ('same', ['A', 'B']), ('different', ['B', 'A'])]:
                row = {'query_index': 7, 'metrics': {'Recall@1': .5, 'Recall@2': 1.0,
                       'Recall@5': 1.0, 'Recall@10': 1.0}, 'all_gold_top5': True, 'all_gold_top10': True}
                entry = {'config': {'name': variant}, 'reports': {dataset: {'per_question': [row]}
                                                                 for dataset in runner.DATASETS}}
                entries.append(entry)
                for dataset in runner.DATASETS:
                    path = temp / 'screen' / dataset / variant / 'result.json'
                    path.parent.mkdir(parents=True)
                    runner.write_json(path, {'results': [{'query_index': 7, 'docs': docs}]})
            chosen, equivalents = runner.unique_confirmation_candidates(entries, temp, 2)
            self.assertEqual([entry['config']['name'] for entry in chosen], ['first', 'different'])
            self.assertEqual(equivalents, {'first': ['same']})

    def test_confirmation_skips_every_candidate_that_breaches_raw_or_weighted_guard(self):
        with tempfile.TemporaryDirectory() as name:
            temp = Path(name)
            summary = {'screen': {'complete': True, 'variants': {
                'bad': {'regression_limit_met': False, 'weak_signal': False}}}}
            args = SimpleNamespace(require_raw_guard=True)
            with patch.object(runner, 'phase_baselines') as baselines:
                runner.confirm(args, {}, temp, {}, summary, temp / 'selection.json')
                baselines.assert_not_called()
            self.assertEqual(summary['selection_status'], 'small_sample_no_guarded_candidate')
            self.assertTrue(summary['confirmation']['skipped'])
            self.assertFalse(summary['automatic_full_run_enabled'])

    def test_representative_assessment_uses_population_calibration(self):
        args = SimpleNamespace(sampling_policy='representative', target_gain=4,
            gain_unit='absolute_pp', max_regression=1, allowed_exceptions=2)
        baseline, candidate = reports(), reports(.67, .77)
        for dataset in runner.DATASETS:
            baseline[dataset]['representative_evaluation'] = {'full_population': {'metrics': {
                'Recall@5': .7, 'Recall@10': .8}}}
            candidate[dataset]['representative_evaluation'] = {'full_population': {'metrics': {
                'Recall@5': .75, 'Recall@10': .85}}}
        self.assertTrue(runner.assess(args, candidate, baseline)['strong_target_met'])
        self.assertFalse(runner.target_assessment(candidate, baseline)['regression_limit_met'])

    def test_representative_indices_use_fixed_split_and_contrasting_smoke(self):
        args = SimpleNamespace(sampling_policy='representative')
        features = [{'hops': 2, 'primary': 'type=bridge'} for _ in range(200)]
        features[20] = {'hops': 2, 'primary': 'type=comparison'}
        contexts = {dataset: {'representative_sampling': {
            'features': features, 'screen_indices': list(range(10, 70)),
            'confirmation_indices': list(range(100, 130))}} for dataset in runner.DATASETS}
        indices = runner.development_indices(args, contexts, {})
        for dataset in runner.DATASETS:
            self.assertEqual(indices['screen'][dataset], list(range(10, 70)))
            self.assertEqual(indices['confirmation'][dataset], list(range(100, 130)))
            self.assertEqual(indices['smoke'][dataset], [10, 20])

    def test_representative_first_run_creates_sampling_reports_and_checks_resume(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / 'outputs') as name:
            out = Path(name)
            args = SimpleNamespace(out_root=str(out), mode='all', screen_size=60, confirm_size=30,
                screen_seed=342, confirm_seed=442, llm_base_url='http://local/v1',
                hippo_root='/hippo', vllm_log=str(out / 'vllm.log'),
                sampling_policy='representative', run_tag='structure_representative',
                variants=','.join(item.name for item in runner.REPRESENTATIVE_VARIANTS))
            Path(args.vllm_log).write_text('')
            contexts = {dataset: {'source': out / 'public', 'data': [{}] * 3, 'hops': [2] * 3,
                'manifest': {'model_dir': 'model', 'source_asset_sha256': {}}}
                for dataset in runner.DATASETS}
            split = {'features': [{'primary': 'bridge', 'hops': 2}] * 3,
                     'screen_indices': [0, 1], 'confirmation_indices': [2]}
            exclusions = {'indices': {dataset: [] for dataset in runner.DATASETS}, 'sources': []}
            response = MagicMock()
            response.__enter__.return_value.status = 200

            def phase_smoke(args, contexts, temp, env, summary, file):
                summary['smoke'] = {'complete': True}

            with patch.object(runner.urllib.request, 'urlopen', return_value=response), \
                    patch.object(runner.previous, 'code_hashes', return_value={}), \
                    patch.object(runner.previous, 'context_for', side_effect=lambda out, dataset, temp: contexts[dataset]), \
                    patch.object(runner.previous, 'environment', return_value={}), \
                    patch('utils.representative_sampling.collect_prior_exclusions', return_value=exclusions), \
                    patch('utils.representative_sampling.make_representative_split', return_value=split), \
                    patch.object(runner, 'protocol_for', return_value={'policy': 'representative'}), \
                    patch.object(runner.previous.experiments, 'asset_hashes', return_value={}), \
                    patch.object(runner, 'smoke', side_effect=phase_smoke), \
                    patch.object(runner, 'screen'), patch.object(runner, 'confirm'), \
                    patch.object(runner.previous, 'full') as full:
                self.assertEqual(runner.run(args), 0)
                summary_path = out / 'metadata/exp4_round2_selection_structure_representative/selection.json'
                saved = runner.read_json(summary_path)
                self.assertEqual(saved['maximum_possible_groups'], 33)
                for dataset, entry in saved['sampling_reports'].items():
                    path = Path(entry['path'])
                    self.assertTrue(path.is_file())
                    self.assertEqual(runner.read_json(path), split)
                    self.assertEqual(runner.previous.experiments.sha256(path), entry['sha256'])
                self.assertEqual(runner.run(args), 0)
                changed = deepcopy(saved)
                changed['indices']['screen']['musique'] = [1, 2]
                runner.write_json(summary_path, changed)
                with self.assertRaisesRegex(ValueError, 'question indices changed'):
                    runner.run(args)
                runner.write_json(summary_path, saved)
                Path(saved['sampling_reports']['musique']['path']).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'sampling report changed'):
                    runner.run(args)
                full.assert_not_called()


if __name__ == '__main__':
    unittest.main()
