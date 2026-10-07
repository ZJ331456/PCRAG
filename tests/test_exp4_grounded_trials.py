from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from utils import exp4_grounded_trials as runner


def reports():
    metrics = dict(zip(runner.RECALLS, (.3, .5, .7, .8, .9, .98)))
    return {dataset: {'retrieval_metrics': dict(metrics), 'difficulty_evaluation': {
        'initial_available_population': {'metrics': dict(metrics)}}} for dataset in runner.DATASETS}


def change(records, dataset, metric, delta, estimator='both'):
    if estimator in ('both', 'raw'):
        records[dataset]['retrieval_metrics'][metric] += delta
    if estimator in ('both', 'calibrated'):
        records[dataset]['difficulty_evaluation']['initial_available_population']['metrics'][metric] += delta


def parent_and_repaired():
    parent = {'results': []}
    for index in (7, 9):
        evidence = dict(plan=[{'id': 's1'}], bindings={'s1': 'Person'}, branch_scores=[], routes=[],
                        search_count=1, llm_plan_calls=1, llm_verification_calls=1,
                        planning_outputs={'raw': 'fixed original plan'}, verification_outputs=[])
        parent['results'].append({'query_index': index, 'retrieval_trace': {'evidence': evidence}})
    repaired = deepcopy(parent)
    for row in repaired['results']:
        trace = row['retrieval_trace']['evidence']
        trace['improvement_failure_recovery'] = {'original_parent_fields': deepcopy(trace)}
        trace['bindings'] = {'r1': 'Changed by recovery'}
        trace['llm_plan_calls'] += 1
    repaired['llm_request_stats'] = {'http_attempts': 2}
    return parent, repaired


class GroundedTrialsTests(unittest.TestCase):
    def test_first_formal_smoke_initializes_new_metadata_and_sampling_namespace(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            out = root / 'outputs' / 'trial'
            out.mkdir(parents=True)
            log = out / 'vllm.log'
            log.touch()
            contexts = {dataset: {'difficulty_sampling': {'screen_indices': [1, 2], 'confirmation_indices': [3]},
                'difficulty_sampling_sha256': 'sampling', 'historical_plan_prune_sha256': 'historical',
                'source': out / dataset, 'manifest': {'model_dir': 'model', 'source_asset_sha256': {}}}
                for dataset in runner.DATASETS}
            response = stack.enter_context(patch.object(runner.urllib.request, 'urlopen')).return_value
            response.__enter__.return_value.status = 200
            stack.enter_context(patch.object(runner, 'ROOT', root))
            stack.enter_context(patch.object(runner, 'build_contexts', return_value=(contexts, {})))
            stack.enter_context(patch.object(runner.previous, 'code_hashes', return_value={}))
            stack.enter_context(patch.object(runner.previous, 'protocol_for', return_value={'algorithm_code_sha256': {}}))
            stack.enter_context(patch.object(runner.previous, 'environment', return_value={}))
            stack.enter_context(patch.object(runner.previous.experiments, 'sha256', return_value='code'))
            stack.enter_context(patch.object(runner.previous.experiments, 'asset_hashes', return_value={}))
            phase = stack.enter_context(patch.object(runner, 'phase_run'))
            args = SimpleNamespace(out_root=str(out), mode='smoke', llm_base_url='http://localhost/v1',
                vllm_log=str(log), hippo_root=str(root / 'hippo'), screen_seed=1142, confirm_seed=1242,
                max_regression=1., target_gain=4., screen_size=90)
            self.assertEqual(runner.run(args), 0)
            phase.assert_called_once()
            metadata = out / 'metadata' / runner.SUMMARY_NAME
            self.assertTrue((metadata / 'selection.json').is_file())
            for dataset in runner.DATASETS:
                saved = json.loads((metadata / 'sampling' / f'{dataset}.json').read_text())
                self.assertEqual(saved, contexts[dataset]['difficulty_sampling'])

    def args(self):
        return SimpleNamespace(target_gain=4, max_regression=1)

    def summary(self, profile_reports):
        baseline = profile_reports['baseline']
        control = profile_reports['dag_control']
        entries = {}
        for profile in runner.PROFILES:
            entry = {'config': runner.asdict(profile), 'reports': profile_reports[profile.name]}
            if profile.name != 'baseline':
                entry['vs_plan_prune'] = runner.compare(self.args(), entry['reports'], baseline)
            if profile.name not in ('baseline', 'dag_control'):
                entry['vs_dag_control'] = runner.compare(self.args(), entry['reports'], control)
            entries[profile.name] = entry
        return {'screen': {'profiles': entries}}

    def test_parent_gate_checks_original_snapshot_while_allowing_real_repair_requests_and_outputs(self):
        parent, repaired = parent_and_repaired()
        gate = runner.validate_recovery_parent(parent, repaired)
        self.assertTrue(gate['validated'])
        self.assertEqual(gate['question_count'], 2)
        self.assertEqual(gate['equal_fields'], list(runner.PARENT_FIELDS))
        self.assertGreater(repaired['llm_request_stats']['http_attempts'], 0)
        repaired['results'].reverse()
        self.assertEqual(gate, runner.validate_recovery_parent(parent, repaired))

    def test_parent_gate_rejects_changed_subset_and_duplicate_indices_in_either_parent_or_child(self):
        parent, repaired = parent_and_repaired()
        variants = []
        changed = deepcopy(repaired)
        changed['results'].pop()
        variants.append((parent, changed))
        changed = deepcopy(repaired)
        changed['results'][0]['query_index'] = 99
        variants.append((parent, changed))
        changed = deepcopy(repaired)
        changed['results'].append(deepcopy(changed['results'][0]))
        variants.append((parent, changed))
        changed = deepcopy(parent)
        changed['results'].append(deepcopy(changed['results'][0]))
        variants.append((changed, repaired))
        for original, result in variants:
            with self.subTest(parent_count=len(original['results']), child_count=len(result['results'])), \
                    self.assertRaises(ValueError):
                runner.validate_recovery_parent(original, result)

    def test_parent_gate_rejects_mutation_of_every_original_plan_binding_route_or_llm_field(self):
        parent, repaired = parent_and_repaired()
        for field in runner.PARENT_FIELDS:
            changed = deepcopy(repaired)
            changed['results'][0]['retrieval_trace']['evidence']['improvement_failure_recovery'][
                'original_parent_fields'][field] = 'changed'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'initial parent outputs differ'):
                runner.validate_recovery_parent(parent, changed)

    def test_parent_gate_rejects_empty_subset_and_missing_snapshot_fields_even_when_parent_value_is_none(self):
        with self.assertRaises(ValueError):
            runner.validate_recovery_parent({'results': []}, {'results': []})
        parent, repaired = parent_and_repaired()
        field = 'planning_outputs'
        parent['results'][0]['retrieval_trace']['evidence'][field] = None
        snapshot = repaired['results'][0]['retrieval_trace']['evidence']['improvement_failure_recovery'][
            'original_parent_fields']
        snapshot[field] = None
        self.assertTrue(runner.validate_recovery_parent(parent, repaired)['validated'])
        del snapshot[field]
        with self.assertRaises(ValueError):
            runner.validate_recovery_parent(parent, repaired)

    def test_comparison_requires_same_metric_improvement_in_raw_and_difficulty_calibration(self):
        baseline = reports()
        for estimator in ('raw', 'calibrated'):
            candidate = deepcopy(baseline)
            change(candidate, 'hotpotqa', 'Recall@5', .02, estimator)
            judged = runner.compare(self.args(), candidate, baseline)
            self.assertTrue(judged['regression_limit_met'])
            self.assertFalse(judged['positive_signal'])
            self.assertEqual(judged['positive_metrics'], [])
        candidate = deepcopy(baseline)
        change(candidate, 'hotpotqa', 'Recall@5', .02, 'raw')
        change(candidate, 'hotpotqa', 'Recall@10', .02, 'calibrated')
        self.assertFalse(runner.compare(self.args(), candidate, baseline)['positive_signal'])
        change(candidate, 'hotpotqa', 'Recall@5', .02, 'calibrated')
        judged = runner.compare(self.args(), candidate, baseline)
        self.assertTrue(judged['positive_signal'])
        self.assertEqual(judged['positive_metrics'], ['hotpotqa/Recall@5'])
        self.assertFalse(judged['large_target_met'])

    def test_one_percentage_point_guard_protects_all_six_metrics_in_both_estimators(self):
        baseline = reports()
        for estimator in ('raw', 'calibrated'):
            for metric in runner.RECALLS:
                candidate = deepcopy(baseline)
                change(candidate, 'hotpotqa', 'Recall@5', .02)
                change(candidate, 'musique', metric, -.011, estimator)
                with self.subTest(metric=metric, estimator=estimator):
                    judged = runner.compare(self.args(), candidate, baseline)
                    self.assertFalse(judged['regression_limit_met'])
                    self.assertFalse(judged['positive_signal'])
        candidate = deepcopy(baseline)
        change(candidate, 'hotpotqa', 'Recall@5', .02)
        change(candidate, 'musique', 'Recall@200', -.01)
        self.assertTrue(runner.compare(self.args(), candidate, baseline)['positive_signal'])

    def test_only_existing_dag_gain_confirms_existing_control_without_claiming_new_contribution(self):
        baseline, control = reports(), reports()
        change(control, '2wikimultihopqa', 'Recall@5', .02)
        summary = self.summary({profile.name: deepcopy(baseline if profile.name == 'baseline' else control)
                                for profile in runner.PROFILES})
        profiles, names = runner.confirmation_profiles(summary)
        self.assertEqual(profiles, (runner.BASE, runner.DAG))
        self.assertEqual(names, ['dag_control'])
        for profile in (runner.SOURCE, runner.RECOVERY, runner.COMBINED):
            self.assertFalse(summary['screen']['profiles'][profile.name]['vs_dag_control']['positive_signal'])

    def test_raw_only_new_gain_does_not_enter_confirmation(self):
        measurements = {profile.name: reports() for profile in runner.PROFILES}
        for profile in (runner.SOURCE, runner.RECOVERY, runner.COMBINED):
            change(measurements[profile.name], 'hotpotqa', 'Recall@5', .02, 'raw')
        profiles, names = runner.confirmation_profiles(self.summary(measurements))
        self.assertEqual(profiles, ())
        self.assertEqual(names, [])

    def test_combined_confirmation_includes_source_parent_even_if_it_is_not_an_effective_candidate(self):
        measurements = {profile.name: reports() for profile in runner.PROFILES}
        change(measurements['witness_recovery'], 'hotpotqa', 'Recall@5', .02)
        profiles, names = runner.confirmation_profiles(self.summary(measurements))
        self.assertEqual(profiles, (runner.BASE, runner.DAG, runner.SOURCE, runner.COMBINED))
        self.assertEqual(names, ['witness_recovery'])
        self.assertLessEqual(len(profiles) * len(runner.DATASETS), 12)

    def test_new_candidate_cannot_preserve_dag_guard_but_breach_plan_prune_guard(self):
        measurements = {profile.name: reports() for profile in runner.PROFILES}
        for profile in (runner.DAG, runner.SOURCE, runner.RECOVERY, runner.COMBINED):
            change(measurements[profile.name], 'musique', 'Recall@200', -.015)
        change(measurements['source_witness'], 'hotpotqa', 'Recall@5', .02)
        summary = self.summary(measurements)
        self.assertTrue(summary['screen']['profiles']['source_witness']['vs_dag_control']['positive_signal'])
        self.assertFalse(summary['screen']['profiles']['source_witness']['vs_plan_prune']['regression_limit_met'])
        self.assertEqual(runner.confirmation_profiles(summary), ((), []))

    def test_phase_dispatch_uses_baseline_for_a_and_b_and_source_for_ab(self):
        parents = {name: {dataset: {'profile': name, 'dataset': dataset} for dataset in runner.DATASETS}
                   for name in ('baseline', 'source_witness')}
        summary = {'parent_caches': {'screen': parents}}
        contexts = {dataset: {'dataset': dataset} for dataset in runner.DATASETS}
        seen = []

        def group(args, context, temp, env, summary, path, phase, profile, parent, snapshot):
            seen.append((profile.name, parent['profile'] if parent else None, snapshot))

        with tempfile.TemporaryDirectory() as name, patch.object(runner, 'run_group', side_effect=group), \
                patch.object(runner, 'assessments'):
            runner.phase_run(self.args(), contexts, Path(name), {}, summary, Path(name) / 'selection.json',
                             'screen', runner.PROFILES)
        expected = {'baseline': (None, True), 'dag_control': ('baseline', False),
                    'source_witness': ('baseline', True), 'failure_recovery': ('baseline', False),
                    'witness_recovery': ('source_witness', False)}
        self.assertEqual(len(seen), 15)
        for profile, parent, snapshot in seen:
            self.assertEqual((parent, snapshot), expected[profile])
        summary = {'parent_caches': {'screen': {'baseline': parents['baseline']}}}
        with tempfile.TemporaryDirectory() as name, patch.object(runner, 'run_group'):
            with self.assertRaisesRegex(ValueError, 'source_witness cache is missing'):
                runner.phase_run(self.args(), contexts, Path(name), {}, summary, Path(name) / 'selection.json',
                                 'screen', (runner.COMBINED,))

    def test_formal_profiles_forward_worker_embedding_token_hop_and_index_reuse_parameters(self):
        args = SimpleNamespace(python='python', llm_base_url='http://127.0.0.1:8035/v1')
        context = {'dataset': 'musique', 'manifest': {'data_path': '/data', 'corpus_path': '/corpus'}}
        for profile in runner.PROFILES:
            command = runner.paired.command(args, context, Path('/case'), [445, 793], profile)
            for key, expected in {'--evidence_improvements': ','.join(profile.flags),
                                  '--embedding_batch_size': '4', '--llm_prefetch_workers': '8',
                                  '--openie_max_workers': '8', '--max_new_tokens': '2048',
                                  '--hop_source': 'benchmark', '--candidate_output_top_k': '200',
                                  '--result_top_k': '10', '--evidence_plan_validation': 'canonical_refs',
                                  '--evidence_plan_routing': 'question_structure'}.items():
                self.assertEqual(command[command.index(key) + 1], expected)
            self.assertIn('--reuse_index', command)
            self.assertNotIn('--force_index_from_scratch', command)


if __name__ == '__main__':
    unittest.main()
