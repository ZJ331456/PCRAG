from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from utils import difficulty_sampling as sampling
from utils import representative_sampling as representative


def samples(count=1000):
    return [{'_id': f'q{i}', 'type': 'bridge' if i % 5 else 'comparison',
             'question': 'When was the author born?' if i % 2 else 'Which country did this person live in?',
             'supporting_facts': [['A', 0], ['B', 0]]} for i in range(count)]


def measurement(index, difficulty):
    values = {'complete5': [.5, .5, 1, 1, 1, 1],
              'missing5_complete10': [.5, .5, .5, 1, 1, 1],
              'missing10_complete200': [.5, .5, .5, .5, .5, 1],
              'missing200': [0, 0, 0, 0, 0, .5]}[difficulty]
    return {'query_index': index, 'metrics': dict(zip(sampling.METRICS, values)),
            'all_gold_top5': values[2] == 1, 'all_gold_top10': values[3] == 1,
            'gold_ranks': [1, 4] if difficulty == 'complete5' else [1, 201]}


def measurements(count=1000):
    return {i: measurement(i, sampling.DIFFICULTIES[(i // 5) % 4]) for i in range(count)}


def split(data=None, rows=None, **options):
    data = samples() if data is None else data
    rows = measurements(len(data)) if rows is None else rows
    return sampling.make_difficulty_split(data, 'hotpotqa', [2] * len(data), [2] * len(data),
                                          rows, **{'excluded_indices': (), **options})


class DifficultySamplingTests(unittest.TestCase):
    def test_random_split_is_deterministic_disjoint_covers_cells_and_preserves_inputs(self):
        data, rows = samples(), measurements()
        original_data, original_rows = deepcopy(data), deepcopy(rows)
        options = dict(excluded_indices=range(576))
        first = split(data, rows, **options)
        self.assertEqual(first, split(data, rows, **options))
        self.assertEqual(first, json.loads(json.dumps(first)))
        self.assertEqual(data, original_data)
        self.assertEqual(rows, original_rows)
        self.assertEqual(first['available_size'], 424)
        screened, confirmed = set(first['screen_indices']), set(first['confirmation_indices'])
        self.assertEqual(len(screened), 90)
        self.assertEqual(len(confirmed), 45)
        self.assertFalse(screened & confirmed)
        self.assertFalse((screened | confirmed) & set(range(576)))
        for phase in ('screen', 'confirmation'):
            self.assertTrue(first[phase]['primary_coverage_complete'])
            self.assertEqual(first[phase]['missing_available_categories']['primary'], [])
            self.assertGreaterEqual(min(first[phase]['primary_quotas'].values()), 1)
            self.assertAlmostEqual(sum(row['weight'] for row in
                                       first[phase]['population_poststratification_weights']), 1)

    def test_exact_marginal_and_conditional_probabilities_match_available_pool_weights(self):
        result = split(excluded_indices=range(300))
        for phase in ('screen', 'confirmation'):
            report = result[phase]
            weights = {row['query_index']: row['weight'] for row in
                       report['initial_available_pool_poststratification_weights']}
            for row in report['inclusion_probabilities']:
                self.assertAlmostEqual(weights[row['query_index']],
                                       1 / result['available_size'] / row['marginal_probability'])
                if phase == 'screen':
                    self.assertEqual(row['marginal_probability'], row['conditional_probability'])
                else:
                    self.assertGreater(row['conditional_probability'], row['marginal_probability'])
            self.assertAlmostEqual(sum(weights.values()), 1)
        features = result['features']
        chosen = result['confirmation_indices']
        metrics = representative.poststratified_metrics(features, chosen,
            [measurements()[index] for index in chosen], target_indices=result['available_indices'],
            metrics=sampling.METRICS)
        self.assertEqual(metrics['metrics'],
                         result['historical_baseline_audit']['confirmation']['available_pool_weighted']['metrics'])

    def test_baseline_difficulty_prevents_easy_stratum_domination_in_a_toy_population(self):
        data = samples(200)
        for row in data:
            row['type'] = 'bridge'
        rows = {i: measurement(i, 'complete5' if i < 160 else 'missing5_complete10') for i in range(200)}
        result = split(data, rows, screen_size=90, confirmation_size=45)
        self.assertEqual(result['historical_difficulty_distribution']['screen'],
                         {'complete5': 72, 'missing5_complete10': 18})
        self.assertEqual(result['historical_difficulty_distribution']['confirmation'],
                         {'complete5': 36, 'missing5_complete10': 9})
        audit = result['historical_baseline_audit']
        self.assertEqual(audit['full_population_metrics']['Recall@5'], .9)
        for phase in ('screen', 'confirmation'):
            self.assertEqual(audit[phase]['raw_metrics']['Recall@5'], .9)
            self.assertAlmostEqual(audit[phase]['available_pool_weighted']['metrics']['Recall@5'], .9)
            self.assertAlmostEqual(audit[phase]['weighted_minus_available_pp']['Recall@5'], 0)

    def test_singleton_difficulty_is_explicitly_merged_before_draw_without_losing_phase_cells(self):
        data = samples(100)
        for row in data:
            row['type'] = 'bridge'
        labels = ['complete5'] * 90 + ['missing5_complete10'] * 7 + ['missing10_complete200'] * 2 + ['missing200']
        rows = {i: measurement(i, label) for i, label in enumerate(labels)}
        result = split(data, rows, screen_size=30, confirmation_size=15)
        merges = [row for row in result['difficulty_stratum_merge_audit'] if row['merged']]
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0]['difficulty_categories'], ['missing10_complete200', 'missing200'])
        self.assertEqual(merges[0]['available_count'], 3)
        for phase in ('screen', 'confirmation'):
            self.assertTrue(result[phase]['primary_coverage_complete'])
            self.assertEqual(result[phase]['population_weighting_error'], None)
            self.assertGreaterEqual(result[phase]['primary_quotas'][merges[0]['primary']], 1)

    def test_difficulty_absent_from_available_pool_is_merged_for_explicit_full_calibration(self):
        data = samples(100)
        for row in data:
            row['type'] = 'bridge'
        rows = {i: measurement(i, 'complete5' if i < 90 else 'missing200') for i in range(100)}
        result = split(data, rows, excluded_indices=range(90, 100), screen_size=30, confirmation_size=15)
        self.assertEqual(len(result['difficulty_stratum_merge_audit']), 1)
        self.assertTrue(result['difficulty_stratum_merge_audit'][0]['merged'])
        self.assertEqual(result['historical_difficulty_distribution']['available'], {'complete5': 90})
        self.assertEqual(result['historical_baseline_audit']['screen']['raw_metrics']['Recall@5'], 1)
        self.assertEqual(result['historical_baseline_audit']['full_population_metrics']['Recall@5'], .9)
        self.assertAlmostEqual(result['historical_baseline_audit']['screen']['calibrated_minus_full_pp']['Recall@5'], 10)

    def test_tiny_phase_budget_merges_difficulty_and_reports_structural_coverage(self):
        result = split(screen_size=8, confirmation_size=4)
        self.assertLessEqual(len(result['available_distribution']['primary']), 4)
        self.assertTrue(any(row['merged'] for row in result['difficulty_stratum_merge_audit']))
        for phase in ('screen', 'confirmation'):
            self.assertEqual(set(result[phase]['original_structure_distribution']['selected']),
                             {'type=bridge', 'type=comparison'})
            self.assertTrue(result[phase]['primary_coverage_complete'])

    def test_entire_structural_class_without_two_available_questions_fails_clearly(self):
        data = samples(100)
        excluded = [i for i in range(100) if data[i]['type'] == 'comparison' and i != 0]
        with self.assertRaisesRegex(ValueError, 'fewer than two available questions'):
            split(data, excluded_indices=excluded, screen_size=20, confirmation_size=10)
        with self.assertRaisesRegex(ValueError, 'every structural class'):
            split(screen_size=10, confirmation_size=1)

    def test_source_and_parameters_hashes_change_with_historical_scores_and_exclusions(self):
        data, rows = samples(), measurements()
        first = split(data, rows)
        changed = deepcopy(rows)
        changed[0] = measurement(0, 'missing200')
        second = split(data, changed)
        self.assertNotEqual(first['source_hashes']['historical_baseline_measurements_sha256'],
                            second['source_hashes']['historical_baseline_measurements_sha256'])
        self.assertNotEqual(first['sampling_protocol_sha256'], second['sampling_protocol_sha256'])
        third = split(data, rows, seed=1143)
        self.assertNotEqual(first['source_hashes']['parameters_sha256'], third['source_hashes']['parameters_sha256'])
        self.assertNotEqual(first['screen_indices'], third['screen_indices'])
        fourth = split(data, rows, excluded_indices=[999])
        self.assertNotEqual(first['source_hashes']['parameters_sha256'], fourth['source_hashes']['parameters_sha256'])

    def test_musique_preserves_all_six_structures_and_benchmark_hops_after_crossing_difficulty(self):
        shapes = {'2hop': ['', '#1'], '3hop1': ['', '#1', '#2'],
                  '3hop2': ['', '', '#1 and #2'], '4hop1': ['', '#1', '#2', '#3'],
                  '4hop2': ['', '', '#1 and #2', '#3'], '4hop3': ['', '#1', '', '#2 and #3']}
        data, hops, gold, rows = [], [], [], {}
        for topology, nodes in shapes.items():
            for index in range(40):
                query_index = len(data)
                data.append({'id': f'{topology}__{index}', 'question': 'Who is the final person?',
                    'question_decomposition': [{'question': text} for text in nodes],
                    'paragraphs': [{'title': 'Same title', 'paragraph_text': f'text{i}', 'is_supporting': True}
                                   for i in range(len(nodes))]})
                hops.append(len(nodes))
                gold.append(len(nodes))
                rows[query_index] = measurement(query_index, sampling.DIFFICULTIES[index % 4])
        result = sampling.make_difficulty_split(data, 'musique', hops, gold, rows, range(0, len(data), 11))
        self.assertEqual(len(result['available_distribution']['primary']), 24)
        for phase in ('screen', 'confirmation'):
            self.assertTrue(result[phase]['primary_coverage_complete'])
            self.assertEqual(len(result[phase]['original_structure_distribution']['selected']), 6)
            self.assertEqual(set(result[phase]['subset_distribution']['hops']), {'2', '3', '4'})
        wiki = [{'_id': f'w{i}', 'type': 'bridge_comparison', 'question': 'Which actor is older?',
                 'supporting_facts': [[str(j), 0] for j in range(4)]} for i in range(200)]
        result = sampling.make_difficulty_split(wiki, '2wikimultihopqa', [2] * 200, [4] * 200,
                                                measurements(200), [])
        self.assertTrue(all(row['hops'] == 2 and row['gold_count'] == 4 for row in result['features']))

    def test_proposed_module_outcomes_do_not_select_questions_or_secondary_greedy_balance(self):
        class Guarded(dict):
            def get(self, key, default=None):
                if key in ('new_module_metrics', 'doc_scores', 'answer', 'proposed_module'):
                    raise AssertionError(f'Forbidden proposed outcome access {key}')
                return super().get(key, default)

        data = [Guarded(row, new_module_metrics={'Recall@5': i % 2}, answer='held out answer')
                for i, row in enumerate(samples())]
        first = split(data)
        changed = deepcopy(data)
        for row in changed:
            row['new_module_metrics'] = {'Recall@5': 1.0}
        second = split(changed)
        self.assertEqual(first['screen_indices'], second['screen_indices'])
        self.assertEqual(first['confirmation_indices'], second['confirmation_indices'])
        self.assertIn('No secondary greedy', first['sampling_policy'])

    def test_invalid_or_missing_baseline_rows_exclusions_and_budgets_are_rejected(self):
        rows = measurements()
        for changed in (list(rows.values())[:-1], list(rows.values())[:-1] + [rows[0]]):
            with self.subTest(case='alignment'), self.assertRaises(ValueError):
                split(rows=changed)
        for metric, value in [('Recall@5', float('nan')), ('Recall@200', .2), ('Recall@1', True)]:
            changed = deepcopy(rows)
            changed[0]['metrics'][metric] = value
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                split(rows=changed)
        changed = deepcopy(rows)
        changed[0]['all_gold_top5'] = False
        with self.assertRaisesRegex(ValueError, 'all-gold'):
            split(rows=changed)
        for options in ({'excluded_indices': [-1]}, {'excluded_indices': [True]}, {'screen_size': 0},
                        {'confirmation_size': 2.5}, {'screen_size': 990, 'confirmation_size': 20}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                split(**options)


if __name__ == '__main__':
    unittest.main()
