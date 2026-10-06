from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from utils import representative_sampling as sampling


def hotpot_samples(count=1000):
    return [{'_id': f'q{i}', 'type': 'bridge' if i < count * .8 else 'comparison',
             'question': ('When was the author born?' if i % 3 == 0 else
                          'Which country did this person live in?' if i % 3 == 1 else 'Who wrote the book?'),
             'supporting_facts': [['A', 0], ['A', 1], ['B', 0]]} for i in range(count)]


class RepresentativeSamplingTests(unittest.TestCase):
    def test_sixty_thirty_split_covers_primary_and_excludes_prior_questions_reproducibly(self):
        samples = hotpot_samples()
        before = deepcopy(samples)
        options = dict(excluded_indices=range(216), seed=342)
        first = sampling.make_representative_split(samples, 'hotpotqa', **options)
        second = sampling.make_representative_split(samples, 'hotpotqa', **options)
        self.assertEqual(first, second)
        self.assertEqual(samples, before)
        screened, confirmed = set(first['screen_indices']), set(first['confirmation_indices'])
        self.assertEqual(len(screened), 60)
        self.assertEqual(len(confirmed), 30)
        self.assertTrue(screened.isdisjoint(confirmed))
        self.assertTrue((screened | confirmed).isdisjoint(range(216)))
        self.assertEqual(first['population_size'], 1000)
        self.assertEqual(first['available_size'], 784)
        self.assertEqual(sum(first['population_distribution']['primary'].values()), 1000)
        self.assertTrue(first['screen']['primary_coverage_complete'])
        self.assertTrue(first['confirmation']['primary_coverage_complete'])
        for phase in ('screen', 'confirmation'):
            self.assertGreaterEqual(min(first[phase]['primary_quotas'].values()), 2)
            weights = first[phase]['population_poststratification_weights']
            self.assertAlmostEqual(sum(row['weight'] for row in weights), 1.0)

    def test_musique_canonicalizes_parent_order_and_preserves_six_main_shapes(self):
        refs = {'2hop': ['', '#1'], '3hop1': ['', '#1', '#2'],
                '3hop2': ['', '', '#1 and #2'], '4hop1': ['', '#1', '#2', '#3'],
                '4hop2': ['', '', '#1 and #2', '#3'], '4hop3': ['', '#1', '', '#2 and #3']}
        samples = []
        for topology, nodes in refs.items():
            for index in range(20):
                questions = [text if index % 2 else text.replace('#1 and #2', '#2 and #1') for text in nodes]
                samples.append({'id': f'{topology}__{index}', 'question': 'Who is the final person?',
                                'question_decomposition': [{'question': text} for text in questions],
                                'paragraphs': [{'title': 'Same title', 'paragraph_text': f'text{i}', 'is_supporting': True}
                                               for i in range(len(nodes))]})
        result = sampling.make_representative_split(samples, 'musique')
        self.assertEqual(len(result['population_distribution']['primary']), 6)
        self.assertEqual(len(result['screen']['subset_distribution']['primary']), 6)
        self.assertEqual(len(result['confirmation']['subset_distribution']['primary']), 6)
        self.assertTrue(all(row['dag_valid'] for row in result['features']))
        self.assertEqual(result['features'][0]['gold_count'], 2)
        self.assertEqual(result['features'][-1]['hops'], 4)

    def test_wiki_gold_count_is_not_treated_as_hop_count(self):
        sample = {'_id': 'wiki', 'type': 'bridge_comparison', 'question': 'Which actor was older?',
                  'supporting_facts': [[str(i), 0] for i in range(4)]}
        feature = sampling.extract_features([sample], '2wikimultihopqa')[0]
        self.assertEqual(feature['hops'], 2)
        self.assertEqual(feature['gold_count'], 4)
        self.assertEqual(feature['terminal_attribute'], 'comparison')

    def test_primary_poststratification_defines_weights_and_rejects_missing_strata(self):
        features = [{'primary': 'a'}] * 8 + [{'primary': 'b'}] * 2
        indices = [0, 1, 8, 9]
        weights = sampling.poststratification_weights(features, indices)
        self.assertEqual([row['weight'] for row in weights], [.4, .4, .1, .1])
        rows = [{'query_index': index, 'metrics': {'Recall@5': float(index >= 8),
                                                  'Recall@10': 1.0}} for index in indices]
        result = sampling.poststratified_metrics(features, indices, rows)
        self.assertAlmostEqual(result['metrics']['Recall@5'], .2)
        self.assertAlmostEqual(result['metrics']['Recall@10'], 1.0)
        self.assertAlmostEqual(result['weight_sum'], 1.0)
        with self.assertRaisesRegex(ValueError, 'missing primary strata'):
            sampling.poststratification_weights(features, [0, 1])
        with self.assertRaisesRegex(ValueError, 'exactly'):
            sampling.poststratified_metrics(features, indices, rows[:-1])

    def test_sampling_does_not_consult_retrieval_scores_or_answers(self):
        class NoRetrieval(dict):
            def get(self, key, default=None):
                if key in ('retrieval_metrics', 'doc_scores', 'candidate_docs', 'answer'):
                    raise AssertionError(f'Sampling read forbidden {key}')
                return super().get(key, default)

        samples = [NoRetrieval(row, retrieval_metrics={'Recall@5': 1.0 if i % 2 else 0.0},
                               doc_scores=[999], answer='gold answer') for i, row in enumerate(hotpot_samples())]
        sampled = sampling.make_representative_split(samples, 'hotpotqa')
        self.assertEqual(len(sampled['screen_indices']), 60)
        self.assertEqual(len(sampled['confirmation_indices']), 30)

    def test_perfect_recall_stays_exactly_one_with_floating_point_weight_accumulation(self):
        features = [{'primary': 'same'}] * 60
        indices = list(range(60))
        rows = [{'query_index': index, 'metrics': {'Recall@5': 1.0, 'Recall@10': 1.0}}
                for index in indices]
        result = sampling.poststratified_metrics(features, indices, rows)
        self.assertGreater(result['weight_sum'], 1.0)
        self.assertEqual(result['metrics'], {'Recall@5': 1.0, 'Recall@10': 1.0})

    def test_rarity_oversampling_and_unavailable_strata_are_explicit(self):
        samples = hotpot_samples(100)
        for index, row in enumerate(samples):
            row['type'] = 'comparison' if index >= 98 else 'bridge'
        result = sampling.make_representative_split(samples, 'hotpotqa', screen_size=20, confirmation_size=10)
        self.assertEqual(result['screen']['primary_quotas']['type=comparison'], 2)
        rare = next(item for item in result['screen']['oversampled_primary_strata']
                    if item['primary'] == 'type=comparison')
        self.assertAlmostEqual(rare['oversampling_factor'], 5.0)
        self.assertEqual(result['confirmation']['unavailable_population_primary_strata'], ['type=comparison'])
        self.assertIsNone(result['confirmation']['population_poststratification_weights'])
        self.assertIn('missing primary strata', result['confirmation']['population_weighting_error'])

    def test_three_prior_rounds_are_merged_with_source_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            directories = ('exp4_improvement_selection', 'exp4_round2_selection',
                           'exp4_round2_selection_relation_plan_fix')
            for offset, directory in enumerate(directories):
                screen = {d: list(range(offset * 72, offset * 72 + 48)) for d in sampling.DATASETS}
                confirm = {d: list(range(offset * 72 + 48, (offset + 1) * 72)) for d in sampling.DATASETS}
                record = ({'screen': {'screen_indices': screen, 'confirmation_indices': confirm}} if offset == 0
                          else {'indices': {'screen': screen, 'confirmation': confirm}})
                path = root / 'metadata' / directory / 'selection.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(record))
            result = sampling.collect_prior_exclusions(root)
            self.assertEqual(len(result['sources']), 3)
            for dataset in sampling.DATASETS:
                self.assertEqual(result['indices'][dataset], list(range(216)))
            self.assertTrue(all(len(source['sha256']) == 64 for source in result['sources']))


if __name__ == '__main__':
    unittest.main()
