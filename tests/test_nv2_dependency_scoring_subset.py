"""CPU checks for score-independent sampling and genuine paired isolation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from utils import nv2_dependency_scoring_subset as runner


def samples():
    questions = ['Were A and B born in the same city?', 'Who wrote the book that was awarded C?',
                 'When was A born?', 'Where is B located?', 'Name the author of C.']
    return [{'question': questions[index % len(questions)], '_id': f'q{index}',
             'gold': 'does not affect sampling', 'type': f'unused-{index}'} for index in range(120)]


def pair_fixture():
    data = samples()
    sampling = runner.choose_subset(data, [2] * len(data), 10)
    results, measurements = [], []
    for index in sampling['indices']:
        evidence = {field: [] for field in runner.STRICT_UPSTREAM}
        evidence['finalizer_input_hash'] = 'frozen-input'
        trace = {field: [] for field in runner.STRICT_RETRIEVAL_UPSTREAM}
        trace['evidence'] = evidence
        results.append({'query_index': index, 'sample_id': f'q{index}', 'question': data[index]['question'],
                        'candidate_docs': ['doc A', 'doc B', 'doc C'], 'retrieval_trace': trace})
        measurements.append({'query_index': index, 'metrics': {metric: .5 for metric in runner.METRICS},
                             'all_gold_top5': False, 'all_gold_top10': True})
    return {'results': results}, {'per_question': measurements}, sampling


class SamplingTests(unittest.TestCase):
    def test_query_hop_strata_are_reproducible_and_do_not_use_gold(self):
        data = samples()
        hops = [2 if i < 60 else 4 for i in range(len(data))]
        first = runner.choose_subset(data, hops, 48, 1031)
        changed = deepcopy(data)
        for sample in changed:
            sample['gold'], sample['type'] = 'changed annotation', 'different-type'
        self.assertEqual(first, runner.choose_subset(changed, hops, 48, 1031))
        self.assertEqual(len(first['indices']), 48)
        self.assertTrue(first['all_strata_covered'])
        self.assertFalse(first['sampling_uses_gold_or_historical_results'])
        self.assertEqual(first['population_distribution'].keys(), first['sample_distribution'].keys())

    def test_smoke_covers_lowest_and_highest_benchmark_hop(self):
        data = samples()
        hops = [2 if i < 60 else 4 for i in range(len(data))]
        selected = runner.choose_subset(data, hops, 48, 1031, smoke=True)
        self.assertEqual(len(selected['indices']), 2)
        self.assertEqual({hops[i] for i in selected['indices']}, {2, 4})

    def test_too_small_sampling_budget_fails_instead_of_dropping_structures(self):
        with self.assertRaisesRegex(ValueError, 'cannot cover'):
            runner.choose_subset(samples(), [2] * 120, 2)


class PairedIsolationTests(unittest.TestCase):
    def test_identical_control_has_exact_zero_delta_and_interval(self):
        result, report, sampling = pair_fixture()
        paired = runner.paired_comparison(result, deepcopy(result), report, deepcopy(report), sampling, 100)
        for metric in runner.ALL_METRICS:
            self.assertEqual(paired['metrics'][metric]['delta'], 0)
            self.assertEqual(paired['metrics'][metric]['paired_stratified_bootstrap_ci95'], [0, 0])
        self.assertTrue(paired['candidate_top200_sets_identical'])
        self.assertFalse(paired['timing_comparison_valid'])

    def test_candidate_pool_change_is_not_reported_as_scoring_gain(self):
        result, report, sampling = pair_fixture()
        candidate = deepcopy(result)
        candidate['results'][0]['candidate_docs'][-1] = 'new document'
        with self.assertRaisesRegex(ValueError, 'Top200 candidate set differs'):
            runner.paired_comparison(result, candidate, report, report, sampling, 100)

    def test_changed_upstream_llm_outputs_fail_pairing(self):
        result, report, sampling = pair_fixture()
        candidate = deepcopy(result)
        candidate['results'][0]['retrieval_trace']['evidence']['verification_outputs'] = ['different response']
        with self.assertRaisesRegex(ValueError, 'cached upstream outputs differ'):
            runner.paired_comparison(result, candidate, report, report, sampling, 100)

    def test_changed_upstream_qd_also_fails_pairing(self):
        result, report, sampling = pair_fixture()
        candidate = deepcopy(result)
        candidate['results'][0]['retrieval_trace']['static_sub_questions'] = ['different question']
        with self.assertRaisesRegex(ValueError, 'retrieval.static_sub_questions'):
            runner.paired_comparison(result, candidate, report, report, sampling, 100)

    def test_missing_finalizer_hash_fails_pairing(self):
        result, report, sampling = pair_fixture()
        candidate = deepcopy(result)
        del candidate['results'][0]['retrieval_trace']['evidence']['finalizer_input_hash']
        with self.assertRaisesRegex(ValueError, 'finalizer_input_hash'):
            runner.paired_comparison(result, candidate, report, report, sampling, 100)

    def test_legal_reorder_with_positive_metric_is_measured_pairwise(self):
        result, report, sampling = pair_fixture()
        candidate, new_report = deepcopy(result), deepcopy(report)
        candidate['results'][0]['candidate_docs'].reverse()
        new_report['per_question'][0]['metrics']['Recall@5'] = 1.0
        paired = runner.paired_comparison(result, candidate, report, new_report, sampling, 100)
        self.assertAlmostEqual(paired['metrics']['Recall@5']['delta'], .05)
        self.assertEqual(paired['metrics']['Recall@5']['wins'], 1)
        self.assertEqual(paired['metrics']['Recall@5']['losses'], 0)
        self.assertTrue(paired['per_question'][0]['ranking_changed'])


if __name__ == '__main__':
    unittest.main()
