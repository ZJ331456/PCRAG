"""CPU checks for score-independent sampling and genuine paired isolation."""
from copy import deepcopy
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

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

    def test_missing_upstream_field_cannot_silently_pass_isolation(self):
        result, report, sampling = pair_fixture()
        candidate = deepcopy(result)
        del candidate['results'][0]['retrieval_trace']['evidence']['planning_outputs']
        with self.assertRaisesRegex(ValueError, 'planning_outputs.availability'):
            runner.paired_comparison(result, candidate, report, report, sampling, 100)

    def test_zero_verification_calls_allow_only_symmetric_absence(self):
        result, report, sampling = pair_fixture()
        for row in result['results']:
            evidence = row['retrieval_trace']['evidence']
            del evidence['verification_outputs']
            evidence['llm_verification_calls'] = 0
        paired = runner.paired_comparison(result, deepcopy(result), report, report, sampling, 100)
        self.assertEqual(len(paired['inactive_upstream_fields']), len(result['results']))
        self.assertIn('verification_outputs', paired['inactive_upstream_fields'][0]['inactive_fields'])
        candidate = deepcopy(result)
        candidate['results'][0]['retrieval_trace']['evidence']['verification_outputs'] = []
        with self.assertRaisesRegex(ValueError, 'verification_outputs.availability'):
            runner.paired_comparison(result, candidate, report, report, sampling, 100)

    def test_active_verification_cannot_omit_outputs_or_replace_them_with_null(self):
        result, report, sampling = pair_fixture()
        for row in result['results']:
            evidence = row['retrieval_trace']['evidence']
            evidence['llm_verification_calls'] = 1
            del evidence['verification_outputs']
        with self.assertRaisesRegex(ValueError, 'required upstream audit fields unavailable'):
            runner.paired_comparison(result, deepcopy(result), report, report, sampling, 100)
        for row in result['results']:
            row['retrieval_trace']['evidence']['verification_outputs'] = None
        with self.assertRaisesRegex(ValueError, 'verification_outputs.null'):
            runner.paired_comparison(result, deepcopy(result), report, report, sampling, 100)

    def test_dense_fallback_allows_only_jointly_inactive_graph_fields(self):
        result, report, sampling = pair_fixture()
        for row in result['results']:
            trace = row['retrieval_trace']
            trace['dense_fallback'] = True
            del trace['static_sub_questions']
        paired = runner.paired_comparison(result, deepcopy(result), report, report, sampling, 100)
        self.assertIn('retrieval.static_sub_questions', paired['inactive_upstream_fields'][0]['inactive_fields'])
        for row in result['results']:
            row['retrieval_trace']['dense_fallback'] = False
        with self.assertRaisesRegex(ValueError, 'required upstream audit fields unavailable'):
            runner.paired_comparison(result, deepcopy(result), report, report, sampling, 100)

    def test_joint_pair_reports_its_actual_candidate_name(self):
        result, report, sampling = pair_fixture()
        paired = runner.paired_comparison(result, result, report, report, sampling, 100,
                                          candidate_name='dependency_joint')
        self.assertIn('dependency_joint', paired['per_question'][0])
        self.assertNotIn('dependency_scoring', paired['per_question'][0])

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


class BaselineReuseTests(unittest.TestCase):
    def protocol(self):
        return dict(datasets=list(runner.DATASETS), smoke=False, index_root='/frozen',
                    build_indexes=False, qa=False, embedding_model='/root/models/NV-Embed-v2',
                    embedding_provider='nvembed', embedding_batch_size=4, nv_embedding_oom_split=False,
                    llm_prefetch_workers=8, openie_max_workers=8, max_new_tokens=2048, thinking=False,
                    hop_source='benchmark', flags=list(runner.FLAGS), vllm_gpu_memory_utilization=.52,
                    vllm_max_model_len=8192, sample_seed=1031, sample_size_per_dataset=96,
                    selected_indices={name: [1, 2] for name in runner.DATASETS},
                    source_sha256={name: {'graph': name} for name in runner.DATASETS},
                    cases=[list(case) for case in runner.CASES])

    def test_default_cases_remain_legacy_and_dependency(self):
        self.assertEqual(runner.case_specs(), runner.CASES)
        self.assertEqual(runner.case_specs('dependency_joint')[1], ('dependency_joint', 'dependency_joint'))
        with self.assertRaisesRegex(ValueError, 'Unknown candidate'):
            runner.case_specs('unimplemented')

    def test_joint_can_reuse_exact_protocol_with_different_candidate_mode(self):
        old, new = self.protocol(), self.protocol()
        new['cases'] = [list(case) for case in runner.case_specs('dependency_joint')]
        new['baseline_results_dir'] = '/prior-results'
        runner.validate_reuse_protocol(old, new)

    def test_reuse_rejects_changed_samples_or_indexes_or_generation_budget(self):
        current = self.protocol()
        for key, replacement in [('selected_indices', {}), ('source_sha256', {}),
                                 ('sample_seed', 42), ('max_new_tokens', 1024),
                                 ('nv_embedding_oom_split', True), ('thinking', True)]:
            with self.subTest(key=key):
                old = deepcopy(current)
                old[key] = replacement
                with self.assertRaisesRegex(ValueError, key):
                    runner.validate_reuse_protocol(old, current)

    def test_formal_baseline_cannot_be_reused_for_two_question_smoke(self):
        old, smoke = self.protocol(), self.protocol()
        smoke.update(smoke=True, sample_size_per_dataset=2)
        with self.assertRaisesRegex(ValueError, 'smoke'):
            runner.validate_reuse_protocol(old, smoke)

    def test_baseline_copy_isolates_mutable_sqlite_and_includes_committed_wal(self):
        # The actual backup helper is CPU-only and makes no API/model calls.
        from utils import exp4_improvements as previous
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'previous', previous, create=True):
            root = Path(directory)
            source, target = root / 'source', root / 'target'
            model = 'model'
            for name in previous.experiments.ASSETS:
                path = source / 'index' / model / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixed graph/vector')
            (source / 'result.json').write_text('original result')
            cache = source / 'index/llm_cache'
            cache.mkdir()
            (cache / 'cache.sqlite.lock').write_text('active lock')
            database = cache / 'cache.sqlite'
            with sqlite3.connect(database) as writer:
                writer.execute('PRAGMA journal_mode=WAL')
                writer.execute('CREATE TABLE values_for_test (value INTEGER)')
                writer.execute('INSERT INTO values_for_test VALUES (1)')
                writer.commit()
                runner.copy_reused_baseline(source, target, model)
                copied = target / 'index/llm_cache/cache.sqlite'
                self.assertNotEqual(database.stat().st_ino, copied.stat().st_ino)
                with sqlite3.connect(copied) as consumer:
                    self.assertEqual(consumer.execute('SELECT value FROM values_for_test').fetchall(), [(1,)])
                    consumer.execute('INSERT INTO values_for_test VALUES (2)')
                    consumer.commit()
                self.assertEqual(writer.execute('SELECT value FROM values_for_test').fetchall(), [(1,)])
                self.assertFalse((target / 'index/llm_cache/cache.sqlite.lock').exists())
                self.assertEqual((source / 'index' / model / 'graph.pickle').stat().st_ino,
                                 (target / 'index' / model / 'graph.pickle').stat().st_ino)
                self.assertNotEqual((source / 'result.json').stat().st_ino, (target / 'result.json').stat().st_ino)
                (target / 'result.json').write_text('private result')
                self.assertEqual((source / 'result.json').read_text(), 'original result')


class JointHookValidationTests(unittest.TestCase):
    def result(self):
        return {'results': [{'query_index': 1, 'retrieval_trace': {'evidence': {
            'improvement_dag_package': {'eligible_proofs': {'s1': [{'doc_id': 1}]}},
            'dependency_joint_selection': {
                'enabled': True, 'mode': 'dependency_joint', 'gold_labels_used': False,
                'extra_requests': 0, 'extra_llm_requests': 0, 'extra_embedding_calls': 0,
                'top200_set_preserved': True, 'document_set_preserved': True,
                'unique_documents': True, 'top2_preserved': True, 'fallback': None,
                'node_denominator': 2, 'terminal_denominator': 1,
                'eligible_proofs': {'s1': [{'doc_id': 1}], 's2': [{'doc_id': 4}]},
                'prefix_optimization': [{
                    'top_k': 5, 'set_changed': True, 'coverage_before': ['s1'],
                    'coverage_after': ['s1', 's2'], 'complete_terminals_before': [],
                    'complete_terminals_after': ['s2'], 'node_coverage_before': .5,
                    'node_coverage_after': 1.0, 'terminal_coverage_before': 0.0,
                    'terminal_coverage_after': 1.0, 'added_doc_ids': [4], 'removed_doc_ids': [3]}],
            }}}}]}

    def test_joint_hook_counts_new_proofs_and_actual_prefix_coverage_gain(self):
        report = runner.validate_joint_selection(self.result())
        self.assertEqual(report['counts']['queries_with_new_proofs'], 1)
        self.assertEqual(report['counts']['new_proof_alternatives'], 1)
        self.assertEqual(report['counts']['queries_changed_top5'], 1)
        self.assertEqual(report['coverage_means_among_optimization_attempts']['top5_terminal_coverage_after'], 1)

    def test_runtime_mode_without_joint_hook_cannot_be_declared_valid(self):
        result = self.result()
        del result['results'][0]['retrieval_trace']['evidence']['dependency_joint_selection']
        with self.assertRaisesRegex(ValueError, 'hook did not export'):
            runner.validate_joint_selection(result)

    def test_joint_no_request_and_no_gold_and_preserved_top2_are_required(self):
        for key, replacement in [('extra_requests', 1), ('gold_labels_used', True),
                                 ('top2_preserved', False), ('top200_set_preserved', False)]:
            with self.subTest(key=key):
                result = self.result()
                result['results'][0]['retrieval_trace']['evidence']['dependency_joint_selection'][key] = replacement
                with self.assertRaisesRegex(ValueError, 'runtime invariants failed'):
                    runner.validate_joint_selection(result)

    def test_changed_prefix_requires_new_terminal_and_preserved_support(self):
        result = self.result()
        prefix = result['results'][0]['retrieval_trace']['evidence']['dependency_joint_selection']['prefix_optimization'][0]
        prefix['complete_terminals_after'] = []
        with self.assertRaisesRegex(ValueError, 'strict terminal coverage gain'):
            runner.validate_joint_selection(result)
        prefix['complete_terminals_after'] = ['s2']
        prefix['coverage_after'] = ['s2']
        with self.assertRaisesRegex(ValueError, 'lost supported ancestor nodes'):
            runner.validate_joint_selection(result)


if __name__ == '__main__':
    unittest.main()
