from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from utils import exp4_improvements as runner


def reports(r5=.7, r10=.8, all5=.4):
    return {name: {'retrieval_metrics': {'Recall@5': r5, 'Recall@10': r10},
                   'all_gold_top5': all5} for name in runner.DATASETS}


class ImprovementRunnerTests(unittest.TestCase):
    def test_stratified_development_and_confirmation_are_disjoint_and_proportional(self):
        samples = [{'n_gold': 2}] * 75 + [{'n_gold': 4}] * 25
        with patch.object(runner.experiments, 'gold_docs', side_effect=lambda sample, name: set(range(sample['n_gold']))):
            screen = runner.stratified_indices(samples, [2] * 100, '2wikimultihopqa', 20)
            again = runner.stratified_indices(samples, [2] * 100, '2wikimultihopqa', 20)
            confirm = runner.stratified_indices(samples, [2] * 100, '2wikimultihopqa', 10, excluded=screen)
        self.assertEqual(screen, again)
        self.assertEqual(len(screen), 20)
        self.assertEqual(sum(samples[index]['n_gold'] == 4 for index in screen), 5)
        self.assertTrue(set(screen).isdisjoint(confirm))

    def test_smoke_uses_easy_and_largest_structure_group(self):
        samples = [{'n_gold': 2}, {'n_gold': 3}, {'n_gold': 4}]
        with patch.object(runner.experiments, 'gold_docs', side_effect=lambda sample, name: set(range(sample['n_gold']))):
            self.assertEqual(runner.stratified_indices(samples, [2, 3, 4], 'musique', 2), [0, 2])

    def test_universal_selection_requires_r5_gain_r10_nonregression_and_dataset_protection(self):
        baseline = reports()
        self.assertTrue(runner.comparison_score(reports(.72, .81, .42), baseline)['eligible'])
        self.assertFalse(runner.comparison_score(reports(.72, .79, .42), baseline)['eligible'])
        self.assertFalse(runner.comparison_score(reports(.7, .81, .42), baseline)['eligible'])
        candidate = reports(.73, .83, .45)
        candidate['hotpotqa']['retrieval_metrics']['Recall@5'] = .67
        self.assertGreater(runner.comparison_score(candidate, baseline)['mean_gain_r5'], 0)
        self.assertFalse(runner.comparison_score(candidate, baseline)['eligible'])

    def test_sqlite_warm_snapshot_includes_wal_and_omits_lock_files(self):
        with tempfile.TemporaryDirectory() as name:
            source, target = Path(name) / 'source', Path(name) / 'target'
            source.mkdir()
            db = source / 'cache.sqlite'
            with sqlite3.connect(db) as connection:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('CREATE TABLE requests (answer TEXT)')
                connection.execute("INSERT INTO requests VALUES ('committed in WAL')")
                connection.commit()
                (source / 'cache.sqlite.lock').write_text('lock')
                runner.snapshot_cache(source, target)
                with closing(sqlite3.connect(target / 'cache.sqlite')) as snapshot:
                    self.assertEqual(snapshot.execute('SELECT answer FROM requests').fetchone()[0],
                                     'committed in WAL')
            self.assertEqual([path.name for path in target.iterdir()], ['cache.sqlite'])

    def test_subset_manifest_keeps_question_indices_and_benchmark_hops(self):
        samples = [{'id': f'id-{i}', 'n_gold': i + 2} for i in range(3)]
        original = {'dataset': 'musique', 'runtime': {'llm_prefetch_workers': 8},
                    'selected_indices': [0, 1, 2]}
        with patch.object(runner.experiments, 'gold_docs', side_effect=lambda sample, name: set(range(sample['n_gold']))):
            subset = runner.subset_manifest(original, samples, [2, 3, 4], [0, 2])
        self.assertEqual(subset['selected_indices'], [0, 2])
        self.assertEqual(subset['benchmark_hops'], [2, 4])
        self.assertEqual(subset['hop_distribution'], {'2': 1, '4': 1})
        self.assertEqual(original['selected_indices'], [0, 1, 2])

    def test_command_preserves_model_call_limits_and_never_rebuilds_index(self):
        args = SimpleNamespace(python='python', llm_base_url='http://local/v1')
        context = {'dataset': 'musique', 'manifest': {'data_path': '/data.json', 'corpus_path': '/corpus.json'}}
        command = runner.command(args, context, Path('/case'), [0, 3], {'planning', 'adaptive'})
        for key, value in [('--embedding_batch_size', '4'), ('--llm_prefetch_workers', '8'),
                           ('--openie_max_workers', '8'), ('--max_new_tokens', '2048'),
                           ('--evidence_improvements', 'planning,adaptive')]:
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn('--reuse_index', command)
        self.assertIn('--hop_source', command)
        self.assertEqual(command[command.index('--hop_source') + 1], 'benchmark')
        self.assertNotIn('--force_index_from_scratch', command)
        self.assertEqual(command[command.index('--candidate_output_top_k') + 1], '200')
        self.assertEqual(command[command.index('--result_top_k') + 1], '10')

    def valid_summary(self):
        return {'chosen_flags': ['planning'],
                'smoke': {'baseline_reproduction': {name: True for name in runner.DATASETS},
                          'historical_prefix_matches': {name: False for name in runner.DATASETS},
                          'variants': {variant: {name: {'validated': True} for name in runner.DATASETS}
                                       for variant in ('baseline', 'all_five')}}}

    def test_smoke_gate_uses_fresh_replay_and_requires_all_datasets(self):
        summary = self.valid_summary()
        runner.require_smoke(summary)
        summary['smoke']['baseline_reproduction']['musique'] = False
        with self.assertRaisesRegex(ValueError, 'successful smoke'):
            runner.require_smoke(summary)

    def test_cleanup_before_full_removes_only_trial_results_and_preserves_warm_cache(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            trial = out / runner.TEMP_NAME
            for child in ('warm_cache', 'smoke', 'screen', 'confirmation'):
                (trial / child).mkdir(parents=True)
                (trial / child / 'sentinel').write_text(child)
            (out / 'cases/original').mkdir(parents=True)
            (out / 'cases/original/sentinel').write_text('keep original results')
            (out / 'shared_indexes').mkdir()
            summary = self.valid_summary()
            summary_file = out / 'selection.json'
            runner.cleanup_trial_results(out, summary, summary_file)
            self.assertEqual([path.name for path in trial.iterdir()], ['warm_cache'])
            self.assertTrue((out / 'cases/original/sentinel').is_file())
            self.assertTrue((out / 'shared_indexes').is_dir())
            self.assertTrue(summary_file.is_file())
            runner.cleanup_trials(out, summary, summary_file)
            self.assertFalse(trial.exists())
            self.assertTrue((out / 'cases/original/sentinel').is_file())

    def test_cleanup_refuses_symlink_outside_temporary_tree(self):
        with tempfile.TemporaryDirectory() as name:
            out, outside = Path(name) / 'run', Path(name) / 'outside'
            out.mkdir()
            outside.mkdir()
            (outside / 'sentinel').write_text('must remain')
            (out / runner.TEMP_NAME).symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'unexpected temporary'):
                runner.cleanup_trials(out, self.valid_summary(), out / 'selection.json')
            self.assertTrue((outside / 'sentinel').is_file())

    def test_repeated_sqlite_snapshots_have_same_starting_file_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            source = out / 'source'
            source.mkdir()
            with closing(sqlite3.connect(source / 'cache.sqlite')) as connection:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('CREATE TABLE requests (answer TEXT)')
                connection.execute("INSERT INTO requests VALUES ('answer')")
                connection.commit()
                runner.snapshot_cache(source, out / 'one')
                runner.snapshot_cache(source, out / 'two')
            self.assertEqual(runner.experiments.cache_hashes(out / 'one'),
                             runner.experiments.cache_hashes(out / 'two'))


if __name__ == '__main__':
    unittest.main()
