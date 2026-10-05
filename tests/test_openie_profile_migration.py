"""Unpublished profile migration must preserve the live source on failure."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathcondrag.index import profile_migration as migration
from pathcondrag.index.openie_checkpoint import OpenIECheckpoint
from pathcondrag.index.shared_index_builder import quality_profile


class OpenIEProfileMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.dataset = 'hotpotqa'
        self.model = self.out / 'shared_indexes' / self.dataset / migration.MODEL_DIR
        self.metadata = self.out / 'metadata' / self.dataset
        self.model.mkdir(parents=True)
        self.metadata.mkdir(parents=True)
        self.corpus = self.out / 'corpus.json'
        self.corpus.write_text(json.dumps([{'title': 'Ada', 'text': 'Ada trained in theater.'}]))
        self.passage = 'Ada\nAda trained in theater.'
        self.key = 'chunk-' + hashlib.md5(self.passage.encode()).hexdigest()
        self.identity = {'corpus_ids_sha256': hashlib.sha256(self.key.encode()).hexdigest(),
            'provenance': {
                'identity': {'model_name': 'qwen3-8b', 'ner_max_tokens': 512, 'triple_max_tokens': 2048},
                'producer': {'mode': 'online', 'class': 'OpenAI', 'endpoint': 'http://localhost'},
                'quality_profile': quality_profile(strict=False, prompt_version='optimized',
                                                   validation_mode='source_verified')}}
        self.manifest = {'openie': self.identity['provenance'], 'kept_native_field': 'unchanged'}
        self.manifest_path = self.model / 'index_manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.protocol = {'openie_validation_mode': 'source_verified', 'kept_protocol_field': 'unchanged'}
        (self.out / 'protocol.json').write_text(json.dumps(self.protocol))
        vector = self.model / 'chunk_embeddings/vdb_chunk.parquet'
        vector.parent.mkdir()
        vector.write_bytes(b'fixture passage vector bytes')
        self.journal = self.model / 'openie_progress.sqlite'
        progress = OpenIECheckpoint(self.journal, self.identity)
        progress.save({'idx': self.key, 'passage': self.passage,
            'extracted_entities': ['Ada'], 'extracted_triples': [['Ada', 'lives in', 'Paris']],
            'openie_metadata': {
                'ner': {'quality_status': 'success', 'complete': True, 'finish_reason': 'stop'},
                'triples': {'semantic_verified': True, 'fresh_extraction_history': [{
                    'stage': 'initial', 'triples': [['Ada', 'trained in', 'theater']],
                    'response': '{"triples":[["Ada","trained in","theater"]]}',
                    'metadata': {'finish_reason': 'stop', 'quality_status': 'success',
                                 'openie_attempt_count': 1}}]}},
            'openie_responses': {'ner': '{"named_entities":["Ada"]}', 'triples': 'filtered'}})
        progress.close()
        self.old_journal_bytes = self.journal.read_bytes()

    def read_identity(self):
        with closing(sqlite3.connect(self.journal.as_uri() + '?mode=ro', uri=True)) as connection:
            return json.loads(connection.execute("SELECT value FROM identity WHERE key='contract'").fetchone()[0])

    def run_migration(self):
        return migration.prepare_structural_resume(self.out, self.dataset, self.corpus)

    def assert_rolled_back(self):
        self.assertEqual(self.read_identity(), self.identity)
        self.assertEqual(self.journal.read_bytes(), self.old_journal_bytes)
        self.assertEqual(json.loads(self.manifest_path.read_text()), self.manifest)
        self.assertEqual(json.loads((self.out / 'protocol.json').read_text()), self.protocol)
        self.assertFalse((self.metadata / 'structural_migration_report.json').exists())
        self.assertEqual(list(self.model.glob('openie_progress.structural_*.sqlite')), [])

    def test_success_publishes_structural_journal_report_and_reuses_original(self):
        report = self.run_migration()
        self.assertEqual(report['reused_ner_count'], 1)
        self.assertEqual(report['reused_initial_triple_count'], 1)
        new_identity = self.read_identity()
        self.assertEqual(new_identity['provenance']['quality_profile']['validation_mode'], 'structural')
        with closing(sqlite3.connect(self.journal.as_uri() + '?mode=ro', uri=True)) as connection:
            row = json.loads(connection.execute('SELECT payload FROM progress').fetchone()[0])
        self.assertEqual(row['extracted_triples'], [['Ada', 'trained in', 'theater']])
        self.assertFalse(row['openie_metadata']['triples']['semantic_verified'])
        backup = Path(report['backup_dir'])
        self.assertEqual((backup / 'original_openie_progress.sqlite').read_bytes(), self.old_journal_bytes)
        self.assertTrue((backup / 'openie_progress.sqlite').is_file())
        self.assertEqual(json.loads(self.manifest_path.read_text())['kept_native_field'], 'unchanged')
        self.assertEqual(json.loads((self.out / 'protocol.json').read_text())['openie_validation_mode'], 'structural')
        self.assertEqual(self.run_migration(), report)

    def test_completed_graph_is_refused_before_any_backup_or_changes(self):
        (self.model / 'graph.pickle').write_bytes(b'existing completed graph')
        with self.assertRaisesRegex(ValueError, 'published or derived'):
            self.run_migration()
        self.assert_rolled_back()
        self.assertEqual(list(self.metadata.glob('source_verified_backup_*')), [])

    def test_first_journal_rename_failure_never_deletes_live_source(self):
        original_rename = Path.rename

        def fail_first(path, target):
            if path == self.journal:
                raise OSError('Injected before source journal moved')
            return original_rename(path, target)

        with mock.patch.object(Path, 'rename', fail_first):
            with self.assertRaisesRegex(OSError, 'before source journal'):
                self.run_migration()
        self.assert_rolled_back()

    def test_manifest_write_failure_restores_old_journal(self):
        original_write = migration._write

        def fail_manifest(path, value):
            if path == self.manifest_path:
                raise OSError('Injected manifest write failure')
            return original_write(path, value)

        with mock.patch.object(migration, '_write', side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, 'manifest write'):
                self.run_migration()
        self.assert_rolled_back()

    def test_report_write_failure_rolls_back_manifest_protocol_and_journal(self):
        original_write = migration._write

        def fail_report(path, value):
            if path == self.metadata / 'structural_migration_report.json':
                raise OSError('Injected migration report write failure')
            return original_write(path, value)

        with mock.patch.object(migration, '_write', side_effect=fail_report):
            with self.assertRaisesRegex(OSError, 'migration report'):
                self.run_migration()
        self.assert_rolled_back()


if __name__ == '__main__':
    unittest.main()
