"""Migration must reuse initial facts rather than semantic-filtered outputs."""

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from utils.openie_checkpoint_migration import (
    migrate_source_verified_checkpoint, validate_migration_identity,
)
from pathcondrag.index.openie_checkpoint import OpenIECheckpoint
from pathcondrag.index.openie.structural_openie import StructuralOpenIE
from pathcondrag.utils.misc_utils import TripleRawOutput


class OpenIECheckpointMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.chunks = {'chunk-a': {'content': 'Ada trained in theater.'},
                       'chunk-b': {'content': 'Bo lived in Paris.'}}
        digest = hashlib.sha256('\n'.join(sorted(self.chunks)).encode()).hexdigest()
        self.old_identity = {'corpus_ids_sha256': digest, 'provenance': {
            'identity': {'model_name': 'qwen3-8b', 'seed': 42, 'ner_max_tokens': 512},
            'producer': {'mode': 'online', 'class': 'OpenAI', 'endpoint': 'localhost'},
            'quality_profile': {'semantic_scope': 'all_final_relations',
                                'ner_prompt_sha256': 'ner-same', 'ner_max_tokens': 512,
                                'ner_recovery': 'ner-v1', 'prompt_sha256': 'triple-same',
                                'triple_max_tokens': 2048}}}
        self.new_identity = copy.deepcopy(self.old_identity)
        self.new_identity['provenance']['quality_profile'].update(
            validation_mode='structural', semantic_scope='none',
            structural_schema='pathcondrag_structural_openie_v1')

    def write_source(self, rows):
        path = self.root / 'source.sqlite'
        progress = OpenIECheckpoint(path, self.old_identity)
        for row in rows:
            progress.save(row)
        progress.close()
        return path

    def initial(self, triples=None, *, calls=1, finish='stop', status='success'):
        return {'stage': 'initial', 'response': '{"triples":[["Ada","trained in","theater"]]}',
                'triples': triples if triples is not None else [['Ada', 'trained in', 'theater']],
                'metadata': {'finish_reason': finish, 'quality_status': status,
                             'openie_attempt_count': calls}}

    def row(self, key='chunk-a', *, initial=None, ner_only=False):
        return {'idx': key, 'passage': self.chunks[key]['content'],
                'extracted_entities': ['Ada'],
                # A different final auditor-selected result must never be copied.
                'extracted_triples': [] if ner_only else [['Ada', 'lived in', 'Rome']],
                'openie_metadata': {
                    'ner': {'quality_status': 'success', 'complete': True, 'finish_reason': 'stop',
                            'ner_max_tokens_used': 512, 'ner_attempts': [{'original': True}]},
                    'triples': {'quality_status': 'pending'} if ner_only else {
                        'source_verified_schema': 'source-v2', 'semantic_verified': True,
                        'fresh_extraction_history': [initial or self.initial()],
                        'semantic_verification_history': [{'complete': True}]}},
                'openie_responses': {'ner': '{"named_entities":["Ada"]}', 'triples': 'audited'}}

    def migrate(self, source, *, new_identity=None, chunks=None):
        target = self.root / 'target.sqlite'
        report = migrate_source_verified_checkpoint(
            source, target, source_identity=self.old_identity,
            target_identity=new_identity or self.new_identity, chunks=chunks or self.chunks)
        with sqlite3.connect('file:' + str(target) + '?mode=ro', uri=True) as connection:
            rows = [json.loads(row[0]) for row in connection.execute('SELECT payload FROM progress')]
            identity = json.loads(connection.execute("SELECT value FROM identity WHERE key='contract'").fetchone()[0])
        return report, rows, identity

    def test_uses_initial_not_final_filtered_facts_and_preserves_ner(self):
        original = self.row()
        source = self.write_source([original])
        before = source.read_bytes()
        report, rows, identity = self.migrate(source)
        self.assertEqual(rows[0]['extracted_triples'], [['Ada', 'trained in', 'theater']])
        self.assertEqual(rows[0]['openie_metadata']['ner'], original['openie_metadata']['ner'])
        metadata = rows[0]['openie_metadata']['triples']
        self.assertTrue(metadata['complete'])
        self.assertFalse(metadata['semantic_verified'])
        self.assertNotIn('semantic_verification_history', metadata)
        self.assertEqual(metadata['validation_scope'], 'structural')
        self.assertEqual(report['reused_initial_triple_count'], 1)
        self.assertEqual(identity, self.new_identity)
        self.assertEqual(source.read_bytes(), before)
        self.assertTrue(StructuralOpenIE.is_verified_complete(TripleRawOutput(
            rows[0]['idx'], rows[0]['openie_responses']['triples'], rows[0]['extracted_triples'], metadata)))

    def test_missing_empty_or_bad_initial_keeps_ner_and_marks_pending(self):
        for variant in ('missing', 'empty', 'bad', 'length'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(dir=self.root) as directory:
                self.root = Path(directory)
                initial = {'empty': self.initial([]), 'bad': self.initial([['Ada', 'is']]),
                           'length': self.initial(finish='length')}.get(variant)
                row = self.row(initial=initial, ner_only=variant == 'missing')
                report, rows, _ = self.migrate(self.write_source([row]))
                self.assertEqual(report['reused_ner_count'], 1)
                self.assertEqual(report['pending_triple_count'], 1)
                self.assertEqual(rows[0]['extracted_triples'], [])
                self.assertFalse(rows[0]['openie_metadata']['triples']['complete'])
                self.assertFalse(rows[0]['openie_metadata']['triples']['semantic_verified'])
                self.root = Path(self.temporary.name)

    def test_earliest_normal_initial_survives_nested_resumed_audit(self):
        earliest = self.initial()
        row = self.row()
        old_metadata = copy.deepcopy(row['openie_metadata']['triples'])
        row['openie_metadata']['triples']['fresh_extraction_history'] = [{
            'stage': 'initial', 'triples': [['Ada', 'lived in', 'Rome']], 'response': 'filtered',
            'metadata': old_metadata},
            {'stage': 'atomic', 'triples': [['Ada', 'trained in', 'London']], 'metadata': {}}]
        report, rows, _ = self.migrate(self.write_source([row]))
        self.assertEqual(rows[0]['extracted_triples'], earliest['triples'])
        self.assertEqual(report['reused_initial_triple_count'], 1)

    def test_compact_and_atomic_without_original_cannot_become_initial(self):
        row = self.row()
        row['openie_metadata']['triples']['fresh_extraction_history'] = [
            {'stage': 'compact', 'triples': self.initial()['triples'], 'metadata': {'finish_reason': 'stop'}},
            {'stage': 'atomic', 'triples': self.initial()['triples'], 'metadata': {'finish_reason': 'stop'}}]
        report, rows, _ = self.migrate(self.write_source([row]))
        self.assertEqual(report['reused_initial_triple_count'], 0)
        self.assertEqual(rows[0]['extracted_triples'], [])

    def test_historical_requests_remain_explicit_new_budget_starts_at_zero(self):
        report, rows, _ = self.migrate(self.write_source([self.row(initial=self.initial(calls=8))]))
        metadata = rows[0]['openie_metadata']['triples']
        self.assertEqual(report['reused_initial_triple_count'], 1)
        self.assertTrue(metadata['migrated_initial'])
        self.assertEqual(metadata['structural_infer_calls'], 0)
        self.assertEqual(metadata['checkpoint_migration']['initial_call_count'], 8)

    def test_changed_triple_prompt_reuses_only_ner(self):
        target_identity = copy.deepcopy(self.new_identity)
        target_identity['provenance']['quality_profile']['prompt_sha256'] = 'different'
        report, rows, _ = self.migrate(self.write_source([self.row()]), new_identity=target_identity)
        self.assertEqual(report['reused_ner_count'], 1)
        self.assertEqual(report['reused_initial_triple_count'], 0)
        self.assertEqual(rows[0]['extracted_triples'], [])

    def test_changed_ner_producer_or_model_contract_refused(self):
        for group, key, value in (('quality_profile', 'ner_prompt_sha256', 'different'),
                                  ('quality_profile', 'ner_max_tokens', 1024),
                                  ('quality_profile', 'ner_recovery', 'different'),
                                  ('producer', 'endpoint', 'different'),
                                  ('identity', 'model_name', 'different')):
            with self.subTest(group=group, key=key):
                target_identity = copy.deepcopy(self.new_identity)
                target_identity['provenance'][group][key] = value
                with self.assertRaises(ValueError):
                    validate_migration_identity(self.old_identity, target_identity)

    def test_source_contract_and_changed_passage_refused_without_target(self):
        source = self.write_source([self.row()])
        wrong = copy.deepcopy(self.old_identity)
        wrong['provenance']['producer']['endpoint'] = 'different'
        with self.assertRaises(ValueError):
            migrate_source_verified_checkpoint(source, self.root / 'target.sqlite',
                source_identity=wrong, target_identity=self.new_identity, chunks=self.chunks)
        changed = copy.deepcopy(self.chunks)
        changed['chunk-a']['content'] = 'Changed source.'
        with self.assertRaisesRegex(RuntimeError, 'passage or row identity'):
            self.migrate(source, chunks=changed)
        self.assertFalse((self.root / 'target.sqlite').exists())

    def test_existing_target_is_never_overwritten(self):
        source = self.write_source([self.row()])
        target = self.root / 'target.sqlite'
        target.write_bytes(b'other owner')
        with self.assertRaisesRegex(ValueError, 'new target'):
            self.migrate(source)
        self.assertEqual(target.read_bytes(), b'other owner')

    def test_actual_stored_contract_must_match_explicit_source(self):
        source = self.write_source([self.row()])
        approved_source = copy.deepcopy(self.old_identity)
        approved_target = copy.deepcopy(self.new_identity)
        approved_source['provenance']['producer']['endpoint'] = 'different'
        approved_target['provenance']['producer']['endpoint'] = 'different'
        with self.assertRaisesRegex(RuntimeError, 'explicitly approved source'):
            migrate_source_verified_checkpoint(source, self.root / 'target.sqlite',
                source_identity=approved_source, target_identity=approved_target, chunks=self.chunks)
        self.assertFalse((self.root / 'target.sqlite').exists())

    def test_target_cannot_relabel_structural_results_as_source_verified(self):
        with self.assertRaisesRegex(ValueError, 'explicit structural'):
            validate_migration_identity(self.old_identity, self.old_identity)


if __name__ == '__main__':
    unittest.main()
