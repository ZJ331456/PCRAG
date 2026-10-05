"""Generic index journals preserve resumable progress across corpus additions."""

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.BaseRAG import BaseRAG
from pathcondrag.index.openie.source_verified_openie import SourceVerifiedOpenIE, SOURCE_VERIFIED_VERSION
from pathcondrag.index.openie.structural_openie import StructuralOpenIE, STRUCTURAL_VERSION
from pathcondrag.utils.misc_utils import NerRawOutput, TripleRawOutput, compute_mdhash_id
from pathcondrag.index.openie.openie_source_evidence import VERIFIER_VERSION
from pathcondrag.index.shared_index_builder import quality_profile
from pathcondrag.index.openie.openie_atomic_recovery import ATOMIC_RECOVERY_IMPLEMENTATION


class InterruptedStage(RuntimeError):
    pass


class OpenIEPrepared(RuntimeError):
    """Stop before vectors or graphs: these tests exercise journal integration."""


class PassageStore:
    def __init__(self):
        self.rows = {}

    def insert_strings(self, passages):
        for passage in passages:
            key = compute_mdhash_id(passage, 'chunk-')
            self.rows[key] = {'hash_id': key, 'content': passage}

    def get_all_id_to_rows(self):
        return copy.deepcopy(self.rows)


class GenericCheckpointTests(unittest.TestCase):
    def runtime(self, directory):
        rag = BaseRAG.__new__(BaseRAG)
        rag.working_dir = str(directory)
        rag.global_config = SimpleNamespace(force_index_from_scratch=False,
            openie_mode='online', llm_name='qwen3-8b', llm_base_url='http://127.0.0.1:8035/v1',
            temperature=0.0, seed=None, save_openie=True, openie_validation_mode='source_verified')
        rag.graph = SimpleNamespace(vcount=lambda: 0)
        rag.chunk_embedding_store = PassageStore()
        rag.openie = SourceVerifiedOpenIE.__new__(SourceVerifiedOpenIE)
        rag.cached_rows = []
        rag.batch_calls = []
        rag.interrupt = False
        rag.load_existing_openie = lambda keys, **kwargs: (copy.deepcopy(rag.cached_rows), list(keys))

        def save(rows):
            rag.cached_rows = copy.deepcopy(rows)
            raise OpenIEPrepared('OpenIE complete; no graph/model operations in this test')

        rag.save_openie_results = save

        def batch(chunks):
            rag.batch_calls.append(list(chunks))
            ner_outputs, triple_outputs = {}, {}
            for key, source in chunks.items():
                fact = ['Alpha', 'is', 'Beta']
                metadata = {'source_verified_schema': SOURCE_VERIFIED_VERSION,
                    'semantic_verifier_contract': VERIFIER_VERSION, 'semantic_verified': True,
                    'complete': True, 'finish_reason': 'stop', 'quality_status': 'success',
                    'semantic_verification_history': [{'complete': True, 'n_unverified': 0,
                        'checks': [{'triple': fact, 'supported': True}]}]}
                ner = NerRawOutput(key, 'NER source response', ['Alpha', 'Beta'],
                                   {'finish_reason': 'stop', 'complete': True})
                triples = TripleRawOutput(key, 'triple source response', [fact], metadata)
                row = {'idx': key, 'passage': source['content'], 'extracted_entities': ner.unique_entities,
                       'extracted_triples': triples.triples,
                       'openie_metadata': {'ner': ner.metadata, 'triples': triples.metadata},
                       'openie_responses': {'ner': ner.response, 'triples': triples.response}}
                rag.openie.checkpoint(row)
                ner_outputs[key], triple_outputs[key] = ner, triples
            if rag.interrupt:
                raise InterruptedStage('Simulated interruption after a committed stage')
            return ner_outputs, triple_outputs

        rag.openie.batch_openie = batch
        return rag

    def journal(self, directory, keys):
        digest = hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()
        return Path(directory) / f'openie_progress_{digest}.sqlite'

    def test_structural_mode_journals_and_reuses_completed_stage_without_extra_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = self.runtime(directory)
            rag.global_config.openie_validation_mode = 'structural'
            rag.openie = StructuralOpenIE.__new__(StructuralOpenIE)

            def batch(chunks):
                rag.batch_calls.append(list(chunks))
                ner, triples = {}, {}
                for key, source in chunks.items():
                    metadata = {'structural_schema': STRUCTURAL_VERSION,
                                'validation_scope': 'structural', 'semantic_verified': False,
                                'structural_infer_calls': 1, 'quality_status': 'success',
                                'complete': True, 'finish_reason': 'stop'}
                    ner[key] = NerRawOutput(key, '{"named_entities":["Alpha","Beta"]}', ['Alpha', 'Beta'],
                                            {'complete': True, 'finish_reason': 'stop'})
                    triples[key] = TripleRawOutput(key, '{"triples":[["Alpha","is","Beta"]]}',
                                                  [['Alpha', 'is', 'Beta']], metadata)
                    rag.openie.checkpoint({'idx': key, 'passage': source['content'],
                                          'extracted_entities': ner[key].unique_entities,
                                          'extracted_triples': triples[key].triples,
                                          'openie_metadata': {'ner': ner[key].metadata, 'triples': metadata}})
                return ner, triples

            rag.openie.batch_openie = batch
            with self.assertRaises(OpenIEPrepared):
                rag.index(['Alpha is Beta.'])
            path = self.journal(directory, rag.chunk_embedding_store.rows)
            with sqlite3.connect(path) as journal:
                identity = json.loads(journal.execute('SELECT value FROM identity WHERE key=?', ('contract',)).fetchone()[0])
            self.assertEqual(identity['quality_profile']['validation_scope'], 'structural')
            self.assertEqual(identity['verifier'], STRUCTURAL_VERSION)
            rag.batch_calls.clear()
            with self.assertRaises(OpenIEPrepared):
                rag.index(['Alpha is Beta.'])
            self.assertEqual(rag.batch_calls, [])
            self.assertFalse(rag.cached_rows[0]['openie_metadata']['triples']['semantic_verified'])

    def test_same_corpus_completed_stage_replayed_after_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = self.runtime(directory)
            rag.interrupt = True
            with self.assertRaises(InterruptedStage):
                rag.index(['Alpha is Beta.'])
            path = self.journal(directory, rag.chunk_embedding_store.rows)
            self.assertTrue(path.is_file())
            self.assertIsNone(rag.openie.checkpoint)
            rag.interrupt = False
            rag.batch_calls.clear()
            with self.assertRaises(OpenIEPrepared):
                rag.index(['Alpha is Beta.'])
            self.assertEqual(rag.batch_calls, [])
            self.assertEqual(rag.cached_rows[0]['openie_responses']['triples'], 'triple source response')
            self.assertEqual(list(Path(directory).glob('openie_progress_*.sqlite')), [path])

    def test_adding_passages_uses_new_journal_and_keeps_old_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = self.runtime(directory)
            with self.assertRaises(OpenIEPrepared):
                rag.index(['Alpha is Beta.'])
            previous = self.journal(directory, rag.chunk_embedding_store.rows)
            previous_bytes = previous.read_bytes()
            rag.batch_calls.clear()
            with self.assertRaises(OpenIEPrepared):
                rag.index(['Gamma is Delta.'])
            added_key = compute_mdhash_id('Gamma is Delta.', 'chunk-')
            self.assertEqual(rag.batch_calls, [[added_key]])
            current = self.journal(directory, rag.chunk_embedding_store.rows)
            self.assertNotEqual(current, previous)
            self.assertTrue(current.is_file())
            self.assertEqual(previous.read_bytes(), previous_bytes)
            self.assertEqual(len(rag.cached_rows), 2)

    def test_changed_quality_contract_cannot_reuse_same_corpus_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = self.runtime(directory)
            with self.assertRaises(OpenIEPrepared):
                rag.index(['Alpha is Beta.'])
            path = self.journal(directory, rag.chunk_embedding_store.rows)
            with sqlite3.connect(path) as journal:
                identity = json.loads(journal.execute(
                    'SELECT value FROM identity WHERE key=?', ('contract',)).fetchone()[0])
            self.assertEqual(identity['quality_profile'], quality_profile())
            changed = copy.deepcopy(quality_profile())
            changed['structured_output'] = 'different-grammar-contract'
            rag.batch_calls.clear()
            with patch('pathcondrag.index.shared_index_builder.quality_profile', return_value=changed):
                with self.assertRaisesRegex(RuntimeError, 'another corpus/producer/quality contract'):
                    rag.index(['Alpha is Beta.'])
            self.assertEqual(rag.batch_calls, [])
            self.assertTrue(path.is_file())

    def test_legacy_profile_is_unchanged_by_current_atomic_implementation(self):
        current, legacy = quality_profile(), quality_profile(legacy=True)
        self.assertEqual(current['atomic_implementation'], ATOMIC_RECOVERY_IMPLEMENTATION)
        self.assertNotIn('atomic_implementation', legacy)
        self.assertNotIn('structured_output', legacy)


if __name__ == '__main__':
    unittest.main()
