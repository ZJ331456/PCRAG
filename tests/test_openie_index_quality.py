"""CPU checks for cached extraction recovery and graph/vector consistency."""

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import igraph as ig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.BaseRAG import BaseRAG
from pathcondrag.utils.misc_utils import (
    NerRawOutput, TripleRawOutput, compute_mdhash_id, normalize_graph_triples,
    reformat_openie_results, text_processing, unicode_text_processing,
)


class MemoryStore:
    def __init__(self, namespace, texts):
        self.namespace = namespace
        self.rows = {}
        self.vectors = {}
        self.encoded = []
        self.insert_strings(texts)
        self.encoded.clear()

    def insert_strings(self, texts):
        for text in texts:
            key = compute_mdhash_id(text, self.namespace + '-')
            if key not in self.rows:
                self.rows[key] = {'hash_id': key, 'content': text}
                self.vectors[key] = [len(text), 1]
                self.encoded.append(text)

    def get_all_ids(self):
        return list(self.rows)

    def get_all_id_to_rows(self):
        return deepcopy(self.rows)

    def delete(self, keys):
        for key in keys:
            del self.rows[key]
            del self.vectors[key]


class OpenIEIndexQualityTests(unittest.TestCase):
    def make_rag(self, directory):
        rag = BaseRAG.__new__(BaseRAG)
        rag.working_dir = str(directory)
        rag.global_config = SimpleNamespace(
            force_openie_from_scratch=False, openie_mode='online',
            save_openie=True, is_directed_graph=False,
        )
        rag.openie_results_path = str(Path(directory) / 'openie.json')
        rag._graph_pickle_filename = str(Path(directory) / 'graph.pickle')
        rag.graph = ig.Graph()
        return rag

    def row(self, passage, triples=None, status='success'):
        return {
            'idx': compute_mdhash_id(passage, 'chunk-'), 'passage': passage,
            'extracted_entities': ['Alpha', 'Beta'],
            'extracted_triples': [] if triples is None else triples,
            'openie_metadata': {'ner': {'finish_reason': 'stop'},
                                'triples': {'quality_status': status}},
        }

    def test_cached_metadata_responses_and_strict_triples_survive_reformat(self):
        row = self.row('passage', [['Alpha', 'r', 'Beta'], ['Alpha', 'r', ''],
                                   {'s': 'Alpha', 'r': 'rel', 'o': 'Beta'}, 'abc'])
        row['openie_responses'] = {'ner': 'NER raw', 'triples': 'triples raw'}
        original = deepcopy(row)
        ner, triples = reformat_openie_results([row])
        key = row['idx']
        self.assertEqual(triples[key].triples, [['Alpha', 'r', 'Beta']])
        self.assertEqual(triples[key].response, 'triples raw')
        self.assertEqual(ner[key].response, 'NER raw')
        self.assertEqual(triples[key].metadata['quality_status'], 'success')
        self.assertEqual(triples[key].metadata['cached_validation']['invalid_triple_count'], 3)
        self.assertEqual(row, original)

    def test_graph_normalization_rejects_empty_fields_after_legacy_normalization(self):
        triples = [['Alpha', 'is', 'Beta'], ['静岡県', 'is', 'Japan'],
                   ['Alpha', 'is', ''], ['Alpha', '...', 'Beta']]
        self.assertEqual(normalize_graph_triples(triples), [['alpha', 'is', 'beta']])
        self.assertEqual(normalize_graph_triples(triples, normalizer=unicode_text_processing),
                         [['alpha', 'is', 'beta'], ['静岡県', 'is', 'japan']])

    def test_index_manifest_selects_unicode_without_changing_legacy_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = self.make_rag(tmp)
            self.assertEqual(rag._resolve_index_text_normalizer()('Straße 静岡県'),
                             text_processing('Straße 静岡県'))
            manifest = Path(tmp) / 'index_manifest.json'
            manifest.write_text(json.dumps({'text_normalization': 'unicode_alnum_casefold_v1'}))
            rag._text_normalizer = rag._resolve_index_text_normalizer()
            self.assertEqual(rag._normalize_index_text('Straße 静岡県'), 'strasse 静岡県')
            manifest.write_text(json.dumps({'text_normalization': 'unknown_future_schema'}))
            with self.assertRaisesRegex(ValueError, 'Unsupported index text_normalization'):
                rag._resolve_index_text_normalizer()

    def test_retry_is_explicit_and_empty_valid_is_not_reprocessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = self.make_rag(tmp)
            rows = [self.row('failed', status='failed'), self.row('empty', status='empty_valid'),
                    self.row('partial', [['Alpha', 'r', 'Beta']], status='partial')]
            Path(rag.openie_results_path).write_text(json.dumps({'docs': rows}))
            keys = [row['idx'] for row in rows]
            _, missing = rag.load_existing_openie(keys)
            self.assertEqual(missing, [])
            _, missing = rag.load_existing_openie(keys, retry_failed=True)
            self.assertEqual(missing, [keys[0], keys[2]])

    def test_merge_replaces_failed_row_and_persists_raw_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = self.make_rag(tmp)
            rows = [self.row('failed', status='failed')]
            key = rows[0]['idx']
            ner = {key: NerRawOutput(key, 'NER response', ['Gamma'], {'n': 1})}
            triples = {key: TripleRawOutput(key, 'triple response', [['Gamma', 'r', 'Delta']],
                                            {'quality_status': 'success', 'recovery_history': ['feedback']})}
            rag.merge_openie_results(rows, {key: {'content': 'failed'}}, ner, triples)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['extracted_triples'], [['Gamma', 'r', 'Delta']])
            self.assertEqual(rows[0]['openie_responses']['triples'], 'triple response')
            self.assertEqual(rows[0]['openie_metadata']['ner'], {'n': 1})

    def test_merge_missing_stage_raises_without_partially_mutating_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = self.make_rag(tmp)
            rows = [self.row('existing')]
            original = deepcopy(rows)
            key = rows[0]['idx']
            with self.assertRaisesRegex(RuntimeError, 'Missing OpenIE stage result'):
                rag.merge_openie_results(rows, {key: {'content': 'existing'}},
                                         {key: NerRawOutput(key, '{}', [], {})}, {})
            self.assertEqual(rows, original)

    def test_atomic_save_preserves_provenance_and_breaks_hardlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = self.make_rag(tmp)
            source = Path(tmp) / 'source.json'
            original = {'docs': [self.row('old')], 'provenance': {'source': 'hipporag'}}
            source.write_text(json.dumps(original))
            os.link(source, rag.openie_results_path)
            rag.load_existing_openie([])
            rag.save_openie_results([self.row('new')])
            self.assertEqual(json.loads(source.read_text()), original)
            self.assertEqual(json.loads(Path(rag.openie_results_path).read_text())['provenance'],
                             original['provenance'])

    def setup_index_fixture(self, rag):
        rows = [self.row('healthy', [['Alpha', 'r', 'Beta']]), self.row('failed', status='failed')]
        Path(rag.openie_results_path).write_text(json.dumps({'docs': rows}))
        rag.chunk_embedding_store = MemoryStore('chunk', ['healthy', 'failed'])
        rag.entity_embedding_store = MemoryStore('entity', ['alpha', 'beta', '', 'obsolete'])
        rag.fact_embedding_store = MemoryStore('fact', [str(('alpha', 'r', 'beta')),
                                                      str(('alpha', 'r', ''))])
        rag.graph.add_vertices(list(rag.chunk_embedding_store.rows) + list(rag.entity_embedding_store.rows))
        failed_key = rows[1]['idx']
        rag.openie = SimpleNamespace(batch_openie=Mock(return_value=(
            {failed_key: NerRawOutput(failed_key, '{}', ['Gamma', 'Delta'], {})},
            {failed_key: TripleRawOutput(failed_key, '{}', [['Gamma', 'r', 'Delta']],
                                        {'quality_status': 'success'})},
        )))
        rag.add_synonymy_edges = Mock()
        rag.ready_to_retrieve = False
        return rows

    def test_existing_graph_repair_requires_explicit_rebuild_before_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = self.make_rag(tmp)
            self.setup_index_fixture(rag)
            with self.assertRaisesRegex(RuntimeError, 'rebuild_graph=True'):
                rag.index(['healthy', 'failed'])
            rag.openie.batch_openie.assert_not_called()

    def test_explicit_rebuild_removes_obsolete_vectors_and_adds_recovered_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = self.make_rag(tmp)
            rows = self.setup_index_fixture(rag)
            kept_key = compute_mdhash_id('alpha', 'entity-')
            kept_vector = list(rag.entity_embedding_store.vectors[kept_key])
            rag.index(['healthy', 'failed'], rebuild_graph=True)
            self.assertNotIn('', [row['content'] for row in rag.entity_embedding_store.rows.values()])
            self.assertNotIn(str(('alpha', 'r', '')), [row['content'] for row in rag.fact_embedding_store.rows.values()])
            self.assertEqual(rag.entity_embedding_store.vectors[kept_key], kept_vector)
            self.assertNotIn('alpha', rag.entity_embedding_store.encoded)
            gamma = compute_mdhash_id('gamma', 'entity-')
            self.assertNotEqual(rag.graph.get_eid(rows[1]['idx'], gamma, error=False), -1)
            self.assertEqual(rag.graph['openie_chunk_digests'].keys(), {row['idx'] for row in rows})
            saved = json.loads(Path(rag.openie_results_path).read_text())['docs']
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[1]['openie_metadata']['triples']['quality_status'], 'success')
            rag.index(['healthy', 'failed'])
            self.assertEqual(rag.openie.batch_openie.call_count, 1)

    def test_digest_detects_same_chunk_fact_changes_without_empty_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = self.make_rag(tmp)
            self.setup_index_fixture(rag)
            rag.index(['healthy', 'failed'], rebuild_graph=True)
            saved_path = Path(rag.openie_results_path)
            saved = json.loads(saved_path.read_text())
            saved['docs'][0]['extracted_triples'] = [['Alpha', 'r', 'Changed']]
            saved_path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(RuntimeError, 'Facts changed'):
                rag.index(['healthy', 'failed'])
            self.assertNotIn(compute_mdhash_id('changed', 'entity-'), rag.entity_embedding_store.rows)

    def test_new_graph_prunes_invalid_vectors_left_in_reused_stores(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = self.make_rag(tmp)
            self.setup_index_fixture(rag)
            # Removing only graph.pickle leaves copied stores behind. Building
            # again must not reintroduce their obsolete blank entity/fact.
            rag.graph = ig.Graph()
            rag.index(['healthy', 'failed'])
            self.assertNotIn(compute_mdhash_id('', 'entity-'), rag.graph.vs['name'])
            self.assertNotIn(str(('alpha', 'r', '')),
                             [row['content'] for row in rag.fact_embedding_store.rows.values()])

    def test_failed_extraction_checkpoints_diagnostics_without_publishing_graph(self):
        for status in ('failed', 'partial'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                rag = self.make_rag(tmp)
                self.setup_index_fixture(rag)
                rag.graph = ig.Graph()
                rag.global_config.save_openie = False
                key = compute_mdhash_id('failed', 'chunk-')
                rag.openie.batch_openie.return_value = (
                    {key: NerRawOutput(key, 'NER source response', ['Gamma'], {})},
                    {key: TripleRawOutput(key, 'bad original model response',
                                          [['Gamma', 'r', 'Delta']] if status == 'partial' else [],
                                          {'quality_status': status, 'openie_skipped': True,
                                           'openie_skip_reason': 'invalid remaining relation'})},
                )
                before_facts = deepcopy(rag.fact_embedding_store.rows)
                with self.assertRaisesRegex(RuntimeError, 'OpenIE incomplete'):
                    rag.index(['healthy', 'failed'], rebuild_graph=True)
                self.assertFalse(Path(rag._graph_pickle_filename).exists())
                self.assertEqual(rag.fact_embedding_store.rows, before_facts)
                saved = json.loads(Path(rag.openie_results_path).read_text())['docs'][1]
                self.assertEqual(saved['openie_metadata']['triples']['quality_status'], status)
                self.assertEqual(saved['openie_responses']['triples'], 'bad original model response')


if __name__ == '__main__':
    unittest.main()
