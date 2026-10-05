"""CPU checks for fresh-index fallback and source-entailment publication gates."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.index.openie.openie_openai import OpenIE
from pathcondrag.index.openie import source_verified_openie as module
from pathcondrag.utils.misc_utils import NerRawOutput, TripleRawOutput
from pathcondrag.index.shared_index_builder import SharedQualityOpenIE, quality_hipporag_class
from pathcondrag.index.openie.openie_semantic_validation import (
    verify_repaired_triples as boolean_verifier,
    SemanticVerificationError as BooleanVerificationError,
)


class FakeLLM:
    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply, {'finish_reason': 'stop'}, False


def result(triples, status='success', **metadata):
    return TripleRawOutput('chunk', json.dumps({'triples': triples}), triples,
                           {'finish_reason': 'stop', 'quality_status': status, **metadata})


SUPPORTED = ['Iris', 'trained in', 'classical theater']
UNSUPPORTED = ['Iris', 'had classical theater training', 'Moon Harbor']
SOURCE = 'Iris had classical theater training and appeared in Moon Harbor.'


def wrapper_verifier(llm, passage, triples, retry_context=''):
    """Test-only verdict bridge; evidence JSON is covered in its own test suite."""
    if triples:
        try:
            return boolean_verifier(llm, passage, triples)
        except BooleanVerificationError as error:
            raise module.SemanticVerificationError(str(error), error.audit_metadata) from error
    response, metadata, _ = llm.infer(
        messages=[{'role': 'system', 'content': 'Independently audit original-source emptiness.'},
                  {'role': 'user', 'content': json.dumps({'SOURCE': passage, 'TRIPLES': []})}],
        max_completion_tokens=2048, temperature=0.0,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}})
    payload = json.loads(response)
    return [], {'complete': True, 'n_unverified': 0, 'n_input': 0, 'n_accepted': 0,
                'n_rejected': 0, 'checks': [], 'raw_response': response,
                'source_no_supported_relations': payload['source_has_supported_relations'] is False}


class SourceVerifiedOpenIETests(unittest.TestCase):
    def call(self, initial, llm, recovered):
        extractor = module.SourceVerifiedOpenIE(llm)
        with patch.object(OpenIE, 'triple_extraction', return_value=initial), \
                patch.object(module, 'verify_repaired_triples', side_effect=wrapper_verifier), \
                patch('pathcondrag.index.openie.openie_atomic_recovery.atomic_recovery',
                      return_value=result([], 'failed', complete=False, openie_skipped=True)):
            with patch.object(module, 'compact_recovery', side_effect=recovered) as fallback:
                output = extractor.triple_extraction('chunk', SOURCE, ['Iris', 'Moon Harbor'])
        return output, fallback

    def test_initial_valid_nonempty_is_audited_before_publication(self):
        initial = result([SUPPORTED])
        llm = FakeLLM(['{"supported":[true]}'])
        output, fallback = self.call(initial, llm, [])
        self.assertIsNot(output, initial)
        self.assertEqual(output.triples, initial.triples)
        self.assertTrue(output.metadata['semantic_verified'])
        self.assertTrue(module.SourceVerifiedOpenIE.is_verified_complete(output))
        fallback.assert_not_called()
        self.assertEqual(len(llm.calls), 1)

    def test_fresh_empty_uses_compact_recovery_and_whole_source_verifier(self):
        recovered = result([SUPPORTED], complete=True, repair_status='success')
        llm = FakeLLM(['{"supported":[true]}'])
        output, fallback = self.call(result([], 'empty_valid'), llm, [recovered])
        self.assertEqual(output.triples, [SUPPORTED])
        self.assertTrue(output.metadata['semantic_verified'])
        self.assertEqual(len(output.metadata['fresh_extraction_history']), 2)
        self.assertEqual(fallback.call_count, 1)
        prompt = json.loads(llm.calls[0]['messages'][1]['content'])
        self.assertEqual(prompt['SOURCE'], SOURCE)
        self.assertEqual(llm.calls[0]['max_completion_tokens'], 2048)
        self.assertFalse(llm.calls[0]['extra_body']['chat_template_kwargs']['enable_thinking'])

    def test_failed_result_recovered_and_initial_diagnostics_retained(self):
        initial = result([], 'failed', openie_skipped=True, recovery_history=[{'raw': 'bad'}])
        llm = FakeLLM(['{"supported":[true]}'])
        output, _ = self.call(initial, llm, [result([SUPPORTED], complete=True)])
        self.assertTrue(output.metadata['complete'])
        self.assertNotIn('openie_skipped', output.metadata)
        self.assertEqual(output.metadata['fresh_extraction_history'][0]['metadata']
                         ['recovery_history'], [{'raw': 'bad'}])

    def test_recovered_initial_filters_unsupported_without_first_compact_pass(self):
        initial = result([UNSUPPORTED, SUPPORTED], quality_recovered=True)
        llm = FakeLLM(['{"supported":[false,true]}'])
        output, fallback = self.call(initial, llm, [])
        fallback.assert_not_called()
        self.assertEqual(output.triples, [SUPPORTED])
        self.assertEqual(output.metadata['semantic_verification_history'][0]['n_rejected'], 1)

    def test_all_rejected_candidates_get_bounded_feedback_then_failure(self):
        recovered = result([UNSUPPORTED], complete=True)
        llm = FakeLLM(['{"supported":[false]}', '{"supported":[false]}'])
        output, fallback = self.call(result([], 'empty_valid'), llm, [recovered, recovered])
        self.assertEqual(fallback.call_count, 2)
        self.assertEqual(output.triples, [])
        self.assertEqual(output.metadata['quality_status'], 'failed')
        self.assertTrue(output.metadata['openie_skipped'])
        self.assertFalse(output.metadata['complete'])
        self.assertIn('rejected', fallback.call_args.args[4])

    def test_legitimate_no_relations_remains_explicit_empty_not_invented(self):
        empty = result([], 'empty_valid', complete=True, repair_status='no_supported_relations')
        llm = FakeLLM(['{"source_has_supported_relations":false}'])
        output, _ = self.call(result([], 'empty_valid'), llm, [empty])
        self.assertEqual(output.triples, [])
        self.assertTrue(output.metadata['complete'])
        self.assertTrue(output.metadata['source_no_supported_relations'])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(json.loads(llm.calls[0]['messages'][1]['content'])['SOURCE'], SOURCE)

    def test_transport_verification_failure_cannot_publish_candidates(self):
        llm = FakeLLM([RuntimeError('backend down')])
        output, _ = self.call(result([], 'empty_valid'), llm,
                              [result([SUPPORTED], complete=True)])
        self.assertTrue(output.metadata['openie_skipped'])
        self.assertFalse(output.metadata['semantic_verified'])
        self.assertEqual(output.triples, [])
        self.assertIn('backend down', output.metadata['openie_skip_reason'])

    def test_dedicated_repair_and_inner_windows_are_not_double_verified(self):
        initial = result([SUPPORTED], quality_recovered=True)
        extractor = module.SourceVerifiedOpenIE(FakeLLM())
        with patch.object(OpenIE, 'triple_extraction', return_value=initial):
            with patch.object(module, 'compact_recovery') as compact:
                self.assertIs(extractor.triple_extraction(
                    'chunk', SOURCE, [], repair_context='Dedicated repair'), initial)
                self.assertIs(extractor.triple_extraction(
                    'chunk', SOURCE, [], _allow_window_recovery=False), initial)
                compact.assert_not_called()

    def test_compact_failure_checkpointed_before_shared_graph_publication(self):
        output, _ = self.call(result([], 'empty_valid'), FakeLLM(),
                              [result([], 'failed', complete=False, openie_skipped=True)] * 2)

        class Native:
            def _current_openie_provenance(self):
                return {'identity': {'tokens': 2048}, 'producer': {'kind': 'native'}}

            def _save_openie_state(self, rows):
                Path(self.openie_state_path).write_text(json.dumps(rows))

        runtime_class = quality_hipporag_class(Native)
        runtime = runtime_class.__new__(runtime_class)
        with tempfile.TemporaryDirectory() as temporary:
            runtime.openie_state_path = str(Path(temporary) / 'state.json')
            with self.assertRaisesRegex(RuntimeError, 'No completed graph was published'):
                runtime.merge_openie_results([], {'chunk': {'content': SOURCE}},
                    {'chunk': NerRawOutput('chunk', '{}', ['Iris'], {'finish_reason': 'stop'})},
                    {'chunk': output})
            saved = json.loads(Path(runtime.openie_state_path).read_text())
            self.assertFalse(saved[0]['openie_metadata']['triples']['complete'])
            self.assertEqual(len(saved[0]['openie_metadata']['triples']['fresh_extraction_history']), 4)

    def test_normal_group_scope_error_recovers_atomic_individual_then_audits_again(self):
        source = ('Iris and Lena are actors. Iris had classical theater training. '
                  'Lena appeared in Moon Harbor.')
        group = ['Iris and Lena', 'trained in', 'classical theater']
        individual = ['Lena', 'appeared in', 'Moon Harbor']
        initial = result([group, SUPPORTED])
        atomic_result = result([SUPPORTED, individual], complete=True)

        def audit(values, decisions):
            return {'complete': True, 'n_unverified': 0,
                    'checks': [{'triple': value, 'supported': supported,
                                **({'rejection_kind': 'subject_scope'} if not supported else {})}
                               for value, supported in zip(values, decisions)]}

        extractor = module.SourceVerifiedOpenIE(FakeLLM())
        with patch.object(OpenIE, 'triple_extraction', return_value=initial), \
                patch.object(module, 'compact_recovery') as compact, \
                patch('pathcondrag.index.openie.openie_atomic_recovery.atomic_recovery',
                      return_value=atomic_result) as atomic, \
                patch.object(module, 'verify_repaired_triples', side_effect=[
                    ([SUPPORTED], audit(initial.triples, [False, True])),
                    ([SUPPORTED, individual], audit(atomic_result.triples, [True, True]))]) as verifier:
            output = extractor.triple_extraction('chunk', source, ['Iris', 'Lena', 'Moon Harbor'])
        compact.assert_not_called()
        atomic.assert_called_once()
        self.assertEqual(verifier.call_count, 2)
        self.assertEqual(output.triples, [SUPPORTED, individual])
        self.assertEqual([entry['stage'] for entry in output.metadata['fresh_extraction_history']],
                         ['initial', 'atomic'])
        self.assertTrue(extractor.is_verified_complete(output))

    def test_shared_extractor_uses_source_verified_class_and_fixed_budgets(self):
        extractor = SharedQualityOpenIE(FakeLLM(), max_workers=8)
        self.assertIsInstance(extractor, module.SourceVerifiedOpenIE)
        self.assertEqual(extractor.worker_limits(), (8, 8))
        self.assertEqual((extractor.ner_max_tokens, extractor.triple_max_tokens), (512, 2048))

    def test_pending_audit_resumes_last_complete_candidates_instead_of_reextracting(self):
        previous = result([], 'failed', openie_skipped=True,
            openie_skip_reason='Evidence transport incomplete', fresh_extraction_history=[
                {'stage': 'compact', 'triples': [SUPPORTED], 'response': 'complete candidate',
                 'metadata': {'finish_reason': 'stop', 'complete': True, 'quality_status': 'success'}},
                {'stage': 'atomic', 'triples': [], 'response': 'unfinished',
                 'metadata': {'complete': False, 'openie_skipped': True, 'finish_reason': 'length'}}])
        original = copy.deepcopy(previous.metadata)
        extractor = module.SourceVerifiedOpenIE(FakeLLM())
        with patch.object(extractor, '_verify_and_recover') as resume:
            extractor.recover_pending_triples('chunk', SOURCE, ['Iris'], previous, 2)
        candidate = resume.call_args.args[3]
        self.assertEqual(candidate.triples, [SUPPORTED])
        self.assertEqual(candidate.response, 'complete candidate')
        self.assertEqual(candidate.metadata['resumed_candidate_stage'], 'compact')
        self.assertEqual(candidate.metadata['openie_skip_reason'], 'Evidence transport incomplete')
        self.assertEqual(previous.metadata, original)

    def test_cuda_knn_releases_encoder_uses_native_float32_and_restores_settings(self):
        import torch
        seen = []

        class Native:
            def add_synonymy_edges(self, keys):
                seen.append((keys, self.embedding_model.model,
                             self.global_config.synonymy_edge_query_batch_size,
                             self.global_config.synonymy_edge_key_batch_size,
                             os.environ.get('HIPPORAG_KNN_DEVICE'),
                             torch.backends.cuda.matmul.allow_tf32))
                return 'native-result'

        runtime_class = quality_hipporag_class(Native)
        runtime = runtime_class.__new__(runtime_class)
        runtime.embedding_model = SimpleNamespace(model=object())
        runtime.global_config = SimpleNamespace(synonymy_edge_query_batch_size=128,
                                                synonymy_edge_key_batch_size=1024,
                                                embedding_batch_size=4)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        with patch.dict(os.environ, {'PATHCONDRAG_SHARED_KNN_DEVICE': 'cuda',
                                     'HIPPORAG_KNN_DEVICE': 'cpu'}):
            with patch.object(torch.cuda, 'is_available', return_value=True):
                with patch.object(torch.cuda, 'empty_cache') as release:
                    self.assertEqual(runtime.add_synonymy_edges(['entity-a']), 'native-result')
                    release.assert_called_once()
            self.assertEqual(os.environ['HIPPORAG_KNN_DEVICE'], 'cpu')
        self.assertEqual(seen, [(['entity-a'], None, 1000, 16384, 'cuda', False)])
        self.assertEqual(runtime.global_config.embedding_batch_size, 4)
        self.assertEqual(runtime.global_config.synonymy_edge_query_batch_size, 128)
        self.assertEqual(runtime.global_config.synonymy_edge_key_batch_size, 1024)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, previous_tf32)

    def test_default_knn_keeps_native_settings_and_encoder(self):
        class Native:
            def add_synonymy_edges(self, keys):
                return keys

        runtime_class = quality_hipporag_class(Native)
        runtime = runtime_class.__new__(runtime_class)
        encoder = object()
        runtime.embedding_model = SimpleNamespace(model=encoder)
        with patch.dict(os.environ, {'PATHCONDRAG_SHARED_KNN_DEVICE': ''}):
            self.assertEqual(runtime.add_synonymy_edges(['entity-a']), ['entity-a'])
        self.assertIs(runtime.embedding_model.model, encoder)


class CheckpointResumeTests(unittest.TestCase):
    """An unpublished checkpoint may retain only completed compatible rows."""

    @staticmethod
    def row(key, **triple_metadata):
        return {'idx': key, 'passage': SOURCE, 'extracted_entities': ['Iris'],
                'extracted_triples': [SUPPORTED],
                'openie_metadata': {
                    'ner': {'finish_reason': 'stop'},
                    'triples': {'finish_reason': 'stop', 'quality_status': 'success',
                                **triple_metadata}},
                'openie_responses': {'ner': '{}', 'triples': '{}'}}

    def runtime(self, directory, rows):
        class Native:
            def _current_openie_provenance(self):
                return {'identity': {'model': 'test', 'max_tokens': 2048},
                        'producer': {'model': 'test', 'endpoint': 'local'}}

            def _validate_openie_provenance(self, stored, source_path):
                if stored.get('identity') != self._current_openie_provenance()['identity']:
                    raise RuntimeError('native identity mismatch')
                return stored

            def load_existing_openie(self, keys, force_reextract=False):
                self.native_load_calls.append(force_reextract)
                if force_reextract:
                    self._openie_provenance = self._current_openie_provenance()
                    return [], set(keys)
                payload = json.loads(Path(self.openie_state_path).read_text())
                self._openie_provenance = self._validate_openie_provenance(
                    payload['provenance'], self.openie_state_path)
                return payload['docs'], set(keys).difference(row['idx'] for row in payload['docs'])

            def _save_openie_state(self, rows):
                Path(self.openie_state_path).write_text(json.dumps({
                    'docs': rows, 'provenance': self._openie_provenance}))

        runtime_class = quality_hipporag_class(Native)
        runtime = runtime_class.__new__(runtime_class)
        runtime.working_dir = str(directory)
        runtime.openie_state_path = str(Path(directory) / 'openie_state.json')
        runtime.openie_results_path = str(Path(directory) / 'legacy_openie.json')
        runtime._graph_pickle_filename = str(Path(directory) / 'graph.pickle')
        runtime.global_config = SimpleNamespace(force_index_from_scratch=True,
                                                force_openie_from_scratch=True)
        runtime.graph = SimpleNamespace(vcount=lambda: 0)
        runtime.entity_embedding_store = SimpleNamespace(get_all_ids=lambda: [])
        runtime.fact_embedding_store = SimpleNamespace(get_all_ids=lambda: [])
        runtime.native_load_calls = []
        runtime._openie_provenance = None
        Path(runtime.openie_state_path).write_text(json.dumps({
            'docs': rows, 'provenance': runtime._current_openie_provenance()}))
        return runtime

    def test_resume_keeps_success_rows_and_repairs_failed_and_missing_in_corpus_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.runtime(temporary, [self.row('good'), self.row('failed',
                                                        quality_status='failed', openie_skipped=True)])
            with patch.dict(os.environ, {'HIPPO_ALLOW_INDEX_RESUME': '1'}):
                rows, pending = runtime.load_existing_openie(
                    ['good', 'missing', 'failed'], force_reextract=True)
            self.assertEqual(runtime.native_load_calls, [False])
            self.assertEqual(pending, ['missing', 'failed'])
            original_good = copy.deepcopy(rows[0])
            diagnostic = json.loads((Path(temporary) / 'openie_resume_diagnostics.json').read_text())
            self.assertEqual(diagnostic['retained_success_count'], 1)
            self.assertEqual(diagnostic['failed_chunk_ids'], ['failed'])
            self.assertEqual(diagnostic['missing_chunk_ids'], ['missing'])
            self.assertEqual(diagnostic['reextract_count'], 2)
            chunks = {key: {'content': SOURCE} for key in pending}
            ner = {key: NerRawOutput(key, '{}', ['Iris'], {'finish_reason': 'stop'})
                   for key in pending}
            triples = {key: TripleRawOutput(key, '{}', [SUPPORTED],
                                           {'finish_reason': 'stop', 'quality_status': 'success'})
                       for key in pending}
            merged = runtime.merge_openie_results(rows, chunks, ner, triples)
            self.assertIs(merged, rows)
            self.assertEqual(rows[0], original_good)
            self.assertEqual([row['idx'] for row in rows], ['good', 'failed', 'missing'])
            self.assertNotIn('openie_skipped', rows[1]['openie_metadata']['triples'])

    def test_without_explicit_permission_the_fresh_flags_still_reextract_everything(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.runtime(temporary, [self.row('good')])
            with patch.dict(os.environ, {'HIPPO_ALLOW_INDEX_RESUME': ''}):
                rows, pending = runtime.load_existing_openie(['good'], force_reextract=True)
            self.assertEqual(rows, [])
            self.assertEqual(pending, {'good'})
            self.assertEqual(runtime.native_load_calls, [True])
            self.assertFalse((Path(temporary) / 'openie_resume_diagnostics.json').exists())

    def test_authorized_resume_without_checkpoint_runs_normal_fresh_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.runtime(temporary, [])
            Path(runtime.openie_state_path).unlink()
            with patch.dict(os.environ, {'HIPPO_ALLOW_INDEX_RESUME': '1'}):
                rows, pending = runtime.load_existing_openie(['new'], force_reextract=True)
            self.assertEqual((rows, pending), ([], {'new'}))
            self.assertEqual(runtime.native_load_calls, [True])

    def test_resume_refuses_any_completed_or_derived_state_and_missing_fresh_flag(self):
        for existing in ('graph_nodes', 'graph_file', 'entity_vectors', 'fact_vectors', 'fresh_flag'):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                runtime = self.runtime(temporary, [self.row('good')])
                if existing == 'graph_nodes':
                    runtime.graph.vcount = lambda: 1
                elif existing == 'graph_file':
                    Path(runtime._graph_pickle_filename).write_text('completed')
                elif existing == 'entity_vectors':
                    runtime.entity_embedding_store.get_all_ids = lambda: ['entity']
                elif existing == 'fact_vectors':
                    runtime.fact_embedding_store.get_all_ids = lambda: ['fact']
                else:
                    runtime.global_config.force_index_from_scratch = False
                with patch.dict(os.environ, {'HIPPO_ALLOW_INDEX_RESUME': '1'}):
                    with self.assertRaises(RuntimeError):
                        runtime.load_existing_openie(['good'], force_reextract=True)
                self.assertEqual(runtime.native_load_calls, [])

    def test_resume_rejects_provenance_changes_out_of_corpus_and_duplicate_rows(self):
        for invalid in ('identity', 'quality_profile', 'producer', 'extra_row', 'duplicate_row'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                runtime = self.runtime(temporary, [self.row('good')])
                path = Path(runtime.openie_state_path)
                payload = json.loads(path.read_text())
                if invalid in ('identity', 'quality_profile', 'producer'):
                    payload['provenance'][invalid] = {'incompatible': True}
                elif invalid == 'extra_row':
                    payload['docs'].append(self.row('outside'))
                else:
                    payload['docs'].append(self.row('good'))
                path.write_text(json.dumps(payload))
                with patch.dict(os.environ, {'HIPPO_ALLOW_INDEX_RESUME': '1'}):
                    with self.assertRaises(RuntimeError):
                        runtime.load_existing_openie(['good'], force_reextract=True)
                self.assertFalse((Path(temporary) / 'openie_resume_diagnostics.json').exists())

    def test_resume_retries_unfinished_invalid_and_unexplained_empty_rows(self):
        rows = [self.row('complete'), self.row('length', finish_reason='length'),
                self.row('invalid'), self.row('empty'), self.row('legitimate_empty',
                    source_no_supported_relations=True, semantic_verified=True, complete=True)]
        rows[2]['extracted_triples'] = [['Iris', 'and', 'Moon Harbor']]
        rows[3]['extracted_triples'] = []
        rows[4]['extracted_triples'] = []
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.runtime(temporary, rows)
            with patch.dict(os.environ, {'HIPPO_ALLOW_INDEX_RESUME': '1'}):
                loaded, pending = runtime.load_existing_openie(
                    [row['idx'] for row in rows], force_reextract=True)
            self.assertEqual(pending, ['length', 'invalid', 'empty'])
            self.assertEqual(loaded, rows)
            self.assertEqual(runtime._openie_resume_diagnostics['retained_success_count'], 2)


if __name__ == '__main__':
    unittest.main()
