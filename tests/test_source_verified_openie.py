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

from pathcondrag.information_extraction.openie_openai import OpenIE
from pathcondrag.information_extraction import source_verified_openie as module
from pathcondrag.utils.misc_utils import NerRawOutput, TripleRawOutput
from pathcondrag.utils.shared_index_builder import SharedQualityOpenIE, quality_hipporag_class


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


class SourceVerifiedOpenIETests(unittest.TestCase):
    def call(self, initial, llm, recovered):
        extractor = module.SourceVerifiedOpenIE(llm)
        with patch.object(OpenIE, 'triple_extraction', return_value=initial):
            with patch.object(module, 'compact_recovery', side_effect=recovered) as fallback:
                output = extractor.triple_extraction('chunk', SOURCE, ['Iris', 'Moon Harbor'])
        return output, fallback

    def test_initial_valid_nonempty_does_not_add_audit_calls(self):
        initial = result([SUPPORTED])
        llm = FakeLLM()
        output, fallback = self.call(initial, llm, [])
        self.assertIs(output, initial)
        fallback.assert_not_called()
        self.assertEqual(llm.calls, [])

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
        llm = FakeLLM()
        output, _ = self.call(result([], 'empty_valid'), llm, [empty])
        self.assertEqual(output.triples, [])
        self.assertTrue(output.metadata['complete'])
        self.assertTrue(output.metadata['source_no_supported_relations'])
        self.assertEqual(llm.calls, [])

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
                              [result([], 'failed', complete=False, openie_skipped=True)])

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
            self.assertEqual(len(saved[0]['openie_metadata']['triples']['fresh_extraction_history']), 2)

    def test_shared_extractor_uses_source_verified_class_and_fixed_budgets(self):
        extractor = SharedQualityOpenIE(FakeLLM(), max_workers=8)
        self.assertIsInstance(extractor, module.SourceVerifiedOpenIE)
        self.assertEqual(extractor.worker_limits(), (8, 8))
        self.assertEqual((extractor.ner_max_tokens, extractor.triple_max_tokens), (512, 2048))

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


if __name__ == '__main__':
    unittest.main()
