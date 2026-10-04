"""CPU-only checks for the Path-owned fresh shared-index entry."""

import copy
import importlib
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
from pathcondrag.information_extraction import source_verified_openie as source_module
from pathcondrag.utils.misc_utils import NerRawOutput, TripleRawOutput
from pathcondrag.utils.shared_index_builder import (
    SharedQualityOpenIE, quality_hipporag_class, quality_profile, run_shared_index_cli,
)


class NativeStub:
    def _current_openie_provenance(self):
        return copy.deepcopy(self.native_provenance)

    def _validate_openie_provenance(self, value, source_path):
        self.assert_native(value)
        return value

    def assert_native(self, value):
        if value['identity'] != self.native_provenance['identity']:
            raise RuntimeError('native identity mismatch')
        if value['producer'] != self.native_provenance['producer']:
            raise RuntimeError('native producer mismatch')

    def _save_openie_state(self, rows):
        Path(self.openie_state_path).write_text(json.dumps({
            'docs': rows, 'provenance': self._openie_provenance,
        }))


class SharedIndexBuilderTests(unittest.TestCase):
    def make_runtime(self, directory):
        runtime_class = quality_hipporag_class(NativeStub)
        runtime = runtime_class.__new__(runtime_class)
        runtime.native_provenance = {
            'identity': {'prompt_schema': 'hipporag_openie_v1', 'triple_max_tokens': 2048},
            'producer': {'mode': 'online', 'class': 'hipporag.llm.openai_gpt.CacheOpenAI',
                         'endpoint': 'http://127.0.0.1:8035/v1', 'region': None},
        }
        runtime.openie_state_path = str(Path(directory) / 'openie_state.json')
        runtime.openie = SharedQualityOpenIE(SimpleNamespace())
        return runtime

    def batch(self, status):
        ner = {'chunk': NerRawOutput('chunk', 'NER raw', ['Alpha', 'Beta'], {'finish_reason': 'stop'})}
        metadata = {'finish_reason': 'stop', 'quality_status': status}
        if status in ('failed', 'partial'):
            metadata['openie_skipped'] = True
        values = [] if status in ('failed', 'empty_valid') else [['Alpha', 'is', 'Beta']]
        if status in ('success', 'empty_valid'):
            metadata.update(source_verified_schema=source_module.SOURCE_VERIFIED_VERSION,
                            semantic_verifier_contract=source_module.VERIFIER_VERSION,
                            semantic_verified=True, complete=True,
                            source_no_supported_relations=status == 'empty_valid',
                            semantic_verification_history=[{
                                'complete': True, 'n_unverified': 0,
                                'source_no_supported_relations': status == 'empty_valid',
                                'checks': [{'triple': value, 'supported': True} for value in values]}])
        triples = {'chunk': TripleRawOutput('chunk', 'triples raw', values, metadata)}
        return ner, triples

    def test_native_identity_and_producer_remain_unchanged_with_explicit_quality_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp)
            provenance = runtime._current_openie_provenance()
            self.assertEqual(provenance['identity'], runtime.native_provenance['identity'])
            self.assertEqual(provenance['producer'], runtime.native_provenance['producer'])
            self.assertEqual(provenance['quality_profile']['name'], 'pathcondrag_openie_quality_v3')
            self.assertEqual(provenance['quality_profile']['semantic_scope'], 'all_final_relations')
            self.assertIn('prompt_sha256', provenance['quality_profile'])
            self.assertEqual(runtime._validate_openie_provenance(provenance, 'state'), provenance)
            provenance['quality_profile']['prompt_sha256'] = 'other prompt'
            with self.assertRaisesRegex(RuntimeError, 'quality profile is incompatible'):
                runtime._validate_openie_provenance(provenance, 'state')

    def test_failed_and_partial_batches_checkpoint_raw_responses_before_abort(self):
        for status in ('failed', 'partial'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                runtime = self.make_runtime(tmp)
                ner, triples = self.batch(status)
                with self.assertRaisesRegex(RuntimeError, 'No completed graph was published'):
                    runtime.merge_openie_results([], {'chunk': {'content': 'passage'}}, ner, triples)
                saved = json.loads(Path(runtime.openie_state_path).read_text())
                self.assertEqual(saved['docs'][0]['openie_metadata']['triples']['quality_status'], status)
                self.assertEqual(saved['docs'][0]['openie_responses']['triples'], 'triples raw')
                self.assertEqual(saved['provenance']['quality_profile'], quality_profile())
                self.assertFalse((Path(tmp) / 'graph.pickle').exists())

    def test_success_and_legitimate_empty_batches_are_accepted(self):
        for status in ('success', 'empty_valid'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                runtime = self.make_runtime(tmp)
                ner, triples = self.batch(status)
                rows = runtime.merge_openie_results([], {'chunk': {'content': 'passage'}}, ner, triples)
                self.assertEqual(len(rows), 1)
                self.assertTrue(Path(runtime.openie_state_path).is_file())

    def test_synthetic_window_completion_keeps_whole_chunk_finish_reason(self):
        extractor = SharedQualityOpenIE(SimpleNamespace())
        metadata = {'finish_reason': 'length', 'quality_status': 'success',
                    'window_recovery_complete': True, 'window_recovery': [{'metadata': {'finish_reason': 'stop'}}]}
        result = TripleRawOutput('chunk', 'window responses', [['Alpha', 'is', 'Beta']], metadata)
        audit = {'complete': True, 'n_unverified': 0,
                 'checks': [{'triple': result.triples[0], 'supported': True}]}
        with patch.object(OpenIE, 'triple_extraction', return_value=result), \
                patch.object(source_module, 'verify_repaired_triples', return_value=(result.triples, audit)):
            restored = extractor.triple_extraction('chunk', 'long passage', ['Alpha'])
        self.assertEqual(restored.metadata['finish_reason'], 'stop')
        self.assertEqual(restored.metadata['whole_chunk_finish_reason'], 'length')
        self.assertEqual(restored.metadata['finish_source'], 'all_windows_completed')
        self.assertEqual(json.loads(restored.response)['extraction_responses'], ['window responses'])
        self.assertTrue(extractor.is_verified_complete(restored))

    def test_constructor_uses_cfg_workers_and_rejects_changed_token_budgets(self):
        with patch.dict(os.environ, {'HIPPO_OPENIE_MAX_WORKERS': '1'}):
            extractor = SharedQualityOpenIE(SimpleNamespace(), max_workers=8)
            self.assertEqual(extractor.worker_limits(), (8, 8))
            self.assertEqual((extractor.ner_max_tokens, extractor.triple_max_tokens), (512, 2048))
        with self.assertRaisesRegex(ValueError, 'NER=512 and triples=2048'):
            SharedQualityOpenIE(SimpleNamespace(), triple_max_tokens=1024)

    def test_cfg_token_budgets_override_legacy_environment_in_actual_requests(self):
        calls = []
        source = 'Alpha is Beta.'
        verdict = {'supported': True, 'quote': source, 'subject_roles': [{
            'source_subject': 'Alpha', 'mention': 'Alpha', 'relation_quote': source,
            'relation_supported': True}], 'reason': 'The source explicitly identifies Alpha as Beta.'}
        for flag in ('relation_supported', 'subject_scope_supported', 'object_scope_supported',
                     'attribution_supported', 'polarity_and_qualifiers_supported'):
            verdict[flag] = True
        evidence = {'source_has_supported_relations': True, 'source_evidence_quote': source,
                    'source_reason': 'The source contains an explicit relation.', 'verdicts': [verdict]}
        responses = iter(['{"named_entities":["Alpha","Beta"]}',
                          '{"triples":[["Alpha","is","Beta"]]}', json.dumps(evidence)])

        def infer(**kwargs):
            calls.append(kwargs)
            return next(responses), {'finish_reason': 'stop'}, False

        llm = SimpleNamespace(infer=infer, llm_config=SimpleNamespace(generate_params={'seed': None}))
        with patch.dict(os.environ, {'HIPPO_OPENIE_NER_MAX_TOKENS': '2048',
                                     'HIPPO_OPENIE_TRIPLE_MAX_TOKENS': '4096'}):
            extractor = SharedQualityOpenIE(llm)
            ner = extractor.ner('chunk', 'Alpha is Beta.')
            triples = extractor.triple_extraction('chunk', 'Alpha is Beta.', ner.unique_entities)
        self.assertEqual([call['max_completion_tokens'] for call in calls], [512, 2048, 2048])
        self.assertEqual(triples.metadata['quality_status'], 'success')
        self.assertTrue(extractor.is_verified_complete(triples))

    def test_ner_retries_keep_512_and_add_validation_feedback(self):
        calls = []
        responses = iter([('{"named_entities":["Alpha"]}', {'finish_reason': 'length'}),
                          ('{"named_entities":["Alpha"]}', {'finish_reason': 'stop'})])

        def infer(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            response, metadata = next(responses)
            return response, metadata, False

        extractor = SharedQualityOpenIE(SimpleNamespace(infer=infer))
        ner = extractor.ner('chunk', 'Alpha is Beta.')
        self.assertEqual(ner.unique_entities, ['Alpha'])
        self.assertEqual([call['max_completion_tokens'] for call in calls], [512, 512])
        self.assertNotEqual(calls[0]['messages'], calls[1]['messages'])
        self.assertTrue(all(call['extra_body']['chat_template_kwargs']['enable_thinking'] is False
                            for call in calls))

    def test_runpy_forwards_cli_and_restores_baseline_symbols_even_on_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'main.py').write_text('# fake baseline entry')
            native_class, native_openie = NativeStub, object()
            module = SimpleNamespace(HippoRAG=native_class, OpenIE=native_openie)
            argv_before, path_before = list(sys.argv), list(sys.path)
            bytecode_before = sys.dont_write_bytecode

            def run_entry(path, run_name):
                self.assertEqual(run_name, '__main__')
                self.assertEqual(sys.argv[1:], ['--eval_mode', 'index_only', '--embedding_batch_size', '4'])
                self.assertIs(module.OpenIE, SharedQualityOpenIE)
                self.assertTrue(issubclass(module.HippoRAG, native_class))
                self.assertTrue(sys.dont_write_bytecode)
                raise SystemExit(0)

            with patch('pathcondrag.utils.shared_index_builder.importlib.import_module', return_value=module), \
                    patch('pathcondrag.utils.shared_index_builder.runpy.run_path', side_effect=run_entry):
                with self.assertRaises(SystemExit):
                    run_shared_index_cli(['--eval_mode', 'index_only', '--embedding_batch_size', '4'], hippo_root=tmp)
            self.assertIs(module.OpenIE, native_openie)
            self.assertIs(module.HippoRAG, native_class)
            self.assertEqual(sys.argv, argv_before)
            self.assertEqual(sys.path, path_before)
            self.assertEqual(sys.dont_write_bytecode, bytecode_before)

    def test_original_baseline_validator_accepts_extra_quality_profile(self):
        baseline_root = Path('/root/baseline/HippoRAG')
        if not (baseline_root / 'src/hipporag/HippoRAG.py').is_file():
            self.skipTest('baseline checkout is unavailable')
        old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
        try:
            sys.path.insert(0, str(baseline_root / 'src'))
            sys.dont_write_bytecode = True
            native_class = importlib.import_module('hipporag.HippoRAG').HippoRAG
            with tempfile.TemporaryDirectory() as tmp:
                runtime = self.make_runtime(tmp)
                stored = runtime._current_openie_provenance()
                baseline_reader = native_class.__new__(native_class)
                baseline_reader._openie_state_identity = lambda: stored['identity']
                baseline_reader._current_openie_provenance = lambda: runtime.native_provenance
                self.assertEqual(baseline_reader._validate_openie_provenance(stored, 'overlay-state'), stored)
        finally:
            sys.path[:] = old_path
            sys.dont_write_bytecode = old_bytecode


if __name__ == '__main__':
    unittest.main()
