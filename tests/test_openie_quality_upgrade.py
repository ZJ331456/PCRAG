"""Exercise real native manifest guards without constructors, models or APIs."""

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathcondrag.utils.shared_index_builder import quality_hipporag_class, quality_profile
from pathcondrag.utils.openie_build_queue import openie_row_is_verified_complete
from pathcondrag.utils.openie_checkpoint import OpenIECheckpoint


class MemoryStore:
    def __init__(self, ids=()):
        self.ids = list(ids)

    def get_all_ids(self):
        return list(self.ids)


class QualityUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path('/root/baseline/HippoRAG')
        if not (root / 'src/hipporag/HippoRAG.py').is_file():
            raise unittest.SkipTest('The read-only baseline checkout is unavailable')
        previous_path, previous_bytecode = list(sys.path), sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(root / 'src'))
            cls.native = importlib.import_module('hipporag.HippoRAG').HippoRAG
            cls.config_class = importlib.import_module('hipporag.utils.config_utils').BaseConfig
            cls.consistency_error = importlib.import_module('hipporag.utils.state_utils').StateConsistencyError
        finally:
            sys.path[:] = previous_path
            sys.dont_write_bytecode = previous_bytecode

    def runtime(self, directory):
        runtime_class = quality_hipporag_class(self.native)
        runtime = runtime_class.__new__(runtime_class)
        runtime.working_dir = str(directory)
        runtime.index_manifest_path = str(Path(directory) / 'index_manifest.json')
        runtime.openie_state_path = str(Path(directory) / 'openie_state.json')
        runtime.openie_results_path = str(Path(directory) / 'top_openie.json')
        runtime._graph_pickle_filename = str(Path(directory) / 'graph.pickle')
        runtime.chunk_metadata_path = str(Path(directory) / 'chunk_metadata.json')
        runtime.global_config = self.config_class(
            save_dir=str(directory), llm_name='qwen3-8b',
            llm_base_url='http://127.0.0.1:8035/v1', openie_mode='online',
            embedding_model_name='/root/models/Qwen3-Embedding-8B',
            embedding_provider='transformers', embedding_batch_size=4,
            force_index_from_scratch=True, force_openie_from_scratch=True,
            openie_ner_max_tokens=512, openie_triple_max_tokens=2048)
        # Components are identities only: no weights or API clients.
        runtime.embedding_model = SimpleNamespace()
        runtime.extraction_llm = SimpleNamespace()
        runtime.text_preprocessor = SimpleNamespace()
        runtime.index_identity = None
        runtime.graph = SimpleNamespace(vcount=lambda: 0)
        runtime.chunk_embedding_store = MemoryStore()
        runtime.entity_embedding_store = MemoryStore()
        runtime.fact_embedding_store = MemoryStore()
        runtime._legacy_quality_manifest_probe = True
        self.native._validate_or_create_index_manifest(runtime)
        runtime._legacy_quality_manifest_probe = False
        original = json.loads(Path(runtime.index_manifest_path).read_text())
        self.assertEqual(original['openie']['quality_profile'], quality_profile(legacy=True))
        Path(runtime.openie_state_path).write_text(json.dumps({
            'provenance': copy.deepcopy(original['openie']),
            'docs': [{'idx': 'chunk-source', 'passage': 'Alpha is Beta.'}]}))
        # Real resumes already have passage vectors, preventing empty rewrite.
        runtime.chunk_embedding_store.ids = ['chunk-source']
        return runtime, original

    def observed_native_calls(self, calls):
        native_method = self.native._validate_or_create_index_manifest

        def invoke(runtime):
            calls.append({'force_openie': runtime.global_config.force_openie_from_scratch,
                          'legacy_probe': getattr(runtime, '_legacy_quality_manifest_probe', False)})
            return native_method(runtime)

        return patch.object(self.native, '_validate_or_create_index_manifest',
                            autospec=True, side_effect=invoke)

    def test_safe_upgrade_uses_real_native_validator_and_only_changes_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, original = self.runtime(directory)
            calls = []
            with patch.dict(os.environ, HIPPO_ALLOW_INDEX_RESUME='1'), self.observed_native_calls(calls):
                runtime._validate_or_create_index_manifest()
            self.assertEqual(calls, [{'force_openie': False, 'legacy_probe': True},
                                     {'force_openie': True, 'legacy_probe': False}])
            self.assertEqual(json.loads((Path(directory) / 'index_manifest.before_quality_upgrade.json').read_text()), original)
            upgraded = json.loads(Path(runtime.index_manifest_path).read_text())
            wanted = copy.deepcopy(original)
            wanted['openie']['quality_profile'] = quality_profile()
            self.assertEqual(upgraded, wanted)
            # Historical extraction is preserved until re-audited.
            oldstate = json.loads(Path(runtime.openie_state_path).read_text())
            self.assertEqual(oldstate['provenance'], original['openie'])
            self.assertTrue(runtime.global_config.force_openie_from_scratch)
            self.assertFalse(runtime._legacy_quality_manifest_probe)

    def test_upgrade_rejects_all_existing_derived_state_without_rewriting(self):
        for state in ('nodes', 'file', 'entity', 'fact'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                runtime, original = self.runtime(directory)
                if state == 'nodes':
                    runtime.graph.vcount = lambda: 1
                elif state == 'file':
                    Path(runtime._graph_pickle_filename).write_text('existing graph')
                elif state == 'entity':
                    runtime.entity_embedding_store.ids = ['entity-source']
                else:
                    runtime.fact_embedding_store.ids = ['fact-source']
                calls = []
                with patch.dict(os.environ, HIPPO_ALLOW_INDEX_RESUME='1'), self.observed_native_calls(calls):
                    with self.assertRaisesRegex(RuntimeError, 'unpublished'):
                        runtime._validate_or_create_index_manifest()
                self.assertEqual(calls, [])
                self.assertEqual(json.loads(Path(runtime.index_manifest_path).read_text()), original)
                self.assertFalse((Path(directory) / 'index_manifest.before_quality_upgrade.json').exists())

    def test_native_validator_refuses_changed_embedding_or_producer(self):
        for field in ('embedding', 'producer'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                runtime, original = self.runtime(directory)
                if field == 'embedding':
                    runtime.global_config.embedding_model_name = '/root/models/other-embedding'
                else:
                    runtime.global_config.llm_base_url = 'http://127.0.0.1:9999/v1'
                calls = []
                with patch.dict(os.environ, HIPPO_ALLOW_INDEX_RESUME='1'), self.observed_native_calls(calls):
                    with self.assertRaisesRegex(self.consistency_error, 'schema/config mismatch'):
                        runtime._validate_or_create_index_manifest()
                self.assertEqual(calls, [{'force_openie': False, 'legacy_probe': True}])
                self.assertEqual(json.loads(Path(runtime.index_manifest_path).read_text()), original)
                self.assertTrue(runtime.global_config.force_openie_from_scratch)
                self.assertFalse(runtime._legacy_quality_manifest_probe)
                self.assertFalse((Path(directory) / 'index_manifest.before_quality_upgrade.json').exists())

    def test_oldstate_mismatch_refused_before_manifest_upgrade(self):
        for field in ('quality_profile', 'producer'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                runtime, original = self.runtime(directory)
                state_path = Path(runtime.openie_state_path)
                state = json.loads(state_path.read_text())
                if field == 'quality_profile':
                    state['provenance'][field]['name'] = 'different-source-contract'
                else:
                    state['provenance'][field]['endpoint'] = 'http://changed/v1'
                state_path.write_text(json.dumps(state))
                calls = []
                with patch.dict(os.environ, HIPPO_ALLOW_INDEX_RESUME='1'), self.observed_native_calls(calls):
                    with self.assertRaises(RuntimeError):
                        runtime._validate_or_create_index_manifest()
                self.assertEqual(calls, [])
                self.assertEqual(json.loads(Path(runtime.index_manifest_path).read_text()), original)
                self.assertFalse((Path(directory) / 'index_manifest.before_quality_upgrade.json').exists())

    def test_without_explicit_resume_native_guard_refuses_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, original = self.runtime(directory)
            calls = []
            with patch.dict(os.environ, HIPPO_ALLOW_INDEX_RESUME=''), self.observed_native_calls(calls):
                with self.assertRaisesRegex(self.consistency_error, 'schema/config mismatch'):
                    runtime._validate_or_create_index_manifest()
            self.assertEqual(calls, [{'force_openie': True, 'legacy_probe': False}])
            self.assertEqual(json.loads(Path(runtime.index_manifest_path).read_text()), original)

    def test_upgrade_requires_both_fresh_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, original = self.runtime(directory)
            runtime.global_config.force_openie_from_scratch = False
            with patch.dict(os.environ, HIPPO_ALLOW_INDEX_RESUME='1'):
                with self.assertRaises(RuntimeError):
                    runtime._validate_or_create_index_manifest()
            self.assertEqual(json.loads(Path(runtime.index_manifest_path).read_text()), original)

    def test_complete_triples_do_not_approve_failed_or_unfinished_ner(self):
        extractor = SimpleNamespace(is_verified_complete=lambda result: True)
        healthy = {'idx': 'chunk', 'passage': 'Alpha is Beta.',
                   'extracted_entities': ['Alpha', 'Beta'],
                   'extracted_triples': [['Alpha', 'is', 'Beta']],
                   'openie_metadata': {'ner': {'finish_reason': 'stop', 'complete': True},
                                       'triples': {'finish_reason': 'stop', 'complete': True}}}
        self.assertTrue(openie_row_is_verified_complete(extractor, healthy))
        for metadata in ({'finish_reason': 'stop', 'error': 'NER timeout'},
                         {'finish_reason': 'stop', 'complete': False},
                         {'finish_reason': 'stop', 'quality_status': 'pending'},
                         {'complete': True}, {'finish_reason': 'length'}):
            with self.subTest(metadata=metadata):
                candidate = copy.deepcopy(healthy)
                candidate['openie_metadata']['ner'] = metadata
                self.assertFalse(openie_row_is_verified_complete(extractor, candidate))

    def test_sqlite_only_resume_keeps_failed_ner_pending_and_reuses_completed_row(self):
        for failed in (True, False):
            with self.subTest(failed_ner=failed), tempfile.TemporaryDirectory() as directory:
                runtime, _ = self.runtime(directory)
                Path(runtime.openie_state_path).unlink()
                runtime._openie_info = None
                runtime._openie_provenance = None
                runtime.openie = SimpleNamespace(is_verified_complete=lambda result: True)
                chunks = {'chunk-source': {'content': 'Alpha is Beta.'}}
                runtime.chunk_embedding_store.get_all_id_to_rows = lambda: chunks
                progress = OpenIECheckpoint(Path(directory) / 'openie_progress.sqlite', {
                    'provenance': runtime._current_openie_provenance(),
                    'corpus_ids_sha256': hashlib.sha256('chunk-source'.encode()).hexdigest(),
                })
                progress.save({'idx': 'chunk-source', 'passage': 'Alpha is Beta.',
                               'extracted_entities': ['Alpha', 'Beta'],
                               'extracted_triples': [['Alpha', 'is', 'Beta']],
                               'openie_metadata': {
                                   'ner': {'finish_reason': 'stop', 'complete': not failed,
                                           **({'error': 'NER timeout'} if failed else {})},
                                   'triples': {'finish_reason': 'stop', 'complete': True}},
                               'openie_responses': {'ner': 'saved NER', 'triples': 'saved triples'}})
                progress.close()
                try:
                    with patch.dict(os.environ, HIPPO_ALLOW_INDEX_RESUME='1'):
                        rows, pending = runtime.load_existing_openie(['chunk-source'], force_reextract=True)
                    self.assertEqual(pending, ['chunk-source'] if failed else [])
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['openie_responses']['triples'], 'saved triples')
                    if not failed:
                        self.assertIsNone(runtime.openie.checkpoint)
                finally:
                    runtime._strict_openie_progress.close()


if __name__ == '__main__':
    unittest.main()
