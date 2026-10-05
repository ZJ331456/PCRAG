"""Publication controls preserve diagnostics, coverage and graph consistency."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from pathcondrag.index.publication_policy import apply_publication_policy, parse_bool
from pathcondrag.index.shared_index_builder import quality_hipporag_class, quality_profile, run_shared_index_cli
from pathcondrag.index.build_report import tolerant_index_build_report
from pathcondrag.utils.config_utils import BaseConfig
from pathcondrag.utils.misc_utils import NerRawOutput, TripleRawOutput, compute_mdhash_id
from utils.new_index_compare import commands
from utils.fresh_index_validation import validate_fresh_index
import test_shared_index_builder as shared_fixture
import test_openie_index_quality as index_fixture
import test_fresh_index_validation as artifact_fixture


class OpenIEPublicationPolicyTests(unittest.TestCase):
    def test_default_config_is_strict_and_validates_explicit_types(self):
        self.assertTrue(BaseConfig().openie_strict)
        self.assertEqual(BaseConfig().openie_prompt_version, 'optimized')
        self.assertFalse(BaseConfig(openie_strict=False).openie_strict)
        with self.assertRaisesRegex(ValueError, 'must be a bool'):
            BaseConfig(openie_strict='false')
        with self.assertRaisesRegex(ValueError, 'origin or optimized'):
            BaseConfig(openie_prompt_version='unknown')
        self.assertFalse(parse_bool('false'))
        self.assertTrue(parse_bool('true'))

    def test_profiles_record_both_prompts_and_keep_failure_policy_separate(self):
        original = quality_profile(prompt_version='origin')
        optimized = quality_profile()
        self.assertNotEqual(original['prompt_sha256'], optimized['prompt_sha256'])
        self.assertNotEqual(original['ner_prompt_sha256'], optimized['ner_prompt_sha256'])
        self.assertTrue(optimized['structured_initial_triples'])
        tolerant = quality_profile(strict=False)
        self.assertFalse(tolerant['openie_strict'])
        self.assertNotEqual(tolerant['failure_policy'], optimized['failure_policy'])
        legacy = quality_profile(legacy=True)
        self.assertNotIn('ner_prompt_sha256', legacy)
        self.assertEqual(legacy['prompt_sha256'], original['prompt_sha256'])

    def test_strict_report_does_not_mutate_original_failed_candidates(self):
        rows = [{'idx': 'chunk', 'extracted_triples': [['A', 'r', 'B']],
                 'extracted_entities': ['A'], 'openie_metadata': {'triples': {'quality_status': 'failed'}}}]
        original = copy.deepcopy(rows)
        report = apply_publication_policy(rows, ['chunk'], strict=True)
        self.assertEqual(rows, original)
        self.assertEqual(report['action'], 'abort')

    def test_tolerant_report_excludes_candidates_without_approving_empty_output(self):
        rows = [{'idx': 'chunk', 'extracted_triples': [['A', 'r', 'B']],
                 'extracted_entities': ['A', 123],
                 'openie_metadata': {'triples': {'quality_status': 'partial', 'semantic_verified': False}}}]
        report = apply_publication_policy(rows, ['chunk'], strict=False)
        self.assertEqual(rows[0]['extracted_triples'], [])
        self.assertEqual(rows[0]['extracted_entities'], ['A'])
        metadata = rows[0]['openie_metadata']
        self.assertEqual(metadata['triples']['quality_status'], 'partial')
        self.assertFalse(metadata['triples']['semantic_verified'])
        self.assertEqual(metadata['publication']['excluded_triples'], [['A', 'r', 'B']])
        self.assertEqual(report['failed_document_count'], 1)
        self.assertEqual(report['excluded_relation_count'], 1)

    def test_tolerant_shared_adapter_saves_actual_failures_and_continues(self):
        fixture = shared_fixture.SharedIndexBuilderTests()
        with tempfile.TemporaryDirectory() as directory:
            original = fixture.make_runtime(directory)
            runtime_class = quality_hipporag_class(shared_fixture.NativeStub, openie_strict=False)
            runtime = runtime_class.__new__(runtime_class)
            runtime.__dict__.update(original.__dict__)
            ner, triples = fixture.batch('partial')
            rows = runtime.merge_openie_results([], {'chunk': {'content': 'passage'}}, ner, triples)
            self.assertEqual(rows[0]['extracted_triples'], [])
            self.assertEqual(rows[0]['openie_metadata']['triples']['quality_status'], 'partial')
            saved = json.loads(Path(runtime.openie_state_path).read_text())
            self.assertEqual(saved['provenance']['quality_profile'], quality_profile(strict=False))
            self.assertEqual(runtime._openie_publication_report['failed_document_count'], 1)
            strict_provenance = copy.deepcopy(saved['provenance'])
            strict_provenance['quality_profile'] = quality_profile()
            with self.assertRaisesRegex(RuntimeError, 'quality profile is incompatible'):
                runtime._validate_openie_provenance(strict_provenance, 'strict state')

    def test_generic_index_keeps_failed_passage_but_excludes_its_fact_vectors(self):
        fixture = index_fixture.OpenIEIndexQualityTests()
        with tempfile.TemporaryDirectory() as directory:
            rag = fixture.make_rag(directory)
            rows = fixture.setup_index_fixture(rag)
            rag.global_config.openie_strict = False
            key = rows[1]['idx']
            rag.openie.batch_openie.return_value = (
                {key: NerRawOutput(key, 'ner response', ['Gamma'], {'finish_reason': 'stop'})},
                {key: TripleRawOutput(key, 'bad response', [['Gamma', 'r', 'Delta']],
                                     {'quality_status': 'partial', 'openie_skipped': True,
                                      'semantic_verified': False})})
            rag.index(['healthy', 'failed'], rebuild_graph=True)
            self.assertTrue(Path(rag._graph_pickle_filename).is_file())
            self.assertIn(key, rag.graph.vs['name'])
            self.assertNotIn(compute_mdhash_id('delta', 'entity-'), rag.graph.vs['name'])
            saved = json.loads(Path(rag.openie_results_path).read_text())
            self.assertEqual(saved['publication_report']['failed_document_count'], 1)
            self.assertEqual(saved['docs'][1]['extracted_triples'], [])
            self.assertEqual(saved['docs'][1]['openie_responses']['triples'], 'bad response')

    def test_false_cli_is_consumed_before_native_parser_and_report_hook_is_local(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'main.py').write_text('# fake native entry')
            native_class, native_openie = shared_fixture.NativeStub, object()
            module = SimpleNamespace(HippoRAG=native_class, OpenIE=native_openie)
            namespace = {}
            exec('def main():\n    return index_build_report\n', namespace)

            def entry(path, run_name):
                self.assertEqual(run_name, '__pathcondrag_shared_main__')
                self.assertEqual(sys.argv[1:], ['--eval_mode', 'index_only'])
                return namespace

            with patch('pathcondrag.index.shared_index_builder.importlib.import_module', return_value=module), \
                    patch('pathcondrag.index.shared_index_builder.runpy.run_path', side_effect=entry):
                report_function = run_shared_index_cli(
                    ['--eval_mode', 'index_only', '--openie_strict', 'false',
                     '--openie_prompt_version', 'origin'], hippo_root=directory)
            self.assertIs(report_function, tolerant_index_build_report)
            self.assertIs(module.HippoRAG, native_class)
            self.assertIs(module.OpenIE, native_openie)

    def test_runner_only_forwards_path_options_to_path_owned_entries(self):
        args = SimpleNamespace(python='python', smoke=True, llm_base_url='http://localhost/v1',
                               hippo_root='/root/baseline/HippoRAG', openie_strict=False,
                               openie_prompt_version='origin')
        builder, cases = commands(args, Path('/tmp/policy'), Path('/tmp/dataset'))
        self.assertEqual(builder[builder.index('--openie_strict') + 1], 'false')
        self.assertNotIn('--openie_strict', cases['hipporag2'])
        self.assertIn('--openie_prompt_version', cases['pathcondrag_original'])

    def test_tolerant_completion_report_keeps_failure_count_and_structural_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_path, manifest_path = Path(directory) / 'graph.pickle', Path(directory) / 'index_manifest.json'
            graph_path.write_text('graph')
            manifest_path.write_text('manifest')
            rows = [{'idx': 'chunk', 'passage': 'passage', 'extracted_entities': [],
                     'extracted_triples': [], 'openie_metadata': {'triples': {'quality_status': 'failed'}}}]
            publication = apply_publication_policy(rows, ['chunk'], strict=False)
            rag = SimpleNamespace(
                chunk_embedding_store=SimpleNamespace(get_all_ids=lambda: ['chunk'], get_all_texts=lambda: ['passage']),
                entity_embedding_store=SimpleNamespace(get_all_ids=lambda: []), graph=SimpleNamespace(vcount=lambda: 1),
                _graph_pickle_filename=str(graph_path), index_manifest_path=str(manifest_path),
                _openie_info=rows, _openie_publication_report=publication,
                _quality_profile=lambda: quality_profile(strict=False),
                llm_model=SimpleNamespace(get_request_stats=lambda: {'http_attempts': 3, 'failures': 1}))
            config = BaseConfig(save_dir=directory, force_index_from_scratch=True, force_openie_from_scratch=True)
            report = tolerant_index_build_report(rag, ['passage'], config, 1)
            self.assertEqual(report['openie_failure_count'], 1)
            self.assertFalse(report['all_openie_verified'])
            self.assertEqual(report['llm_request_stats']['failures'], 1)
            self.assertFalse(report['runtime_config']['openie_strict'])
            rag.graph.vcount = lambda: 2
            with self.assertRaisesRegex(RuntimeError, 'different node counts'):
                tolerant_index_build_report(rag, ['passage'], config, 1)

    def test_tolerant_artifact_check_requires_explicit_exclusion_and_exact_graph(self):
        fixture = artifact_fixture.FreshIndexValidationTests()
        fixture.setUp()
        try:
            passage = 'Failed extraction passage'
            failed_row = {'idx': artifact_fixture.key(passage, 'chunk'), 'passage': passage,
                          'extracted_entities': [], 'extracted_triples': [],
                          'openie_metadata': {'ner': {'finish_reason': 'stop'},
                                              'triples': {'quality_status': 'failed', 'error': 'bad JSON'},
                                              'publication': {'openie_strict': False, 'skipped_from_graph': True}}}
            fixture.rows.append(failed_row)
            fixture.profile['openie_strict'] = False
            fixture.write_fixture()
            docs = {fixture.passage, passage}
            with self.assertRaisesRegex(ValueError, 'Incomplete fresh triples metadata'):
                validate_fresh_index(fixture.out, fixture.index, docs)
            report = validate_fresh_index(fixture.out, fixture.index, docs, strict=False)
            self.assertEqual(report['failed_chunks_excluded_from_graph'], 1)
            self.assertEqual(report['vectors']['chunk']['count'], 2)
            failed_row['extracted_triples'] = [['Wrong', 'r', 'Fact']]
            fixture.write_fixture()
            with self.assertRaisesRegex(ValueError, 'not excluded from graph inputs'):
                validate_fresh_index(fixture.out, fixture.index, docs, strict=False)
        finally:
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
