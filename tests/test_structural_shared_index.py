"""Structural mode keeps publication integrity without semantic audit calls."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from pathcondrag.index.openie.structural_openie import STRUCTURAL_VERSION, RECOVERY_BUDGET
from pathcondrag.index.shared_index_builder import (
    SharedStructuralOpenIE, quality_hipporag_class, quality_profile, run_shared_index_cli,
)
from pathcondrag.utils.misc_utils import NerRawOutput, TripleRawOutput
from utils.fresh_index_validation import validate_fresh_index
import test_shared_index_builder as shared_fixture
import test_fresh_index_validation as artifact_fixture
import test_multidataset_protocol as protocol_fixture


def structural_metadata(calls=1):
    return {'finish_reason': 'stop', 'quality_status': 'success', 'complete': True,
            'structural_schema': STRUCTURAL_VERSION, 'validation_scope': 'structural',
            'semantic_verified': False, 'requires_semantic_verification': False,
            'structural_infer_calls': calls, 'attempt_count': calls}


class StructuralSharedIndexTests(unittest.TestCase):
    def test_profile_separates_semantics_and_triple_request_budget(self):
        semantic = quality_profile()
        structural = quality_profile(validation_mode='structural', strict=False)
        self.assertEqual(semantic['semantic_scope'], 'all_final_relations')
        self.assertEqual(structural['validation_scope'], 'structural')
        self.assertFalse(structural['semantic_verified'])
        self.assertEqual(structural['recovery_budget'], RECOVERY_BUDGET)
        self.assertEqual(structural['recovery_budget_scope'], 'triple_stage_new_profile_logical_infer_calls')
        self.assertNotIn('fresh_recovery_semantic_verifier', structural)
        self.assertNotEqual(semantic['name'], structural['name'])
        with self.assertRaisesRegex(ValueError, 'no legacy'):
            quality_profile(validation_mode='structural', legacy=True)

    def test_shared_constructor_preserves_tokens_and_declared_workers(self):
        extractor = SharedStructuralOpenIE(SimpleNamespace(), max_workers=8)
        self.assertEqual(extractor.worker_limits(), (8, 8))
        self.assertEqual((extractor.ner_max_tokens, extractor.triple_max_tokens), (512, 2048))
        with self.assertRaisesRegex(ValueError, 'NER=512 and triples=2048'):
            SharedStructuralOpenIE(SimpleNamespace(), triple_max_tokens=1024)

    def test_factory_accepts_structural_success_without_semantic_verdict(self):
        fixture = shared_fixture.SharedIndexBuilderTests()
        with tempfile.TemporaryDirectory() as directory:
            previous = fixture.make_runtime(directory)
            runtime_class = quality_hipporag_class(shared_fixture.NativeStub, validation_mode='structural')
            runtime = runtime_class.__new__(runtime_class)
            runtime.__dict__.update(previous.__dict__)
            runtime.openie = SharedStructuralOpenIE(SimpleNamespace())
            ner = {'chunk': NerRawOutput('chunk', 'NER', ['Alpha', 'Beta'], {'finish_reason': 'stop'})}
            triples = {'chunk': TripleRawOutput('chunk', 'triples', [['Alpha', 'is', 'Beta']],
                                                structural_metadata())}
            rows = runtime.merge_openie_results([], {'chunk': {'content': 'Alpha is Beta.'}}, ner, triples)
            self.assertEqual(rows[0]['extracted_triples'], [['Alpha', 'is', 'Beta']])
            self.assertFalse(rows[0]['openie_metadata']['triples']['semantic_verified'])
            self.assertEqual(runtime._current_openie_provenance()['quality_profile']['validation_scope'],
                             'structural')

    def test_cli_selects_structural_extractor_and_does_not_forward_unknown_native_option(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'main.py').write_text('# native stub')
            module = SimpleNamespace(HippoRAG=shared_fixture.NativeStub, OpenIE=object())
            originals = module.HippoRAG, module.OpenIE
            entry_globals = {'index_build_report': lambda *args: self.fail('Native semantic report must not run')}
            exec("def main():\n    return index_build_report('rag', ['source'], 'config', 1.0)\n", entry_globals)

            def entry(path, run_name):
                self.assertEqual(run_name, '__pathcondrag_shared_main__')
                self.assertIs(module.OpenIE, SharedStructuralOpenIE)
                self.assertNotIn('--openie_validation_mode', sys.argv)
                return entry_globals

            with patch('pathcondrag.index.shared_index_builder.importlib.import_module', return_value=module), \
                    patch('pathcondrag.index.shared_index_builder.runpy.run_path', side_effect=entry), \
                    patch('pathcondrag.index.build_report.tolerant_index_build_report', return_value='completed') as report:
                result = run_shared_index_cli(['--eval_mode', 'index_only',
                                              '--openie_validation_mode', 'structural'], hippo_root=directory)
                self.assertEqual(run_shared_index_cli(['--eval_mode', 'index_only'], hippo_root=directory),
                                 'completed')
                self.assertEqual(report.call_count, 2)
                report.assert_called_with('rag', ['source'], 'config', 1.0)
                self.assertIs(entry_globals['main'].__globals__['index_build_report'], report)
            self.assertEqual(result, 'completed')
            self.assertEqual((module.HippoRAG, module.OpenIE), originals)

    def artifact(self):
        fixture = artifact_fixture.FreshIndexValidationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.profile = quality_profile(validation_mode='structural')
        fixture.provenance = {'quality_profile': fixture.profile}
        fixture.rows[0]['openie_metadata']['triples'] = structural_metadata()
        fixture.write_fixture()
        return fixture

    def test_fresh_structural_artifacts_keep_exact_graph_and_vector_checks(self):
        fixture = self.artifact()
        report = fixture.validate()
        self.assertEqual(report['validation_scope'], 'structural')
        self.assertFalse(report['semantic_verified'])
        self.assertEqual(report['structurally_checked_chunks'], 1)
        self.assertEqual(report['audited_chunks'], 0)
        self.assertTrue(report['edge_validation']['fact_provenance_exact'])
        graph_path = fixture.model / 'graph.pickle'
        graph = artifact_fixture.ig.Graph.Read_Pickle(str(graph_path))
        graph.add_vertex('unknown-corrupt-node')
        graph.write_pickle(str(graph_path))
        with self.assertRaisesRegex(ValueError, 'vertex identities'):
            fixture.validate()

    def test_semantic_claims_and_excess_structural_budget_are_rejected(self):
        fixture = self.artifact()
        for update in ({'semantic_verified': True}, {'structural_infer_calls': RECOVERY_BUDGET + 1},
                       {'semantic_verification_history': [{'complete': True}]}):
            with self.subTest(update=update):
                fixture.rows[0]['openie_metadata']['triples'] = {**structural_metadata(), **update}
                fixture.write_fixture()
                with self.assertRaises(ValueError):
                    fixture.validate()

    def test_structural_tolerant_failure_remains_explicit_and_unverified(self):
        fixture = self.artifact()
        fixture.profile['openie_strict'] = False
        passage = 'A failed source passage.'
        fixture.rows.append({'idx': artifact_fixture.key(passage, 'chunk'), 'passage': passage,
                             'extracted_entities': [], 'extracted_triples': [],
                             'openie_metadata': {'ner': {'finish_reason': 'stop'},
                                                 'triples': {'quality_status': 'failed', 'complete': False,
                                                             'semantic_verified': False, 'error': 'invalid JSON'},
                                                 'publication': {'openie_strict': False,
                                                                 'skipped_from_graph': True}}})
        fixture.write_fixture()
        report = validate_fresh_index(fixture.out, fixture.index, {fixture.passage, passage}, strict=False)
        self.assertEqual(report['failed_chunks_excluded_from_graph'], 1)
        self.assertEqual(report['vectors']['chunk']['count'], 2)
        self.assertFalse(report['semantic_verified'])
        fixture.rows[-1]['extracted_triples'] = [['Wrong', 'is', 'Fact']]
        fixture.write_fixture()
        with self.assertRaisesRegex(ValueError, 'not excluded'):
            validate_fresh_index(fixture.out, fixture.index, {fixture.passage, passage}, strict=False)

    def test_freeze_uses_manifest_structural_scope_without_semantic_audits(self):
        fixture = protocol_fixture.MultiDatasetProtocolTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        args, _, manifest = fixture.prepare('hotpotqa')
        freeze, rows, path = fixture.fake_build(args, manifest)
        for row in rows[1:]:
            row['extracted_triples'] = [['Document', 'contains', 'Text']]
            row['openie_metadata']['triples'] = structural_metadata()
        fixture.write(path, {'docs': rows})
        index_manifest_path = Path(args.source_index) / manifest['model_dir'] / 'index_manifest.json'
        index_manifest = protocol_fixture.protocol.read_json(index_manifest_path)
        index_manifest['openie']['quality_profile'] = quality_profile(validation_mode='structural', strict=False)
        fixture.write(index_manifest_path, index_manifest)
        protocol_fixture.protocol.freeze_index(freeze)
        report = protocol_fixture.protocol.read_json(Path(args.out_root) / 'index_build_report.json')
        self.assertEqual(report['validation_scope'], 'structural')
        self.assertFalse(report['semantic_verified'])
        self.assertEqual(report['openie_failure_count'], 1)
        self.assertEqual(protocol_fixture.protocol.index_ready(args), 0)


if __name__ == '__main__':
    unittest.main()
