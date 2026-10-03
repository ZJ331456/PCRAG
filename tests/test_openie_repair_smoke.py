"""CPU-only checks of smoke commands, export gates and sequential execution."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from utils import openie_repair_smoke as smoke  # noqa: E402


def fixture_dataset():
    corpus = [{'title': f'D{i}', 'text': f'Text{i}'} for i in range(200)]
    data = [{'id': f'2hop__{i}', 'question': f'Question {i}', 'answer': 'Answer',
             'question_decomposition': [{'question': 'First?'}, {'question': 'Second?'}],
             'paragraphs': [dict(corpus[0], is_supporting=True), dict(corpus[1], is_supporting=True)]}
            for i in range(815)]
    return data, corpus


def fixture_result(name, data, corpus):
    docs = [row['title'] + '\n' + row['text'] for row in corpus]
    metrics = {f'Recall@{k}': 0.5 if k == 1 else 1.0 for k in (1, 2, 5, 10, 20, 200)}
    rows = []
    for index in smoke.SELECTED_INDICES:
        rows.append({'query_index': index, 'sample_id': data[index]['id'],
                     'question': data[index]['question'], 'benchmark_hops': 2,
                     'docs': docs[:10], 'doc_scores': [0.9] * 10,
                     'candidate_docs': docs, 'candidate_doc_scores': [0.9] * 200,
                     'gold_docs': docs[:2], 'retrieval_metrics': metrics,
                     'gold_document_ranks': [{'doc': docs[i], 'rank': i + 1} for i in range(2)],
                     'all_gold_in_top5': True, 'all_gold_in_top10': True, 'retrieval_trace': {}})
    result = {'selected_indices': smoke.SELECTED_INDICES, 'sample_size_effective': 3,
              'result_top_k': 10, 'candidate_output_top_k': 200, 'eval_mode': 'retrieve',
              'qa_metrics': {}, 'retrieval_metrics': metrics, 'results': rows,
              'runtime_config': {'embedding_batch_size': 4, 'openie_max_workers': 8,
                                 'llm_prefetch_workers': 8, 'max_new_tokens': 2048,
                                 'embedding_model_name': smoke.EMBEDDING_MODEL, 'llm_name': 'qwen3-8b'},
              'llm_request_stats': {'failures': 0, 'max_in_flight': 8, 'http_attempts': 3},
              'n_docs': len(corpus), 'indexed_docs': len(corpus)}
    if name != 'hipporag2':
        result.update(hop_source='benchmark', hop_distribution={'2': 3})
        result['runtime_config']['improvement_stage'] = 4
    return result


class RepairedIndexSmokeTests(unittest.TestCase):
    def setUp(self):
        self.data, self.corpus = fixture_dataset()
        self.corpus_docs = {row['title'] + '\n' + row['text'] for row in self.corpus}
        self.hops = [2] * len(self.data)

    def test_native_commands_share_index_and_fixed_budgets(self):
        args = SimpleNamespace(python='/rag/python', hippo_root='/readonly/HippoRAG',
                               datasets_dir='/datasets', llm_base_url='http://local/v1', out_root='/out')
        commands = smoke.build_commands(args, Path('/out/indices.json'), Path('/index'), Path('/out/smoke'))
        self.assertEqual([item['name'] for item in commands], ['hipporag2', 'exp4_dependency_binding'])
        for item in commands:
            command = item['command']
            self.assertEqual(command[:3], ['/rag/python', '-B', '-u'])
            for flag, value in (('--save_dir', '/index'), ('--embedding_batch_size', '4'),
                                ('--openie_max_workers', '8'), ('--llm_prefetch_workers', '8'),
                                ('--result_top_k', '10'), ('--candidate_output_top_k', '200'),
                                ('--retrieval_top_k', '200'), ('--sample_size', '3'),
                                ('--sample_indices_file', '/out/indices.json'), ('--eval_mode', 'retrieve')):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn('--reuse_index', command)
            self.assertNotIn('--force_index_from_scratch', command)
            self.assertNotIn('--force_openie_from_scratch', command)
        path_command = commands[1]['command']
        self.assertEqual(path_command[path_command.index('--hop_source') + 1], 'benchmark')
        self.assertEqual(path_command[path_command.index('--improvement_stage') + 1], '4')
        self.assertEqual(path_command[path_command.index('--max_new_tokens') + 1], '2048')
        self.assertIn('--save_dir_exact', commands[0]['command'])

    def test_child_environment_overrides_excess_concurrency_and_disables_bytecode(self):
        with patch.dict(os.environ, {'PATHCONDRAG_LLM_MAX_IN_FLIGHT': '16',
                                     'HIPPORAG_LLM_MAX_IN_FLIGHT': '16'}):
            environment = smoke.child_environment()
        self.assertEqual(environment['PATHCONDRAG_LLM_MAX_IN_FLIGHT'], '8')
        self.assertEqual(environment['HIPPORAG_LLM_MAX_IN_FLIGHT'], '8')
        self.assertEqual(environment['PYTHONDONTWRITEBYTECODE'], '1')

    def test_valid_detailed_three_question_exports_pass_for_both_consumers(self):
        for name in ('hipporag2', 'exp4_dependency_binding'):
            report = smoke.validate_case(fixture_result(name, self.data, self.corpus), name,
                                         self.data, self.corpus_docs, self.hops)
            self.assertEqual(report['n_samples'], 3)
            self.assertEqual(report['result_top_k'], 10)
            self.assertEqual(report['retrieval_metrics']['Recall@5'], 1.0)

    def test_export_gate_rejects_wrong_hops_detail_count_or_llm_failures(self):
        for defect in ('hop', 'detail', 'failure', 'token', 'corpus'):
            result = fixture_result('exp4_dependency_binding', self.data, self.corpus)
            if defect == 'hop':
                result['hop_source'] = 'estimated'
            elif defect == 'detail':
                result['results'][0]['docs'] = result['results'][0]['docs'][:5]
            elif defect == 'failure':
                result['llm_request_stats']['failures'] = 1
            elif defect == 'token':
                result['runtime_config']['max_new_tokens'] = 256
            else:
                result['indexed_docs'] = 10
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                smoke.validate_case(result, 'exp4_dependency_binding', self.data,
                                    self.corpus_docs, self.hops)

    def prepare_directory(self, directory):
        project = Path(directory) / 'PathCondRAG'
        out = project / 'outputs' / 'repair'
        index = out / 'repaired_index'
        for asset in smoke.ASSETS:
            target = index / smoke.MODEL_DIR / asset
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(asset)
        smoke.write_json(index / smoke.MODEL_DIR / 'index_manifest.json', {
            'text_normalization': 'unicode_alnum_casefold_v1',
            'embedding': {'model_name': smoke.EMBEDDING_MODEL},
            'openie': {'quality_repair': {'schema': 'repair_v2'}},
        })
        smoke.write_json(out / 'validation_report.json', {
            'complete': True, 'repaired_index': str(index),
            'asset_sha256': smoke.asset_hashes(index),
        })
        args = SimpleNamespace(out_root=str(out), index_root='', python='/rag/python',
                               hippo_root='/readonly/HippoRAG', datasets_dir='/datasets',
                               llm_base_url='http://local/v1')
        return project, out, index, args

    def test_mock_subprocesses_run_sequentially_and_publish_only_validated_results(self):
        with tempfile.TemporaryDirectory() as directory:
            project, out, index, args = self.prepare_directory(directory)
            calls = []

            def native(command, **kwargs):
                name = 'hipporag2' if command[3].endswith('/main.py') else 'exp4_dependency_binding'
                calls.append(name)
                self.assertTrue(kwargs['check'])
                self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
                self.assertEqual(command[command.index('--save_dir') + 1], str(index))
                smoke.write_json(command[command.index('--output') + 1],
                                 fixture_result(name, self.data, self.corpus))

            with patch.object(smoke, 'ROOT', project), patch.object(
                    smoke, 'validated_dataset', return_value=(self.data, self.corpus_docs, self.hops)), patch.object(
                    smoke.subprocess, 'run', side_effect=native):
                result = smoke.run(args)
            self.assertEqual(calls, ['hipporag2', 'exp4_dependency_binding'])
            self.assertTrue(result['graph_vectors_openie_unchanged'])
            self.assertEqual(json.loads((out / 'retrieval_smoke/selected_indices.json').read_text()),
                             [39, 789, 814])
            self.assertTrue(smoke.read_json(out / 'retrieval_smoke_report.json')['complete'])

    def test_failed_subprocess_cannot_leave_a_stale_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            project, out, _, args = self.prepare_directory(directory)
            report = out / 'retrieval_smoke_report.json'
            smoke.write_json(report, {'complete': True, 'old': True})
            with patch.object(smoke, 'ROOT', project), patch.object(
                    smoke, 'validated_dataset', return_value=(self.data, self.corpus_docs, self.hops)), patch.object(
                    smoke.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['native'])):
                with self.assertRaises(subprocess.CalledProcessError):
                    smoke.run(args)
            self.assertFalse(report.exists())

    def test_mutated_repaired_index_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            project, _, index, args = self.prepare_directory(directory)
            (index / smoke.MODEL_DIR / 'graph.pickle').write_text('changed')
            with patch.object(smoke, 'ROOT', project), patch.object(smoke.subprocess, 'run') as native:
                with self.assertRaisesRegex(ValueError, 'changed since'):
                    smoke.run(args)
                native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
