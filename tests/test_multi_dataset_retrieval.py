"""CPU checks for the multi-dataset runner's experiment boundaries."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

from utils import improvement_experiments as experiments
from utils import multi_dataset_retrieval as runner


class MultiDatasetRunnerTests(unittest.TestCase):
    def test_all_commands_preserve_budgets_and_separate_indexes(self):
        args = SimpleNamespace(python='/python', hippo_root='/hippo', smoke=False,
                               openie_strict=False, openie_prompt_version='optimized',
                               llm_base_url='http://local/v1')
        for name in runner.DATASETS:
            metadata, index = Path('/output/metadata') / name, Path('/output/shared_indexes') / name
            builder, cases = runner.commands(args, metadata, index, Path('/datasets'), name)
            self.assertEqual(set(cases), set(runner.CASES))
            for command in (builder, *cases.values()):
                self.assertEqual(command[command.index('--dataset') + 1], name)
                self.assertEqual(command[command.index('--sample_size') + 1], '0')
                self.assertEqual(command[command.index('--embedding_batch_size') + 1], '4')
                self.assertEqual(command[command.index('--openie_max_workers') + 1], '8')
                self.assertEqual(command[command.index('--llm_prefetch_workers') + 1], '8')
            self.assertEqual(builder[builder.index('--save_dir') + 1], str(index))
            self.assertEqual(builder[builder.index('--openie_strict') + 1], 'false')
            self.assertEqual(builder[builder.index('--eval_mode') + 1], 'index_only')
            for case_name, command in cases.items():
                self.assertIn('--reuse_index', command)
                self.assertEqual(command[command.index('--eval_mode') + 1], 'retrieve')
                self.assertEqual(command[command.index('--result_top_k') + 1], '10')
                self.assertEqual(command[command.index('--candidate_output_top_k') + 1], '200')
                if case_name != 'hipporag2':
                    self.assertEqual(command[command.index('--hop_source') + 1], 'benchmark')
                    self.assertEqual(command[command.index('--max_new_tokens') + 1], '2048')
                    self.assertEqual(command[command.index('--improvement_stage') + 1],
                                     '0' if case_name == 'pathcondrag_original' else '4')

    def test_linked_case_omits_build_journal_but_keeps_private_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            metadata, source = runner.layout(out, 'hotpotqa')
            for name in experiments.ASSETS:
                asset = source / experiments.MODEL_DIR / name
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_text(name)
            (source / 'llm_cache').mkdir()
            (source / 'llm_cache' / 'cache.json').write_text('source cache')
            (source / experiments.MODEL_DIR / 'openie_progress.sqlite').write_text('large journal')
            snapshot = metadata / 'initial_llm_cache'
            snapshot.mkdir()
            (snapshot / 'cache.json').write_text('common cache')
            manifest = {'source_index': str(source), 'model_dir': experiments.MODEL_DIR,
                        'source_asset_sha256': experiments.asset_hashes(source),
                        'initial_cache_sha256': experiments.cache_hashes(snapshot)}
            (metadata / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(experiments, 'index_ready', return_value=0):
                runner.initialize_linked_case(metadata, 'hipporag2')
            index = out / 'cases' / 'hotpotqa' / 'hipporag2' / 'index'
            self.assertFalse((index / experiments.MODEL_DIR / 'openie_progress.sqlite').exists())
            for name in experiments.ASSETS:
                self.assertEqual((index / experiments.MODEL_DIR / name).stat().st_ino,
                                 (source / experiments.MODEL_DIR / name).stat().st_ino)
            cache = index / 'llm_cache' / 'cache.json'
            self.assertEqual(cache.read_text(), 'common cache')
            cache.write_text('changed by this method')
            self.assertEqual((snapshot / 'cache.json').read_text(), 'common cache')
            self.assertEqual((source / 'llm_cache' / 'cache.json').read_text(), 'source cache')

    def test_incomplete_case_archived_validated_case_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata, _ = runner.layout(Path(temporary), 'musique')
            case = metadata / 'cases' / 'hipporag2'
            case.mkdir()
            (case / 'result.json').write_text('partial')
            runner.archive_incomplete_case(metadata, 'hipporag2')
            self.assertFalse(case.exists())
            archived = list((metadata / 'incomplete_attempts').iterdir())
            self.assertEqual((archived[0] / 'result.json').read_text(), 'partial')
            case.mkdir()
            (case / 'validated.ok').write_text('{}')
            with self.assertRaises(ValueError):
                runner.archive_incomplete_case(metadata, 'hipporag2')

    def test_smoke_keeps_original_gold_and_question_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, output = base / 'source', base / 'run_smoke'
            source.mkdir()
            runner.layout(output, 'hotpotqa')
            rows = [{'title': f'Doc{i}', 'text': 'A source passage with intact words. ' * 4}
                    for i in range(205)]
            samples = [{'_id': f'q{i}', 'question': f'Question {i}?', 'type': 'bridge',
                        'supporting_facts': [[f'Doc{i}', 0]],
                        'context': [[f'Doc{i}', [rows[i]['text']]]]} for i in range(2)]
            (source / 'hotpotqa.json').write_text(json.dumps(samples))
            (source / 'hotpotqa_corpus.json').write_text(json.dumps(rows))
            directory = runner.smoke_dataset(output, source, 'hotpotqa')
            self.assertEqual(json.loads((directory / 'hotpotqa.json').read_text()), samples)
            _, docs, hops = experiments.validated_dataset(directory / 'hotpotqa.json',
                                                          directory / 'hotpotqa_corpus.json')
            self.assertEqual(len(docs), 20)
            self.assertEqual(hops, [2, 2])

    def test_failed_case_keeps_running_other_methods_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / 'outputs' / 'three_datasets'
            out.mkdir(parents=True)
            vlog = out / 'vllm.log'
            vlog.write_text('')
            args = SimpleNamespace(out_root=str(out), datasets_dir='/data', smoke=False,
                hippo_root='/hippo', runtime_deps='/runtime', python='/python',
                llm_base_url='http://local/v1', vllm_log=str(vlog),
                openie_strict=False, openie_prompt_version='optimized')

            def prepare(options):
                metadata = Path(options.out_root)
                (metadata / 'index_build_report.json').write_text(json.dumps({'openie_failure_count': 1}))

            calls = []

            def execute(command, logfile, env):
                calls.append(str(logfile))
                if logfile.name == 'hotpotqa_hipporag2.log':
                    raise RuntimeError('injected request failure')
                return 1

            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * 1024 ** 3)), \
                    patch.object(runner.urllib.request, 'urlopen') as server, \
                    patch.object(runner, 'code_hashes', return_value={}), \
                    patch.object(runner, 'environment', return_value={}), \
                    patch.object(experiments, 'prepare', side_effect=prepare), \
                    patch.object(experiments, 'index_ready', return_value=0), \
                    patch.object(experiments, 'validated_dataset', return_value=([], set(), [])), \
                    patch('utils.fresh_index_validation.validate_fresh_index'), \
                    patch.object(runner, 'frozen_hashes', return_value={'stable': 'hash'}), \
                    patch.object(experiments, 'ready', return_value=1), \
                    patch.object(runner, 'initialize_linked_case'), \
                    patch.object(runner, 'execute', side_effect=execute), \
                    patch.object(experiments, 'report'), \
                    patch.object(experiments, 'summary'), \
                    patch.object(runner, 'publish_summary'), \
                    patch.object(runner.LOG, 'exception'):
                server.return_value.__enter__.return_value.status = 200
                self.assertEqual(runner.run(args), 1)
            self.assertEqual(len(calls), 9)
            self.assertTrue(any('musique_exp4_dependency_binding.log' in name for name in calls))
            stages = json.loads((out / 'stage_status.json').read_text())
            self.assertEqual(stages['hotpotqa/index']['openie_failure_count'], 1)
            self.assertEqual(stages['hotpotqa/hipporag2']['state'], 'failed')
            self.assertEqual(stages['musique/exp4_dependency_binding']['state'], 'ready')
            self.assertFalse((out / 'completed.ok').exists())
            self.assertTrue((out / 'failed_stages.json').is_file())


if __name__ == '__main__':
    unittest.main()
