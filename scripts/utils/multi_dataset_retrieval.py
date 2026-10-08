"""Build three reusable indexes, then run nine retrieval-only comparisons.

All methods start with the same dataset, frozen graph/vectors and private copy
of the same initial LLM cache. Extraction failures are tolerated by default;
corrupt artifacts and failed retrieval jobs are recorded as actual failures.
"""

import argparse
import logging
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from pathcondrag.index.publication_policy import parse_bool

from . import improvement_experiments as experiments
from .common import read_json, write_json
from .new_index_compare import (
    ALL_ASSETS, CASES, PC_ARGUMENTS, ROOT, code_hashes, environment as qwen_environment, execute,
)

DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique')
DEFAULT_OUT = ROOT / 'outputs/3multi_hop_datasets_results_10_5'
DEFAULT_RUNTIME = Path('/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513')
SMOKE_POSITIONS = {'hotpotqa': (140, 901), '2wikimultihopqa': (866, 966), 'musique': (808, 930)}
LOG = logging.getLogger('multi_dataset_retrieval')


def environment(args):
    """Keep NV-Embed in its own runtime; Qwen embeddings need isolated deps."""
    model = getattr(args, 'embedding_model', experiments.EMBEDDING_MODEL)
    provider = getattr(args, 'embedding_provider', experiments.DEFAULT_EMBEDDING_PROVIDER)
    if provider != 'nvembed' and 'NV-Embed' not in model:
        return qwen_environment(args)
    env = os.environ.copy()
    # Never put Qwen3's transformers/accelerate bundle before the NV runtime.
    parts = [part for part in env.get('PYTHONPATH', '').split(os.pathsep)
             if part and 'runtime_deps_qwen3' not in part
             and Path(part).resolve() != DEFAULT_RUNTIME.resolve()]
    if parts:
        env['PYTHONPATH'] = os.pathsep.join(parts)
    else:
        env.pop('PYTHONPATH', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='42',
               TOKENIZERS_PARALLELISM='false',
               OPENAI_API_KEY=env.get('OPENAI_API_KEY', 'EMPTY'),
               PATHCONDRAG_LLM_MAX_IN_FLIGHT='8', HIPPORAG_LLM_MAX_IN_FLIGHT='8',
               HIPPO_OPENIE_MAX_WORKERS='8', HIPPO_OPENIE_NER_WORKERS='8',
               HIPPO_OPENIE_TRIPLE_WORKERS='8', HIPPO_OPENIE_NER_MAX_TOKENS='512',
               HIPPO_OPENIE_TRIPLE_MAX_TOKENS='2048', HIPPO_OPENIE_QUALITY_MAX_RETRIES='2',
               PATHCONDRAG_SHARED_KNN_DEVICE='cpu', HIPPORAG_KNN_DEVICE='cpu',
               HIPPO_EMBEDDING_MODEL_NAME=model, HIPPO_EMBEDDING_BASE_URL='',
               HIPPO_ROOT=getattr(args, 'hippo_root', '/root/baseline/HippoRAG'))
    return env


def sample_id(sample, position):
    return sample.get('id', sample.get('_id', position))


def smoke_dataset(out, source, name):
    """Keep original questions/gold passages plus 20 intact source passages."""
    from eval_utils import get_gold_docs

    data, _, hops = experiments.validated_dataset(
        source / f'{name}.json', source / f'{name}_corpus.json', dataset_name=name)
    preferred = SMOKE_POSITIONS.get(name)
    positions = (list(preferred) if preferred and max(preferred) < len(data) else
                 [hops.index(2), hops.index(4)] if 2 in hops and 4 in hops else
                 list(range(min(2, len(data)))))
    samples = [data[position] for position in positions]
    corpus = read_json(source / f'{name}_corpus.json')
    by_text = {experiments.passage_text(row): row for row in corpus}
    destination = out / 'smoke_datasets'
    dataset_file = destination / f'{name}.json'
    corpus_file = destination / f'{name}_corpus.json'
    provenance_file = out / 'metadata' / name / 'smoke_provenance.json'
    fixtures = (dataset_file, corpus_file, provenance_file)
    if any(path.exists() for path in fixtures):
        if not all(path.is_file() for path in fixtures):
            raise ValueError(f'{name}: incomplete smoke fixtures; inspect before retrying')
        provenance = read_json(provenance_file)
        expected = {
            'original_indices': positions,
            'original_sample_ids': [sample_id(data[i], i) for i in positions],
            'source_dataset_sha256': experiments.sha256(source / f'{name}.json'),
            'source_corpus_sha256': experiments.sha256(source / f'{name}_corpus.json'),
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{name}: smoke source provenance changed')
        saved = read_json(corpus_file)
        texts = [experiments.passage_text(row) for row in saved]
        if (read_json(dataset_file) != samples or len(saved) < 20
                or provenance.get('corpus_docs') != len(saved)
                or len(set(texts)) != len(texts)
                or any(row != by_text.get(text) for row, text in zip(saved, texts))):
            raise ValueError(f'{name}: existing smoke fixtures differ from their original sources')
        experiments.validated_dataset(dataset_file, corpus_file, dataset_name=name)
        manifest_file = out / 'metadata' / name / 'manifest.json'
        if manifest_file.is_file():
            manifest = read_json(manifest_file)
            if (experiments.sha256(dataset_file) != manifest['data_sha256']
                    or experiments.sha256(corpus_file) != manifest['corpus_sha256']):
                raise ValueError(f'{name}: existing smoke fixture SHA differs from the frozen manifest')
        # Preserve the exact corpus order and serialized bytes on resume.
        # A changed process hash seed must never rewrite a frozen fixture.
        return destination
    selected = {}
    for documents in get_gold_docs(samples, name):
        for document in sorted(documents):
            selected[document] = by_text[document]
    # Include the historical NV2 OOM passages and a longest intact passage.
    # Short-only fixtures cannot exercise full-length encoder memory peaks.
    if 'NV-Embed' in os.environ.get('HIPPO_EMBEDDING_MODEL_NAME', ''):
        historical = {'hotpotqa': 'Nashville Terminal Subdivision',
                      '2wikimultihopqa': 'Pattom A. Thanu Pillai'}
        title = historical.get(name)
        for row in corpus:
            text = experiments.passage_text(row)
            if title and text.partition('\n')[0].strip() == title:
                selected[text] = row
        longest = max(corpus, key=lambda row: len(experiments.passage_text(row)))
        selected[experiments.passage_text(longest)] = longest
    # A bounded smoke still uses complete original passages and original gold.
    for row in sorted(corpus, key=lambda item: (len(experiments.passage_text(item)),
                                               experiments.passage_text(item))):
        text = experiments.passage_text(row)
        if len(text.partition('\n')[2]) >= 80:
            selected[text] = row
        if len(selected) >= 20:
            break
    if len(selected) < 20:
        raise ValueError(f'{name}: smoke needs 20 unique source passages')
    destination.mkdir(parents=True, exist_ok=True)
    write_json(dataset_file, samples)
    write_json(corpus_file, list(selected.values()))
    write_json(provenance_file, {
        'original_indices': positions,
        'original_sample_ids': [sample_id(data[i], i) for i in positions],
        'source_dataset_sha256': experiments.sha256(source / f'{name}.json'),
        'source_corpus_sha256': experiments.sha256(source / f'{name}_corpus.json'),
        'corpus_docs': len(selected),
    })
    return destination


def layout(out, name):
    metadata = out / 'metadata' / name
    metadata.mkdir(parents=True, exist_ok=True)
    cases = out / 'cases' / name
    cases.mkdir(parents=True, exist_ok=True)
    alias = metadata / 'cases'
    if not alias.exists():
        alias.symlink_to(cases, target_is_directory=True)
    elif not alias.is_symlink() or alias.resolve() != cases.resolve():
        raise ValueError(f'Case directory alias differs: {alias}')
    (metadata / 'logs').mkdir(exist_ok=True)
    return metadata, out / 'shared_indexes' / name


def commands(args, metadata, index, datasets, name):
    options = ['--openie_strict', str(args.openie_strict).lower(),
               '--openie_prompt_version', args.openie_prompt_version]
    validation = ['--openie_validation_mode', getattr(args, 'openie_validation_mode', 'structural')]
    embedding_model = getattr(args, 'embedding_model', experiments.EMBEDDING_MODEL)
    embedding_provider = getattr(args, 'embedding_provider', experiments.DEFAULT_EMBEDDING_PROVIDER)
    embedding_batch_size = str(getattr(args, 'embedding_batch_size', experiments.DEFAULT_EMBEDDING_BATCH_SIZE))
    common = ['--dataset', name, '--sample_size', '2' if args.smoke else '0',
              '--sample_seed', '42', '--sample_indices_file', str(metadata / 'selected_indices.json'),
              '--llm_name', 'qwen3-8b', '--llm_base_url', args.llm_base_url,
              '--embedding_batch_size', embedding_batch_size,
              '--openie_max_workers', '8', '--llm_prefetch_workers', '8']
    builder = [args.python, '-B', '-u', str(ROOT / 'scripts/build_shared_index.py')] + common + options + validation + [
        '--datasets_dir', str(datasets), '--rag_type', 'hipporag',
        '--embedding_name', embedding_model, '--embedding_provider', embedding_provider,
        '--openie_mode', 'online', '--eval_mode', 'index_only',
        '--force_index_from_scratch', 'true', '--force_openie_from_scratch', 'true',
        '--save_dir_exact', '--save_dir', str(index),
        '--output', str(metadata / 'index_build_result.json')]
    retrieval = common + ['--eval_mode', 'retrieve', '--retrieval_top_k', '200',
                          '--result_top_k', '10', '--candidate_output_top_k', '200', '--reuse_index']
    cases = {}
    for case_name in CASES:
        case = metadata / 'cases' / case_name
        if case_name == 'hipporag2':
            main = (ROOT / 'scripts/nvembed_baseline.py'
                    if embedding_provider == 'nvembed' or 'NV-Embed' in embedding_model
                    else Path(args.hippo_root) / 'main.py')
            command = [args.python, '-B', '-u', str(main)] + retrieval + [
                '--datasets_dir', str(datasets), '--rag_type', 'hipporag',
                '--embedding_name', embedding_model, '--embedding_provider', embedding_provider,
                '--save_dir_exact']
        else:
            command = [args.python, '-B', '-u', str(ROOT / 'scripts/eval_dataset.py')] + retrieval + PC_ARGUMENTS + options + validation + [
                '--data_path', str(datasets / f'{name}.json'),
                '--corpus_path', str(datasets / f'{name}_corpus.json'),
                '--corpus_mode', 'full', '--qa_top_k', '5', '--max_qa_steps', '1', '--max_new_tokens', '2048',
                '--embedding_model_name', embedding_model,
                '--improvement_stage', '0' if case_name == 'pathcondrag_original' else '4',
                '--stratified_eval', '--stratified_output', str(case / 'stratified.json')]
        cases[case_name] = command + ['--save_dir', str(case / 'index'), '--output', str(case / 'result.json')]
    return builder, cases


def log_offset(path):
    if not path.is_file():
        raise ValueError(f'vLLM request log is missing: {path}')
    return path.stat().st_size


def archive_incomplete_case(metadata, case_name):
    case = metadata / 'cases' / case_name
    if not case.exists():
        return
    if (case / 'validated.ok').exists():
        raise ValueError(f'Validated case cannot be replaced: {case_name}')
    archives = metadata / 'incomplete_attempts'
    archives.mkdir(exist_ok=True)
    target = archives / f'{case_name}_{time.time_ns()}'
    case.rename(target)
    LOG.warning('[previous-incomplete-case] %s', target)


def initialize_linked_case(metadata, case_name):
    """Share immutable large assets; keep request caches and JSON files private."""
    manifest = experiments.manifest_at(metadata)
    if experiments.index_ready(SimpleNamespace(out_root=str(metadata))) != 0:
        raise ValueError('Shared source index is not frozen')
    source = Path(manifest['source_index'])
    case = metadata / 'cases' / case_name
    if case.exists():
        raise ValueError(f'Case already exists: {case}')
    linked = {source / manifest['model_dir'] / name for name in experiments.ASSETS}

    def copy_file(original, destination):
        if Path(original) in linked:
            os.link(original, destination)
            return destination
        return shutil.copy2(original, destination)

    case.mkdir(parents=True)
    index = case / 'index'
    shutil.copytree(source, index, copy_function=copy_file,
                    ignore=shutil.ignore_patterns('*.lock', 'metrics*.json', 'eval_results*',
                                                 'openie_progress*', '*.sqlite*'))
    if (index / 'llm_cache').exists():
        shutil.rmtree(index / 'llm_cache')
    shutil.copytree(metadata / 'initial_llm_cache', index / 'llm_cache')
    if experiments.asset_hashes(index, manifest['model_dir']) != manifest['source_asset_sha256']:
        raise ValueError('Linked graph/vectors differ from frozen source')
    if experiments.cache_hashes(index / 'llm_cache') != manifest['initial_cache_sha256']:
        raise ValueError('Private cache differs from common initial snapshot')
    write_json(case / 'before.json', {'asset_sha256': manifest['source_asset_sha256'],
                                     'initial_cache_sha256': manifest['initial_cache_sha256']})


def frozen_hashes(index):
    candidates = list(index.glob('*/index_manifest.json'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one model-specific index manifest under {index}')
    directory = candidates[0].parent
    hashes = {name: experiments.sha256(directory / name) for name in ALL_ASSETS}
    hashes['openie_results_ner_qwen3-8b.json'] = experiments.sha256(index / 'openie_results_ner_qwen3-8b.json')
    return hashes


def publish_summary(out, statuses):
    reports = {}
    lines = ['| Dataset | Case | R@1 | R@2 | R@5 | R@10 | All gold@5 | Seconds |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in DATASETS:
        reports[name] = {}
        for case_name in CASES:
            report_file = out / 'cases' / name / case_name / 'report.json'
            if not report_file.is_file():
                continue
            report = read_json(report_file)
            reports[name][case_name] = {key: value for key, value in report.items() if key != 'per_question'}
            values = [f"{report['retrieval_metrics'][f'Recall@{k}']:.4f}" for k in (1, 2, 5, 10)]
            lines.append(f'| {name} | {case_name} | ' + ' | '.join(values)
                         + f" | {report['all_gold_top5']:.4f} | {report['seconds']} |")
    write_json(out / 'comparison.json', {'datasets': reports, 'stages': statuses})
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def run(args):
    args.embedding_model = getattr(args, 'embedding_model', experiments.EMBEDDING_MODEL)
    args.embedding_provider = getattr(args, 'embedding_provider', experiments.DEFAULT_EMBEDDING_PROVIDER)
    args.embedding_batch_size = getattr(args, 'embedding_batch_size', experiments.DEFAULT_EMBEDDING_BATCH_SIZE)
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(ROOT / 'outputs') or out == ROOT / 'outputs':
        raise ValueError('Output must be a dedicated directory under PathCondRAG/outputs')
    if args.smoke and 'smoke' not in out.name:
        raise ValueError('Smoke output must have "smoke" in its name, separate from full indexes')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    (out / 'shared_indexes').mkdir(exist_ok=True)
    if shutil.disk_usage(out).free < 5 * 1024 ** 3:
        raise ValueError('At least 5 GiB of free disk is needed')
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('vLLM service is not ready')
    server_log = Path(args.vllm_log)
    log_offset(server_log)
    baseline_code = code_hashes(Path(args.hippo_root))
    env = environment(args)
    env['HIPPO_ALLOW_INDEX_RESUME'] = '1'
    env['HIPPO_EMBEDDING_MODEL_NAME'] = args.embedding_model
    env['HIPPO_EMBEDDING_BASE_URL'] = ''
    prepared, statuses = {}, {}
    stage_file = out / 'stage_status.json'
    protocol = {'datasets': list(DATASETS), 'cases': list(CASES), 'smoke': args.smoke,
                'openie_strict': args.openie_strict, 'openie_prompt_version': args.openie_prompt_version,
                'openie_validation_mode': getattr(args, 'openie_validation_mode', 'structural'),
                'embedding_model': args.embedding_model,
                'embedding_provider': args.embedding_provider,
                'embedding_batch_size': args.embedding_batch_size,
                'llm_name': 'qwen3-8b', 'llm_workers': 8, 'ner_max_tokens': 512,
                'triple_max_tokens': 2048, 'enable_thinking': False,
                'retrieval_top_k': 200, 'result_top_k': 10, 'candidate_output_top_k': 200}
    protocol_file = out / 'protocol.json'
    if protocol_file.exists() and read_json(protocol_file) != protocol:
        raise ValueError('Existing run protocol differs; use a different output directory')
    write_json(protocol_file, protocol)
    (out / 'completed.ok').unlink(missing_ok=True)

    def record(stage, state, **details):
        statuses[stage] = {'state': state, **details}
        write_json(stage_file, statuses)
        LOG.info('[stage] %s %s %s', stage, state, details)

    # Finish all three source indexes before any retrieval method is launched.
    for name in DATASETS:
        stage = f'{name}/index'
        try:
            metadata, index = layout(out, name)
            datasets = (smoke_dataset(out, Path(args.datasets_dir), name)
                        if args.smoke else Path(args.datasets_dir))
            options = SimpleNamespace(out_root=str(metadata), source_index=str(index),
                shared_output_root=str(out), dataset=name,
                data_path=str(datasets / f'{name}.json'), corpus_path=str(datasets / f'{name}_corpus.json'),
                sample_size=2 if args.smoke else 0, sample_seed=42, sample_indices_file=None,
                cases=' '.join(CASES), build_shared_index=True,
                embedding_model=args.embedding_model,
                embedding_provider=args.embedding_provider,
                embedding_batch_size=args.embedding_batch_size)
            experiments.prepare(options)
            builder, cases = commands(args, metadata, index, datasets, name)
            if experiments.index_ready(options) != 0:
                start = log_offset(server_log)
                record(stage, 'running', source_index=str(index), log_start=start)
                elapsed = execute(builder, out / 'logs' / f'{name}_index_build.log', env)
                _, documents, _ = experiments.validated_dataset(options.data_path, options.corpus_path, dataset_name=name)
                from .fresh_index_validation import validate_fresh_index
                validate_fresh_index(metadata, index, documents, strict=args.openie_strict)
                experiments.freeze_index(SimpleNamespace(out_root=str(metadata), elapsed=elapsed,
                    vllm_log=str(server_log), log_start=start, log_end=log_offset(server_log),
                    openie_strict=args.openie_strict))
            else:
                from .fresh_index_validation import validate_fresh_index
                _, documents, _ = experiments.validated_dataset(options.data_path, options.corpus_path, dataset_name=name)
                validate_fresh_index(metadata, index, documents, strict=args.openie_strict)
            prepared[name] = (metadata, index, cases, frozen_hashes(index))
            report = read_json(metadata / 'index_build_report.json')
            record(stage, 'ready', source_index=str(index),
                   openie_failure_count=report.get('openie_failure_count', 0),
                   openie_strict=args.openie_strict)
        except Exception as error:
            LOG.exception('[failed] %s', stage)
            record(stage, 'failed', error=f'{type(error).__name__}: {error}')

    if getattr(args, 'indexes_only', False):
        failed = {stage: value for stage, value in statuses.items() if value['state'] != 'ready'}
        if failed:
            write_json(out / 'failed_stages.json', failed)
            return 1
        write_json(out / 'indexes_completed.ok', {
            'datasets': list(DATASETS), 'smoke': args.smoke,
            'embedding_model': args.embedding_model,
            'embedding_batch_size': args.embedding_batch_size,
        })
        LOG.info('[done] all three shared indexes validated: %s', out)
        return 0

    for name in DATASETS:
        if name not in prepared:
            for case_name in CASES:
                record(f'{name}/{case_name}', 'blocked', reason='Source index failed its integrity checks')
            continue
        metadata, index, cases, hashes = prepared[name]
        for case_name in CASES:
            stage = f'{name}/{case_name}'
            case_args = SimpleNamespace(out_root=str(metadata), name=case_name)
            try:
                if experiments.ready(case_args) == 0:
                    if frozen_hashes(index) != hashes or frozen_hashes(metadata / 'cases' / case_name / 'index') != hashes:
                        raise ValueError('Validated case OpenIE assets differ from the frozen source')
                    record(stage, 'ready', reused=True)
                    continue
                archive_incomplete_case(metadata, case_name)
                initialize_linked_case(metadata, case_name)
                start = log_offset(server_log)
                record(stage, 'running')
                elapsed = execute(cases[case_name], out / 'logs' / f'{name}_{case_name}.log', env)
                experiments.report(SimpleNamespace(out_root=str(metadata), name=case_name,
                    elapsed=elapsed, vllm_log=str(server_log), log_start=start, log_end=log_offset(server_log)))
                if frozen_hashes(index) != hashes or frozen_hashes(metadata / 'cases' / case_name / 'index') != hashes:
                    raise ValueError('Retrieval modified a frozen graph/vector/OpenIE asset')
                if code_hashes(Path(args.hippo_root)) != baseline_code:
                    raise ValueError('Original HippoRAG source changed during retrieval')
                record(stage, 'ready', seconds=elapsed)
            except Exception as error:
                LOG.exception('[failed] %s', stage)
                record(stage, 'failed', error=f'{type(error).__name__}: {error}')
        if all(statuses[f'{name}/{case_name}']['state'] == 'ready' for case_name in CASES):
            experiments.summary(SimpleNamespace(out_root=str(metadata)))
    publish_summary(out, statuses)
    failed = {stage: value for stage, value in statuses.items() if value['state'] in ('failed', 'blocked')}
    if failed:
        write_json(out / 'failed_stages.json', failed)
        LOG.error('[incomplete] %s stage(s) failed or blocked; see %s', len(failed), out / 'failed_stages.json')
        return 1
    write_json(out / 'completed.ok', {'datasets': list(DATASETS), 'cases': list(CASES),
               'smoke': args.smoke, 'openie_strict': args.openie_strict})
    (out / 'failed_stages.json').unlink(missing_ok=True)
    LOG.info('[done] all three indexes and nine retrieval cases validated: %s', out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root')
    parser.add_argument('--datasets-dir', default='/root/datasets')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--runtime-deps', default=str(DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=str(DEFAULT_OUT / 'logs/vllm.log'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--indexes-only', action='store_true',
                        help='Build and validate all three indexes; defer retrieval to a later invocation.')
    parser.add_argument('--openie-strict', '--openie_strict', type=parse_bool, default=False)
    parser.add_argument('--openie-prompt-version', '--openie_prompt_version',
                        choices=['origin', 'optimized'], default='optimized')
    parser.add_argument('--openie-validation-mode', '--openie_validation_mode',
                        choices=['structural', 'source_verified'], default='structural',
                        help='Structural checks match the normal HippoRAG extraction cost; '
                             'source_verified adds expensive LLM evidence audits.')
    parser.add_argument('--embedding-model', '--embedding_model',
                        default=experiments.EMBEDDING_MODEL)
    parser.add_argument('--embedding-provider', '--embedding_provider',
                        default=experiments.DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument('--embedding-batch-size', '--embedding_batch_size', type=int,
                        default=experiments.DEFAULT_EMBEDDING_BATCH_SIZE)
    args = parser.parse_args(argv)
    args.out_root = args.out_root or str(DEFAULT_OUT.with_name(DEFAULT_OUT.name + '_smoke') if args.smoke else DEFAULT_OUT)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)
