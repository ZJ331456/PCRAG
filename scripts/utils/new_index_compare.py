"""One fresh-index workflow for a two-question check and the full benchmark."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from . import improvement_experiments as experiments
from .common import read_json, write_json
from pathcondrag.index.publication_policy import parse_bool

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'outputs/pathcondrag_new_index_10_4'
DEFAULT_RUNTIME = ROOT / 'outputs/openie_quality_repair_qwen3_b4_20261003/runtime_deps'
CASES = ('hipporag2', 'pathcondrag_original', 'exp4_dependency_binding')
ALL_ASSETS = experiments.ASSETS + ('openie_state.json', 'index_manifest.json', 'chunk_metadata.json')
LOG = logging.getLogger('new_index_compare')

PC_ARGUMENTS = [
    '--use_iterative_retrieval', '--iterative_round1_top_docs', '1',
    '--iterative_round2_seed_top_k', '5', '--iterative_seed_mode', 'idf_novel',
    '--iterative_merge_alpha', '0.45', '--iterative_min_seed_entities', '1',
    '--use_query_decomposition', '--use_path_conditioned_qd', '--qd_min_hops', '2',
    '--qd_sub_retrieval_top_k', '3', '--pcqd_ground_top_docs', '5', '--pcqd_entity_top_k', '3',
    '--pcqd_path_score_threshold', '0.60', '--pcqd_weight_base', '0.40',
    '--pcqd_weight_static', '0.20', '--pcqd_weight_path', '0.40',
    '--hop_source', 'benchmark', '--hop_force_max', '4', '--qd_max_sub_questions', '4',
]


def code_hashes(hippo_root):
    listing = subprocess.run(['git', '-C', str(hippo_root), 'ls-files', '-z', '--', '*.py'],
                             check=True, capture_output=True).stdout
    return {name.decode(): experiments.sha256(hippo_root / name.decode())
            for name in listing.split(b'\0') if name and (hippo_root / name.decode()).is_file()}


def all_hashes(index):
    return {name: experiments.sha256(index / experiments.MODEL_DIR / name) for name in ALL_ASSETS}


def smoke_dataset(out, datasets):
    """Keep two original 2/4-hop questions and at least 200 original passages."""
    data, corpus, hops = experiments.validated_dataset(datasets / 'musique.json',
                                                       datasets / 'musique_corpus.json')
    positions = [hops.index(2), hops.index(4)]
    samples = [data[index] for index in positions]
    originals = read_json(datasets / 'musique_corpus.json')
    by_text = {experiments.passage_text(row): row for row in originals}
    chosen = {}
    for sample in samples:
        for paragraph in sample['paragraphs']:
            text = experiments.passage_text(paragraph)
            chosen[text] = by_text[text]
    # Short, intact passages make this a bounded end-to-end check. No chunk
    # text, benchmark label, retrieval budget or model budget is rewritten.
    ordered = sorted(originals, key=lambda row: (len(experiments.passage_text(row)),
                                                experiments.passage_text(row)))
    for row in ordered:
        text = experiments.passage_text(row)
        if len(text.partition('\n')[2]) >= 80:
            chosen[text] = row
        if len(chosen) >= 200:
            break
    if len(chosen) < 200:
        raise ValueError('Smoke needs 200 unique source passages for the unchanged candidate export')
    directory = out / 'smoke_dataset'
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'musique.json', samples)
    write_json(directory / 'musique_corpus.json', list(chosen.values()))
    write_json(out / 'smoke_provenance.json', {'original_indices': positions,
               'original_sample_ids': [sample['id'] for sample in samples],
               'source_dataset_sha256': experiments.sha256(datasets / 'musique.json'),
               'source_corpus_sha256': experiments.sha256(datasets / 'musique_corpus.json')})
    return directory


def environment(args):
    env = os.environ.copy()
    runtime = Path(args.runtime_deps).resolve()
    if not (runtime / 'transformers').is_dir():
        raise ValueError(f'Qwen3-compatible isolated runtime is missing: {runtime}')
    env['PYTHONPATH'] = str(runtime) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               OPENAI_API_KEY=env.get('OPENAI_API_KEY', 'EMPTY'),
               PATHCONDRAG_LLM_MAX_IN_FLIGHT='8', HIPPORAG_LLM_MAX_IN_FLIGHT='8',
               HIPPO_OPENIE_MAX_WORKERS='8', HIPPO_OPENIE_NER_WORKERS='8',
               HIPPO_OPENIE_TRIPLE_WORKERS='8', HIPPO_OPENIE_NER_MAX_TOKENS='512',
               HIPPO_OPENIE_TRIPLE_MAX_TOKENS='2048', HIPPO_OPENIE_QUALITY_MAX_RETRIES='2',
               PATHCONDRAG_SHARED_KNN_DEVICE='cuda', HIPPORAG_KNN_DEVICE='cuda')
    if (Path(args.out_root) / 'shared_hipporag2_index').exists():
        env['HIPPO_ALLOW_INDEX_RESUME'] = '1'
    return env


def execute(command, logfile, env):
    LOG.info('[command] %s', ' '.join(command))
    started = time.monotonic()
    if logfile.is_file() and logfile.stat().st_size:
        previous = logfile.with_name(f'{logfile.stem}.attempt_{time.time_ns()}.log')
        logfile.rename(previous)
        LOG.info('[previous-attempt-log] %s', previous)
    with logfile.open('w', encoding='utf-8') as stream:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
    return int(time.monotonic() - started)


def initialize_linked_case(out, name):
    """Link only frozen graph/vectors; keep JSON and request caches private."""
    manifest = experiments.manifest_at(out)
    source = Path(manifest['source_index'])
    case = out / 'cases' / name
    if case.exists():
        raise ValueError(f'Incomplete case already exists: {case}; inspect it before rerunning')
    if experiments.index_ready(SimpleNamespace(out_root=str(out))) != 0:
        raise ValueError('Shared index is not frozen')
    linked = {source / manifest['model_dir'] / name for name in experiments.ASSETS}

    def copy_file(original, destination):
        if Path(original) in linked:
            os.link(original, destination)
            return destination
        return shutil.copy2(original, destination)

    case.mkdir(parents=True)
    index = case / 'index'
    shutil.copytree(source, index, copy_function=copy_file,
                    ignore=shutil.ignore_patterns('*.lock', 'metrics*.json', 'eval_results*'))
    shutil.rmtree(index / 'llm_cache')
    shutil.copytree(out / 'initial_llm_cache', index / 'llm_cache')
    if experiments.asset_hashes(index) != manifest['source_asset_sha256']:
        raise ValueError('Linked case graph/vectors differ from the frozen index')
    if experiments.cache_hashes(index / 'llm_cache') != manifest['initial_cache_sha256']:
        raise ValueError('Private LLM cache differs from the common snapshot')
    write_json(case / 'before.json', {'asset_sha256': experiments.asset_hashes(index),
               'initial_cache_sha256': manifest['initial_cache_sha256']})


def commands(args, out, datasets):
    extraction_options = ['--openie_strict', str(getattr(args, 'openie_strict', True)).lower(),
                          '--openie_prompt_version', getattr(args, 'openie_prompt_version', 'optimized')]
    common = ['--dataset', 'musique', '--sample_size', '2' if args.smoke else '0',
              '--sample_seed', '42', '--sample_indices_file', str(out / 'selected_indices.json'),
              '--llm_name', 'qwen3-8b', '--llm_base_url', args.llm_base_url,
              '--embedding_batch_size', '4', '--openie_max_workers', '8', '--llm_prefetch_workers', '8']
    builder = [args.python, '-B', '-u', str(ROOT / 'scripts/build_shared_index.py')] + common + extraction_options + [
        '--datasets_dir', str(datasets), '--rag_type', 'hipporag',
        '--embedding_name', experiments.EMBEDDING_MODEL, '--embedding_provider', 'transformers',
        '--openie_mode', 'online', '--eval_mode', 'index_only',
        '--force_index_from_scratch', 'true', '--force_openie_from_scratch', 'true',
        '--save_dir_exact', '--save_dir', str(out / 'shared_hipporag2_index'),
        '--output', str(out / 'index_build_result.json')]
    retrieval = common + ['--eval_mode', 'retrieve', '--retrieval_top_k', '200',
                          '--result_top_k', '10', '--candidate_output_top_k', '200', '--reuse_index']
    cases = {}
    for name in CASES:
        case = out / 'cases' / name
        if name == 'hipporag2':
            command = [args.python, '-B', '-u', str(Path(args.hippo_root) / 'main.py')] + retrieval + [
                '--datasets_dir', str(datasets), '--rag_type', 'hipporag',
                '--embedding_name', experiments.EMBEDDING_MODEL, '--embedding_provider', 'transformers',
                '--save_dir_exact']
        else:
            command = [args.python, '-B', '-u', str(ROOT / 'scripts/eval_dataset.py')] + retrieval + PC_ARGUMENTS + extraction_options + [
                '--data_path', str(datasets / 'musique.json'), '--corpus_path', str(datasets / 'musique_corpus.json'),
                '--corpus_mode', 'full', '--qa_top_k', '5', '--max_qa_steps', '1', '--max_new_tokens', '2048',
                '--embedding_model_name', experiments.EMBEDDING_MODEL,
                '--improvement_stage', '0' if name == 'pathcondrag_original' else '4',
                '--stratified_eval', '--stratified_output', str(case / 'stratified.json')]
        cases[name] = command + ['--save_dir', str(case / 'index'), '--output', str(case / 'result.json')]
    return builder, cases


def run(args):
    out, datasets, hippo = Path(args.out_root).resolve(), Path(args.datasets_dir).resolve(), Path(args.hippo_root).resolve()
    if not out.is_relative_to(ROOT / 'outputs') or out == ROOT / 'outputs':
        raise ValueError('Output must be a dedicated directory under PathCondRAG/outputs')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    if shutil.disk_usage(out).free < 3 * 1024 ** 3:
        raise ValueError('At least 3 GiB of free disk is needed for the fresh index and exports')
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('The existing vLLM service is not ready')
    before_code = code_hashes(hippo)
    if args.smoke:
        datasets = smoke_dataset(out, datasets)
    prepare = SimpleNamespace(out_root=str(out), source_index=str(out / 'shared_hipporag2_index'),
        data_path=str(datasets / 'musique.json'), corpus_path=str(datasets / 'musique_corpus.json'),
        sample_size=2 if args.smoke else 0, sample_seed=42, sample_indices_file=None,
        cases=' '.join(CASES), build_shared_index=True)
    experiments.prepare(prepare)
    env = environment(args)
    builder, cases = commands(args, out, datasets)
    server_log = Path(args.vllm_log)
    if experiments.index_ready(prepare) != 0:
        log_start = server_log.stat().st_size
        elapsed = execute(builder, out / 'logs/index_build.log', env)
        from .fresh_index_validation import validate_fresh_index
        _, documents, _ = experiments.validated_dataset(prepare.data_path, prepare.corpus_path)
        validate_fresh_index(out, out / 'shared_hipporag2_index', documents, strict=args.openie_strict)
        experiments.freeze_index(SimpleNamespace(out_root=str(out), elapsed=elapsed, vllm_log=str(server_log),
                                                log_start=log_start, log_end=server_log.stat().st_size,
                                                openie_strict=args.openie_strict))
    else:
        from .fresh_index_validation import validate_fresh_index
        _, documents, _ = experiments.validated_dataset(prepare.data_path, prepare.corpus_path)
        validate_fresh_index(out, out / 'shared_hipporag2_index', documents, strict=args.openie_strict)
    source = out / 'shared_hipporag2_index'
    frozen = all_hashes(source)
    write_json(out / 'fresh_index_identity.json', {'builder': str(ROOT / 'scripts/build_shared_index.py'),
               'fresh_corpus': str(datasets / 'musique_corpus.json'), 'asset_sha256': frozen,
               'baseline_code_sha256': before_code, 'embedding_batch_size': 4, 'llm_workers': 8,
               'thinking': False, 'triple_max_tokens': 2048, 'smoke': args.smoke,
               'openie_strict': args.openie_strict, 'openie_prompt_version': args.openie_prompt_version})
    for name in CASES:
        case_args = SimpleNamespace(out_root=str(out), name=name)
        if experiments.ready(case_args) == 0:
            continue
        initialize_linked_case(out, name)
        log_start = server_log.stat().st_size
        elapsed = execute(cases[name], out / 'logs' / (name + '.log'), env)
        experiments.report(SimpleNamespace(out_root=str(out), name=name, elapsed=elapsed,
            vllm_log=str(server_log), log_start=log_start, log_end=server_log.stat().st_size))
        if all_hashes(source) != frozen or all_hashes(out / 'cases' / name / 'index') != frozen:
            raise ValueError(f'{name} modified a frozen graph/vector/OpenIE asset')
        if code_hashes(hippo) != before_code:
            raise ValueError('The original HippoRAG code changed during retrieval')
    experiments.summary(SimpleNamespace(out_root=str(out)))
    LOG.info('[done] fresh PathCondRAG quality index and all three retrieval cases validated: %s', out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', default=str(DEFAULT_OUT))
    parser.add_argument('--datasets-dir', default='/root/datasets')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--runtime-deps', default=str(DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default='/root/eval/logs/vllm_qwen3.log')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--openie_strict', '--openie-strict', type=parse_bool, default=True)
    parser.add_argument('--openie_prompt_version', '--openie-prompt-version',
                        choices=['origin', 'optimized'], default='optimized')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(args)
    return 0
