"""Sequential, bounded retrieval checks for a validated repaired shared index."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .improvement_experiments import validate_result, validated_dataset

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / 'outputs/openie_quality_repair_qwen3_b4_20261003'
SELECTED_INDICES = [39, 789, 814]
MODEL_DIR = 'qwen3-8b__root_models_Qwen3-Embedding-8B'
EMBEDDING_MODEL = '/root/models/Qwen3-Embedding-8B'
ASSETS = (
    'openie_state.json', 'index_manifest.json', 'chunk_metadata.json', 'graph.pickle',
    'chunk_embeddings/vdb_chunk.parquet', 'entity_embeddings/vdb_entity.parquet',
    'fact_embeddings/vdb_fact.parquet',
)
LOG = logging.getLogger('openie.repair.retrieval_smoke')

# Keep the stage-four retrieval settings aligned with PC3_COMMON in the full
# seven-case runner. hop_force_max is only a ceiling; per-query hops come from
# --hop_source benchmark and the selected MuSiQue annotations.
EXP4_ARGUMENTS = [
    '--use_iterative_retrieval',
    '--iterative_round1_top_docs', '1', '--iterative_round2_seed_top_k', '5',
    '--iterative_seed_mode', 'idf_novel', '--iterative_merge_alpha', '0.45',
    '--iterative_min_seed_entities', '1',
    '--use_query_decomposition', '--use_path_conditioned_qd',
    '--qd_min_hops', '2', '--qd_sub_retrieval_top_k', '3',
    '--pcqd_ground_top_docs', '5', '--pcqd_entity_top_k', '3',
    '--pcqd_path_score_threshold', '0.60',
    '--pcqd_weight_base', '0.40', '--pcqd_weight_static', '0.20',
    '--pcqd_weight_path', '0.40',
    '--hop_source', 'benchmark', '--hop_force_max', '4', '--qd_max_sub_questions', '4',
    '--improvement_stage', '4',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def asset_hashes(index_root):
    return {name: sha256(Path(index_root) / MODEL_DIR / name) for name in ASSETS}


def child_environment():
    environment = os.environ.copy()
    environment.update({
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'HIPPORAG_KNN_DEVICE': 'cpu',
        'PATHCONDRAG_LLM_MAX_IN_FLIGHT': '8', 'HIPPORAG_LLM_MAX_IN_FLIGHT': '8',
        'HIPPO_OPENIE_MAX_WORKERS': '8', 'HIPPO_OPENIE_NER_WORKERS': '8',
        'HIPPO_OPENIE_TRIPLE_WORKERS': '8',
    })
    environment.setdefault('OPENAI_API_KEY', 'EMPTY')
    return environment


def build_commands(args, indices_path, index_root, smoke_root):
    """Return two shell-free commands; the caller executes them sequentially."""
    common = [
        '--dataset', 'musique', '--sample_size', '3', '--sample_seed', '42',
        '--sample_indices_file', str(indices_path),
        '--llm_name', 'qwen3-8b', '--llm_base_url', args.llm_base_url,
        '--embedding_batch_size', '4', '--openie_max_workers', '8',
        '--llm_prefetch_workers', '8', '--eval_mode', 'retrieve',
        '--retrieval_top_k', '200', '--result_top_k', '10',
        '--candidate_output_top_k', '200', '--reuse_index',
        '--save_dir', str(index_root),
    ]
    specifications = []
    for name in ('hipporag2', 'exp4_dependency_binding'):
        case = smoke_root / name
        output = case / 'result.json'
        if name == 'hipporag2':
            command = [args.python, '-B', '-u', str(Path(args.hippo_root) / 'main.py')]
            command += common + [
                '--datasets_dir', str(args.datasets_dir), '--rag_type', 'hipporag',
                '--embedding_name', EMBEDDING_MODEL, '--embedding_provider', 'transformers',
                '--save_dir_exact', '--output', str(output),
            ]
            # Native main.py fixes max_new_tokens=2048; its parser has no flag
            # for changing that budget. The result validator checks the value.
        else:
            command = [args.python, '-B', '-u', str(ROOT / 'scripts/eval_dataset.py')]
            command += common + EXP4_ARGUMENTS + [
                '--data_path', str(Path(args.datasets_dir) / 'musique.json'),
                '--corpus_path', str(Path(args.datasets_dir) / 'musique_corpus.json'),
                '--corpus_mode', 'full', '--qa_top_k', '5', '--max_qa_steps', '1',
                '--max_new_tokens', '2048', '--embedding_model_name', EMBEDDING_MODEL,
                '--output', str(output), '--stratified_eval',
                '--stratified_output', str(case / 'stratified.json'),
            ]
        specifications.append({'name': name, 'command': command, 'output': output,
                               'log': Path(args.out_root) / 'logs' / f'retrieval_smoke_{name}.log'})
    return specifications


def validate_case(result, name, data, corpus, hops):
    """Reuse exact-string export/Recall validation from the full experiment."""
    protocol = {
        'selected_indices': SELECTED_INDICES,
        'benchmark_hops': [hops[index] for index in SELECTED_INDICES],
        'hop_distribution': {},
        'runtime': {'embedding_batch_size': 4, 'openie_max_workers': 8,
                    'llm_prefetch_workers': 8, 'embedding_model_name': EMBEDDING_MODEL},
    }
    for hop in protocol['benchmark_hops']:
        key = str(hop)
        protocol['hop_distribution'][key] = protocol['hop_distribution'].get(key, 0) + 1
    if result.get('eval_mode') != 'retrieve' or result.get('qa_metrics'):
        raise ValueError('Smoke validation must perform retrieval only')
    measurements, metrics = validate_result(result, protocol, name, data, corpus)
    stats = result.get('llm_request_stats')
    if (not isinstance(stats, dict) or type(stats.get('failures')) is not int
            or stats['failures'] != 0 or stats.get('max_in_flight') != 8):
        raise ValueError('Missing LLM statistics, nonzero failures or incorrect concurrency cap')
    indexed_count = result.get('n_docs') if name == 'hipporag2' else result.get('indexed_docs')
    if indexed_count != len(corpus):
        raise ValueError('Smoke must use the complete MuSiQue corpus')
    return {'complete': True, 'n_samples': len(measurements),
            'retrieval_metrics': metrics, 'llm_request_stats': stats,
            'per_question': measurements, 'result_top_k': 10, 'candidate_output_top_k': 200,
            'indexed_docs': indexed_count}


def run(args):
    out = Path(args.out_root).resolve()
    index_root = Path(args.index_root).resolve() if args.index_root else out / 'repaired_index'
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Smoke outputs must stay under PathCondRAG/outputs')
    if index_root == out or out.is_relative_to(index_root):
        raise ValueError('Smoke output root must not overwrite or live inside the repaired index')
    validation = read_json(out / 'validation_report.json')
    if not validation.get('complete') or validation.get('repaired_index') != str(index_root):
        raise ValueError('Run index validation before the retrieval smoke checks')
    before = asset_hashes(index_root)
    if validation.get('asset_sha256') != before:
        raise ValueError('Repaired index changed since its successful validation')
    manifest = read_json(index_root / MODEL_DIR / 'index_manifest.json')
    if (manifest.get('text_normalization') != 'unicode_alnum_casefold_v1'
            or manifest.get('embedding', {}).get('model_name') != EMBEDDING_MODEL
            or not manifest.get('openie', {}).get('quality_repair')):
        raise ValueError('Smoke input must be the declared repaired Unicode Qwen index')
    data, corpus, hops = validated_dataset(
        Path(args.datasets_dir) / 'musique.json', Path(args.datasets_dir) / 'musique_corpus.json')
    if any(index >= len(data) for index in SELECTED_INDICES):
        raise ValueError('The requested original dataset indices are unavailable')
    smoke_root = out / 'retrieval_smoke'
    indices_path = smoke_root / 'selected_indices.json'
    write_json(indices_path, SELECTED_INDICES)
    final_report = out / 'retrieval_smoke_report.json'
    if final_report.exists():
        final_report.unlink()
    specifications = build_commands(args, indices_path, index_root, smoke_root)
    reports = {}
    for specification in specifications:
        name, output, logfile = (specification[key] for key in ('name', 'output', 'log'))
        output.parent.mkdir(parents=True, exist_ok=True)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        # A failed subprocess must never cause an old successful result to be
        # reported as the current validation.
        if output.exists():
            output.unlink()
        LOG.info('[smoke] %s samples=%s; batch=4 workers=8 retrieve-only', name, SELECTED_INDICES)
        started = time.monotonic()
        with logfile.open('w', encoding='utf-8') as stream:
            subprocess.run(specification['command'], check=True, cwd=str(ROOT),
                           env=child_environment(), stdout=stream, stderr=subprocess.STDOUT)
        report = validate_case(read_json(output), name, data, corpus, hops)
        report.update(seconds=time.monotonic() - started, result_path=str(output), log_path=str(logfile))
        reports[name] = report
        if asset_hashes(index_root) != before:
            raise ValueError(f'{name} mutated frozen repaired graph/vector/OpenIE assets')
        LOG.info('[smoke done] %s Recall@5=%s failures=0', name, report['retrieval_metrics']['Recall@5'])
    final = {'complete': True, 'repaired_index': str(index_root),
             'selected_indices': SELECTED_INDICES, 'cases': reports,
             'graph_vectors_openie_unchanged': True, 'asset_sha256': before,
             'purpose': 'three affected-question integration check; not a full-benchmark improvement claim'}
    write_json(final_report, final)
    LOG.info('[validated] both native consumers loaded the same repaired index')
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', default=str(OUT_DEFAULT))
    parser.add_argument('--index-root', default='', help='Default: OUT_ROOT/repaired_index')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--datasets-dir', default='/root/datasets')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--python', default=sys.executable, help='Interpreter with the rag dependencies')
    args = parser.parse_args(argv)
    args.out_root = str(Path(args.out_root).resolve())
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(args)
    return 0
