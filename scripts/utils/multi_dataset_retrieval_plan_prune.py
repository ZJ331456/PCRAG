"""Full three-dataset retrieval for exp4 + plan_prune on frozen shared indexes.

Reuses the public indexes under shared_indexes/ and writes validated cases to
cases/<dataset>/exp4_dependency_binding_plan_prune. Configuration matches the
round2 plan_prune profile: planning + plan_prune with question_structure routing
and canonical_refs validation.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
from dataclasses import asdict
from pathlib import Path

from . import exp4_improvements as previous
from . import exp4_round2 as round2
from .common import read_json, write_json

ROOT = previous.ROOT
DATASETS = previous.DATASETS
DEFAULT_OUT = previous.DEFAULT_OUT
DEFAULT_RUNTIME = previous.DEFAULT_RUNTIME
CASE_NAME = 'exp4_dependency_binding_plan_prune'
SUPPORT_NAME = '_plan_prune_full_support'
VARIANT = round2.Variant(
    'plan_prune',
    ('planning', 'plan_prune'),
    plan_validation='canonical_refs',
    plan_routing='question_structure',
)
LOG = logging.getLogger('multi_dataset_retrieval_plan_prune')


def publish_summary(out, reports, statuses):
    lines = [
        '| Dataset | Case | R@1 | R@2 | R@5 | R@10 | R@20 | R@200 | All gold@5 | Seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    comparison = {'case': CASE_NAME, 'variant': asdict(VARIANT), 'datasets': {}, 'stages': statuses}
    for dataset in DATASETS:
        report = reports.get(dataset)
        if not report:
            continue
        metrics = report['retrieval_metrics']
        values = [f"{metrics[f'Recall@{k}']:.4f}" for k in (1, 2, 5, 10, 20, 200)]
        lines.append(
            f'| {dataset} | {CASE_NAME} | ' + ' | '.join(values)
            + f" | {report['all_gold_top5']:.4f} | {report['seconds']} |")
        baseline_path = out / 'cases' / dataset / 'exp4_dependency_binding' / 'report.json'
        baseline = read_json(baseline_path) if baseline_path.is_file() else None
        deltas = {}
        if baseline:
            for key in ('Recall@1', 'Recall@2', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@200'):
                deltas[key] = (metrics[key] - baseline['retrieval_metrics'][key]) * 100
        comparison['datasets'][dataset] = {
            'plan_prune': {k: v for k, v in report.items() if k != 'per_question'},
            'delta_pp_vs_exp4': deltas,
        }
    write_json(out / 'metadata' / f'{CASE_NAME}_comparison.json', comparison)
    markdown = out / 'metadata' / f'{CASE_NAME}_comparison.md'
    markdown.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def run(args):
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(ROOT / 'outputs') or out == ROOT / 'outputs':
        raise ValueError('Output must be a dedicated directory under PathCondRAG/outputs')
    if not (out / 'shared_indexes').is_dir():
        raise ValueError(f'Missing shared indexes under {out / "shared_indexes"}')
    free = shutil.disk_usage(out).free
    if free < 3 * 1024 ** 3:
        raise ValueError('At least 3 GiB of free disk is needed for private case indexes')
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('vLLM service is not ready')
    if not Path(args.vllm_log).is_file():
        raise ValueError(f'vLLM request log is missing: {args.vllm_log}')

    support = out / SUPPORT_NAME
    if support.is_symlink():
        raise ValueError('Refusing a symlinked support directory')
    support.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    (out / 'metadata').mkdir(exist_ok=True)

    protocol = {
        'case': CASE_NAME,
        'variant': asdict(VARIANT),
        'datasets': list(DATASETS),
        'reuse_shared_indexes': True,
        'shared_indexes': str(out / 'shared_indexes'),
        'baseline_cache_policy': (
            'private identical snapshot of original exp4 completed cache; timings are warm-cache'),
        'index_cleanup_policy': (
            'Private indexes removed after validation; Top200/Top10/traces/reports retained'),
    }
    protocol_file = out / 'metadata' / f'{CASE_NAME}_protocol.json'
    if protocol_file.exists() and read_json(protocol_file) != protocol:
        raise ValueError('Existing plan_prune full protocol differs; inspect before retrying')
    write_json(protocol_file, protocol)

    before_code = previous.code_hashes(Path(args.hippo_root))
    env = previous.environment(args)
    env['PYTHONHASHSEED'] = '42'
    contexts = {dataset: previous.context_for(out, dataset, support) for dataset in DATASETS}

    statuses, reports = {}, {}
    status_file = out / 'metadata' / f'{CASE_NAME}_stage_status.json'
    completed_marker = out / 'metadata' / f'{CASE_NAME}_completed.ok'

    def record(stage, state, **details):
        statuses[stage] = {'state': state, **details}
        write_json(status_file, statuses)
        LOG.info('[stage] %s %s %s', stage, state, details)

    for dataset, context in contexts.items():
        stage = f'{dataset}/{CASE_NAME}'
        case = out / 'cases' / dataset / CASE_NAME
        indices = list(context['manifest']['selected_indices'])
        try:
            print(f'[full-start] {dataset}/{CASE_NAME} n={len(indices)} '
                  f'flags={",".join(VARIANT.flags)} '
                  f'routing={VARIANT.plan_routing} validation={VARIANT.plan_validation}',
                  flush=True)
            report = round2.run_case(args, context, case, indices, VARIANT, env, cleanup=True)
            reports[dataset] = report
            record(stage, 'ready', seconds=report.get('seconds'), n_samples=report.get('n_samples'),
                   recall_at_5=report['retrieval_metrics']['Recall@5'],
                   recall_at_10=report['retrieval_metrics']['Recall@10'])
        except Exception as error:
            LOG.exception('[failed] %s', stage)
            record(stage, 'failed', error=f'{type(error).__name__}: {error}')

    if previous.code_hashes(Path(args.hippo_root)) != before_code:
        raise ValueError('Original HippoRAG source changed during plan_prune full retrieval')
    for context in contexts.values():
        expected = context['manifest']['source_asset_sha256']
        actual = previous.experiments.asset_hashes(context['source'], context['manifest']['model_dir'])
        if actual != expected:
            raise ValueError('Public frozen source index changed during plan_prune full retrieval')

    publish_summary(out, reports, statuses)
    failed = {stage: value for stage, value in statuses.items() if value['state'] != 'ready'}
    if failed:
        write_json(out / 'metadata' / f'{CASE_NAME}_failed_stages.json', failed)
        LOG.error('[incomplete] %s stage(s) failed; see metadata/%s_failed_stages.json',
                  len(failed), CASE_NAME)
        return 1
    write_json(completed_marker, {
        'datasets': list(DATASETS),
        'case': CASE_NAME,
        'variant': asdict(VARIANT),
        'n_samples': {dataset: reports[dataset]['n_samples'] for dataset in DATASETS},
    })
    (out / 'metadata' / f'{CASE_NAME}_failed_stages.json').unlink(missing_ok=True)
    LOG.info('[done] full plan_prune retrieval validated for all datasets under %s', out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', default=str(DEFAULT_OUT))
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--runtime-deps', default=str(DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=str(DEFAULT_OUT / 'logs/vllm.log'))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)
