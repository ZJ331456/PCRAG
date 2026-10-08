"""Full exp4 + plan_prune + dag_package retrieval using frozen public indexes.

Use --smoke for two questions per dataset in an isolated disposable directory.
Both modes keep Qwen3-Embedding-8B batch=4, LLM workers=8, benchmark hops,
2048 generation tokens and Top10/Top200 exports. No indexing or QA is run.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
from dataclasses import asdict
from copy import copy
from pathlib import Path

from . import exp4_improvements as previous
from . import exp4_round2 as round2
from .common import read_json, write_json

CASE_NAME = 'exp4_dependency_binding_plan_prune_dag_package'
SMOKE_NAME = '_plan_prune_dag_package_smoke'
VARIANT = round2.Variant(
    'plan_prune_dag_package', ('planning', 'plan_prune', 'dag_package'),
    plan_validation='canonical_refs', plan_routing='question_structure',
)
SMOKE_INDICES = {'hotpotqa': [197, 572], '2wikimultihopqa': [180, 312], 'musique': [93, 567]}
PARENT_FIELDS = ('plan', 'bindings', 'branch_scores', 'routes', 'search_count',
                 'llm_plan_calls', 'llm_verification_calls', 'planning_outputs', 'verification_outputs')
LOG = logging.getLogger('plan_prune_dag_package_full')


def validate_dag_case(case):
    """Check that the enabled finalizer executed and preserved its parent invariants."""
    result = read_json(case / 'result.json')
    promoted = 0
    for row in result['results']:
        diagnostic = row['retrieval_trace']['evidence'].get('improvement_dag_package') or {}
        if (diagnostic.get('enabled') is not True or diagnostic.get('extra_requests') != 0
                or diagnostic.get('gold_labels_used') is not False):
            raise ValueError(f"DAG finalizer missing or invalid on question {row['query_index']}")
        for key in ('top2_preserved', 'top200_set_preserved', 'document_set_preserved', 'unique_documents'):
            if diagnostic.get(key) is not True:
                raise ValueError(f"DAG invariant {key} failed on question {row['query_index']}")
        promoted += bool(diagnostic.get('promotions'))
    return {'validated': True, 'question_count': len(result['results']),
            'questions_with_promotions': promoted, 'direct_extra_llm_requests': 0}


def historical_comparison(out, dataset, case, indices, report):
    """Compare matched historical plan_prune records; do not claim a fresh control."""
    reference = out / 'cases' / dataset / 'exp4_dependency_binding_plan_prune'
    if not (reference / 'validated.ok').is_file():
        return {'available': False, 'reason': 'Validated historical plan_prune export is missing'}
    raw = read_json(reference / 'result.json')
    rows = {row['query_index']: row for row in raw['results']}
    if any(index not in rows for index in indices):
        raise ValueError(f'{dataset}: historical plan_prune question subset is incomplete')
    saved = previous.verify_trial_report(reference)
    measurements = {row['query_index']: row for row in saved['per_question']}
    matched = previous.aggregate_measurements([measurements[index] for index in indices])
    current = read_json(case / 'result.json')
    differences = []
    for row in current['results']:
        old = rows[row['query_index']]['retrieval_trace']['evidence']
        new = row['retrieval_trace']['evidence']
        changed = [field for field in PARENT_FIELDS if old.get(field) != new.get(field)]
        if changed:
            differences.append({'query_index': row['query_index'], 'fields': changed})
    return {
        'available': True, 'reference_result': str(reference / 'result.json'),
        'reference_result_sha256': previous.experiments.sha256(reference / 'result.json'),
        'scope': 'Historical plan_prune on identical question indices; not a fresh paired LLM control',
        'matched_samples': len(indices), 'reference_metrics': matched['retrieval_metrics'],
        'delta_pp': {key: (value - matched['retrieval_metrics'][key]) * 100
                     for key, value in report['retrieval_metrics'].items()},
        'compared_parent_fields': list(PARENT_FIELDS),
        'parent_outputs_identical': not differences,
        'parent_output_different_questions': differences,
    }


def publish_summary(work, reports, statuses):
    lines = [
        '| Dataset | Case | R@1 | R@2 | R@5 | R@10 | R@20 | R@200 | All gold@5 | Seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset, report in reports.items():
        values = [f"{report['retrieval_metrics'][f'Recall@{k}']:.4f}" for k in (1, 2, 5, 10, 20, 200)]
        lines.append(f'| {dataset} | {CASE_NAME} | ' + ' | '.join(values)
                     + f" | {report['all_gold_top5']:.4f} | {report['seconds']} |")
    summary = {
        'case': CASE_NAME, 'variant': asdict(VARIANT), 'stages': statuses,
        'datasets': {dataset: {key: value for key, value in report.items() if key != 'per_question'}
                     for dataset, report in reports.items()},
    }
    write_json(work / 'metadata' / f'{CASE_NAME}_comparison.json', summary)
    (work / 'metadata' / f'{CASE_NAME}_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def run(args):
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(previous.ROOT / 'outputs') or out == previous.ROOT / 'outputs':
        raise ValueError('Output must be a dedicated directory under PathCondRAG/outputs')
    if not (out / 'shared_indexes').is_dir():
        raise ValueError(f'Missing public shared indexes: {out / "shared_indexes"}')
    if shutil.disk_usage(out).free < 3 * 1024 ** 3:
        raise ValueError('At least 3 GiB free disk is required')
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('vLLM service is not ready')
    if not Path(args.vllm_log).is_file():
        raise ValueError(f'vLLM request log is missing: {args.vllm_log}')

    work = out / SMOKE_NAME if args.smoke else out
    support = work / '_plan_prune_dag_package_support'
    if work.is_symlink() or support.is_symlink():
        raise ValueError('Refusing symlinked workflow directories')
    support.mkdir(parents=True, exist_ok=True)
    (work / 'metadata').mkdir(exist_ok=True)
    before_code = previous.code_hashes(Path(args.hippo_root))
    contexts = {dataset: previous.context_for(out, dataset, support) for dataset in previous.DATASETS}
    for context in contexts.values():
        # Link graph/vector assets from the public index, never rebuild them.
        context['clone_source'] = context['source']
    embedding_runtime = next(iter(contexts.values()))['manifest']['runtime']

    def selected_for(dataset, context):
        if not args.smoke:
            return list(context['manifest']['selected_indices'])
        preferred = SMOKE_INDICES[dataset]
        # Full-index smoke keeps the historical two-question fixtures; tiny
        # disposable smoke indexes fall back to the shared-index sample.
        if max(preferred) < len(context['data']):
            return preferred
        return list(context['manifest']['selected_indices'])

    protocol = {
        'case': CASE_NAME, 'variant': dict(asdict(VARIANT), flags=list(VARIANT.flags)),
        'smoke': args.smoke,
        'shared_indexes': str(out / 'shared_indexes'), 'datasets': list(previous.DATASETS),
        'embedding_model': embedding_runtime['embedding_model_name'],
        'embedding_provider': embedding_runtime.get('embedding_provider'),
        'embedding_batch_size': embedding_runtime['embedding_batch_size'],
        'llm_prefetch_workers': 8, 'openie_max_workers': 8,
        'hop_source': 'benchmark', 'max_new_tokens': 2048, 'thinking': False,
        'result_top_k': 10, 'candidate_output_top_k': 200, 'eval_mode': 'retrieve',
        'cache_policy': 'Private original-exp4 warm-cache snapshot, matching the existing full plan_prune runner',
        'selected_indices': {dataset: selected_for(dataset, context)
                             for dataset, context in contexts.items()},
    }
    protocol_file = work / 'metadata' / f'{CASE_NAME}_protocol.json'
    if protocol_file.exists() and read_json(protocol_file) != protocol:
        raise ValueError('Existing workflow protocol differs; inspect before retrying')
    write_json(protocol_file, protocol)
    if 'NV-Embed' in str(embedding_runtime.get('embedding_model_name', '')):
        from .multi_dataset_retrieval import environment as nv_environment
        nv_args = copy(args)
        nv_args.embedding_model = embedding_runtime['embedding_model_name']
        nv_args.embedding_provider = embedding_runtime.get('embedding_provider', 'nvembed')
        env = nv_environment(nv_args)
    else:
        env = previous.environment(args)
    env['PYTHONHASHSEED'] = '42'
    reports, statuses = {}, {}
    for dataset, context in contexts.items():
        indices = protocol['selected_indices'][dataset]
        if not args.smoke and set(indices) != set(range(len(context['data']))):
            raise ValueError(f'{dataset}: manifest does not cover the entire dataset')
        case = work / 'cases' / dataset / CASE_NAME
        try:
            print(f'[{"smoke" if args.smoke else "full"}-start] {dataset}/{CASE_NAME} '
                  f'n={len(indices)} flags={",".join(VARIANT.flags)}', flush=True)
            report = round2.run_case(args, context, case, indices, VARIANT, env, cleanup=True)
            dag_validation = validate_dag_case(case)
            write_json(case / 'dag_validation.json', dag_validation)
            comparison = historical_comparison(out, dataset, case, indices, report)
            write_json(case / 'comparison_vs_plan_prune.json', comparison)
            reports[dataset] = dict(report, comparison_vs_plan_prune=comparison,
                                   dag_validation=dag_validation)
            statuses[dataset] = {'state': 'ready', 'n_samples': report['n_samples'],
                                 'seconds': report['seconds']}
        except Exception as error:
            LOG.exception('[failed] %s', dataset)
            (case / 'validated.ok').unlink(missing_ok=True)
            statuses[dataset] = {'state': 'failed', 'error': f'{type(error).__name__}: {error}'}
        write_json(work / 'metadata' / f'{CASE_NAME}_stage_status.json', statuses)
        publish_summary(work, reports, statuses)

    if previous.code_hashes(Path(args.hippo_root)) != before_code:
        raise ValueError('Original HippoRAG source changed during retrieval')
    for context in contexts.values():
        actual = previous.experiments.asset_hashes(context['source'], context['manifest']['model_dir'])
        if actual != context['manifest']['source_asset_sha256']:
            raise ValueError('Public frozen source index changed during retrieval')
    if any(stage['state'] != 'ready' for stage in statuses.values()):
        return 1
    write_json(work / 'metadata' / f'{CASE_NAME}_completed.ok', {
        'case': CASE_NAME, 'smoke': args.smoke,
        'n_samples': {dataset: report['n_samples'] for dataset, report in reports.items()},
    })
    LOG.info('[done] all three %s retrieval cases validated under %s',
             'smoke' if args.smoke else 'full', work)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', default=str(previous.DEFAULT_OUT))
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--runtime-deps', default=str(previous.DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=str(previous.DEFAULT_OUT / 'logs/vllm.log'))
    parser.add_argument('--smoke', action='store_true', help='Check two questions per dataset in a disposable namespace')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)
