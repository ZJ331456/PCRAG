"""Retrieve all three datasets with support protection and semantic veto.

The shared indexes are read only. Use --smoke for two questions per dataset
under --smoke-out-root; the same configuration is used for the full run.
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
from . import multi_dataset_retrieval_plan_prune_dag_package as dag_runner
from .common import read_json, write_json
from .exp4_http_audit import locate


CASE_NAME = 'exp4_dependency_binding_plan_prune_dag_package_support_semantic_veto'
REFERENCE_CASE = dag_runner.CASE_NAME
SMOKE_NAME = '_support_semantic_veto_smoke'
SUPPORT_NAME = '_support_semantic_veto_support'
SMOKE_INDICES = {'hotpotqa': [145, 339], '2wikimultihopqa': [161, 532], 'musique': [716, 889]}
VARIANT = round2.Variant(
    'plan_prune_dag_package_support_semantic_veto',
    ('planning', 'plan_prune', 'dag_package', 'support_semantic_veto'),
    plan_validation='canonical_refs', plan_routing='question_structure',
)
LOG = logging.getLogger('support_semantic_veto_full')


def validate_finalizer(case, result=None):
    """Require the opt-in finalizer to run on every exported question."""
    result = read_json(case / 'result.json') if result is None else result
    dag_promotions = 0
    base_proposals = 0
    support_vetoes = 0
    semantic_vetoes = 0
    promotions = 0
    requests = 0
    for row in result['results']:
        dag = row['retrieval_trace']['evidence'].get('improvement_dag_package') or {}
        if (dag.get('enabled') is not True or dag.get('extra_requests') != 0
                or dag.get('gold_labels_used') is not False):
            raise ValueError(f"DAG finalizer missing on question {row['query_index']}")
        for invariant in ('top2_preserved', 'top200_set_preserved', 'document_set_preserved', 'unique_documents'):
            if dag.get(invariant) is not True:
                raise ValueError(f"DAG invariant {invariant} failed on question {row['query_index']}")
        dag_promotions += bool(dag.get('promotions'))
        diagnostic = row['retrieval_trace']['evidence'].get('improvement_support_semantic_veto') or {}
        if diagnostic.get('enabled') is not True or diagnostic.get('gold_labels_used') is not False:
            raise ValueError(f"Support/semantic finalizer missing on question {row['query_index']}")
        for invariant in ('top2_preserved', 'top10_set_preserved',
                          'top200_set_preserved', 'unique_documents', 'suffix_preserved'):
            if diagnostic.get(invariant) is not True:
                raise ValueError(f"Support/semantic invariant {invariant} failed on question {row['query_index']}")
        count = diagnostic.get('extra_requests')
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 2:
            raise ValueError(f"Unexpected extra request count on question {row['query_index']}: {count}")
        requests += count
        promotions += bool(diagnostic.get('promotions'))
        proposed = bool((diagnostic.get('base_proposal') or {}).get('promotions'))
        base_proposals += proposed
        support_vetoes += proposed and bool((diagnostic.get('support_guard') or {}).get('veto_reason'))
        semantic_vetoes += proposed and (diagnostic.get('independent_verification') or {}).get('approved') is not True
    return {'validated': True, 'question_count': len(result['results']),
            'questions_with_promotions': promotions,
            'base_proposals': base_proposals, 'support_vetoes': support_vetoes,
            'semantic_vetoes': semantic_vetoes, 'logical_extra_llm_requests': requests,
            'dag_validation': {'validated': True, 'question_count': len(result['results']),
                               'questions_with_promotions': dag_promotions, 'direct_extra_llm_requests': 0}}


def paired_parent_comparison(context, case, report, result=None):
    """Evaluate the exact pre-finalizer ranking; labels enter only this audit."""
    from .exp4_anchor_companion_trials import evaluate

    result = read_json(case / 'result.json') if result is None else result
    parent_rows, final_rows, changes = [], [], []
    for row in result['results']:
        index = row['query_index']
        diagnostic = row['retrieval_trace']['evidence']['improvement_support_semantic_veto']
        old = diagnostic.get('original_parent_doc_ids')
        new = diagnostic.get('final_doc_ids')
        candidates = row['candidate_docs']
        if (not isinstance(old, list) or len(old) != len(candidates) or len(set(old)) != len(old)
                or any(type(doc_id) is not int or doc_id < 0 for doc_id in old)
                or not isinstance(new, list) or len(new) != len(candidates)
                or len(set(new)) != len(new) or set(new) != set(old)
                or any(type(doc_id) is not int for doc_id in new)
                or len(set(candidates)) != len(candidates)):
            raise ValueError(f'Question {index}: invalid original physical source-ID ranking')
        by_id = dict(zip(new, candidates))
        expected = list(old)
        promotions = diagnostic.get('promotions')
        if diagnostic.get('accepted') is True:
            if not isinstance(promotions, list) or len(promotions) != 1:
                raise ValueError(f'Question {index}: accepted finalizer must make exactly one swap')
            promoted, victim = promotions[0]['doc_id'], promotions[0]['victim']
            a, b = old.index(promoted), old.index(victim)
            if not 5 <= a < 10 or not 2 <= b < 5:
                raise ValueError(f'Question {index}: swap exceeded its documented prefix scope')
            expected[a], expected[b] = expected[b], expected[a]
        elif promotions != []:
            raise ValueError(f'Question {index}: abstention retained a promotion')
        if (new != expected or new[:2] != old[:2] or set(new[:10]) != set(old[:10])
                or new[10:] != old[10:] or len(new) != len(set(new))):
            raise ValueError(f'Question {index}: exported documents differ from the recorded physical-ID swap')
        protected = (diagnostic.get('support_guard') or {}).get('protected_doc_ids', [])
        if not set(protected) <= set(new[:5]):
            raise ValueError(f'Question {index}: exported Top5 lost protected source support')
        measured_row = dict(row, dataset=context['dataset'])
        parent = evaluate(context['data'], measured_row, [by_id[doc_id] for doc_id in old])
        final = evaluate(context['data'], measured_row, candidates)
        for metric in ('Recall@1', 'Recall@2', 'Recall@10', 'Recall@20', 'Recall@200'):
            if parent['metrics'][metric] != final['metrics'][metric]:
                raise ValueError(f'Question {index}: finalizer changed protected {metric}')
        if parent['all_gold_top10'] != final['all_gold_top10']:
            raise ValueError(f'Question {index}: finalizer changed complete Top10 evidence')
        parent_rows.append(parent)
        final_rows.append(final)
        if new != old:
            changes.append({'query_index': index, 'parent_top5_doc_ids': old[:5],
                            'final_top5_doc_ids': new[:5],
                            'recall5_delta': final['metrics']['Recall@5'] - parent['metrics']['Recall@5'],
                            'all_gold_top5_delta': int(final['all_gold_top5']) - int(parent['all_gold_top5'])})
    parent, final = previous.aggregate_measurements(parent_rows), previous.aggregate_measurements(final_rows)
    if any(abs(value - report['retrieval_metrics'][metric]) > 1e-10
           for metric, value in final['retrieval_metrics'].items()):
        raise ValueError('Paired final metrics differ from the independently validated retrieval report')
    comparison = {
        'validated': True, 'scope': 'Exact real parent ranking from this same retrieval run; no repeated LLM calls',
        'labels_used_by_selector': False, 'labels_used_only_for_offline_evaluation': True,
        'result_sha256': previous.experiments.sha256(case / 'result.json'),
        'n_samples': len(parent_rows), 'parent_metrics': parent['retrieval_metrics'],
        'final_metrics': final['retrieval_metrics'],
        'delta_pp': {metric: (final['retrieval_metrics'][metric] - value) * 100
                     for metric, value in parent['retrieval_metrics'].items()},
        'parent_all_gold': {key: parent[key] for key in ('all_gold_top5', 'all_gold_top10')},
        'final_all_gold': {key: final[key] for key in ('all_gold_top5', 'all_gold_top10')},
        'all_gold_delta_pp': {key: (final[key] - parent[key]) * 100
                              for key in ('all_gold_top5', 'all_gold_top10')},
        'protected_metrics_identical': True,
        'recall5_wins': sum(row['recall5_delta'] > 0 for row in changes),
        'recall5_losses': sum(row['recall5_delta'] < 0 for row in changes),
        'recall5_neutral_swaps': sum(row['recall5_delta'] == 0 for row in changes),
        'changes': changes, 'parent_per_question': parent_rows,
    }
    comparison['regression_limit_0_01_met'] = (
        all(value >= -1 - 1e-8 for value in comparison['delta_pp'].values())
        and all(value >= -1 - 1e-8 for value in comparison['all_gold_delta_pp'].values()))
    return comparison


def retain_http_audit(case, report):
    """Keep request-log bytes when the service stdout exists only in /proc."""
    segment = report['vllm_log_slice']
    retained = case / 'vllm_http.log'
    audit_file = case / 'http_audit.json'
    if audit_file.exists():
        saved = read_json(audit_file)
        if (saved.get('original_log_segment') != segment
                or saved.get('http_status_in_log') != report['http_status_in_log']
                or not retained.is_file()
                or saved.get('retained_log_sha256') != previous.experiments.sha256(retained)):
            raise ValueError('Retained HTTP audit proof changed')
        return
    with Path(segment['path']).open('rb') as stream:
        stream.seek(segment['start'])
        blob = stream.read(segment['end'] - segment['start'])
    if len(blob) != segment['end'] - segment['start']:
        raise ValueError('vLLM log no longer contains the recorded request segment')
    if retained.exists() and retained.read_bytes() != blob:
        raise ValueError('Retained HTTP audit segment differs from the original')
    retained.write_bytes(blob)
    write_json(audit_file, {
        'original_log_segment': segment, 'retained_log': str(retained),
        'retained_log_sha256': previous.experiments.sha256(retained),
        'http_status_in_log': report['http_status_in_log'],
    })


def historical_comparison(out, dataset, case, indices, report, result=None):
    """Describe matched historical DAG metrics and upstream output differences."""
    reference = out / 'cases' / dataset / REFERENCE_CASE
    if not (reference / 'validated.ok').is_file():
        return {'available': False, 'reason': f'Validated historical {REFERENCE_CASE} is missing'}
    saved = previous.verify_trial_report(reference)
    measurements = {row['query_index']: row for row in saved['per_question']}
    raw = read_json(reference / 'result.json')
    rows = {row['query_index']: row for row in raw['results']}
    if any(index not in rows or index not in measurements for index in indices):
        raise ValueError(f'{dataset}: historical DAG question subset is incomplete')
    matched = previous.aggregate_measurements([measurements[index] for index in indices])
    differences = []
    result = read_json(case / 'result.json') if result is None else result
    for row in result['results']:
        old = rows[row['query_index']]['retrieval_trace']['evidence']
        new = row['retrieval_trace']['evidence']
        changed = [field for field in dag_runner.PARENT_FIELDS if old.get(field) != new.get(field)]
        if changed:
            differences.append({'query_index': row['query_index'], 'fields': changed})
    return {
        'available': True, 'reference_case': REFERENCE_CASE,
        'reference_result': str(reference / 'result.json'),
        'reference_result_sha256': previous.experiments.sha256(reference / 'result.json'),
        'scope': 'Historical DAG export on matched indices; not a fresh paired LLM control',
        'matched_samples': len(indices), 'reference_metrics': matched['retrieval_metrics'],
        'delta_pp': {metric: (value - matched['retrieval_metrics'][metric]) * 100
                     for metric, value in report['retrieval_metrics'].items()},
        'all_gold_delta_pp': {key: (report[key] - matched[key]) * 100
                              for key in ('all_gold_top5', 'all_gold_top10')},
        'compared_parent_fields': list(dag_runner.PARENT_FIELDS),
        'parent_outputs_identical': not differences,
        'parent_output_different_questions': differences,
    }


def publish_summary(work, reports, statuses):
    lines = [
        '| Dataset | R@1 | R@2 | R@5 | R@10 | R@20 | R@200 | All gold@5 | All gold@10 | Seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset, report in reports.items():
        values = [f"{report['retrieval_metrics'][f'Recall@{k}']:.4f}" for k in (1, 2, 5, 10, 20, 200)]
        lines.append(f'| {dataset} | ' + ' | '.join(values)
                     + f" | {report['all_gold_top5']:.4f} | {report['all_gold_top10']:.4f} | {report['seconds']} |")
    write_json(work / 'metadata' / f'{CASE_NAME}_comparison.json', {
        'case': CASE_NAME, 'variant': asdict(VARIANT), 'stages': statuses,
        'datasets': {dataset: previous.compact(report) for dataset, report in reports.items()},
    })
    (work / 'metadata' / f'{CASE_NAME}_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def run(args):
    out = Path(args.out_root).resolve()
    outputs = previous.ROOT / 'outputs'
    if not out.is_relative_to(outputs) or out == outputs:
        raise ValueError('Output must be a dedicated directory under PathCondRAG/outputs')
    if not (out / 'shared_indexes').is_dir():
        raise ValueError(f'Missing frozen shared indexes: {out / "shared_indexes"}')
    if shutil.disk_usage(out).free < 3 * 1024 ** 3:
        raise ValueError('At least 3 GiB free disk is required')
    if not args.smoke and args.smoke_out_root:
        raise ValueError('--smoke-out-root requires --smoke')
    work = (Path(args.smoke_out_root).resolve() if args.smoke_out_root else out / SMOKE_NAME) if args.smoke else out
    if args.smoke and (not work.is_relative_to(outputs) or work in (outputs, out)
                       or any(work.is_relative_to(out / name) for name in ('cases', 'shared_indexes', 'metadata', 'logs'))):
        raise ValueError('Smoke output must be a dedicated disposable directory, separate from production artifacts')
    support = work / SUPPORT_NAME
    if work.is_symlink() or support.is_symlink():
        raise ValueError('Refusing symlinked workflow directories')
    args.vllm_log = args.vllm_log or locate(args.llm_base_url)
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('vLLM service is not ready')
    if not Path(args.vllm_log).is_file():
        raise ValueError(f'vLLM request log is missing: {args.vllm_log}')
    support.mkdir(parents=True, exist_ok=True)
    (work / 'metadata').mkdir(exist_ok=True)
    before_code = previous.code_hashes(Path(args.hippo_root))
    contexts = {dataset: previous.context_for(out, dataset, support) for dataset in previous.DATASETS}
    runtime = next(iter(contexts.values()))['manifest']['runtime']
    if int(runtime['embedding_batch_size']) <= 0:
        raise ValueError('The frozen index must record a positive embedding batch size')
    selected = {}
    for dataset, context in contexts.items():
        if context['manifest']['runtime'] != runtime:
            raise ValueError(f'{dataset}: dataset embedding runtime differs')
        context['clone_source'] = context['source']
        preferred = SMOKE_INDICES[dataset]
        indices = list(preferred if args.smoke and max(preferred) < len(context['data'])
                       else context['manifest']['selected_indices'])
        if len(indices) != len(set(indices)) or any(not 0 <= index < len(context['data']) for index in indices):
            raise ValueError(f'{dataset}: invalid selected question indices')
        if not args.smoke and set(indices) != set(range(len(context['data']))):
            raise ValueError(f'{dataset}: index manifest does not cover the entire dataset')
        selected[dataset] = indices
    protocol = {
        'case': CASE_NAME, 'variant': dict(asdict(VARIANT), flags=list(VARIANT.flags)),
        'smoke': args.smoke, 'shared_indexes': str(out / 'shared_indexes'),
        'datasets': list(previous.DATASETS), 'embedding_model': runtime['embedding_model_name'],
        'embedding_provider': runtime.get('embedding_provider'),
        'embedding_batch_size': runtime['embedding_batch_size'],
        'llm_prefetch_workers': 8, 'openie_max_workers': 8,
        'hop_source': 'benchmark', 'max_new_tokens': 2048, 'thinking': False,
        'result_top_k': 10, 'candidate_output_top_k': 200, 'eval_mode': 'retrieve',
        'cache_policy': 'Private original-exp4 warm-cache snapshot, matching the existing full DAG runner',
        'selected_indices': selected,
    }
    protocol_file = work / 'metadata' / f'{CASE_NAME}_protocol.json'
    if protocol_file.exists() and read_json(protocol_file) != protocol:
        raise ValueError('Existing workflow protocol differs; inspect before retrying')
    write_json(protocol_file, protocol)
    if 'NV-Embed' in runtime['embedding_model_name']:
        from .multi_dataset_retrieval import environment as nv_environment
        nv_args = copy(args)
        nv_args.embedding_model = runtime['embedding_model_name']
        nv_args.embedding_provider = runtime.get('embedding_provider', 'nvembed')
        env = nv_environment(nv_args)
    else:
        env = previous.environment(args)
    env['PYTHONHASHSEED'] = '42'
    reports, statuses = {}, {}
    for dataset, context in contexts.items():
        case = work / 'cases' / dataset / CASE_NAME
        try:
            print(f'[{"smoke" if args.smoke else "full"}-start] {dataset}/{CASE_NAME} '
                  f'n={len(selected[dataset])} flags={",".join(VARIANT.flags)}', flush=True)
            report = round2.run_case(args, context, case, selected[dataset], VARIANT, env, cleanup=True)
            retain_http_audit(case, report)
            result = read_json(case / 'result.json')
            validation = validate_finalizer(case, result)
            write_json(case / 'support_semantic_veto_validation.json', validation)
            paired = paired_parent_comparison(context, case, report, result)
            write_json(case / 'paired_parent_comparison.json', paired)
            comparison = historical_comparison(out, dataset, case, selected[dataset], report, result)
            write_json(case / 'comparison_vs_dag_package.json', comparison)
            reports[dataset] = dict(report, comparison_vs_dag_package=comparison,
                                   paired_parent_comparison={key: value for key, value in paired.items()
                                                             if key != 'parent_per_question'},
                                   support_semantic_veto_validation=validation)
            del result
            statuses[dataset] = {'state': 'ready', 'n_samples': report['n_samples'], 'seconds': report['seconds']}
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
            raise ValueError('Frozen source index changed during retrieval')
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
    parser.add_argument('--vllm-log', help='vLLM log path; defaults to the local server stdout descriptor')
    parser.add_argument('--smoke', action='store_true', help='Check two questions per dataset in a disposable namespace')
    parser.add_argument('--smoke-out-root', help='Dedicated smoke output directory under PathCondRAG/outputs')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)
