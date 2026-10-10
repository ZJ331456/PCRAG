"""Build three shared NV-Embed-v2 indexes, then evaluate five retrieval methods.

--smoke runs the same stages on two intact questions and twenty intact source
passages per dataset, in a disposable output directory. Full runs cover every
question, reuse the frozen dataset index, retain Top10/Top200 and omit QA.

--methods legacy_dependency_joint runs only the new composition against the
three existing full indexes; it never enters the index-building workflow.
"""
from __future__ import annotations

import argparse
import logging
import os
from copy import copy
from pathlib import Path

from pathcondrag.index.publication_policy import parse_bool

from . import multi_dataset_retrieval as base
from . import multi_dataset_retrieval_plan_prune_dag_package as dag
from . import multi_dataset_retrieval_support_semantic_veto as semantic
from .common import read_json, write_json
from .exp4_http_audit import locate


DEFAULT_OUT = base.ROOT / 'outputs/3multi_hop_datasets_results_with_nv2_10_8'
METHODS = (*base.CASES, dag.CASE_NAME, semantic.CASE_NAME)
JOINT_CASE_NAME = 'legacy_dependency_joint'
LOG = logging.getLogger('multi_dataset_nv2')


def case_path(out, dataset, method, smoke):
    work = out
    if smoke and method == dag.CASE_NAME:
        work = out / dag.SMOKE_NAME
    elif smoke and method == semantic.CASE_NAME:
        work = out / semantic.SMOKE_NAME
    return work / 'cases' / dataset / method


def publish_summary(out, stages, smoke):
    """Aggregate the fifteen independently validated reports without reranking."""
    reports, completed = {}, 0
    lines = ['| Dataset | Method | R@1 | R@2 | R@5 | R@10 | All gold@5 | Seconds | HTTP attempts |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in base.DATASETS:
        reports[dataset] = {}
        for method in METHODS:
            case = case_path(out, dataset, method, smoke)
            if not (case / 'validated.ok').is_file():
                continue
            report = read_json(case / 'report.json')
            if report.get('validated') is not True:
                raise ValueError(f'Case report is not validated: {case}')
            reports[dataset][method] = {
                **{key: value for key, value in report.items() if key != 'per_question'},
                'case_path': str(case),
            }
            completed += 1
            values = [f"{report['retrieval_metrics'][f'Recall@{k}']:.4f}" for k in (1, 2, 5, 10)]
            lines.append(f'| {dataset} | {method} | ' + ' | '.join(values)
                         + f" | {report['all_gold_top5']:.4f} | {report['seconds']}"
                         + f" | {report['llm_request_stats'].get('http_attempts', 0)} |")
    summary = {'datasets': reports, 'stages': stages, 'smoke': smoke,
               'expected_case_count': len(base.DATASETS) * len(METHODS),
               'validated_case_count': completed}
    write_json(out / 'comparison.json', summary)
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return completed


def retain_request_logs(out, smoke):
    """Retain actual service-log slices before its process or file disappears."""
    for dataset in base.DATASETS:
        metadata = out / 'metadata' / dataset
        report_path = metadata / 'index_build_report.json'
        if report_path.is_file():
            semantic.retain_http_audit(metadata, read_json(report_path))
        for method in METHODS:
            case = case_path(out, dataset, method, smoke)
            if (case / 'validated.ok').is_file():
                semantic.retain_http_audit(case, read_json(case / 'report.json'))


def run(args):
    if getattr(args, 'methods', None) == [JOINT_CASE_NAME]:
        from .multi_dataset_retrieval_dependency_joint import run as run_joint
        return run_joint(args)
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(base.ROOT / 'outputs') or out == base.ROOT / 'outputs':
        raise ValueError('Output must be a dedicated directory under PathCondRAG/outputs')
    if args.smoke and 'smoke' not in out.name:
        raise ValueError('Smoke requires a separate output directory with "smoke" in its name')
    if args.embedding_provider != 'nvembed' or 'NV-Embed-v2' not in args.embedding_model:
        raise ValueError('This runner requires the NV-Embed-v2 model and nvembed provider')
    if args.embedding_batch_size <= 0:
        raise ValueError('Embedding batch size must be positive')
    args.vllm_log = args.vllm_log or locate(args.llm_base_url)
    base.log_offset(Path(args.vllm_log))
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'datasets': list(base.DATASETS), 'methods': list(METHODS), 'smoke': args.smoke,
        'embedding_model': args.embedding_model, 'embedding_provider': args.embedding_provider,
        'embedding_batch_size': args.embedding_batch_size, 'llm_name': 'qwen3-8b',
        'nv_embedding_oom_split': parse_bool(os.environ.get('PATHCONDRAG_NVEMBED_OOM_SPLIT', 'false')),
        'llm_base_url': args.llm_base_url, 'llm_workers': 8, 'index_workers': 8,
        'max_new_tokens': 2048, 'thinking': False, 'hop_source': 'benchmark',
        'openie_strict': args.openie_strict, 'openie_prompt_version': args.openie_prompt_version,
        'openie_validation_mode': args.openie_validation_mode,
        'eval_mode': 'retrieve', 'retrieval_top_k': 200,
        'result_top_k': 10, 'candidate_output_top_k': 200,
        'stages': ['build_all_shared_indexes', 'retrieve_original_three_methods',
                   'retrieve_plan_prune_dag_package', 'retrieve_support_semantic_veto'],
    }
    protocol_file = out / 'nv2_workflow_protocol.json'
    if protocol_file.is_file() and read_json(protocol_file) != protocol:
        raise ValueError('Existing NV2 workflow protocol differs; inspect before retrying')
    write_json(protocol_file, protocol)
    (out / 'completed.ok').unlink(missing_ok=True)
    stages = {}
    frozen = {}

    def record(name, state, **details):
        stages[name] = {'state': state, **details}
        write_json(out / 'nv2_workflow_stage_status.json', stages)
        LOG.info('[workflow-stage] %s %s %s', name, state, details)

    index_args = copy(args)
    index_args.indexes_only = True
    base_args = copy(args)
    base_args.indexes_only = False
    secondary_args = copy(args)
    secondary_args.smoke_out_root = None
    functions = (
        ('build_all_shared_indexes', base.run, index_args),
        ('retrieve_original_three_methods', base.run, base_args),
        ('retrieve_plan_prune_dag_package', dag.run, secondary_args),
        ('retrieve_support_semantic_veto', semantic.run, secondary_args),
    )
    for name, function, options in functions:
        record(name, 'running')
        try:
            status = function(options)
            retain_request_logs(out, args.smoke)
            if status:
                record(name, 'failed', returncode=status)
                (out / 'completed.ok').unlink(missing_ok=True)
                publish_summary(out, stages, args.smoke)
                return status
            if name == 'build_all_shared_indexes':
                frozen = {dataset: base.frozen_hashes(out / 'shared_indexes' / dataset)
                          for dataset in base.DATASETS}
                write_json(out / 'metadata' / 'nv2_frozen_index_sha256.json', frozen)
            elif any(base.frozen_hashes(out / 'shared_indexes' / dataset) != frozen[dataset]
                     for dataset in base.DATASETS):
                raise ValueError('A retrieval stage changed a frozen graph/vector/OpenIE/manifest asset')
            # The original three-method runner has its own completion marker;
            # the unified workflow is complete only after all fifteen cases.
            (out / 'completed.ok').unlink(missing_ok=True)
            record(name, 'ready')
        except Exception as error:
            record(name, 'failed', error=f'{type(error).__name__}: {error}')
            (out / 'completed.ok').unlink(missing_ok=True)
            LOG.exception('[workflow-failed] %s', name)
            publish_summary(out, stages, args.smoke)
            return 1
    count = publish_summary(out, stages, args.smoke)
    if count != len(base.DATASETS) * len(METHODS):
        raise ValueError(f'Expected fifteen validated retrieval cases, found {count}')
    write_json(out / 'completed.ok', {
        'datasets': list(base.DATASETS), 'methods': list(METHODS),
        'validated_case_count': count, 'smoke': args.smoke,
    })
    LOG.info('[done] three NV2 indexes and fifteen retrieval cases validated: %s', out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root')
    parser.add_argument('--datasets-dir', default='/root/datasets')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--runtime-deps', default=str(base.DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', help='Actual server request log; default: local vLLM stdout descriptor')
    parser.add_argument('--embedding-model', default='/root/models/NV-Embed-v2')
    parser.add_argument('--embedding-provider', default='nvembed')
    parser.add_argument('--embedding-batch-size', type=int, default=4)
    parser.add_argument('--openie-strict', '--openie_strict', type=parse_bool, default=False)
    parser.add_argument('--openie-prompt-version', '--openie_prompt_version',
                        choices=('origin', 'optimized'), default='optimized')
    parser.add_argument('--openie-validation-mode', '--openie_validation_mode',
                        choices=('structural', 'source_verified'), default='structural')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=(JOINT_CASE_NAME,),
                        help='Run only legacy_dependency_joint using the existing shared indexes')
    parser.add_argument('--smoke-out-root',
                        help='Separate test output for --methods legacy_dependency_joint --smoke')
    args = parser.parse_args(argv)
    if args.methods and args.methods != [JOINT_CASE_NAME]:
        parser.error('Specify legacy_dependency_joint once')
    if args.smoke_out_root and not (args.smoke and args.methods == [JOINT_CASE_NAME]):
        parser.error('--smoke-out-root requires --methods legacy_dependency_joint --smoke')
    args.out_root = args.out_root or str(DEFAULT_OUT.with_name(DEFAULT_OUT.name + '_smoke')
                                        if args.smoke and not args.methods else DEFAULT_OUT)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)
