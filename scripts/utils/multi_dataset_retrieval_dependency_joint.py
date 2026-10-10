"""Run legacy + dependency joint selection on three complete frozen NV2 indexes.

The legacy selector is executed inside the existing dependency_joint mode.
This driver performs normal retrieval, never indexes, replays traces or starts
another baseline. --smoke evaluates two questions using the same full corpus.
"""
from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
import shutil

from .common import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
CASE_NAME = 'legacy_dependency_joint'
SMOKE_NAME = '_legacy_dependency_joint_smoke'
SCORING_MODE = 'dependency_joint'
FLAGS = ('planning', 'plan_prune', 'dag_package', 'support_semantic_veto')
SMOKE_INDICES = {'hotpotqa': [52, 700], '2wikimultihopqa': [32, 75], 'musique': [270, 56]}
LOG = logging.getLogger('multi_dataset_dependency_joint')


def dependencies():
    """Keep model imports out of model-free command/layout tests."""
    global previous, round2, semantic, base, validate_joint_selection, locate
    from . import exp4_improvements as previous
    from . import exp4_round2 as round2
    from . import multi_dataset_retrieval_support_semantic_veto as semantic
    from . import multi_dataset_retrieval as base
    from .nv2_dependency_scoring_subset import validate_joint_selection
    from .exp4_http_audit import locate


def work_directory(out, smoke=False, smoke_out_root=None):
    outputs = ROOT / 'outputs'
    out = Path(out).resolve()
    if out == outputs or not out.is_relative_to(outputs):
        raise ValueError('Source output must be a dedicated directory under PathCondRAG/outputs')
    if smoke_out_root and not smoke:
        raise ValueError('--smoke-out-root requires --smoke')
    work = (Path(smoke_out_root).resolve() if smoke_out_root else out / SMOKE_NAME) if smoke else out
    if smoke and (work == out or work == outputs or not work.is_relative_to(outputs)
                  or any(work.is_relative_to(out / name) for name in ('cases', 'shared_indexes', 'metadata', 'logs'))
                  or out.is_relative_to(work)):
        raise ValueError('Smoke output must be separate from source indexes and existing production artifacts')
    return work


def selected_indices(dataset, data, hops, manifest, smoke=False):
    indices = list(manifest['selected_indices'])
    if smoke:
        preferred = SMOKE_INDICES[dataset]
        if max(preferred) < len(data):
            indices = list(preferred)
        elif 2 in hops and 4 in hops:
            indices = [hops.index(2), hops.index(4)]
        else:
            indices = indices[:2]
        if len(indices) != 2:
            raise ValueError(f'{dataset}: smoke needs exactly two indexed questions')
    elif set(indices) != set(range(len(data))):
        raise ValueError(f'{dataset}: full retrieval must cover every question')
    if len(indices) != len(set(indices)) or any(type(index) is not int or not 0 <= index < len(data) for index in indices):
        raise ValueError(f'{dataset}: invalid selected indices')
    return indices


def command(args, context, case, indices, variant):
    return round2.command(args, context, case, indices, variant) + ['--evidence_scoring_mode', SCORING_MODE]


def validate_case(case, report, result):
    config = result.get('runtime_config') or {}
    expected = {'evidence_scoring_mode': SCORING_MODE, 'evidence_plan_validation': 'canonical_refs',
                'evidence_plan_routing': 'question_structure', 'embedding_batch_size': 4,
                'openie_max_workers': 8, 'llm_prefetch_workers': 8, 'max_new_tokens': 2048}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{case}: joint retrieval configuration differs')
    # Older runtime configs do not serialize the chat-template flag. The
    # shared-index protocol records False and the existing LLM backend keeps
    # it disabled; explicitly enabled values must still fail validation.
    if config.get('embedding_model_name') != '/root/models/NV-Embed-v2' or config.get('enable_thinking', False) is not False:
        raise ValueError(f'{case}: embedding model or disabled-thinking configuration differs')
    joint = validate_joint_selection(result)
    finalizer = semantic.validate_finalizer(case, result)
    write_json(case / 'dependency_joint_selection_validation.json', joint)
    write_json(case / 'support_semantic_veto_validation.json', finalizer)
    report.update(scoring_mode=SCORING_MODE, improvements=list(FLAGS),
                  dependency_joint_selection_validation={key: value for key, value in joint.items() if key != 'per_question'},
                  support_semantic_veto_validation=finalizer,
                  execution_mode='normal_retrieval_no_frozen_replay',
                  cache_policy='Private snapshot of existing original exp4 cache; shared graph/vector/OpenIE assets are frozen.')
    return report


def run_case(args, context, case, indices, variant, env):
    manifest = previous.subset_manifest(context['manifest'], context['data'], context['hops'], indices)
    if (case / 'validated.ok').is_file():
        report = round2.verify_saved_case(context, case, manifest, variant)
        if report.get('scoring_mode') != SCORING_MODE or report.get('execution_mode') != 'normal_retrieval_no_frozen_replay':
            raise ValueError(f'{case}: saved case is not normal joint retrieval')
        result = read_json(case / 'result.json')
        if (validate_joint_selection(result) != read_json(case / 'dependency_joint_selection_validation.json')
                or semantic.validate_finalizer(case, result) != read_json(case / 'support_semantic_veto_validation.json')):
            raise ValueError(f'{case}: saved joint/finalizer proof changed')
        semantic.retain_http_audit(case, report)
        if not read_json(case / 'validated.ok').get('cleaned_private_index'):
            round2.cleanup_private_index(context, case)
        return report
    previous.initialize_case(context, case)
    write_json(case / 'selected_indices.json', list(indices))
    write_json(case / 'manifest.json', manifest)
    write_json(case / 'round2_config.json', asdict(variant))
    start = Path(args.vllm_log).stat().st_size
    elapsed = previous.execute(command(args, context, case, indices, variant), case / 'run.log', env)
    end = Path(args.vllm_log).stat().st_size
    try:
        report, result = previous.validate_case(args, context, case, manifest, FLAGS, elapsed, start, end)
        report = validate_case(case, report, result)
        report['round2_config'] = asdict(variant)
        write_json(case / 'report.json', report)
        marker = read_json(case / 'validated.ok')
        marker.update(report_sha256=previous.experiments.sha256(case / 'report.json'),
                      round2_config_sha256=previous.experiments.sha256(case / 'round2_config.json'))
        write_json(case / 'validated.ok', marker)
        semantic.retain_http_audit(case, report)
        round2.cleanup_private_index(context, case)
        return report
    except Exception:
        (case / 'validated.ok').unlink(missing_ok=True)
        raise


def publish_summary(work, reports, statuses):
    lines = ['| Dataset | Case | R@1 | R@2 | R@5 | R@10 | R@20 | R@200 | All gold@5 | All gold@10 | Seconds |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset, report in reports.items():
        values = [f'{report["retrieval_metrics"][f"Recall@{k}"]:.4f}' for k in (1, 2, 5, 10, 20, 200)]
        values += [f'{report["all_gold_top5"]:.4f}', f'{report["all_gold_top10"]:.4f}', str(report['seconds'])]
        lines.append(f'| {dataset} | {CASE_NAME} | ' + ' | '.join(values) + ' |')
    write_json(work / 'metadata' / f'{CASE_NAME}_comparison.json',
               {'case': CASE_NAME, 'scoring_mode': SCORING_MODE, 'stages': statuses,
                'datasets': {dataset: previous.compact(report) for dataset, report in reports.items()}})
    (work / 'metadata' / f'{CASE_NAME}_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def run(args):
    dependencies()
    out = Path(args.out_root).resolve()
    work = work_directory(out, args.smoke, getattr(args, 'smoke_out_root', None))
    if args.embedding_model != '/root/models/NV-Embed-v2' or args.embedding_provider != 'nvembed' or args.embedding_batch_size != 4:
        raise ValueError('Joint retrieval requires NV-Embed-v2/nvembed and embedding batch 4')
    if not (out / 'shared_indexes').is_dir():
        raise ValueError('All three existing shared indexes must be available; this workflow never builds indexes')
    support = work / '_legacy_dependency_joint_support'
    if work.is_symlink() or support.is_symlink() or shutil.disk_usage(out).free < 3 * 1024 ** 3:
        raise ValueError('Unsafe output symlink or insufficient free disk')
    support.mkdir(parents=True, exist_ok=True)
    (work / 'metadata').mkdir(exist_ok=True)
    args.vllm_log = args.vllm_log or locate(args.llm_base_url)
    base.log_offset(Path(args.vllm_log))
    contexts = {dataset: previous.context_for(out, dataset, support) for dataset in previous.DATASETS}
    before_code = previous.code_hashes(Path(args.hippo_root))
    frozen = {dataset: base.frozen_hashes(context['source']) for dataset, context in contexts.items()}
    selections = {}
    for dataset, context in contexts.items():
        runtime = context['manifest']['runtime']
        expected = {'embedding_model_name': '/root/models/NV-Embed-v2', 'embedding_provider': 'nvembed',
                    'embedding_batch_size': 4, 'llm_prefetch_workers': 8, 'openie_max_workers': 8,
                    'max_new_tokens': 2048, 'enable_thinking': False}
        if any(runtime.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{dataset}: frozen index runtime differs')
        context['clone_source'] = context['source']
        selections[dataset] = selected_indices(dataset, context['data'], context['hops'], context['manifest'], args.smoke)
    variant = round2.Variant(CASE_NAME, FLAGS, plan_validation='canonical_refs', plan_routing='question_structure')
    protocol = {'case': CASE_NAME, 'scoring_mode': SCORING_MODE, 'variant': dict(asdict(variant), flags=list(FLAGS)),
                'smoke': args.smoke, 'shared_indexes': str(out / 'shared_indexes'), 'build_indexes': False,
                'datasets': list(previous.DATASETS), 'selected_indices': selections,
                'embedding_model': args.embedding_model, 'embedding_provider': args.embedding_provider,
                'embedding_batch_size': 4, 'nv_embedding_oom_split': False, 'llm_prefetch_workers': 8,
                'openie_max_workers': 8, 'max_new_tokens': 2048, 'thinking': False, 'hop_source': 'benchmark',
                'hop_force_max': 4, 'eval_mode': 'retrieve', 'result_top_k': 10, 'candidate_output_top_k': 200,
                'execution_mode': 'normal_retrieval_no_frozen_replay', 'source_index_sha256': frozen,
                'cache_policy': 'Private existing original-exp4 snapshot; no separately rerun baseline.'}
    protocol_file = work / 'metadata' / f'{CASE_NAME}_protocol.json'
    if protocol_file.is_file() and read_json(protocol_file) != protocol:
        raise ValueError('Existing joint workflow protocol differs; inspect before retrying')
    write_json(protocol_file, protocol)
    env = base.environment(args)
    env['PATHCONDRAG_NVEMBED_OOM_SPLIT'] = 'false'
    reports, statuses = {}, {}
    completion = work / 'metadata' / f'{CASE_NAME}_completed.ok'
    completion.unlink(missing_ok=True)
    for dataset, context in contexts.items():
        case = work / 'cases' / dataset / CASE_NAME
        try:
            print(f'[{"smoke" if args.smoke else "full"}-start] {dataset}/{CASE_NAME} n={len(selections[dataset])} mode={SCORING_MODE}', flush=True)
            reports[dataset] = run_case(args, context, case, selections[dataset], variant, env)
            if base.frozen_hashes(context['source']) != frozen[dataset]:
                raise ValueError(f'{dataset}: frozen graph/vector/OpenIE/manifest changed')
            statuses[dataset] = {'state': 'ready', 'n_samples': reports[dataset]['n_samples'], 'seconds': reports[dataset]['seconds']}
        except Exception as error:
            LOG.exception('[failed] %s', dataset)
            (case / 'validated.ok').unlink(missing_ok=True)
            statuses[dataset] = {'state': 'failed', 'error': f'{type(error).__name__}: {error}'}
        write_json(work / 'metadata' / f'{CASE_NAME}_stage_status.json', statuses)
        publish_summary(work, reports, statuses)
    if previous.code_hashes(Path(args.hippo_root)) != before_code:
        raise ValueError('Original HippoRAG source changed')
    if any(base.frozen_hashes(context['source']) != frozen[dataset] for dataset, context in contexts.items()):
        raise ValueError('A shared index changed during joint retrieval')
    if any(state['state'] != 'ready' for state in statuses.values()):
        return 1
    write_json(completion, {'case': CASE_NAME, 'scoring_mode': SCORING_MODE, 'smoke': args.smoke,
                            'datasets': {dataset: report['n_samples'] for dataset, report in reports.items()},
                            'validated_case_count': 3, 'normal_retrieval': True})
    LOG.info('[done] three normal legacy/dependency joint retrieval cases validated: %s', work)
    return 0
