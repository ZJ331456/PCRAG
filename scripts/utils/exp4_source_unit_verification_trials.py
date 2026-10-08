"""Paired concise-review trials using the same fixed proposal and sampling policy."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import time

from . import exp4_swap_verification_trials as base
from .common import read_json
from .exp4_swap_verification_analysis import _check_code


META = 'exp4_source_unit_verification_selection'
TEMP = '_exp4_source_unit_verification_trials'
PROFILES = ('baseline', 'protected_joint', 'verified_source_units', 'protected_verified_source_units')
CODE_PATHS = dict(base.CODE_PATHS, concise_verifier='src/pathcondrag/evidence_source_unit_verifier.py',
    source_validator='src/pathcondrag/evidence_swap_verifier_v2.py',
    concise_runner='scripts/utils/exp4_source_unit_verification_trials.py',
    provenance_audit='scripts/utils/exp4_swap_verification_analysis.py')


def _hashes():
    return {key: base.previous.experiments.sha256(base.previous.ROOT / path)
            for key, path in CODE_PATHS.items()}


def prepare(args):
    out = Path(args.out_root).resolve()
    target = out / 'metadata' / META / 'selection.json'
    if target.is_file():
        summary = read_json(target)
        if summary['code_sha256'] != _hashes() or summary['protocol']['llm_base_url'] != args.llm_base_url:
            raise ValueError('This source units namespace belongs to a different frozen implementation or endpoint')
        base._check_file(summary['parent']['path'], summary['parent']['sha256'], 'v1 protocol')
        return summary
    parent_path = out / 'metadata' / base.META / 'selection.json'
    parent = read_json(parent_path)
    audit = _check_code(parent)
    if parent['protocol']['generation'] != base.GENERATION:
        raise ValueError('Generation settings differ from the original paired trial')
    summary = {'protocol': dict(parent['protocol'], profiles=list(PROFILES),
            scope='Same proposals, support guard and predetermined 60/30/30 sampling; concise independent reviewer',
            verifier_changes=['Model selects original sentence IDs and shared literal name options; code reconstructs exact source spans',
                              'Final answer-bearing relation explicitly qualifies as a needed source fact; same 2000 visible characters',
                              'All three proposal IDs required in review response'],
            warning='source units was developed on observed screen/regression data BEFORE any new confirmation response'),
        'datasets': deepcopy(parent['datasets']), 'code_sha256': _hashes(),
        'parent': {'path': str(parent_path), 'sha256': base.previous.experiments.sha256(parent_path)},
        'parent_unrelated_helper_edit_audit': audit, 'phase_results': {}}
    if summary['protocol']['llm_base_url'] != args.llm_base_url:
        raise ValueError('The endpoint must match the fixed proposal experiment')
    base._write(target, summary)
    print('[prepared source units] Same predetermined samples; no fresh-confirmation outcomes read', flush=True)
    return summary


def run_dataset(args, summary, llm, tokenizer, dataset, phase):
    from pathcondrag.evidence_typed_joint_selector import build_joint_prompt, apply_joint_selection
    from pathcondrag.evidence_support_guard import support_guard
    from pathcondrag.evidence_source_unit_verifier import build_verifier_prompt, validate_verification

    out = Path(args.out_root).resolve()
    record = summary['datasets'][dataset]
    if summary['code_sha256'] != _hashes():
        raise ValueError('Code changed during the frozen source units trial')
    original, parents, documents, text_to_id, samples, sampling, indices = base._load_dataset(
        out, record, phase, args.pilot_size)
    contexts, prompts = {}, {}
    for index in indices:
        parent = parents[index]
        order = [text_to_id[text] for text in parent['candidate_docs']]
        trace = deepcopy(parent['retrieval_trace']['evidence'])
        state = {'evidence_trace': trace, '_evidence_plan': trace.get('plan', []),
            '_evidence_winning_bindings': trace.get('bindings', {}), '_evidence_beams': trace.get('branch_scores', [])}
        before = base._json_sha(state)
        messages = build_joint_prompt(parent['question'], order, state, documents.__getitem__)
        contexts[index] = {'query': parent['question'], 'old': order, 'state': state, 'state_sha256': before}
        if messages is not None:
            prompts[index] = messages
        if base._json_sha(state) != before:
            raise ValueError('Proposer mutated the original state')
    metadata = out / 'metadata' / META
    server_log = Path(args.vllm_log or out / 'logs/vllm_joint_selector.log')
    proposals, proposer_stats = base._fetch_stage(llm, tokenizer, prompts, metadata, dataset, phase,
        'proposer', record, summary, server_log)
    review_prompts = {}
    for index, ctx in contexts.items():
        response = proposals.get(index)
        proposed, diag = apply_joint_selection(ctx['query'], ctx['old'], ctx['state'], documents.__getitem__,
            response['response'] if response else '{"swap":null}',
            response['metadata'].get('finish_reason') if response else 'stop')
        allowed, guard = support_guard(ctx['query'], ctx['old'], ctx['state'], documents.__getitem__, diag)
        ctx.update(proposed=proposed, proposal=diag, guard_allowed=allowed, guard=guard)
        if proposed != ctx['old']:
            messages = build_verifier_prompt(ctx['query'], ctx['old'], ctx['state'], documents.__getitem__, diag)
            if messages is None:
                raise ValueError('Accepted proposal has no source units review prompt')
            review_prompts[index] = messages
    reviews, review_stats = base._fetch_stage(llm, tokenizer, review_prompts, metadata, dataset, phase,
        'verifier', record, summary, server_log)
    for index, ctx in contexts.items():
        response = reviews.get(index)
        approved, diag = validate_verification(ctx['query'], ctx['old'], ctx['state'], documents.__getitem__,
            ctx['proposal'], response['response'] if response else '{}',
            response['metadata'].get('finish_reason') if response else 'stop')
        ctx.update(verifier_approved=approved, verifier=diag)
        if base._json_sha(ctx['state']) != ctx['state_sha256']:
            raise ValueError('Review/guard mutated the original state')
    stage_path = metadata / 'stage_reports' / dataset / f'{phase}.json'
    stages = {'dataset': dataset, 'phase': phase, 'proposer': proposer_stats, 'verifier': review_stats,
        'code_sha256': summary['code_sha256'], 'source_result_sha256': record['source_result_sha256']}
    base._write(stage_path, stages)
    baseline_rows = [base.evaluate(samples, dict(parents[index], dataset=dataset), parents[index]['candidate_docs'])
                     for index in indices]
    baseline = base.aggregate(baseline_rows, sampling, partial=phase == 'pilot', include_all_gold=True)
    baseline_by_id = {row['query_index']: row for row in baseline_rows}
    reports = {}
    for profile in PROFILES:
        measured_rows, output, changes = [], [], []
        decisions, vetoes = Counter(), Counter()
        for index, ctx in contexts.items():
            old = ctx['old']
            proposed = ctx['proposed'] != old
            accepted = proposed and profile != 'baseline'
            if profile in ('protected_joint', 'protected_verified_source_units'):
                accepted = accepted and ctx['guard_allowed']
            if profile in ('verified_source_units', 'protected_verified_source_units'):
                accepted = accepted and ctx['verifier_approved']
            new = ctx['proposed'] if accepted else old
            if (new[:2] != old[:2] or set(new[:10]) != set(old[:10]) or new[10:] != old[10:]
                    or len(new) != len(set(new)) or base._json_sha(ctx['state']) != ctx['state_sha256']):
                raise ValueError('Bounded ranking/state invariant violated')
            protected = 'protected_doc_ids' if profile in ('protected_joint', 'protected_verified_source_units') else 'protected_dag_doc_ids'
            if not set(ctx['guard'].get(protected, [])) <= set(new[:5]):
                raise ValueError('Protected source evidence removed')
            measured = base.evaluate(samples, dict(parents[index], dataset=dataset), [documents[doc_id] for doc_id in new])
            before = baseline_by_id[index]
            if any(measured['metrics'][key] != before['metrics'][key] for key in base.RECALLS if key != 'Recall@5'):
                raise ValueError('A fixed-prefix Recall metric changed')
            diagnostic = {'profile': profile, 'accepted': bool(accepted), 'base_proposal': ctx['proposal'],
                'support_guard': ctx['guard'], 'independent_verification': ctx['verifier'],
                'gold_labels_used': False, 'shared_stage_report': str(stage_path)}
            output.append(base._output_row(parents[index], old, new, documents, measured, diagnostic, profile))
            measured_rows.append(measured)
            decisions['base_accepted' if proposed else 'base_abstained'] += 1
            decisions['applied' if accepted else 'unchanged'] += 1
            if proposed and not ctx['verifier_approved']:
                vetoes[ctx['verifier']['rejection_reason']] += 1
            if new != old:
                changes.append({'query_index': index, 'delta_r5': measured['metrics']['Recall@5'] - before['metrics']['Recall@5'],
                    'delta_all_gold5': int(measured['all_gold_top5']) - int(before['all_gold_top5']), 'diagnostic': diagnostic})
        report = base.aggregate(measured_rows, sampling, partial=phase == 'pilot', include_all_gold=True)
        raw = base._metric_delta(report, baseline, weighted=False)
        weighted = base._metric_delta(report, baseline, weighted=True)
        logical = {'proposer': 0 if profile == 'baseline' else proposer_stats['logical_calls'],
            'verifier': review_stats['logical_calls'] if profile in ('verified_source_units', 'protected_verified_source_units') else 0}
        report.update(dataset=dataset, phase=phase, profile=profile, raw_delta=raw, weighted_delta=weighted,
            regression_limit_met=all(value >= -.01 - 1e-12 for values in (raw, weighted or {}) for value in values.values()),
            logical_llm_calls=logical, observed_shared_stage_stats=stages, shared_stage_report=str(stage_path),
            shared_stage_report_sha256=base.previous.experiments.sha256(stage_path),
            changed_questions=len(changes), wins=sum(x['delta_r5'] > 0 for x in changes),
            losses=sum(x['delta_r5'] < 0 for x in changes), neutral=sum(x['delta_r5'] == 0 for x in changes),
            changes=changes, decision_counts=dict(decisions), verifier_vetoes=dict(vetoes),
            evaluation_scope='frozen_candidate_paired_swap_verification_source units', retrieval_seconds=None,
            code_sha256=summary['code_sha256'],
            phase_scope='Partial screening sample, unweighted' if phase == 'pilot' else summary['protocol']['phase_scope'][phase])
        result = {key: original[key] for key in ('dataset', 'eval_mode', 'indexed_docs', 'result_top_k',
            'candidate_output_top_k', 'hop_source') if key in original}
        result.update(results=output, selected_indices=indices, sample_size_requested=len(indices),
            sample_size_effective=len(indices), retrieval_metrics=report['retrieval_metrics'], retrieval_seconds=None,
            source_result=record['source_result'], source_result_sha256=record['source_result_sha256'],
            upstream_runtime_config=original.get('runtime_config'), evaluation_scope=report['evaluation_scope'],
            postprocessing_config={'profile': profile, 'generation': base.GENERATION, 'workers': 8,
                'code_sha256': summary['code_sha256'], 'shared_stage_report': str(stage_path)},
            logical_llm_calls=logical, observed_shared_stage_stats=stages)
        case = out / TEMP / phase / dataset / profile
        (case / 'validated.ok').unlink(missing_ok=True)
        base._write(case / 'result.json', result)
        base._write(case / 'report.json', report)
        base._write(case / 'validated.ok', {'result_sha256': base.previous.experiments.sha256(case / 'result.json'),
            'report_sha256': base.previous.experiments.sha256(case / 'report.json'), 'code_sha256': summary['code_sha256'],
            'sampling_sha256': record['sampling_sha256'], 'source_result_sha256': record['source_result_sha256'],
            'selected_indices': indices, 'stage_report': str(stage_path),
            'stage_report_sha256': base.previous.experiments.sha256(stage_path)})
        reports[profile] = report
        print(f'[{phase} source units] {dataset}/{profile} n={len(indices)} R5={report["retrieval_metrics"]["Recall@5"]:.5f} '
            f'delta={raw["Recall@5"]:+.5f} weighted={(weighted or {}).get("Recall@5")} '
            f'wins/losses={report["wins"]}/{report["losses"]} guard={report["regression_limit_met"]}', flush=True)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', default=str(base.previous.DEFAULT_OUT))
    parser.add_argument('--phase', choices=('prepare', 'pilot', 'screen', 'regression', 'confirmation', 'all'), default='screen')
    parser.add_argument('--pilot-size', type=int, default=8)
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=None)
    args = parser.parse_args(argv)
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(base.previous.ROOT / 'outputs') or out == base.previous.ROOT / 'outputs':
        raise ValueError('Output must be a dedicated project outputs directory')
    if not 1 <= args.pilot_size <= 60:
        raise ValueError('Pilot size must be 1..60')
    summary = prepare(args)
    if args.phase == 'prepare':
        return 0
    from pathcondrag.llm.openai_gpt import CacheOpenAI
    from pathcondrag.utils.config_utils import BaseConfig
    from transformers import AutoTokenizer
    os.environ.setdefault('OPENAI_API_KEY', 'sk-local')
    tokenizer = AutoTokenizer.from_pretrained('/root/models/Qwen3-8B', trust_remote_code=True, local_files_only=True)
    config = BaseConfig(llm_name='qwen3-8b', llm_base_url=args.llm_base_url, temperature=0., seed=0,
        max_new_tokens=2048, num_gen_choices=1, save_dir=summary['protocol']['cache_save_dir'], max_retry_attempts=1)
    llm = CacheOpenAI.from_experiment_config(config)
    llm.max_retries = 1
    phases = ('screen', 'regression', 'confirmation') if args.phase == 'all' else (args.phase,)
    try:
        for phase in phases:
            for dataset in base.previous.DATASETS:
                reports = run_dataset(args, summary, llm, tokenizer, dataset, phase)
                for profile, report in reports.items():
                    summary['phase_results'].setdefault(phase, {}).setdefault(profile, {})[dataset] = {
                        key: value for key, value in report.items() if key not in ('per_question', 'changes')}
                base._write(out / 'metadata' / META / 'selection.json', summary)
    finally:
        llm.openai_client.close()
    return 0
