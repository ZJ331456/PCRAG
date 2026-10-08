"""CPU attribution from saved proposals/reviews; no model or retrieval calls.

Run with scripts and src on PYTHONPATH, using ``python -B -m
utils.exp4_swap_verification_analysis``. The plain_joint control is a replay of
the same saved proposal, not an additional experiment or independent model draw.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
import subprocess

from . import exp4_swap_verification_trials as trial
from .common import read_json


def _check_code(summary):
    """Audit frozen code; tolerate only a separately edited, unused CLI builder.

    Retrieval runners may be edited while this CPU attribution is prepared.
    Their command builder is never called by the frozen-candidate experiment.
    A matching historical blob and identical remaining AST are required; this
    exception cannot authorize experiment resume or alter its saved protocol.
    """
    current = trial._code_hashes()
    drift = {key: {'frozen': value, 'current': current[key]}
             for key, value in summary['code_sha256'].items() if value != current[key]}
    if not drift:
        return {}
    if set(drift) != {'experiment_helpers'}:
        raise ValueError('Frozen algorithm/evaluation code changed before attribution')
    relative = trial.CODE_PATHS['experiment_helpers']
    expected = summary['code_sha256']['experiment_helpers']
    revisions = subprocess.check_output(['git', 'log', '-30', '--format=%H', '--', relative],
                                        cwd=trial.previous.ROOT, text=True).splitlines()
    frozen = None
    import hashlib
    for revision in revisions:
        blob = subprocess.check_output(['git', 'show', f'{revision}:{relative}'], cwd=trial.previous.ROOT)
        if hashlib.sha256(blob).hexdigest() == expected:
            frozen = blob.decode('utf-8')
            break
    if frozen is None:
        raise ValueError('Cannot find the recorded historical helper source')
    def used_ast(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body
                     if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != 'command']
        return ast.dump(tree, include_attributes=False)
    if used_ast(frozen) != used_ast((trial.previous.ROOT / relative).read_text()):
        raise ValueError('Changes extend beyond the unused command builder')
    drift['experiment_helpers'].update(unused_function='command', historical_revision=revision,
        verification='All other helper AST nodes identical; no retrieval command is executed in CPU attribution')
    return drift


def _response(metadata, dataset, index, stage, messages, record, summary, phase):
    path = metadata / 'responses' / dataset / f'{index}_{stage}.json'
    value = read_json(path)
    manifest = {'stage': stage, 'messages_sha256': trial._json_sha(messages),
                'generation': trial.GENERATION, 'llm_base_url': summary['protocol']['llm_base_url'],
                'source_result_sha256': record['source_result_sha256'], 'code_sha256': summary['code_sha256']}
    if (value['messages'] != messages or value['prompt_manifest'] != manifest
            or value['prompt_manifest_sha256'] != trial._json_sha(manifest)
            or not any(item['phase'] == phase and item['response_sha256'] == trial._json_sha(value['response'])
                       for item in value['observations'])):
        raise ValueError(f'Saved {stage} response/provenance mismatch: {dataset}/{phase}/{index}')
    return value['response'], value['metadata'].get('finish_reason')


def _stage_cost(path, expected_hash):
    """Audit one shared stage report; profiles never multiply observed costs."""
    trial._check_file(path, expected_hash, 'shared stage report')
    stages, observed = read_json(path), {}
    for name in ('proposer', 'verifier'):
        stage = stages[name]
        segment = stage['http_log_segment']
        with open(segment['path'], 'rb') as stream:
            stream.seek(segment['start'])
            blob = stream.read(segment['end'] - segment['start']).decode('utf-8', 'replace')
        status = Counter(re.findall(r'POST /v1/chat/completions HTTP/1\.1" (\d{3})', blob))
        stats = stage['observed_request_stats']
        if (dict(status) != stage['http_status_in_log'] or set(status) - {'200'}
                or sum(status.values()) != stats['http_attempts'] or stats['failures'] or stats['retries']):
            raise ValueError(f'Shared HTTP stage audit failed: {path}/{name}')
        observed[name] = {'logical_calls': stage['logical_calls'], 'observed_request_stats': stats,
                          'http_status_in_log': dict(status), 'seconds': stage['seconds']}
    return observed


def analyze_dataset(out, metadata, summary, dataset, phase):
    from pathcondrag.evidence_typed_joint_selector import build_joint_prompt, apply_joint_selection
    from pathcondrag.evidence_support_guard import support_guard
    from pathcondrag.evidence_swap_verifier import build_verifier_prompt, validate_verification

    record = summary['datasets'][dataset]
    original, parents, documents, text_to_id, samples, sampling, indices = trial._load_dataset(
        out, record, phase, pilot_size=2)
    reports = trial._resumed(out, dataset, phase, record, summary, indices)
    if reports is None:
        raise ValueError(f'All four profiles must be completed: {dataset}/{phase}')
    exports = {}
    for profile in trial.PROFILES:
        exported = read_json(out / trial.TEMP / phase / dataset / profile / 'result.json')
        if len(exported['results']) != len(indices) or exported['selected_indices'] != indices:
            raise ValueError('Completed profile has a different phase size/order')
        exports[profile] = {row['query_index']: row for row in exported['results']}
    if any(set(rows) != set(indices) for rows in exports.values()):
        raise ValueError('Completed profile indices differ from the frozen phase')
    metrics, baseline_metrics = [], []
    observed_metrics = {profile: [] for profile in trial.PROFILES}
    blocked = {profile: Counter() for profile in trial.PROFILES}
    applied = {profile: Counter() for profile in trial.PROFILES}
    joint_vetoes, details = Counter(), []
    for index in indices:
        parent = parents[index]
        old = [text_to_id[text] for text in parent['candidate_docs']]
        trace = deepcopy(parent['retrieval_trace']['evidence'])
        state = {'evidence_trace': trace, '_evidence_plan': trace.get('plan', []),
                 '_evidence_winning_bindings': trace.get('bindings', {}),
                 '_evidence_beams': trace.get('branch_scores', [])}
        before = trial._json_sha(state)
        messages = build_joint_prompt(parent['question'], old, state, documents.__getitem__)
        response, finish = ('{"swap": null}', 'stop') if messages is None else _response(
            metadata, dataset, index, 'proposer', messages, record, summary, phase)
        plain, proposal = apply_joint_selection(parent['question'], old, state, documents.__getitem__, response, finish)
        guard_allowed, guard = support_guard(parent['question'], old, state, documents.__getitem__, proposal)
        review_messages = build_verifier_prompt(parent['question'], old, state, documents.__getitem__, proposal)
        if plain != old and review_messages is None:
            raise ValueError('Accepted proposal has no frozen independent review')
        review_response, review_finish = ('{}', 'stop') if plain == old else _response(
            metadata, dataset, index, 'verifier', review_messages, record, summary, phase)
        approved, review = validate_verification(parent['question'], old, state, documents.__getitem__,
                                                 proposal, review_response, review_finish)
        if trial._json_sha(state) != before:
            raise ValueError('Counterfactual replay mutated upstream state')
        if (plain[:2] != old[:2] or set(plain[:10]) != set(old[:10]) or plain[10:] != old[10:]
                or len(plain) != len(set(plain))):
            raise ValueError('Counterfactual ranking invariant failed')
        evaluate = lambda ids: trial.evaluate(samples, dict(parent, dataset=dataset),
                                               [documents[doc_id] for doc_id in ids])
        baseline, counterfactual = evaluate(old), evaluate(plain)
        baseline_metrics.append(baseline)
        metrics.append(counterfactual)
        delta = counterfactual['metrics']['Recall@5'] - baseline['metrics']['Recall@5']
        kind = 'gain' if delta > 0 else 'loss' if delta < 0 else 'neutral'
        if plain != old:
            joint_vetoes[('guard_and_verifier' if not guard_allowed and not approved else
                          'guard_only' if not guard_allowed else 'verifier_only' if not approved else 'neither')] += 1
        expected = {'baseline': old, 'protected_joint': plain if guard_allowed else old,
                    'verified_joint': plain if approved else old,
                    'protected_verified_joint': plain if guard_allowed and approved else old}
        decisions = {}
        for profile, ranking in expected.items():
            exported = exports[profile][index]
            actual = [text_to_id[text] for text in exported['candidate_docs']]
            if actual != ranking or exported['docs'] != [documents[doc_id] for doc_id in ranking[:10]]:
                raise ValueError(f'Frozen profile differs from saved-response replay: {dataset}/{phase}/{profile}/{index}')
            measured = evaluate(ranking)
            if (exported['retrieval_metrics'] != measured['metrics']
                    or exported['all_gold_in_top5'] != measured['all_gold_top5']
                    or exported['all_gold_in_top10'] != measured['all_gold_top10']):
                raise ValueError('Exported per-question metrics differ from the reconstructed ranking')
            if profile != 'baseline':
                saved = exported['retrieval_trace']['evidence']['improvement_swap_verification']
                if (saved['base_proposal'] != proposal or saved['support_guard'] != guard
                        or saved['independent_verification'] != review):
                    raise ValueError('Saved decision diagnostics differ from the frozen response replay')
            observed_metrics[profile].append(measured)
            if plain != old:
                (applied[profile] if ranking == plain else blocked[profile])[kind] += 1
            decisions[profile] = ranking == plain and plain != old
        if plain != old:
            details.append({'query_index': index, 'plain_delta_r5': delta, 'plain_outcome': kind,
                            'guard_allowed': guard_allowed, 'verifier_approved': approved,
                            'guard_veto_reason': guard.get('veto_reason'),
                            'verifier_veto_reason': review.get('rejection_reason'), 'applied': decisions})
    baseline = trial.aggregate(baseline_metrics, sampling, include_all_gold=True)
    plain_report = trial.aggregate(metrics, sampling, include_all_gold=True)
    attribution = {}
    for profile in trial.PROFILES:
        recomputed = trial.aggregate(observed_metrics[profile], sampling, include_all_gold=True)
        actual_report = reports[profile]
        for field in ('retrieval_metrics', 'all_gold_top5', 'all_gold_top10',
                      'available_pool_weighted', 'available_pool_weighted_all_gold'):
            if recomputed[field] != actual_report[field]:
                raise ValueError(f'Frozen profile metrics disagree: {dataset}/{phase}/{profile}/{field}')
        attribution[profile] = {'blocked_plain_proposals': {key: blocked[profile][key] for key in ('gain', 'loss', 'neutral')},
            'applied_plain_proposals': {key: applied[profile][key] for key in ('gain', 'loss', 'neutral')},
            'raw_delta_vs_plain': trial._metric_delta(recomputed, plain_report, weighted=False),
            'weighted_delta_vs_plain': trial._metric_delta(recomputed, plain_report, weighted=True),
            'raw_delta_vs_baseline': trial._metric_delta(recomputed, baseline, weighted=False),
            'weighted_delta_vs_baseline': trial._metric_delta(recomputed, baseline, weighted=True),
            'note': 'Baseline applies no added proposal; its blocked counts are a control, not a module veto.'}
    stage_paths = {(report['shared_stage_report'], report['shared_stage_report_sha256']) for report in reports.values()}
    if len(stage_paths) != 1:
        raise ValueError('Profiles reference different supposedly shared API stages')
    stage_path, stage_hash = stage_paths.pop()
    return {'selected_indices': indices, 'n_samples': len(indices), 'weighting_pool_size': sampling['available_size'],
            'sampling_sha256': record['sampling_sha256'], 'source_result_sha256': record['source_result_sha256'],
            'baseline': baseline, 'plain_joint_counterfactual': plain_report,
            'plain_raw_delta_vs_baseline': trial._metric_delta(plain_report, baseline, weighted=False),
            'plain_weighted_delta_vs_baseline': trial._metric_delta(plain_report, baseline, weighted=True),
            'profiles': attribution, 'veto_overlap': dict(joint_vetoes), 'per_proposal': details,
            'shared_stage_report': stage_path, 'shared_stage_report_sha256': stage_hash,
            'shared_stage_cost_counted_once': _stage_cost(stage_path, stage_hash)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', default=str(trial.previous.DEFAULT_OUT))
    parser.add_argument('--phases', nargs='+', choices=('screen', 'regression', 'confirmation'),
                        default=('screen', 'regression', 'confirmation'))
    args = parser.parse_args(argv)
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(trial.previous.ROOT / 'outputs') or out == trial.previous.ROOT / 'outputs':
        raise ValueError('Output must be a dedicated project outputs directory')
    if len(args.phases) != len(set(args.phases)):
        raise ValueError('Repeated phases would double-count API cost')
    metadata = out / 'metadata' / trial.META
    summary_path = metadata / 'selection.json'
    summary = read_json(summary_path)
    code_drift = _check_code(summary)
    trial._check_file(summary['parent']['path'], summary['parent']['sha256'], 'typed parent protocol')
    result = {'method': 'Same saved-response CPU counterfactual; no new HTTP, retrieval, embedding, indexing or QA',
              'limits': ['Plain joint reuses the same draw; this is not an independent model experiment.',
                         'Screen/regression are observed development data; confirmation is fresh only for these frozen modules.',
                         'Weights describe each remaining pool (289/199/169), not an untouched full benchmark.',
                         'Separate verifier prompt uses the same model; its errors need not be statistically independent.'],
              'source_protocol_sha256': trial.previous.experiments.sha256(summary_path),
              'analysis_script_sha256': trial.previous.experiments.sha256(Path(__file__)),
              'frozen_code_sha256': summary['code_sha256'], 'phase_results': {}, 'new_http_requests': 0}
    result['unrelated_helper_edit_audit'] = code_drift
    total_http, total_status = 0, Counter()
    for phase in args.phases:
        result['phase_results'][phase] = {}
        for dataset in summary['datasets']:
            report = analyze_dataset(out, metadata, summary, dataset, phase)
            result['phase_results'][phase][dataset] = report
            for stage in report['shared_stage_cost_counted_once'].values():
                total_http += stage['observed_request_stats']['http_attempts']
                total_status.update(stage['http_status_in_log'])
            print(f'[attribution] {phase}/{dataset}: saved-response replay validated', flush=True)
    result.update(observed_trial_http_attempts_counted_once=total_http,
                  observed_trial_http_status_counted_once=dict(total_status))
    if trial.previous.experiments.sha256(summary_path) != result['source_protocol_sha256']:
        raise ValueError('Protocol was modified while attribution was running')
    target = metadata / 'component_attribution.json'
    trial._write(target, result)
    print('[saved]', target, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
