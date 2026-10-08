"""Audit saved decisions and attribute support protection versus semantic veto."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import re

from . import exp4_semantic_veto_trials as trial
from .common import read_json
from .exp4_swap_verification_analysis import _response


def analyze(out):
    from pathcondrag.evidence_typed_joint_selector import build_joint_prompt, apply_joint_selection
    from pathcondrag.evidence_support_guard import support_guard
    from pathcondrag.evidence_semantic_swap_gate import build_verifier_prompt, validate_verification

    base = trial.base
    metadata = out / 'metadata' / trial.META
    protocol_path = metadata / 'selection.json'
    protocol = read_json(protocol_path)
    if protocol['code_sha256'] != trial._hashes():
        raise ValueError('The frozen semantic trial implementation changed')
    base._check_file(protocol['parent']['path'], protocol['parent']['sha256'], 'v1 protocol')
    source_protocol_hash = base.previous.experiments.sha256(protocol_path)
    archive = read_json(metadata / 'http_audit/archive_index.json')
    segments = {(r['dataset'], r['phase'], r['stage']): r for r in archive['segments']}
    phases = ('screen', 'regression', 'confirmation')
    result = {'method': 'Paired same-response CPU reconstruction; no new API, embedding or retrieval',
        'source_protocol_sha256': source_protocol_hash,
        'analysis_source_sha256': base.previous.experiments.sha256(Path(__file__)),
        'primary_profile': protocol['pre_confirmation_choice']['primary_profile'],
        'phase_results': {}, 'observed_http_attempts_counted_once': 0, 'http_status_counted_once': {}}
    total_status = Counter()
    for phase in phases:
        result['phase_results'][phase] = {}
        for dataset, record in protocol['datasets'].items():
            original, parents, documents, text_to_id, samples, sampling, indices = base._load_dataset(out, record, phase, 8)
            reports, exports = {}, {}
            stage_paths = set()
            for profile in trial.PROFILES:
                case = out / trial.TEMP / phase / dataset / profile
                marker = read_json(case / 'validated.ok')
                if (marker['code_sha256'] != protocol['code_sha256'] or marker['selected_indices'] != indices
                        or marker['sampling_sha256'] != record['sampling_sha256']
                        or marker['source_result_sha256'] != record['source_result_sha256']):
                    raise ValueError('Completed profile provenance differs')
                for name in ('result', 'report'):
                    base._check_file(case / f'{name}.json', marker[f'{name}_sha256'], name)
                base._check_file(marker['stage_report'], marker['stage_report_sha256'], 'shared API stages')
                stage_paths.add((marker['stage_report'], marker['stage_report_sha256']))
                reports[profile] = read_json(case / 'report.json')
                exported = read_json(case / 'result.json')
                if exported['selected_indices'] != indices or len(exported['results']) != len(indices):
                    raise ValueError('Exported sample differs')
                exports[profile] = {row['query_index']: row for row in exported['results']}
            if len(stage_paths) != 1:
                raise ValueError('Profiles do not share exactly one API stage report')
            stage_path, stage_hash = stage_paths.pop()
            stage_report = read_json(stage_path)
            for stage in ('proposer', 'verifier'):
                saved = segments[dataset, phase, stage]
                if saved['stage_report_sha256'] != stage_hash:
                    raise ValueError('HTTP archive belongs to another API stage')
                base._check_file(saved['archive_path'], saved['segment_sha256'], 'HTTP audit segment')
                statuses = Counter(re.findall(r'POST /v1/chat/completions HTTP/1\.1" (\d{3})',
                    Path(saved['archive_path']).read_text(errors='replace')))
                stats = stage_report[stage]['observed_request_stats']
                if dict(statuses) != saved['http_status'] or sum(statuses.values()) != stats['http_attempts']:
                    raise ValueError('Retained HTTP bytes disagree with client counters')
                if set(statuses) - {'200'} or stats['failures'] or stats['retries']:
                    raise ValueError('API errors invalidate this trial')
                result['observed_http_attempts_counted_once'] += stats['http_attempts']
                total_status.update(statuses)
            evaluated = {p: [] for p in (*trial.PROFILES, 'plain_joint')}
            counts = {p: Counter() for p in trial.PROFILES}
            details = []
            for index in indices:
                parent = parents[index]
                old = [text_to_id[text] for text in parent['candidate_docs']]
                trace = deepcopy(parent['retrieval_trace']['evidence'])
                state = {'evidence_trace': trace, '_evidence_plan': trace.get('plan', []),
                    '_evidence_winning_bindings': trace.get('bindings', {}), '_evidence_beams': trace.get('branch_scores', [])}
                before = base._json_sha(state)
                messages = build_joint_prompt(parent['question'], old, state, documents.__getitem__)
                response, finish = ('{"swap":null}', 'stop') if messages is None else _response(
                    metadata, dataset, index, 'proposer', messages, record, protocol, phase)
                plain, proposal = apply_joint_selection(parent['question'], old, state, documents.__getitem__, response, finish)
                guard_allowed, guard = support_guard(parent['question'], old, state, documents.__getitem__, proposal)
                review_response, review_finish = ('{}', 'stop') if plain == old else _response(metadata, dataset, index,
                    'verifier', build_verifier_prompt(parent['question'], old, state, documents.__getitem__, proposal),
                    record, protocol, phase)
                review_allowed, review = validate_verification(parent['question'], old, state, documents.__getitem__,
                    proposal, review_response, review_finish)
                if base._json_sha(state) != before:
                    raise ValueError('A replay mutated upstream state')
                evaluate = lambda ids: base.evaluate(samples, dict(parent, dataset=dataset), [documents[i] for i in ids])
                old_metrics, plain_metrics = evaluate(old), evaluate(plain)
                evaluated['plain_joint'].append(plain_metrics)
                delta = plain_metrics['metrics']['Recall@5'] - old_metrics['metrics']['Recall@5']
                outcome = 'gain' if delta > 0 else 'loss' if delta < 0 else 'neutral'
                expected = {'baseline': old, 'protected_joint': plain if guard_allowed else old,
                    'semantic_veto': plain if review_allowed else old,
                    'protected_semantic_veto': plain if guard_allowed and review_allowed else old}
                applied = {}
                for profile, order in expected.items():
                    row = exports[profile][index]
                    if [text_to_id[text] for text in row['candidate_docs']] != order or row['docs'] != [documents[i] for i in order[:10]]:
                        raise ValueError('Ranking differs from reconstructed decisions')
                    if order[:2] != old[:2] or set(order[:10]) != set(old[:10]) or order[10:] != old[10:]:
                        raise ValueError('Fixed-rank invariant failed')
                    measured = evaluate(order)
                    if (row['retrieval_metrics'] != measured['metrics']
                            or row['all_gold_in_top5'] != measured['all_gold_top5']
                            or row['all_gold_in_top10'] != measured['all_gold_top10']):
                        raise ValueError('Saved per-query metrics disagree with the reconstructed order')
                    evaluated[profile].append(measured)
                    applied[profile] = plain != old and order == plain
                    if plain != old:
                        counts[profile][('applied_' if applied[profile] else 'blocked_') + outcome] += 1
                    if profile != 'baseline':
                        saved = row['retrieval_trace']['evidence']['improvement_swap_verification']
                        if saved['base_proposal'] != proposal or saved['support_guard'] != guard or saved['independent_verification'] != review:
                            raise ValueError('Saved diagnostics disagree with the paired replay')
                if plain != old:
                    details.append({'query_index': index, 'plain_delta_r5': delta, 'guard_allowed': guard_allowed,
                        'review_allowed': review_allowed, 'applied': applied})
            aggregates = {p: base.aggregate(rows, sampling, include_all_gold=True) for p, rows in evaluated.items()}
            attribution = {}
            for profile in trial.PROFILES:
                for key in ('retrieval_metrics', 'all_gold_top5', 'all_gold_top10', 'available_pool_weighted', 'available_pool_weighted_all_gold'):
                    if reports[profile][key] != aggregates[profile][key]:
                        raise ValueError('Saved aggregate metrics disagree with CPU reconstruction')
                attribution[profile] = {'proposal_outcomes': dict(counts[profile]),
                    'raw_delta_vs_plain': base._metric_delta(aggregates[profile], aggregates['plain_joint'], weighted=False),
                    'weighted_delta_vs_plain': base._metric_delta(aggregates[profile], aggregates['plain_joint'], weighted=True),
                    'raw_delta_vs_baseline': reports[profile]['raw_delta'],
                    'weighted_delta_vs_baseline': reports[profile]['weighted_delta']}
            result['phase_results'][phase][dataset] = {'n_samples': len(indices),
                'baseline': aggregates['baseline'], 'plain_joint_counterfactual': aggregates['plain_joint'],
                'profiles': attribution, 'per_proposal': details, 'stage_report_sha256': stage_hash,
                'sampling_sha256': record['sampling_sha256'], 'source_result_sha256': record['source_result_sha256']}
            print(f'[audited] {phase}/{dataset}: four decisions, plain counterfactual, metrics and HTTP', flush=True)
    result['http_status_counted_once'] = dict(total_status)
    if base.previous.experiments.sha256(protocol_path) != source_protocol_hash:
        raise ValueError('Protocol changed during analysis')
    base._write(metadata / 'component_attribution.json', result)
    print('[done] All paired replay and archive checks passed', flush=True)


if __name__ == '__main__':
    analyze(trial.base.previous.DEFAULT_OUT)
