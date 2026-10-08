"""Paired support-retention and independent swap-review trials on frozen DAG output.

The local API proposes and reviews bounded replacements; all four profiles reuse
the same responses. No embedding model, indexing or QA is run. Historical labels
enter offline sampling/evaluation only, never either model prompt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from itertools import product
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import pyarrow.parquet as pq

from . import difficulty_sampling as difficulty
from . import exp4_improvements as previous
from .common import read_json, write_json
from .exp4_anchor_companion_trials import (
    RECALLS, _joint_http_audit, _joint_log_offset, _joint_stats_delta, _json_sha,
    aggregate, evaluate,
)


META = 'exp4_swap_verification_selection'
TEMP = '_exp4_swap_verification_trials'
PROFILES = ('baseline', 'protected_joint', 'verified_joint', 'protected_verified_joint')
GENERATION = {'model': 'qwen3-8b', 'temperature': 0., 'seed': 0, 'max_new_tokens': 2048,
              'num_gen_choices': 1, 'response_format': {'type': 'json_object'}, 'enable_thinking': False}
CODE_PATHS = {
    'runner': 'scripts/utils/exp4_swap_verification_trials.py',
    'core_proposer': 'src/pathcondrag/evidence_joint_selector.py',
    'typed_gate': 'src/pathcondrag/evidence_typed_joint_selector.py',
    'support_guard': 'src/pathcondrag/evidence_support_guard.py',
    'swap_verifier': 'src/pathcondrag/evidence_swap_verifier.py',
    'anchor_helpers': 'src/pathcondrag/evidence_anchor_companion.py',
    'selection_helpers': 'src/pathcondrag/evidence_selection.py',
    'binding_helpers': 'src/pathcondrag/evidence_binding.py',
    'retrieval_helpers': 'src/pathcondrag/evidence_retrieval.py',
    'sampling_helpers': 'scripts/utils/difficulty_sampling.py',
    'weighting_helpers': 'scripts/utils/representative_sampling.py',
    'report_helpers': 'scripts/utils/exp4_anchor_companion_trials.py',
    'experiment_helpers': 'scripts/utils/exp4_improvements.py',
    'dataset_helpers': 'scripts/utils/improvement_experiments.py',
    'common_helpers': 'scripts/utils/common.py',
    'llm_client': 'src/pathcondrag/llm/openai_gpt.py',
}


def _write(path, value):
    """Publish a complete JSON file atomically using the common serializer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def _code_hashes():
    return {key: previous.experiments.sha256(previous.ROOT / path) for key, path in CODE_PATHS.items()}


def _fresh_sampling(parent, size, seed):
    """One unused phase; preserve every structure still represented in its pool."""
    old_screen = set(parent['screen_indices'])
    old_confirmation = set(parent['old_confirmation_indices'])
    typed_confirmation = set(parent['confirmation_indices'])
    excluded = old_screen | old_confirmation | typed_confirmation
    available = [index for index in parent['available_indices'] if index not in excluded]
    features = deepcopy(parent['features'])
    labels = difficulty.DIFFICULTIES
    full, pool = {}, {}
    for row in features:
        full.setdefault(row['structure_primary'], Counter())[labels.index(row['historical_difficulty'])] += 1
    for index in available:
        row = features[index]
        pool.setdefault(row['structure_primary'], Counter())[labels.index(row['historical_difficulty'])] += 1
    mapping, strata, unavailable = {}, [], []
    for structure in sorted(full):
        if not pool.get(structure):
            unavailable.append({'structure_primary': structure, 'full_population_count': sum(full[structure].values()),
                                'available_count': 0, 'quota': 0,
                                'reason': 'All questions of this structure were previously observed; no borrowed cases'})
            continue
        choices = []
        for keep in product((False, True), repeat=3):
            edges = (0, *(i + 1 for i, yes in enumerate(keep) if yes), 4)
            bins = tuple(tuple(range(a, b)) for a, b in zip(edges, edges[1:])
                         if sum(full[structure][i] for i in range(a, b)))
            if all(sum(pool[structure][i] for i in group) >= 1 for group in bins):
                choices.append(bins)
        for group in min(choices, key=lambda bins: (-len(bins), bins)):
            names = [labels[i] for i in group]
            primary = structure + '|difficulty=' + '+'.join(names)
            for name in names:
                mapping[structure, name] = primary
            strata.append({'structure_primary': structure, 'primary': primary, 'difficulty_categories': names,
                           'merged': len(group) > 1, 'available_count': sum(pool[structure][i] for i in group),
                           'available_counts_by_difficulty': {labels[i]: pool[structure][i] for i in group}})
    groups = {}
    for row in features:
        row['primary'] = mapping.get((row['structure_primary'], row['historical_difficulty']),
                                     'unavailable:' + row['structure_primary'])
    for index in available:
        groups.setdefault(features[index]['primary'], []).append(index)
    counts = {key: len(values) for key, values in groups.items()}
    quotas = difficulty._quotas(counts, size, counts)
    selected = difficulty._draw(groups, quotas, seed)
    if set(selected) & excluded:
        raise ValueError('Fresh confirmation overlaps previously observed questions')
    return {'dataset': parent['dataset'], 'features': features, 'feature_sha256': difficulty._hash(features),
            'available_indices': available, 'available_size': len(available), 'selected_indices': selected,
            'seed': seed, 'quotas': quotas, 'difficulty_stratum_merge_audit': strata,
            'unavailable_original_structures': unavailable,
            'coverage': difficulty._coverage(features, selected, available, available, quotas, 'confirmation'),
            'sampling_policy': 'Single-phase strict random sampling without replacement within structure/difficulty cells; '
                               'minimum one per available cell; contiguous difficulty merging; unavailable structures excluded explicitly',
            'sampling_protocol_sha256': difficulty._hash({'available': available, 'selected': selected,
                                                         'seed': seed, 'quotas': quotas, 'strata': strata}),
            'limits': ['Fresh for these modules, not an untouched benchmark test.',
                       'Coverage concerns available original structures, not exhausted full-population structures.']}


def _check_file(path, expected, description):
    if previous.experiments.sha256(path) != expected:
        raise ValueError(f'Frozen {description} changed: {path}')


def prepare(args):
    out = Path(args.out_root).resolve()
    metadata = out / 'metadata' / META
    path = metadata / 'selection.json'
    if path.is_file():
        summary = read_json(path)
        if (summary['code_sha256'] != _code_hashes() or summary['protocol']['llm_base_url'] != args.llm_base_url
                or summary['protocol']['confirmation_size'] != args.confirmation_size
                or summary['protocol']['confirmation_seed'] != args.confirmation_seed):
            raise ValueError('Code, endpoint or sampling differs from this frozen trial namespace')
        _check_file(summary['parent']['path'], summary['parent']['sha256'], 'typed parent protocol')
        return summary
    parent_path = out / 'metadata/exp4_typed_joint_selector_selection/selection.json'
    parent = read_json(parent_path)
    if parent['protocol']['generation'] != GENERATION or parent['protocol']['llm_base_url'] != args.llm_base_url:
        raise ValueError('Proposal generation must match the frozen typed experiment/cache')
    summary = {'protocol': {'profiles': list(PROFILES), 'screen_size': 60, 'regression_size': 30,
                'confirmation_size': args.confirmation_size, 'confirmation_seed': args.confirmation_seed,
                'llm_base_url': args.llm_base_url, 'generation': GENERATION, 'workers': 8, 'max_model_len': 8192,
                'tokenizer_path': '/root/models/Qwen3-8B',
                'cache_save_dir': str(out / 'metadata/exp4_joint_selector_selection'),
                'candidate_policy': 'Top2 fixed, Top10 set fixed, suffix rank11+ fixed; no new retrieval',
                'regression_limit': .01, 'regression_metrics': list(RECALLS) + ['all_gold_top5', 'all_gold_top10'],
                'phase_scope': {'screen': 'Observed original 60 development questions; weighting pool 289',
                                'regression': 'Observed typed 30 confirmation questions now used as regression checks; weighting pool 199',
                                'confirmation': 'New unused draw from remaining 169; no full-test or exhausted-structure coverage claim'},
                'api_cost_policy': 'Stage costs observed once and shared across profiles; logical calls recorded separately',
                'verification_policy': 'Independent review for every accepted base proposal, including local-guard vetoes',
                'warning': 'Exploratory development; frozen modules must not be redesigned after fresh confirmation.'},
               'code_sha256': _code_hashes(), 'parent': {'path': str(parent_path),
                    'sha256': previous.experiments.sha256(parent_path)}, 'datasets': {}, 'phase_results': {}}
    for dataset, record in parent['datasets'].items():
        _check_file(record['sampling_file'], record['sampling_sha256'], 'typed sampling')
        old_sampling = read_json(record['sampling_file'])
        fresh = _fresh_sampling(old_sampling, args.confirmation_size, args.confirmation_seed)
        phases = {'screen': {'indices': record['screen_indices'], 'sampling': old_sampling},
                  'regression': {'indices': record['confirmation_indices'],
                                 'sampling': old_sampling['fresh_confirmation_sampling']},
                  'confirmation': {'indices': fresh['selected_indices'], 'sampling': fresh}}
        if len(phases['screen']['indices']) != 60 or len(phases['regression']['indices']) != 30:
            raise ValueError('Expected frozen 60 screening and 30 observed regression questions')
        sampling_path = metadata / f'{dataset}_sampling.json'
        _write(sampling_path, {'dataset': dataset, 'phases': phases,
                               'parent_sampling': {'path': record['sampling_file'], 'sha256': record['sampling_sha256']}})
        manifest_path = out / 'metadata' / dataset / 'manifest.json'
        manifest = read_json(manifest_path)
        _check_file(record['source_result'], record['source_result_sha256'], 'DAG retrieval result')
        previous.verify_trial_report(Path(record['source_result']).parent)
        summary['datasets'][dataset] = {
            'source_result': record['source_result'], 'source_result_sha256': record['source_result_sha256'],
            'source_report': str(Path(record['source_result']).with_name('report.json')),
            'source_report_sha256': record['source_report_sha256'],
            'sampling_file': str(sampling_path), 'sampling_sha256': previous.experiments.sha256(sampling_path),
            'manifest_file': str(manifest_path), 'manifest_sha256': previous.experiments.sha256(manifest_path),
            'data_path': manifest['data_path'], 'data_sha256': previous.experiments.sha256(manifest['data_path']),
            'fresh_pool_size': fresh['available_size'], 'unavailable_original_structures': fresh['unavailable_original_structures']}
    _write(path, summary)
    print('[prepared]', {dataset: record['fresh_pool_size'] for dataset, record in summary['datasets'].items()}, flush=True)
    return summary


def _load_dataset(out, record, phase, pilot_size):
    for path_key, hash_key in (('source_result', 'source_result_sha256'), ('source_report', 'source_report_sha256'),
                               ('sampling_file', 'sampling_sha256'), ('manifest_file', 'manifest_sha256'),
                               ('data_path', 'data_sha256')):
        _check_file(record[path_key], record[hash_key], path_key)
    original = read_json(record['source_result'])
    rows = {row['query_index']: row for row in original['results']}
    manifest = read_json(record['manifest_file'])
    parquet = out / 'shared_indexes' / original['dataset'] / manifest['model_dir'] / 'chunk_embeddings/vdb_chunk.parquet'
    _check_file(parquet, manifest['source_asset_sha256']['chunk_embeddings/vdb_chunk.parquet'], 'passage index')
    documents = pq.read_table(parquet, columns=['content']).column('content').to_pylist()
    text_to_id = {text: index for index, text in enumerate(documents)}
    if len(text_to_id) != len(documents):
        raise ValueError('Duplicate passage texts make physical doc IDs ambiguous')
    phases = read_json(record['sampling_file'])['phases']
    selected = phases['screen' if phase == 'pilot' else phase]
    indices = selected['indices'][:pilot_size] if phase == 'pilot' else selected['indices']
    return original, rows, documents, text_to_id, read_json(record['data_path']), selected['sampling'], indices


def _preflight(tokenizer, messages):
    count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    if count + 2048 > 8192:
        raise ValueError(f'Prompt requires {count}+2048 tokens, exceeding unchanged 8192 context before HTTP')
    return count


def _fetch_stage(llm, tokenizer, prompts, metadata, dataset, phase, stage, record, summary, server_log):
    """One bounded stage; profile reuse never triggers another API request."""
    started, before = time.monotonic(), llm.get_request_stats()
    log_start = _joint_log_offset(server_log)
    token_counts = {index: _preflight(tokenizer, messages) for index, messages in prompts.items()}
    responses = {}

    def request(index, messages):
        manifest = {'stage': stage, 'messages_sha256': _json_sha(messages), 'generation': GENERATION,
                    'llm_base_url': llm.llm_base_url, 'source_result_sha256': record['source_result_sha256'],
                    'code_sha256': summary['code_sha256']}
        path = metadata / 'responses' / dataset / f'{index}_{stage}.json'
        prior = read_json(path) if path.is_file() else None
        if prior and prior['prompt_manifest_sha256'] != _json_sha(manifest):
            raise ValueError(f'Frozen {stage} prompt changed: {dataset}/{index}')
        response, info, cached = llm.infer(messages, response_format={'type': 'json_object'})
        value = {'dataset': dataset, 'query_index': index, 'stage': stage, 'messages': messages,
                 'response': response, 'metadata': info, 'cache_hit': bool(cached),
                 'prompt_manifest': manifest, 'prompt_manifest_sha256': _json_sha(manifest),
                 'prompt_tokens_preflight': token_counts[index],
                 'observations': (prior or {}).get('observations', []) + [
                     {'phase': phase, 'cache_hit': bool(cached), 'metadata': info, 'response_sha256': _json_sha(response)}]}
        _write(path, value)
        return index, value

    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(request, index, messages): index for index, messages in prompts.items()}
        try:
            for future in as_completed(pending):
                index, response = future.result()
                responses[index] = response
                print(f'[{phase}/{stage}] {dataset} {len(responses)}/{len(prompts)} '
                      f'index={index} cache={response["cache_hit"]}', flush=True)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    stats = _joint_stats_delta(before, llm.get_request_stats())
    if stats['failures'] or stats['retries']:
        raise RuntimeError(f'{stage} request failure invalidates the trial: {stats}')
    log_end, statuses = _joint_http_audit(server_log, log_start, stats['http_attempts'])
    return responses, {'logical_calls': len(prompts), 'observed_request_stats': stats,
                       'seconds': time.monotonic() - started, 'http_status_in_log': statuses,
                       'http_log_segment': {'path': str(server_log), 'start': log_start, 'end': log_end},
                       'max_prompt_tokens': max(token_counts.values(), default=0), 'profile_costs_shared': True}


def _metric_delta(report, baseline, *, weighted):
    if weighted:
        if report['available_pool_weighted'] is None:
            return None
        delta = {key: report['available_pool_weighted']['metrics'][key]
                      - baseline['available_pool_weighted']['metrics'][key] for key in RECALLS}
        delta.update({key: report['available_pool_weighted_all_gold']['metrics'][key]
                           - baseline['available_pool_weighted_all_gold']['metrics'][key]
                      for key in ('all_gold_top5', 'all_gold_top10')})
        return delta
    delta = {key: report['retrieval_metrics'][key] - baseline['retrieval_metrics'][key] for key in RECALLS}
    delta.update({key: report[key] - baseline[key] for key in ('all_gold_top5', 'all_gold_top10')})
    return delta


def _output_row(parent, old, new, documents, measured, diagnostic, profile):
    row = deepcopy(parent)
    if profile == 'baseline':
        return row
    candidates = [documents[doc_id] for doc_id in new]
    row.update(docs=candidates[:10], candidate_docs=candidates, retrieval_metrics=measured['metrics'],
               all_gold_in_top5=measured['all_gold_top5'], all_gold_in_top10=measured['all_gold_top10'],
               gold_document_ranks=[{'doc': text, 'rank': candidates.index(text) + 1 if text in candidates else None}
                                    for text in parent['gold_docs']])
    if new != old:
        row['candidate_doc_scores'] = [(len(new) - rank) / len(new) for rank in range(len(new))]
        row['doc_scores'] = row['candidate_doc_scores'][:10]
        prefix = {item['doc_id']: item for item in parent['retrieval_trace']['evidence'].get('selected_prefix', [])}
        row['retrieval_trace']['evidence']['selected_prefix'] = [
            dict(prefix.get(doc_id, {'doc_id': doc_id, 'selection_source': profile}),
                 original_greedy_rank=old.index(doc_id) + 1) for doc_id in new[:5]]
    row['retrieval_trace']['evidence']['improvement_swap_verification'] = diagnostic
    return row


def _resumed(out, dataset, phase, record, summary, indices):
    reports = {}
    for profile in PROFILES:
        case = out / TEMP / phase / dataset / profile
        marker_path = case / 'validated.ok'
        if not marker_path.is_file():
            return None
        marker = read_json(marker_path)
        if (marker['code_sha256'] != summary['code_sha256'] or marker['sampling_sha256'] != record['sampling_sha256']
                or marker['source_result_sha256'] != record['source_result_sha256']
                or marker['selected_indices'] != indices):
            raise ValueError('Existing output belongs to a different frozen trial')
        _check_file(case / 'result.json', marker['result_sha256'], 'completed result')
        _check_file(case / 'report.json', marker['report_sha256'], 'completed report')
        _check_file(marker['stage_report'], marker['stage_report_sha256'], 'shared API stage report')
        reports[profile] = read_json(case / 'report.json')
    return reports


def run_dataset(args, summary, llm, tokenizer, dataset, phase):
    from pathcondrag.evidence_typed_joint_selector import build_joint_prompt, apply_joint_selection
    from pathcondrag.evidence_support_guard import support_guard
    from pathcondrag.evidence_swap_verifier import build_verifier_prompt, validate_verification

    out = Path(args.out_root).resolve()
    metadata, record = out / 'metadata' / META, summary['datasets'][dataset]
    original, rows, documents, text_to_id, samples, sampling, indices = _load_dataset(out, record, phase, args.pilot_size)
    resumed = _resumed(out, dataset, phase, record, summary, indices) if args.resume else None
    if resumed:
        print(f'[resumed] {dataset}/{phase}: four exact validated profiles', flush=True)
        return resumed
    contexts, prompts = {}, {}
    for index in indices:
        parent = rows[index]
        old = [text_to_id[text] for text in parent['candidate_docs']]
        trace = deepcopy(parent['retrieval_trace']['evidence'])
        state = {'evidence_trace': trace, '_evidence_plan': trace.get('plan', []),
                 '_evidence_winning_bindings': trace.get('bindings', {}), '_evidence_beams': trace.get('branch_scores', [])}
        before = _json_sha(state)
        messages = build_joint_prompt(parent['question'], old, state, documents.__getitem__)
        if _json_sha(state) != before:
            raise ValueError('Proposer prompt mutated upstream state')
        contexts[index] = {'query': parent['question'], 'old': old, 'state': state, 'state_sha256': before}
        if messages is not None:
            prompts[index] = messages
    server_log = Path(args.vllm_log or out / 'logs/vllm_joint_selector.log')
    proposals, proposer_stats = _fetch_stage(llm, tokenizer, prompts, metadata, dataset, phase,
                                            'proposer', record, summary, server_log)
    verifier_prompts = {}
    for index, context in contexts.items():
        response = proposals.get(index)
        proposed, diagnostic = apply_joint_selection(context['query'], context['old'], context['state'], documents.__getitem__,
            response['response'] if response else '{"swap": null}', response['metadata'].get('finish_reason') if response else 'stop')
        allowed, guard = support_guard(context['query'], context['old'], context['state'], documents.__getitem__, diagnostic)
        context.update(proposed=proposed, proposal=diagnostic, guard_allowed=allowed, guard=guard)
        if proposed != context['old']:
            messages = build_verifier_prompt(context['query'], context['old'], context['state'], documents.__getitem__, diagnostic)
            if messages is None:
                raise ValueError('Accepted base proposal could not produce an independent review prompt')
            verifier_prompts[index] = messages  # Review even if the local guard vetoed it.
        if _json_sha(context['state']) != context['state_sha256']:
            raise ValueError('Proposal/guard mutated upstream state')
    reviews, verifier_stats = _fetch_stage(llm, tokenizer, verifier_prompts, metadata, dataset, phase,
                                          'verifier', record, summary, server_log)
    for index, context in contexts.items():
        response = reviews.get(index)
        approved, review = validate_verification(context['query'], context['old'], context['state'], documents.__getitem__,
            context['proposal'], response['response'] if response else '{}',
            response['metadata'].get('finish_reason') if response else 'stop')
        context.update(verifier_approved=approved, verifier=review)
        if _json_sha(context['state']) != context['state_sha256']:
            raise ValueError('Independent review mutated upstream state')
    stage_path = metadata / 'stage_reports' / dataset / f'{phase}.json'
    stages = {'dataset': dataset, 'phase': phase, 'proposer': proposer_stats, 'verifier': verifier_stats,
              'code_sha256': summary['code_sha256'], 'source_result_sha256': record['source_result_sha256']}
    _write(stage_path, stages)
    baseline_rows = [evaluate(samples, dict(rows[index], dataset=dataset), rows[index]['candidate_docs']) for index in indices]
    baseline_by_index = {row['query_index']: row for row in baseline_rows}
    baseline = aggregate(baseline_rows, sampling, partial=phase == 'pilot', include_all_gold=True)
    reports = {}
    for profile in PROFILES:
        started, output, measured_rows, changes = time.monotonic(), [], [], []
        decisions, guard_vetoes, review_vetoes = Counter(), Counter(), Counter()
        for index, context in contexts.items():
            old = context['old']
            proposed = context['proposed'] != old
            allowed = proposed and profile != 'baseline'
            if profile in ('protected_joint', 'protected_verified_joint'):
                allowed = allowed and context['guard_allowed']
            if profile in ('verified_joint', 'protected_verified_joint'):
                allowed = allowed and context['verifier_approved']
            new = context['proposed'] if allowed else old
            if (new[:2] != old[:2] or set(new[:10]) != set(old[:10]) or new[10:] != old[10:]
                    or len(new) != len(set(new)) or _json_sha(context['state']) != context['state_sha256']):
                raise ValueError(f'Ranking/state invariant failed: {dataset}/{phase}/{profile}/{index}')
            if not set(context['guard'].get('protected_dag_doc_ids', [])) <= set(new[:5]):
                raise ValueError('A profile removed protected DAG source evidence')
            if profile in ('protected_joint', 'protected_verified_joint') and not set(
                    context['guard'].get('protected_doc_ids', [])) <= set(new[:5]):
                raise ValueError('Protected profile removed existing support or a question anchor')
            measured = evaluate(samples, dict(rows[index], dataset=dataset), [documents[doc_id] for doc_id in new])
            expected = baseline_by_index[index]
            if any(measured['metrics'][key] != expected['metrics'][key] for key in RECALLS if key != 'Recall@5'):
                raise ValueError('Protected Recall changed despite fixed Top2/Top10')
            delta = measured['metrics']['Recall@5'] - expected['metrics']['Recall@5']
            diagnostic = {'profile': profile, 'accepted': bool(allowed), 'base_proposal': context['proposal'],
                          'support_guard': context['guard'], 'independent_verification': context['verifier'],
                          'gold_labels_used': False, 'shared_stage_report': str(stage_path)}
            output.append(_output_row(rows[index], old, new, documents, measured, diagnostic, profile))
            measured_rows.append(measured)
            decisions['base_accepted' if proposed else 'base_abstained'] += 1
            decisions['applied' if allowed else 'unchanged'] += 1
            if proposed and not context['guard_allowed']:
                guard_vetoes[context['guard'].get('veto_reason') or 'unknown'] += 1
            if proposed and not context['verifier_approved']:
                review_vetoes[context['verifier'].get('rejection_reason') or 'unknown'] += 1
            if new != old:
                changes.append({'query_index': index, 'question': rows[index]['question'], 'delta_r5': delta,
                                'delta_all_gold5': int(measured['all_gold_top5']) - int(expected['all_gold_top5']),
                                'diagnostic': diagnostic})
        report = aggregate(measured_rows, sampling, partial=phase == 'pilot', include_all_gold=True)
        raw_delta, weighted_delta = _metric_delta(report, baseline, weighted=False), _metric_delta(report, baseline, weighted=True)
        logical_calls = {'proposer': 0 if profile == 'baseline' else proposer_stats['logical_calls'],
                         'verifier': verifier_stats['logical_calls'] if profile in ('verified_joint', 'protected_verified_joint') else 0}
        report.update(profile=profile, dataset=dataset, phase=phase, evaluation_scope='frozen_candidate_paired_swap_verification',
            selector_seconds=time.monotonic() - started, retrieval_seconds=None,
            shared_stage_report=str(stage_path), shared_stage_report_sha256=previous.experiments.sha256(stage_path),
            logical_llm_calls=logical_calls, observed_shared_stage_stats=stages,
            api_attribution='Observed stage costs are shared, counted once per dataset/phase, not additive across profiles.',
            changed_questions=len(changes), wins=sum(row['delta_r5'] > 0 for row in changes),
            losses=sum(row['delta_r5'] < 0 for row in changes), neutral=sum(row['delta_r5'] == 0 for row in changes),
            all_gold5_wins=sum(row['delta_all_gold5'] > 0 for row in changes),
            all_gold5_losses=sum(row['delta_all_gold5'] < 0 for row in changes),
            changes=changes, decision_counts=dict(decisions), support_guard_vetoes=dict(guard_vetoes),
            verifier_vetoes=dict(review_vetoes), raw_delta=raw_delta, weighted_delta=weighted_delta,
            regression_limit_met=all(value >= -.01 - 1e-12 for values in (raw_delta, weighted_delta or {}) for value in values.values()),
            regression_check_scope='Raw pilot only' if phase == 'pilot' else 'Raw and weighted six Recall plus All gold@5/@10',
            phase_scope='First screening indices, partial coverage, no weighting' if phase == 'pilot'
                        else summary['protocol']['phase_scope'][phase], code_sha256=summary['code_sha256'])
        stats = ({'http_attempts': 0, 'cache_hits': 0, 'failures': 0, 'retries': 0, 'max_in_flight': 8,
                  'attribution': 'Baseline invokes neither added model stage'} if profile == 'baseline' else
                 {'attribution': 'Shared observed stages; no unique per-profile HTTP allocation',
                  'logical_calls': logical_calls, 'shared_stage_report': str(stage_path),
                  'stages': {'proposer': proposer_stats['observed_request_stats'], 'verifier': verifier_stats['observed_request_stats']}})
        result = {key: original[key] for key in ('dataset', 'eval_mode', 'indexed_docs', 'result_top_k',
                  'candidate_output_top_k', 'hop_source') if key in original}
        result.update(results=output, selected_indices=indices, sample_size_requested=len(indices),
            sample_size_effective=len(indices), retrieval_metrics=report['retrieval_metrics'],
            retrieval_seconds=None, selector_seconds=report['selector_seconds'],
            source_result=record['source_result'], source_result_sha256=record['source_result_sha256'],
            upstream_runtime_config=original.get('runtime_config'), evaluation_scope=report['evaluation_scope'],
            postprocessing_config={'profile': profile, 'generation': GENERATION, 'workers': 8,
                                   'code_sha256': summary['code_sha256'], 'shared_stage_report': str(stage_path)},
            llm_request_stats=stats, hop_distribution=dict(Counter(str(row['benchmark_hops']) for row in output)))
        case = out / TEMP / phase / dataset / profile
        (case / 'validated.ok').unlink(missing_ok=True)
        _write(case / 'result.json', result)
        _write(case / 'report.json', report)
        _write(case / 'validated.ok', {'result_sha256': previous.experiments.sha256(case / 'result.json'),
            'report_sha256': previous.experiments.sha256(case / 'report.json'), 'code_sha256': summary['code_sha256'],
            'sampling_sha256': record['sampling_sha256'], 'source_result_sha256': record['source_result_sha256'],
            'selected_indices': indices,
            'stage_report': str(stage_path), 'stage_report_sha256': previous.experiments.sha256(stage_path)})
        reports[profile] = report
        print(f'[{phase}] {dataset}/{profile} n={len(indices)} R5={report["retrieval_metrics"]["Recall@5"]:.5f} '
              f'delta={raw_delta["Recall@5"]:+.5f} Allgold5delta={raw_delta["all_gold_top5"]:+.5f} '
              f'wins/losses={report["wins"]}/{report["losses"]} guard={report["regression_limit_met"]}', flush=True)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', default=str(previous.DEFAULT_OUT))
    parser.add_argument('--phase', choices=('prepare', 'pilot', 'screen', 'regression', 'confirmation', 'all'), default='screen')
    parser.add_argument('--pilot-size', type=int, default=2)
    parser.add_argument('--confirmation-size', type=int, default=30)
    parser.add_argument('--confirmation-seed', type=int, default=2558)
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=None)
    parser.add_argument('--resume', action='store_true', help='Reuse only exact validated four-profile output and stage hashes')
    args = parser.parse_args(argv)
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(previous.ROOT / 'outputs') or out == previous.ROOT / 'outputs':
        raise ValueError('Output must be a dedicated path under PathCondRAG/outputs')
    if not 1 <= args.pilot_size <= 60 or args.confirmation_size < 1:
        raise ValueError('Invalid pilot/confirmation size')
    summary = prepare(args)
    if args.phase == 'prepare':
        return 0
    from pathcondrag.llm.openai_gpt import CacheOpenAI
    from pathcondrag.utils.config_utils import BaseConfig
    from transformers import AutoTokenizer
    if urlsplit(args.llm_base_url).hostname in ('127.0.0.1', 'localhost', '::1'):
        os.environ.setdefault('OPENAI_API_KEY', 'sk-local')
    tokenizer = AutoTokenizer.from_pretrained('/root/models/Qwen3-8B', trust_remote_code=True, local_files_only=True)
    config = BaseConfig(llm_name='qwen3-8b', llm_base_url=args.llm_base_url, temperature=0., seed=0,
                        max_new_tokens=2048, num_gen_choices=1, save_dir=summary['protocol']['cache_save_dir'], max_retry_attempts=1)
    llm = CacheOpenAI.from_experiment_config(config)
    llm.max_retries = 1
    phases = ('screen', 'regression', 'confirmation') if args.phase == 'all' else (args.phase,)
    try:
        for phase in phases:
            for dataset in previous.DATASETS:
                reports = run_dataset(args, summary, llm, tokenizer, dataset, phase)
                for profile, report in reports.items():
                    summary['phase_results'].setdefault(phase, {}).setdefault(profile, {})[dataset] = {
                        key: value for key, value in report.items() if key not in ('per_question', 'changes')}
                _write(out / 'metadata' / META / 'selection.json', summary)
    finally:
        llm.openai_client.close()
    return 0
