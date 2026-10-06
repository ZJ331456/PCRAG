"""Evaluate the second improvement round with paired small-sample controls.

This runner never launches full retrieval. Screening and confirmation remain
exploratory because their questions come from the existing evaluation files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from . import exp4_improvements as previous
from .common import read_json, write_json


ROOT = previous.ROOT
DATASETS = previous.DATASETS
TEMP_NAME = '_exp4_round2_trials'
SUMMARY_NAME = 'exp4_round2_selection'
METRICS = ('Recall@5', 'Recall@10')


@dataclass(frozen=True)
class Variant:
    name: str
    flags: tuple[str, ...] = ()
    adaptive_mode: str = 'both'
    binding_validation: str = 'legacy'


BASELINE = Variant('baseline')
SMOKE_COMBINATION = Variant('all_combination_strict', previous.FLAGS, 'both', 'strict_relation')
VARIANTS = (
    Variant('closure_fixed', ('closure',)),
    Variant('adaptive_verify', ('adaptive',), 'verify_only'),
    Variant('adaptive_beam', ('adaptive',), 'beam_only'),
    Variant('adaptive_both', ('adaptive',)),
    Variant('planning_selection', ('planning', 'selection')),
    Variant('binding_strict', ('binding',), binding_validation='strict_relation'),
    Variant('binding_strict_adaptive_verify', ('binding', 'adaptive'), 'verify_only', 'strict_relation'),
    Variant('binding_strict_adaptive_both', ('binding', 'adaptive'), 'both', 'strict_relation'),
)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prior_exclusions(out):
    """Keep previously tuned round1 questions out of round2 development."""
    source = out / 'metadata/exp4_improvement_selection/selection.json'
    earlier = read_json(source)['screen']
    indices = {dataset: sorted(set(earlier['screen_indices'][dataset]) |
                               set(earlier['confirmation_indices'][dataset])) for dataset in DATASETS}
    return {'source': str(source), 'source_sha256': previous.experiments.sha256(source),
            'excluded_indices': indices,
            'reason': 'Exclude all round1 screening and confirmation questions from round2 screening/confirmation.'}


def target_assessment(reports, baselines, *, target_gain=4.0, gain_unit='absolute_pp',
                      max_regression=1.0, allowed_exceptions=2):
    """Assess the requested gain without reducing thresholds at metric ceilings.

    Regression limits always use absolute percentage points. The paper target
    additionally needs a qualifying metric on every dataset.
    """
    if gain_unit not in ('absolute_pp', 'relative_percent', 'error_reduction'):
        raise ValueError('Unknown gain unit')
    details = {}
    reached = 0
    protected = True
    for dataset in DATASETS:
        details[dataset] = {}
        for metric in METRICS:
            old = float(baselines[dataset]['retrieval_metrics'][metric])
            new = float(reports[dataset]['retrieval_metrics'][metric])
            if not (0 <= old <= 1 and 0 <= new <= 1):
                raise ValueError('Recall values must be in [0, 1]')
            delta_pp = (new - old) * 100
            denominator = 1.0 if gain_unit == 'absolute_pp' else (
                old if gain_unit == 'relative_percent' else 1 - old)
            gain = (new - old) / denominator * 100 if denominator > 0 else None
            maximum = (1 - old) / denominator * 100 if denominator > 0 else None
            achievable = maximum is not None and maximum >= target_gain - 1e-8
            met = gain is not None and gain >= target_gain - 1e-8
            protected = protected and delta_pp >= -max_regression - 1e-8
            reached += int(met)
            details[dataset][metric] = {
                'baseline': old, 'candidate': new, 'delta_pp': delta_pp,
                'gain': gain, 'gain_unit': gain_unit, 'target_gain': target_gain,
                'maximum_possible_gain': maximum, 'unreachable': not achievable,
                'target_met': met, 'regression_limit_met': delta_pp >= -max_regression - 1e-8}
    every_dataset = all(any(row['target_met'] for row in rows.values()) for rows in details.values())
    required = len(DATASETS) * len(METRICS) - allowed_exceptions
    mean5 = sum(details[name]['Recall@5']['delta_pp'] for name in DATASETS) / len(DATASETS)
    mean10 = sum(details[name]['Recall@10']['delta_pp'] for name in DATASETS) / len(DATASETS)
    strong = reached >= required and every_dataset and protected
    weak = mean5 > 1e-6 and mean10 >= -max_regression - 1e-8 and protected
    return {'strong_target_met': strong, 'strict_target_met': reached == 6 and protected,
            'weak_signal': weak, 'qualified_metrics': reached, 'required_metrics': required,
            'every_dataset_has_target_gain': every_dataset, 'regression_limit_met': protected,
            'mean_gain_r5_pp': mean5, 'mean_gain_r10_pp': mean10,
            'score': .7 * mean5 + .3 * mean10, 'metric_details': details,
            'unreachable_metrics': [f'{name}/{metric}' for name in DATASETS for metric in METRICS
                                    if details[name][metric]['unreachable']],
            'exception_metrics': [f'{name}/{metric}' for name in DATASETS for metric in METRICS
                                  if not details[name][metric]['target_met']]}


def assess(args, reports, baselines):
    return target_assessment(reports, baselines, target_gain=args.target_gain,
                             gain_unit=args.gain_unit, max_regression=args.max_regression,
                             allowed_exceptions=args.allowed_exceptions)


def command(args, context, case, indices, variant):
    return previous.command(args, context, case, indices, frozenset(variant.flags)) + [
        '--evidence_adaptive_mode', variant.adaptive_mode,
        '--evidence_binding_validation', variant.binding_validation]


def verify_saved_case(context, case, manifest, variant):
    marker = read_json(case / 'validated.ok')
    previous.verify_trial_report(case)
    if read_json(case / 'manifest.json') != manifest:
        raise ValueError(f'Question subset changed: {case}')
    if (read_json(case / 'round2_config.json') != json.loads(json.dumps(asdict(variant))) or
            marker.get('round2_config_sha256') != previous.experiments.sha256(case / 'round2_config.json')):
        raise ValueError(f'Variant configuration changed: {case}')
    expected = context['manifest']['source_asset_sha256']
    before = read_json(case / 'before.json')
    if (previous.experiments.asset_hashes(context['source'], manifest['model_dir']) != expected or
            before['asset_sha256'] != expected or before['initial_cache_sha256'] != context['cache_hashes']):
        raise ValueError(f'Public index or initial cache changed: {case}')
    if marker.get('cleaned_private_index') or (marker.get('private_index_cleanup_pending') and
                                              not (case / 'index').exists()):
        proof = case / 'private_index_cleanup.json'
        if marker.get('cleanup_proof_sha256') != previous.experiments.sha256(proof):
            raise ValueError(f'Private index cleanup proof changed: {case}')
        saved = read_json(proof)
        if (saved['asset_sha256_at_cleanup'] != expected or
                saved['initial_cache_sha256'] != context['cache_hashes'] or (case / 'index').exists()):
            raise ValueError(f'Private index cleanup state differs: {case}')
        if not marker.get('cleaned_private_index'):
            marker.update(cleaned_private_index=True, private_index_cleanup_pending=False)
            write_json(case / 'validated.ok', marker)
    elif previous.experiments.asset_hashes(case / 'index', manifest['model_dir']) != expected:
        raise ValueError(f'Private index changed: {case}')
    return read_json(case / 'report.json')


def cleanup_private_index(context, case):
    """Keep exports and signed cleanup evidence; unlink only this private index."""
    marker = read_json(case / 'validated.ok')
    if marker.get('cleaned_private_index'):
        return
    index = case / 'index'
    expected = context['manifest']['source_asset_sha256']
    if (index.is_symlink() or not index.is_dir() or
            previous.experiments.asset_hashes(index, context['manifest']['model_dir']) != expected):
        raise ValueError(f'Refusing cleanup of an unverified private index: {index}')
    before = read_json(case / 'before.json')
    proof = case / 'private_index_cleanup.json'
    write_json(proof, {'asset_sha256_at_cleanup': expected,
                       'initial_cache_sha256': before['initial_cache_sha256'],
                       'scope': 'only private case/index; exported Top200/Top10, traces and reports retained'})
    marker.update(private_index_cleanup_pending=True,
                  cleanup_proof_sha256=previous.experiments.sha256(proof))
    write_json(case / 'validated.ok', marker)
    shutil.rmtree(index)
    marker.update(cleaned_private_index=True, private_index_cleanup_pending=False)
    write_json(case / 'validated.ok', marker)


def run_case(args, context, case, indices, variant, env, *, cleanup=True):
    manifest = previous.subset_manifest(context['manifest'], context['data'], context['hops'], indices)
    if (case / 'validated.ok').is_file():
        report = verify_saved_case(context, case, manifest, variant)
        if cleanup and not read_json(case / 'validated.ok').get('cleaned_private_index'):
            cleanup_private_index(context, case)
        return report
    # The existing initializer rejects unfinished directories rather than
    # overwriting checkpoints or pretending a partial result is complete.
    previous.initialize_case(context, case)
    write_json(case / 'selected_indices.json', list(indices))
    write_json(case / 'manifest.json', manifest)
    write_json(case / 'round2_config.json', asdict(variant))
    start = Path(args.vllm_log).stat().st_size
    elapsed = previous.execute(command(args, context, case, indices, variant), case / 'run.log', env)
    end = Path(args.vllm_log).stat().st_size
    report, result = previous.validate_case(args, context, case, manifest,
                                           frozenset(variant.flags), elapsed, start, end)
    runtime = result.get('runtime_config', {})
    for key, expected in [('evidence_adaptive_mode', variant.adaptive_mode),
                          ('evidence_binding_validation', variant.binding_validation)]:
        if runtime.get(key) != expected:
            # Do not retain the old helper's marker after a round-specific
            # validation failure; the exported result remains for inspection.
            (case / 'validated.ok').unlink()
            raise ValueError(f'{case}: {key}={runtime.get(key)} != {expected}')
    report['round2_config'] = asdict(variant)
    write_json(case / 'report.json', report)
    marker = read_json(case / 'validated.ok')
    marker.update(report_sha256=previous.experiments.sha256(case / 'report.json'),
                  round2_config_sha256=previous.experiments.sha256(case / 'round2_config.json'))
    write_json(case / 'validated.ok', marker)
    if cleanup:
        cleanup_private_index(context, case)
    return report


def update_progress(summary, summary_file):
    plans = summary['group_plan']
    completed = sum(item.get('validated') is True for item in plans)
    summary['progress'] = {'planned_groups': len(plans), 'completed_groups': completed,
                           'remaining_groups': len(plans) - completed,
                           'maximum_possible_groups': 48,
                           'optional_groups_note': 'Up to 3 evidence-based combination groups and '
                                                   'up to 9 confirmation groups are added after screening.'}
    write_json(summary_file, summary)
    print(f'[round2-progress] planned={len(plans)} completed={completed} '
          f'remaining={len(plans)-completed} maximum=48', flush=True)


def register_groups(summary, summary_file, phase, variants, count):
    known = {item['id'] for item in summary['group_plan']}
    for variant in variants:
        for dataset in DATASETS:
            name = f'{phase}/{dataset}/{variant.name}'
            if name not in known:
                summary['group_plan'].append({'id': name, 'phase': phase, 'dataset': dataset,
                                             'variant': variant.name, 'samples': count, 'validated': False})
    update_progress(summary, summary_file)


def mark_complete(summary, summary_file, phase, dataset, variant):
    name = f'{phase}/{dataset}/{variant.name}'
    for item in summary['group_plan']:
        if item['id'] == name:
            item['validated'] = True
            update_progress(summary, summary_file)
            return
    raise ValueError(f'Unplanned group completed: {name}')


def phase_baselines(args, contexts, temp, env, summary, summary_file, phase, indices):
    records = summary.setdefault(phase, {})
    reports = records.setdefault('baselines', {})
    phase_contexts = {}
    for dataset, context in contexts.items():
        case = temp / phase / dataset / BASELINE.name
        reports[dataset] = run_case(args, context, case, indices[dataset], BASELINE, env, cleanup=False)
        cache = temp / 'phase_start_cache' / phase / dataset
        saved_hash = records.setdefault('phase_start_cache_sha256', {}).get(dataset)
        if not cache.is_dir():
            if read_json(case / 'validated.ok').get('cleaned_private_index'):
                raise ValueError(f'{dataset}/{phase}: required phase-start cache was deleted')
            previous.snapshot_cache(case / 'index/llm_cache', cache)
        current_hash = previous.experiments.cache_hashes(cache)
        if saved_hash is not None and saved_hash != current_hash:
            raise ValueError(f'{dataset}/{phase}: shared variant starting cache changed')
        records['phase_start_cache_sha256'][dataset] = current_hash
        phase_contexts[dataset] = dict(context, cache=cache, cache_hashes=current_hash)
        write_json(summary_file, summary)
        cleanup_private_index(context, case)
        mark_complete(summary, summary_file, phase, dataset, BASELINE)
    return phase_contexts, reports


def evaluate_variant(args, contexts, temp, env, summary, summary_file, phase, indices, variant, baselines):
    records = summary.setdefault(phase, {}).setdefault('variants', {})
    entry = records.setdefault(variant.name, {'config': asdict(variant), 'reports': {}})
    for dataset, context in contexts.items():
        entry['reports'][dataset] = run_case(args, context, temp / phase / dataset / variant.name,
                                             indices[dataset], variant, env)
        mark_complete(summary, summary_file, phase, dataset, variant)
    entry.update(assess(args, entry['reports'], baselines))
    entry['paired_comparison'] = previous.comparison_score(entry['reports'], baselines)
    write_json(summary_file, summary)
    print(f'[round2-{phase}] {variant.name} target={entry["strong_target_met"]} '
          f'strict6={entry["strict_target_met"]} weak_signal={entry["weak_signal"]} '
          f'R5gain_pp={entry["mean_gain_r5_pp"]:+.3f} R10gain_pp={entry["mean_gain_r10_pp"]:+.3f} '
          f'unreachable={entry["unreachable_metrics"]}', flush=True)
    return entry


def smoke(args, contexts, temp, env, summary, summary_file):
    indices = summary['indices']['smoke']
    if summary.get('smoke', {}).get('complete'):
        for dataset, context in contexts.items():
            manifest = previous.subset_manifest(context['manifest'], context['data'], context['hops'], indices[dataset])
            verify_saved_case(context, temp / 'smoke' / dataset / 'baseline', manifest, BASELINE)
            cache = temp / 'phase_start_cache/smoke' / dataset
            hashes = previous.experiments.cache_hashes(cache)
            if hashes != summary['smoke']['phase_start_cache_sha256'][dataset]:
                raise ValueError(f'{dataset}: smoke starting cache changed')
            phase_context = dict(context, cache=cache, cache_hashes=hashes)
            for variant in (Variant('baseline_replay'), SMOKE_COMBINATION):
                verify_saved_case(phase_context, temp / 'smoke' / dataset / variant.name, manifest, variant)
        print('[round2-smoke-resume] 9 completed groups revalidated; no smoke requests rerun.', flush=True)
        return
    phase_contexts, baselines = phase_baselines(args, contexts, temp, env, summary, summary_file, 'smoke', indices)
    replay = Variant('baseline_replay')
    for variant in (replay, SMOKE_COMBINATION):
        for dataset, context in phase_contexts.items():
            case = temp / 'smoke' / dataset / variant.name
            run_case(args, context, case, indices[dataset], variant, env)
            if variant == replay:
                baseline = read_json(temp / 'smoke' / dataset / 'baseline/result.json')
                replay_result = read_json(case / 'result.json')
                if ([row['docs'] for row in baseline['results']] !=
                        [row['docs'] for row in replay_result['results']]):
                    raise ValueError(f'{dataset}: original exp4 smoke replay changed Top10')
            mark_complete(summary, summary_file, 'smoke', dataset, variant)
    summary['smoke']['complete'] = True
    write_json(summary_file, summary)


def variant_from_config(config):
    return Variant(config['name'], tuple(config['flags']), config['adaptive_mode'], config['binding_validation'])


def combined_candidate(screened):
    """Combine only settings with measured positive standalone/group evidence."""
    evidence = []
    for name in ('closure_fixed', 'planning_selection', 'binding_strict'):
        if screened[name]['weak_signal']:
            evidence.append(screened[name])
    adaptive = [screened[name] for name in ('adaptive_verify', 'adaptive_beam', 'adaptive_both')
                if screened[name]['weak_signal']]
    if adaptive:
        evidence.append(max(adaptive, key=lambda item: (item['strong_target_met'], item['score'])))
    if len(evidence) < 2:
        return None, []
    flags = tuple(flag for flag in previous.FLAGS
                  if any(flag in item['config']['flags'] for item in evidence))
    mode = next((item['config']['adaptive_mode'] for item in evidence
                 if 'adaptive' in item['config']['flags']), 'both')
    candidate = Variant('combined_supported', flags, mode,
                        'strict_relation' if 'binding' in flags else 'legacy')
    if any((candidate.flags, candidate.adaptive_mode, candidate.binding_validation) ==
           (tuple(item['config']['flags']), item['config']['adaptive_mode'], item['config']['binding_validation'])
           for name, item in screened.items() if name != 'combined_supported'):
        return None, []
    return candidate, [item['config']['name'] for item in evidence]


def screen(args, contexts, temp, env, summary, summary_file):
    indices = summary['indices']['screen']
    phase_contexts, baselines = phase_baselines(args, contexts, temp, env, summary, summary_file, 'screen', indices)
    for variant in VARIANTS:
        evaluate_variant(args, phase_contexts, temp, env, summary, summary_file, 'screen', indices, variant, baselines)
    combined, basis = combined_candidate(summary['screen']['variants'])
    summary['combination_basis'] = basis
    if combined:
        register_groups(summary, summary_file, 'screen', [combined], args.screen_size)
        evaluate_variant(args, phase_contexts, temp, env, summary, summary_file, 'screen', indices, combined, baselines)
    summary['screen']['complete'] = True
    write_json(summary_file, summary)


def confirm(args, contexts, temp, env, summary, summary_file):
    if not summary.get('screen', {}).get('complete'):
        raise ValueError('Finish round2 screening before confirmation')
    entries = summary['screen']['variants'].values()
    positive = [item for item in entries if item['weak_signal']]
    # Confirm at least the best exploratory candidate, even if the requested
    # large gain is absent. This does not label it effective or meet the target.
    ranked = sorted(positive or list(entries),
                    key=lambda item: (-int(item['strong_target_met']), -item['score'], item['config']['name']))
    chosen = ranked[:2] if positive else ranked[:1]
    variants = [variant_from_config(item['config']) for item in chosen]
    plan = [asdict(variant) for variant in variants]
    if summary.get('confirmation_candidates') and summary['confirmation_candidates'] != plan:
        raise ValueError('Confirmation candidates changed during resume')
    summary['confirmation_candidates'] = plan
    register_groups(summary, summary_file, 'confirmation', [BASELINE, *variants], args.confirm_size)
    indices = summary['indices']['confirmation']
    phase_contexts, baselines = phase_baselines(args, contexts, temp, env, summary, summary_file,
                                               'confirmation', indices)
    entries = [evaluate_variant(args, phase_contexts, temp, env, summary, summary_file,
                                'confirmation', indices, variant, baselines) for variant in variants]
    summary['confirmed_target_variants'] = [entry['config']['name'] for entry in entries if entry['strong_target_met']]
    summary['confirmed_weak_signal_variants'] = [entry['config']['name'] for entry in entries
                                                 if entry['weak_signal'] and not entry['strong_target_met']]
    summary['selection_status'] = ('small_sample_target_confirmed' if summary['confirmed_target_variants']
                                    else 'small_sample_target_not_met')
    summary['confirmation']['complete'] = True
    summary['automatic_full_run_enabled'] = False
    write_json(summary_file, summary)
    print(f'[round2-finished-small-only] {summary["selection_status"]}; '
          'full retrieval is not started; all small-sample exports retained.', flush=True)


def protocol_for(args, contexts):
    protocol = previous.protocol_for(args, contexts)
    for relative in ('scripts/exp4_round2.py', 'scripts/utils/exp4_round2.py', 'scripts/run_exp4_round2.sh'):
        protocol['algorithm_code_sha256'][relative] = previous.experiments.sha256(ROOT / relative)
    protocol.update(round=2, screen_seed=args.screen_seed, confirmation_seed=args.confirm_seed,
                    previous_round_exclusions=args.previous_round_exclusions,
                    variant_configs=[asdict(variant) for variant in VARIANTS],
                    target={'gain_unit': args.gain_unit, 'gain': args.target_gain,
                            'max_regression_absolute_pp': args.max_regression,
                            'allowed_exceptions': args.allowed_exceptions, 'reference': args.reference,
                            'every_dataset_needs_target_metric': True})
    return json.loads(json.dumps(protocol))


def development_indices(args, contexts, exclusions):
    indices = {'smoke': {}, 'screen': {}, 'confirmation': {}}
    for dataset, context in contexts.items():
        excluded = exclusions['excluded_indices'][dataset]
        indices['smoke'][dataset] = previous.stratified_indices(context['data'], context['hops'], dataset, 2)
        indices['screen'][dataset] = previous.stratified_indices(
            context['data'], context['hops'], dataset, args.screen_size,
            seed=args.screen_seed, excluded=excluded)
        indices['confirmation'][dataset] = previous.stratified_indices(
            context['data'], context['hops'], dataset, args.confirm_size, seed=args.confirm_seed,
            excluded=set(excluded) | set(indices['screen'][dataset]) | set(indices['smoke'][dataset]))
    return indices


def run(args):
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(ROOT / 'outputs') or out == ROOT / 'outputs':
        raise ValueError('Output root must be a dedicated PathCondRAG outputs directory')
    temp = out / TEMP_NAME
    if temp.is_symlink():
        raise ValueError('Refusing a symlinked round2 temporary directory')
    temp.mkdir(parents=True, exist_ok=True)
    summary_file = out / 'metadata' / SUMMARY_NAME / 'selection.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('Existing vLLM service is unavailable')
    if not Path(args.vllm_log).is_file():
        raise ValueError('vLLM request log does not exist')
    before_code = previous.code_hashes(Path(args.hippo_root))
    contexts = {dataset: previous.context_for(out, dataset, temp) for dataset in DATASETS}
    args.previous_round_exclusions = prior_exclusions(out)
    protocol = protocol_for(args, contexts)
    if summary_file.is_file():
        summary = read_json(summary_file)
        if summary.get('protocol') != protocol or summary.get('protocol_sha256') != signature(protocol):
            raise ValueError('Round2 algorithm, runner, runtime, source indexes, cache or target protocol changed')
    else:
        indices = development_indices(args, contexts, args.previous_round_exclusions)
        summary = {'schema_version': 1, 'round': 2, 'protocol': protocol,
                   'protocol_sha256': signature(protocol), 'indices': indices, 'group_plan': [],
                   'previous_round_exclusions': args.previous_round_exclusions,
                   'selection_status': 'pending_small_sample_evaluation',
                   'automatic_full_run_enabled': False,
                   'development_overlap_final_evaluation': True,
                   'paper_claim_warning': 'Exploratory tuning on existing evaluation files; '
                                          'not an untouched test or final improvement claim.',
                   'cache_policy': 'Every variant uses the same private phase-baseline cache; '
                                   'warm-cache wall time is not a fair efficiency comparison.',
                   'index_cleanup_policy': 'Private indexes removed after validation/cache snapshots; '
                                           'exported results, traces, reports and signed cleanup proofs retained.'}
        register_groups(summary, summary_file, 'smoke', [BASELINE, Variant('baseline_replay'), SMOKE_COMBINATION], 2)
        register_groups(summary, summary_file, 'screen', [BASELINE, *VARIANTS], args.screen_size)
    env = previous.environment(args)
    env['PYTHONHASHSEED'] = '42'
    if args.mode in ('smoke', 'all'):
        smoke(args, contexts, temp, env, summary, summary_file)
    if args.mode in ('screen', 'all'):
        if not summary.get('smoke', {}).get('complete'):
            raise ValueError('A completed round2 smoke is required before screening')
        screen(args, contexts, temp, env, summary, summary_file)
    if args.mode in ('confirm', 'all'):
        confirm(args, contexts, temp, env, summary, summary_file)
    if previous.code_hashes(Path(args.hippo_root)) != before_code:
        raise ValueError('Original HippoRAG source changed during round2')
    for context in contexts.values():
        if previous.experiments.asset_hashes(context['source'], context['manifest']['model_dir']) != context['manifest']['source_asset_sha256']:
            raise ValueError('Public frozen source index changed during round2')
    update_progress(summary, summary_file)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'screen', 'confirm', 'all'), default='all')
    parser.add_argument('--screen-size', type=int, default=48)
    parser.add_argument('--confirm-size', type=int, default=24)
    parser.add_argument('--screen-seed', type=int, default=142)
    parser.add_argument('--confirm-seed', type=int, default=242)
    parser.add_argument('--gain-unit', choices=('absolute_pp', 'relative_percent', 'error_reduction'), default='absolute_pp')
    parser.add_argument('--target-gain', type=float, default=4.0)
    parser.add_argument('--max-regression', type=float, default=1.0, help='Absolute percentage points')
    parser.add_argument('--allowed-exceptions', type=int, default=2)
    parser.add_argument('--reference', choices=('matched_exp4',), default='matched_exp4')
    parser.add_argument('--out-root', '--output-root', default=str(previous.DEFAULT_OUT))
    parser.add_argument('--runtime-deps', default=str(previous.DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=str(previous.DEFAULT_OUT / 'logs/vllm.log'))
    args = parser.parse_args(argv)
    if (args.target_gain <= 0 or args.max_regression < 0 or not 0 <= args.allowed_exceptions <= 5 or
            args.screen_size < 2 or args.confirm_size < 1):
        parser.error('Invalid target, regression limit, exception count or sample size')
    return run(args)
