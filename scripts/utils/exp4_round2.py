"""Evaluate the second improvement round with paired small-sample controls.

This runner never launches full retrieval. Screening and confirmation remain
exploratory because their questions come from the existing evaluation files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
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
    plan_validation: str = 'strict'
    plan_routing: str = 'all'
    support_mode: str = 'tail_only'
    terminal_mode: str = 'tail_only'


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
FOLLOWUP_VARIANTS = (
    Variant('planning_selection_refs', ('planning', 'selection'), plan_validation='canonical_refs'),
    Variant('binding_conservative', ('binding',), binding_validation='conservative_relation'),
    Variant('binding_conservative_adaptive_verify', ('binding', 'adaptive'), 'verify_only', 'conservative_relation'),
    Variant('binding_conservative_adaptive_both', ('binding', 'adaptive'), 'both', 'conservative_relation'),
)
REPRESENTATIVE_VARIANTS = (
    Variant('structure_routed', ('planning',), plan_validation='canonical_refs', plan_routing='question_structure'),
    Variant('support_tail', ('support',)),
    Variant('support_swap', ('support',), support_mode='bounded_swap'),
    Variant('structure_support', ('planning', 'support'), plan_validation='canonical_refs',
            plan_routing='question_structure', support_mode='bounded_swap'),
)
FAILURE_VARIANTS = (
    Variant('terminal_tail', ('terminal',)),
    Variant('terminal_prefix', ('terminal',), terminal_mode='prefix'),
    Variant('plan_prune', ('planning', 'plan_prune'), plan_validation='canonical_refs',
            plan_routing='question_structure'),
    Variant('chain_atomic', ('planning', 'plan_prune'), plan_validation='canonical_refs',
            plan_routing='dependency_depth'),
    Variant('chain_terminal', ('planning', 'plan_prune', 'terminal'), plan_validation='canonical_refs',
            plan_routing='dependency_depth', terminal_mode='prefix'),
)
BRIDGE_VARIANTS = (
    Variant('package', ('planning', 'plan_prune', 'package'), plan_validation='canonical_refs',
            plan_routing='question_structure'),
    Variant('bridge_recovery', ('planning', 'plan_prune', 'bridge_recovery'),
            plan_validation='canonical_refs', plan_routing='question_structure'),
    Variant('bridge_package', ('planning', 'plan_prune', 'bridge_recovery', 'package'),
            plan_validation='canonical_refs', plan_routing='question_structure'),
)
EXPORTED_RECALLS = ('Recall@1', 'Recall@2', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@200')
BRIDGE_SMOKE_FIXTURES = {'hotpotqa': [572, 723], '2wikimultihopqa': [262, 584], 'musique': [93, 567]}


def baseline_variant(args, name='baseline'):
    profile = getattr(args, 'baseline_profile', 'original_exp4')
    if profile == 'original_exp4':
        return Variant(name)
    if profile == 'plan_prune':
        return replace(next(item for item in FAILURE_VARIANTS if item.name == profile), name=name)
    raise ValueError('Unknown baseline profile')


def selection_policy(args):
    policy = getattr(args, 'selection_policy', 'original')
    if policy not in ('original', 'hotpot_bridge'):
        raise ValueError('Unknown selection policy')
    if policy == 'hotpot_bridge' and (
            getattr(args, 'baseline_profile', 'original_exp4') != 'plan_prune' or
            getattr(args, 'sampling_policy', 'hop_support') != 'representative' or
            not getattr(args, 'require_raw_guard', False) or
            set(protected_metrics(args)) != set(EXPORTED_RECALLS)):
        raise ValueError('Hotpot bridge selection needs plan_prune, representative sampling and both six-Recall guards')
    return policy


def namespace(run_tag=None):
    if run_tag is not None and not re.fullmatch(r'[A-Za-z0-9_]{1,80}', run_tag):
        raise ValueError('Run tag must contain only letters, digits or underscores')
    suffix = '_' + run_tag if run_tag else ''
    return SUMMARY_NAME + suffix, TEMP_NAME + suffix


def selected_variants(args):
    requested = getattr(args, 'variants', None)
    if requested is None:
        return VARIANTS
    names = [name.strip() for name in requested.split(',') if name.strip()]
    known = {item.name: item for item in (*VARIANTS, *FOLLOWUP_VARIANTS, *REPRESENTATIVE_VARIANTS,
                                         *FAILURE_VARIANTS, *BRIDGE_VARIANTS)}
    if not names or len(names) != len(set(names)) or any(name not in known for name in names):
        raise ValueError('Variants must be unique known profile names')
    return tuple(known[name] for name in names)


def smoke_combination(args):
    if any(item in BRIDGE_VARIANTS for item in selected_variants(args)):
        return replace(BRIDGE_VARIANTS[-1], name='bridge_package_smoke')
    if any(item in FAILURE_VARIANTS for item in selected_variants(args)):
        return Variant('chain_terminal_smoke', ('planning', 'plan_prune', 'terminal'),
                       plan_validation='canonical_refs', plan_routing='dependency_depth', terminal_mode='prefix')
    if any(item in REPRESENTATIVE_VARIANTS for item in selected_variants(args)):
        return Variant('structure_support_smoke', ('planning', 'support'),
                       plan_validation='canonical_refs', plan_routing='question_structure',
                       support_mode='bounded_swap')
    if any(item in FOLLOWUP_VARIANTS for item in selected_variants(args)):
        return Variant('all_combination_conservative_refs', previous.FLAGS, 'both',
                       'conservative_relation', 'canonical_refs')
    return SMOKE_COMBINATION


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def comma_values(value):
    return tuple(item.strip() for item in value.split(',') if item.strip())


def protected_metrics(args):
    metrics = comma_values(getattr(args, 'protected_metrics', ','.join(METRICS)))
    if not metrics or len(metrics) != len(set(metrics)) or any(
            metric not in EXPORTED_RECALLS
            for metric in metrics):
        raise ValueError('Protected metrics must be unique exported Recall names')
    return metrics


def exclusion_tags(args):
    tags = comma_values(getattr(args, 'exclude_run_tags', ''))
    if len(tags) != len(set(tags)):
        raise ValueError('Excluded run tags must be unique')
    for tag in tags:
        namespace(tag)
    if getattr(args, 'run_tag', None) in tags:
        raise ValueError('A run cannot exclude its own namespace')
    return tags


def prior_exclusions(out, run_tag=None):
    """Keep previously tuned round1 questions out of round2 development."""
    source = out / 'metadata/exp4_improvement_selection/selection.json'
    earlier = read_json(source)['screen']
    indices = {dataset: sorted(set(earlier['screen_indices'][dataset]) |
                               set(earlier['confirmation_indices'][dataset])) for dataset in DATASETS}
    sources = [{'source': str(source), 'source_sha256': previous.experiments.sha256(source)}]
    if run_tag:
        round2_source = out / 'metadata' / SUMMARY_NAME / 'selection.json'
        prior = read_json(round2_source)
        for dataset in DATASETS:
            indices[dataset] = sorted(set(indices[dataset]) | set(prior['indices']['screen'][dataset]) |
                                      set(prior['indices']['confirmation'][dataset]))
        sources.append({'source': str(round2_source), 'source_sha256': previous.experiments.sha256(round2_source)})
    return {'source': str(source), 'source_sha256': previous.experiments.sha256(source), 'sources': sources,
            'excluded_indices': indices,
            'reason': 'Exclude all prior screening/confirmation questions; tagged followups also exclude original round2.'}


def target_assessment(reports, baselines, *, target_gain=4.0, gain_unit='absolute_pp',
                      max_regression=1.0, allowed_exceptions=2, protected_metrics=METRICS):
    """Assess the requested gain without reducing thresholds at metric ceilings.

    Regression limits always use absolute percentage points. The paper target
    additionally needs a qualifying metric on every dataset.
    """
    if gain_unit not in ('absolute_pp', 'relative_percent', 'error_reduction'):
        raise ValueError('Unknown gain unit')
    details = {}
    reached = 0
    protected = True
    protection_details = {}
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
            reached += int(met)
            details[dataset][metric] = {
                'baseline': old, 'candidate': new, 'delta_pp': delta_pp,
                'gain': gain, 'gain_unit': gain_unit, 'target_gain': target_gain,
                'maximum_possible_gain': maximum, 'unreachable': not achievable,
                'target_met': met, 'regression_limit_met': delta_pp >= -max_regression - 1e-8}
        protection_details[dataset] = {}
        for metric in protected_metrics:
            old = float(baselines[dataset]['retrieval_metrics'][metric])
            new = float(reports[dataset]['retrieval_metrics'][metric])
            if not (0 <= old <= 1 and 0 <= new <= 1):
                raise ValueError('Protected Recall values must be in [0, 1]')
            delta_pp = (new - old) * 100
            meets = delta_pp >= -max_regression - 1e-8
            protection_details[dataset][metric] = {
                'baseline': old, 'candidate': new, 'delta_pp': delta_pp, 'regression_limit_met': meets}
            protected = protected and meets
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
            'protected_metrics': list(protected_metrics), 'protection_metric_details': protection_details,
            'unreachable_metrics': [f'{name}/{metric}' for name in DATASETS for metric in METRICS
                                    if details[name][metric]['unreachable']],
            'exception_metrics': [f'{name}/{metric}' for name in DATASETS for metric in METRICS
                                  if not details[name][metric]['target_met']]}


def assess(args, reports, baselines):
    original_reports, original_baselines = reports, baselines
    if getattr(args, 'sampling_policy', 'hop_support') == 'representative':
        reports = {dataset: dict(report, retrieval_metrics=report['representative_evaluation']['full_population']['metrics'])
                   for dataset, report in reports.items()}
        baselines = {dataset: dict(report, retrieval_metrics=report['representative_evaluation']['full_population']['metrics'])
                     for dataset, report in baselines.items()}
    assessment = target_assessment(reports, baselines, target_gain=args.target_gain,
                             gain_unit=args.gain_unit, max_regression=args.max_regression,
                             allowed_exceptions=args.allowed_exceptions, protected_metrics=protected_metrics(args))
    if getattr(args, 'require_raw_guard', False):
        raw = target_assessment(original_reports, original_baselines, target_gain=args.target_gain,
            gain_unit=args.gain_unit, max_regression=args.max_regression,
            allowed_exceptions=args.allowed_exceptions, protected_metrics=protected_metrics(args))
        assessment['weighted_regression_limit_met'] = assessment['regression_limit_met']
        assessment['raw_regression_limit_met'] = raw['regression_limit_met']
        assessment['regression_limit_met'] = assessment['regression_limit_met'] and raw['regression_limit_met']
        for key in ('strong_target_met', 'strict_target_met', 'weak_signal'):
            assessment[key] = assessment[key] and raw['regression_limit_met']
    return assessment


def hotpot_bridge_assessment(args, reports, baselines, assessment):
    """Require new Hotpot gain over the paired plan_prune control.

    Benefits already delivered by the control on 2Wiki are never a signal for
    these modules. Both estimators must improve the same Hotpot metric, while
    every exported Recall on all datasets remains under the paired guards.
    """
    per_dataset = {}
    for dataset in DATASETS:
        per_dataset[dataset] = {}
        for metric in METRICS:
            raw_delta = (reports[dataset]['retrieval_metrics'][metric] -
                         baselines[dataset]['retrieval_metrics'][metric]) * 100
            weighted_delta = assessment['metric_details'][dataset][metric]['delta_pp']
            per_dataset[dataset][metric] = {'raw_delta_pp': raw_delta,
                                          'calibrated_delta_pp': weighted_delta}
    improved = [metric for metric, deltas in per_dataset['hotpotqa'].items()
                if deltas['raw_delta_pp'] > 1e-6 and deltas['calibrated_delta_pp'] > 1e-6]
    signal = bool(improved) and assessment['regression_limit_met']
    score = sum(weight * min(per_dataset['hotpotqa'][metric].values())
                for metric, weight in [('Recall@5', .7), ('Recall@10', .3)])
    return {'hotpot_bridge_signal': signal, 'hotpot_improved_metrics': improved,
            'component_gain_details': per_dataset, 'component_selection_score': score,
            'component_reference': 'fresh_paired_plan_prune',
            'selection_policy': selection_policy(args),
            'gain_not_a_four_percentage_point_target_claim': True}


def historical_reference_report(context, indices):
    """Matched questions from the old original-exp4 export, for context only."""
    report = previous.baseline_subset(context, indices)
    sampling = context.get('representative_sampling')
    if sampling is not None and len(indices) > 2:
        from .representative_sampling import poststratified_metrics
        report['representative_evaluation'] = {'full_population': poststratified_metrics(
            sampling['features'], indices, report['per_question'], metrics=tuple(report['retrieval_metrics']))}
    report['reference_scope'] = 'historical_original_exp4_export_on_matched_questions'
    report['warning'] = ('Historical generation, cache and runtime conditions differ; '
                         'not a fresh paired causal control or a confirmed four-point claim.')
    return report


def historical_plan_prune_report(context, indices):
    report = previous.aggregate_measurements([
        context['historical_plan_prune_measurements'][i] for i in indices])
    sampling = context.get('representative_sampling')
    if sampling is not None and len(indices) > 2:
        from .representative_sampling import poststratified_metrics
        report['representative_evaluation'] = {'full_population': poststratified_metrics(
            sampling['features'], indices, report['per_question'], metrics=tuple(report['retrieval_metrics']))}
    report.update(reference_scope='user_completed_full_plan_prune_export_on_matched_questions',
                  warning='Historical full export is a reference; fresh paired plan_prune is the causal control.')
    return report


def validate_package_parent_outputs(parent, result):
    """Package is a finalizer: its inputs must match the evaluated B parent."""
    before = {row['query_index']: row for row in parent['results']}
    after = {row['query_index']: row for row in result['results']}
    if before.keys() != after.keys() or len(before) != len(parent['results']) or len(after) != len(result['results']):
        raise ValueError('Package parent question subset differs')
    fields = ('plan', 'bindings', 'branch_scores', 'routes', 'search_count',
              'llm_plan_calls', 'llm_verification_calls', 'planning_outputs', 'verification_outputs')
    for index in before:
        old = before[index]['retrieval_trace']['evidence']
        new = after[index]['retrieval_trace']['evidence']
        if any(old.get(field) != new.get(field) for field in fields):
            raise ValueError(f'Package parent retrieval or LLM outputs differ on question {index}')
    if result['llm_request_stats'].get('http_attempts', 0):
        raise ValueError('Combined package run generated fresh LLM outputs instead of reusing its B parent')
    return {'question_count': len(before), 'equal_parent_fields': list(fields),
            'fresh_http_attempts': 0, 'validated': True}


def command(args, context, case, indices, variant):
    base = previous.command(args, context, case, indices, frozenset(variant.flags))
    # The earlier driver knows only its original five flags. Preserve the
    # explicit profile here so newer opt-in flags cannot silently disappear.
    base[base.index('--evidence_improvements') + 1] = ','.join(variant.flags)
    return base + [
        '--evidence_adaptive_mode', variant.adaptive_mode,
        '--evidence_binding_validation', variant.binding_validation,
        '--evidence_plan_validation', variant.plan_validation,
        '--evidence_plan_routing', variant.plan_routing,
        '--evidence_support_mode', variant.support_mode,
        '--evidence_terminal_mode', variant.terminal_mode]


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
                          ('evidence_binding_validation', variant.binding_validation),
                          ('evidence_plan_validation', variant.plan_validation),
                          ('evidence_plan_routing', variant.plan_routing),
                          ('evidence_support_mode', variant.support_mode),
                          ('evidence_terminal_mode', variant.terminal_mode)]:
        if runtime.get(key) != expected:
            # Do not retain the old helper's marker after a round-specific
            # validation failure; the exported result remains for inspection.
            (case / 'validated.ok').unlink()
            raise ValueError(f'{case}: {key}={runtime.get(key)} != {expected}')
    report['round2_config'] = asdict(variant)
    report['improvements'] = list(variant.flags)
    if context.get('parent_profile_cache'):
        report['parent_profile_cache'] = context['parent_profile_cache']
        report['cache_policy'] = 'Reuse validated same-phase bridge_recovery outputs for package isolation; warm-cache timing.'
        parent = context['parent_profile_cache']
        source = Path(parent['source_result_path'])
        if previous.experiments.sha256(source) != parent['source_result_sha256']:
            (case / 'validated.ok').unlink()
            raise ValueError('Package parent result changed')
        try:
            report['package_parent_output_validation'] = validate_package_parent_outputs(read_json(source), result)
        except (ValueError, KeyError):
            (case / 'validated.ok').unlink()
            raise
    if getattr(args, 'sampling_policy', 'hop_support') == 'representative' and len(indices) > 2:
        from .representative_sampling import poststratified_metrics
        sampling = context['representative_sampling']
        available_key = ('confirmation_available_indices' if case.parent.parent.name == 'confirmation'
                         else 'available_indices')
        groups = {}
        for row in report['per_question']:
            primary = sampling['features'][row['query_index']]['primary']
            groups.setdefault(primary, []).append(row)
        report['representative_evaluation'] = {
            'full_population': poststratified_metrics(sampling['features'], indices, report['per_question'],
                metrics=tuple(report['retrieval_metrics'])),
            'available_population': poststratified_metrics(
                sampling['features'], indices, report['per_question'], target_indices=sampling[available_key],
                metrics=tuple(report['retrieval_metrics'])),
            'by_primary_stratum': {primary: previous.compact(previous.aggregate_measurements(rows))
                                  for primary, rows in groups.items()},
            'warning': 'Primary-stratum calibrated descriptive estimates; not an unbiased guarantee or untouched test.'}
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
    maximum = summary.get('maximum_possible_groups', 48)
    summary['progress'] = {'planned_groups': len(plans), 'completed_groups': completed,
                           'remaining_groups': len(plans) - completed,
                           'maximum_possible_groups': maximum,
                           'optional_groups_note': ('The combined profile is already screened; up to 9 confirmation groups.'
                               if summary.get('sampling_policy') == 'representative' else
                               'Up to 3 evidence-based combination groups and up to 9 confirmation groups are added after screening.')}
    write_json(summary_file, summary)
    print(f'[round2-progress] planned={len(plans)} completed={completed} '
          f'remaining={len(plans)-completed} maximum={maximum}', flush=True)


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
    baseline = baseline_variant(args)
    records['baseline_profile'] = getattr(args, 'baseline_profile', 'original_exp4')
    for dataset, context in contexts.items():
        case = temp / phase / dataset / baseline.name
        reports[dataset] = run_case(args, context, case, indices[dataset], baseline, env, cleanup=False)
        if getattr(args, 'baseline_profile', 'original_exp4') != 'original_exp4':
            records.setdefault('historical_original_exp4_diagnostic', {})[dataset] = historical_reference_report(
                context, indices[dataset])
        if context.get('historical_plan_prune_measurements'):
            records.setdefault('historical_full_plan_prune_diagnostic', {})[dataset] = historical_plan_prune_report(
                context, indices[dataset])
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
        mark_complete(summary, summary_file, phase, dataset, baseline)
    return phase_contexts, reports


def package_parent_context(context, parent, dataset, phase):
    if not parent or previous.experiments.cache_hashes(Path(parent['path'])) != parent['sha256']:
        raise ValueError(f'{dataset}/{phase}: validated bridge_recovery parent cache is required and immutable')
    return dict(context, cache=Path(parent['path']), cache_hashes=parent['sha256'], parent_profile_cache=parent)


def freeze_bridge_parent_cache(case, cache, saved=None):
    if not cache.is_dir():
        if read_json(case / 'validated.ok').get('cleaned_private_index'):
            raise ValueError(f'{case}: bridge parent cache missing after private index cleanup')
        previous.snapshot_cache(case / 'index/llm_cache', cache)
    hashes = previous.experiments.cache_hashes(cache)
    result_hash = previous.experiments.sha256(case / 'result.json')
    if saved is not None and (hashes != saved['sha256'] or result_hash != saved['source_result_sha256']):
        raise ValueError(f'{case}: bridge parent cache or source result changed')
    return {'path': str(cache), 'sha256': hashes, 'parent_profile': 'bridge_recovery',
            'source_result_path': str(case / 'result.json'),
            'source_result_sha256': result_hash,
            'policy': 'Same phase and question subset; AB reuses B plans, repair and verification outputs.'}


def evaluate_variant(args, contexts, temp, env, summary, summary_file, phase, indices, variant, baselines):
    records = summary.setdefault(phase, {}).setdefault('variants', {})
    entry = records.setdefault(variant.name, {'config': asdict(variant), 'reports': {}})
    for dataset, context in contexts.items():
        paired_bridge = selection_policy(args) == 'hotpot_bridge'
        if paired_bridge and variant.name == 'bridge_package':
            parent = summary[phase].get('bridge_parent_caches', {}).get(dataset)
            context = package_parent_context(context, parent, dataset, phase)
        case = temp / phase / dataset / variant.name
        freeze_parent = paired_bridge and variant.name == 'bridge_recovery'
        entry['reports'][dataset] = run_case(args, context, case,
                                             indices[dataset], variant, env, cleanup=not freeze_parent)
        if freeze_parent:
            cache = temp / 'bridge_parent_cache' / phase / dataset
            parent_records = summary[phase].setdefault('bridge_parent_caches', {})
            parent_records[dataset] = freeze_bridge_parent_cache(case, cache, parent_records.get(dataset))
            write_json(summary_file, summary)
            cleanup_private_index(context, case)
        mark_complete(summary, summary_file, phase, dataset, variant)
    entry.update(assess(args, entry['reports'], baselines))
    if selection_policy(args) == 'hotpot_bridge':
        entry.update(hotpot_bridge_assessment(args, entry['reports'], baselines, entry))
        historical = summary[phase]['historical_original_exp4_diagnostic']
        entry['historical_original_exp4_target_diagnostic'] = assess(args, entry['reports'], historical)
        entry['historical_original_exp4_target_diagnostic']['causal_control'] = False
        entry['historical_original_exp4_target_diagnostic']['warning'] = historical['hotpotqa']['warning']
    if getattr(args, 'sampling_policy', 'hop_support') == 'representative':
        entry['raw_sample_assessment'] = target_assessment(entry['reports'], baselines,
            target_gain=args.target_gain, gain_unit=args.gain_unit, max_regression=args.max_regression,
            allowed_exceptions=args.allowed_exceptions, protected_metrics=protected_metrics(args))
    entry['assessment_estimator'] = ('primary_poststratified_full_population'
        if getattr(args, 'sampling_policy', 'hop_support') == 'representative' else 'raw_sample_macro')
    entry['paired_comparison'] = previous.comparison_score(entry['reports'], baselines)
    entry['raw_all_gold_deltas_pp'] = {dataset: {
        metric: (entry['reports'][dataset][metric] - baselines[dataset][metric]) * 100
        for metric in ('all_gold_top5', 'all_gold_top10')
        if metric in entry['reports'][dataset] and metric in baselines[dataset]}
        for dataset in DATASETS}
    write_json(summary_file, summary)
    print(f'[round2-{phase}] {variant.name} target={entry["strong_target_met"]} '
          f'strict6={entry["strict_target_met"]} weak_signal={entry["weak_signal"]} '
          f'R5gain_pp={entry["mean_gain_r5_pp"]:+.3f} R10gain_pp={entry["mean_gain_r10_pp"]:+.3f} '
          f'guard={entry["regression_limit_met"]} '
          f'Hotpot_new_signal={entry.get("hotpot_bridge_signal")} '
          f'unreachable={entry["unreachable_metrics"]}', flush=True)
    return entry


def smoke(args, contexts, temp, env, summary, summary_file):
    indices = summary['indices']['smoke']
    combination = smoke_combination(args)
    baseline_variant_config = baseline_variant(args)
    replay = baseline_variant(args, 'baseline_replay')
    if summary.get('smoke', {}).get('complete'):
        for dataset, context in contexts.items():
            manifest = previous.subset_manifest(context['manifest'], context['data'], context['hops'], indices[dataset])
            verify_saved_case(context, temp / 'smoke' / dataset / 'baseline', manifest, baseline_variant_config)
            cache = temp / 'phase_start_cache/smoke' / dataset
            hashes = previous.experiments.cache_hashes(cache)
            if hashes != summary['smoke']['phase_start_cache_sha256'][dataset]:
                raise ValueError(f'{dataset}: smoke starting cache changed')
            phase_context = dict(context, cache=cache, cache_hashes=hashes)
            for variant in (replay, combination):
                verify_saved_case(phase_context, temp / 'smoke' / dataset / variant.name, manifest, variant)
        print('[round2-smoke-resume] 9 completed groups revalidated; no smoke requests rerun.', flush=True)
        return
    phase_contexts, baselines = phase_baselines(args, contexts, temp, env, summary, summary_file, 'smoke', indices)
    for variant in (replay, combination):
        for dataset, context in phase_contexts.items():
            case = temp / 'smoke' / dataset / variant.name
            run_case(args, context, case, indices[dataset], variant, env)
            if variant == replay:
                baseline = read_json(temp / 'smoke' / dataset / 'baseline/result.json')
                replay_result = read_json(case / 'result.json')
                if ([row['docs'] for row in baseline['results']] !=
                        [row['docs'] for row in replay_result['results']]):
                    raise ValueError(f'{dataset}: {baseline_variant_config.name}/'
                                     f'{getattr(args, "baseline_profile", "original_exp4")} smoke replay changed Top10')
            mark_complete(summary, summary_file, 'smoke', dataset, variant)
    summary['smoke']['complete'] = True
    write_json(summary_file, summary)


def variant_from_config(config):
    return Variant(config['name'], tuple(config['flags']), config['adaptive_mode'],
                   config['binding_validation'], config.get('plan_validation', 'strict'),
                   config.get('plan_routing', 'all'), config.get('support_mode', 'tail_only'),
                   config.get('terminal_mode', 'tail_only'))


def combined_candidate(screened):
    """Combine only settings with measured positive standalone/group evidence."""
    evidence = []
    for names in [('closure_fixed',), ('planning_selection', 'planning_selection_refs'),
                  ('binding_strict', 'binding_conservative')]:
        candidates = [screened[name] for name in names if name in screened and screened[name]['weak_signal']]
        if candidates:
            evidence.append(max(candidates, key=lambda item: (item['strong_target_met'], item['score'])))
    adaptive = [item for name, item in screened.items() if name != 'combined_supported' and
                'adaptive' in item['config']['flags'] and item['weak_signal']]
    if adaptive:
        evidence.append(max(adaptive, key=lambda item: (item['strong_target_met'], item['score'])))
    if len(evidence) < 2:
        return None, []
    flags = tuple(flag for flag in previous.FLAGS
                  if any(flag in item['config']['flags'] for item in evidence))
    mode = next((item['config']['adaptive_mode'] for item in evidence
                 if 'adaptive' in item['config']['flags']), 'both')
    binding = next((item['config']['binding_validation'] for item in evidence
                    if 'adaptive' in item['config']['flags'] and 'binding' in item['config']['flags']), None)
    if binding is None:
        binding = next((item['config']['binding_validation'] for item in evidence
                        if 'binding' in item['config']['flags']), 'legacy')
    evidence = [item for item in evidence if 'binding' not in item['config']['flags'] or
                item['config']['binding_validation'] == binding]
    plan = 'canonical_refs' if any(item['config'].get('plan_validation') == 'canonical_refs'
                                   for item in evidence) else 'strict'
    candidate = Variant('combined_supported', flags, mode,
                        binding, plan)
    if any((candidate.flags, candidate.adaptive_mode, candidate.binding_validation, candidate.plan_validation) ==
           (tuple(item['config']['flags']), item['config']['adaptive_mode'], item['config']['binding_validation'],
            item['config'].get('plan_validation', 'strict'))
           for name, item in screened.items() if name != 'combined_supported'):
        return None, []
    return candidate, [item['config']['name'] for item in evidence]


def screen(args, contexts, temp, env, summary, summary_file):
    indices = summary['indices']['screen']
    phase_contexts, baselines = phase_baselines(args, contexts, temp, env, summary, summary_file, 'screen', indices)
    for variant in selected_variants(args):
        evaluate_variant(args, phase_contexts, temp, env, summary, summary_file, 'screen', indices, variant, baselines)
    combined, basis = ((None, []) if getattr(args, 'sampling_policy', 'hop_support') == 'representative'
                      else combined_candidate(summary['screen']['variants']))
    summary['combination_basis'] = basis
    if combined:
        register_groups(summary, summary_file, 'screen', [combined], args.screen_size)
        evaluate_variant(args, phase_contexts, temp, env, summary, summary_file, 'screen', indices, combined, baselines)
    summary['screen']['complete'] = True
    write_json(summary_file, summary)


def screening_output_signature(temp, entry):
    """Identify identical evaluated rankings, rather than equal macro scores."""
    outputs = {}
    for dataset in DATASETS:
        report = entry['reports'].get(dataset, {})
        rows = report.get('per_question')
        result_path = temp / 'screen' / dataset / entry['config']['name'] / 'result.json'
        if not rows or not result_path.is_file():
            return None
        result = read_json(result_path)
        rankings = {row['query_index']: row['docs'] for row in result['results']}
        indices = {row['query_index'] for row in rows}
        if set(rankings) != indices or len(indices) != len(rows):
            raise ValueError(f'{dataset}: candidate deduplication rows differ from exported rankings')
        outputs[dataset] = [{'query_index': row['query_index'], 'metrics': row['metrics'],
                             'all_gold_top5': row['all_gold_top5'],
                             'all_gold_top10': row['all_gold_top10'],
                             'docs': rankings[row['query_index']]}
                            for row in sorted(rows, key=lambda row: row['query_index'])]
    return signature(outputs)


def unique_confirmation_candidates(ranked, temp, maximum):
    chosen, seen, equivalents = [], {}, {}
    for entry in ranked:
        digest = screening_output_signature(temp, entry)
        name = entry['config']['name']
        if digest is not None and digest in seen:
            equivalents.setdefault(seen[digest], []).append(name)
            continue
        if len(chosen) < maximum:
            chosen.append(entry)
            if digest is not None:
                seen[digest] = name
        # Keep scanning to record all equivalent profiles, but never add a
        # third distinct candidate. Equal averages alone are not deduplicated.
    return chosen, equivalents


def confirm(args, contexts, temp, env, summary, summary_file):
    if not summary.get('screen', {}).get('complete'):
        raise ValueError('Finish round2 screening before confirmation')
    entries = list(summary['screen']['variants'].values())
    if getattr(args, 'require_raw_guard', False):
        entries = [item for item in entries if item['regression_limit_met']]
        if not entries:
            summary.update(confirmation_candidates=[], confirmation_equivalent_profiles={},
                confirmed_target_variants=[], confirmed_weak_signal_variants=[],
                selection_status='small_sample_no_guarded_candidate', automatic_full_run_enabled=False)
            summary['confirmation'] = {'complete': True, 'skipped': True,
                'reason': 'Every screening candidate breached the paired raw or weighted regression limit.'}
            write_json(summary_file, summary)
            print('[round2-finished-small-only] No candidate passed both regression guards; '
                  'confirmation and full retrieval are not started.', flush=True)
            return
    bridge_policy = selection_policy(args) == 'hotpot_bridge'
    positive = [item for item in entries if item.get('hotpot_bridge_signal', False)] if bridge_policy else [
        item for item in entries if item['weak_signal']]
    if bridge_policy and not positive:
        summary.update(confirmation_candidates=[], confirmation_equivalent_profiles={},
            confirmed_target_variants=[], confirmed_weak_signal_variants=[],
            confirmed_hotpot_bridge_variants=[], selected_new_module_profiles=[],
            selection_status='small_sample_no_guarded_hotpot_gain', automatic_full_run_enabled=False)
        summary['confirmation'] = {'complete': True, 'skipped': True,
            'reason': 'No profile improved a Hotpot R@5/R@10 metric in both raw and calibrated '
                      'means while retaining all six paired Recall guards on every dataset.'}
        write_json(summary_file, summary)
        print('[round2-finished-small-only] No guarded new Hotpot gain over plan_prune; '
              'confirmation and full retrieval are not started.', flush=True)
        return
    # Confirm at least the best exploratory candidate, even if the requested
    # large gain is absent. This does not label it effective or meet the target.
    ranked = sorted(positive or list(entries),
                    key=lambda item: (-int(item['strong_target_met']),
                        -item.get('component_selection_score', item['score']), item['config']['name']))
    chosen, equivalents = unique_confirmation_candidates(ranked, temp, 2 if positive else 1)
    parent_controls = []
    if bridge_policy and any(item['config']['name'] == 'bridge_package' for item in chosen):
        combined = next(item for item in chosen if item['config']['name'] == 'bridge_package')
        parent = summary['screen']['variants'].get('bridge_recovery')
        if parent is None:
            raise ValueError('Combined package confirmation requires a screened bridge_recovery parent')
        # At most two profiles remain. The B control is mandatory to isolate
        # package from newly generated repair output on confirmation questions.
        chosen = [parent, combined]
        if not parent.get('hotpot_bridge_signal', False):
            parent_controls = ['bridge_recovery']
    if bridge_policy:
        summary['confirmation_parent_controls'] = parent_controls
    summary['confirmation_equivalent_profiles'] = equivalents
    variants = [variant_from_config(item['config']) for item in chosen]
    plan = json.loads(json.dumps([asdict(variant) for variant in variants]))
    if summary.get('confirmation_candidates') and summary['confirmation_candidates'] != plan:
        raise ValueError('Confirmation candidates changed during resume')
    summary['confirmation_candidates'] = plan
    register_groups(summary, summary_file, 'confirmation', [baseline_variant(args), *variants], args.confirm_size)
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
    if bridge_policy:
        summary['confirmed_hotpot_bridge_variants'] = [entry['config']['name'] for entry in entries
            if entry['hotpot_bridge_signal'] and entry['config']['name'] not in parent_controls]
        summary['selected_new_module_profiles'] = [
            {'profile': entry['config']['name'], 'new_modules': [flag for flag in entry['config']['flags']
               if flag not in baseline_variant(args).flags]} for entry in entries
            if entry['hotpot_bridge_signal'] and entry['config']['name'] not in parent_controls]
        summary['selection_status'] = ('small_sample_hotpot_component_confirmed'
            if summary['confirmed_hotpot_bridge_variants'] else 'small_sample_no_hotpot_component_confirmed')
        summary['global_target_status'] = ('paired_four_point_target_confirmed'
            if summary['confirmed_target_variants'] else 'paired_four_point_target_not_met')
    summary.setdefault('confirmation', {})['complete'] = True
    summary['automatic_full_run_enabled'] = False
    write_json(summary_file, summary)
    print(f'[round2-finished-small-only] {summary["selection_status"]}; '
          'full retrieval is not started; all small-sample exports retained.', flush=True)


def protocol_for(args, contexts):
    protocol = previous.protocol_for(args, contexts)
    for relative in ('scripts/exp4_round2.py', 'scripts/utils/exp4_round2.py', 'scripts/run_exp4_round2.sh'):
        protocol['algorithm_code_sha256'][relative] = previous.experiments.sha256(ROOT / relative)
    if selection_policy(args) == 'hotpot_bridge':
        relative = 'scripts/run_exp4_bridge_trials.sh'
        protocol['algorithm_code_sha256'][relative] = previous.experiments.sha256(ROOT / relative)
    if getattr(args, 'sampling_policy', 'hop_support') == 'representative':
        relative = 'scripts/utils/representative_sampling.py'
        protocol['algorithm_code_sha256'][relative] = previous.experiments.sha256(ROOT / relative)
        protocol['representative_sampling'] = {dataset: {
            'feature_sha256': context['representative_sampling']['feature_sha256'],
            'screen_indices': context['representative_sampling']['screen_indices'],
            'confirmation_indices': context['representative_sampling']['confirmation_indices']}
            for dataset, context in contexts.items()}
    protocol.update(round=2, run_tag=getattr(args, 'run_tag', None),
                    sampling_policy=getattr(args, 'sampling_policy', 'hop_support'),
                    exclude_run_tags=list(exclusion_tags(args)),
                    target_metric_estimator=('primary_poststratified_full_population'
                        if getattr(args, 'sampling_policy', 'hop_support') == 'representative' else 'raw_sample_macro'),
                    screen_seed=args.screen_seed, confirmation_seed=args.confirm_seed,
                    previous_round_exclusions=args.previous_round_exclusions,
                    variant_configs=[asdict(variant) for variant in selected_variants(args)],
                    smoke_combination=asdict(smoke_combination(args)),
                    target={'gain_unit': args.gain_unit, 'gain': args.target_gain,
                            'max_regression_absolute_pp': args.max_regression,
                            'protected_metrics': list(protected_metrics(args)),
                            'require_raw_guard': getattr(args, 'require_raw_guard', False),
                            'allowed_exceptions': args.allowed_exceptions, 'reference': args.reference,
                            'every_dataset_needs_target_metric': True})
    if getattr(args, 'baseline_profile', 'original_exp4') != 'original_exp4' or selection_policy(args) != 'original':
        protocol.update(baseline_profile=getattr(args, 'baseline_profile', 'original_exp4'),
            baseline_config=asdict(baseline_variant(args)), selection_policy=selection_policy(args),
            component_gain_reference='fresh_paired_plan_prune',
            historical_original_exp4_scope='matched questions; diagnostic only; not a fresh causal control')
        if selection_policy(args) == 'hotpot_bridge':
            protocol['smoke_fixture_policy'] = {
                'indices': BRIDGE_SMOKE_FIXTURES,
                'scope': 'Previously inspected integration fixtures; excluded from screening/confirmation.',
                'retriever_inputs': 'question, benchmark hop value and corpus index; no fixture/gold labels'}
            protocol['combination_isolation'] = {
                'standalone_package_parent': 'Fresh plan_prune phase-baseline cache',
                'combined_package_parent': 'Immutable same-phase completed bridge_recovery cache',
                'confirmation_rule': 'If AB is chosen, run B then AB; B may be a noncandidate parent control.',
                'time_comparison': 'Different warm-cache histories; not a wall-time efficiency comparison.'}
        if contexts:
            protocol['historical_original_exp4_export_sha256'] = {
                dataset: {filename: previous.experiments.sha256(
                    Path(args.out_root) / 'cases' / dataset / 'exp4_dependency_binding' / filename)
                    for filename in ('result.json', 'report.json')}
                for dataset in contexts}
            if selection_policy(args) == 'hotpot_bridge':
                protocol['historical_full_plan_prune_export_sha256'] = {
                    dataset: {filename: previous.experiments.sha256(
                        Path(args.out_root) / 'cases' / dataset / 'exp4_dependency_binding_plan_prune' / filename)
                        for filename in ('result.json', 'report.json')}
                    for dataset in contexts}
    return json.loads(json.dumps(protocol))


def development_indices(args, contexts, exclusions):
    indices = {'smoke': {}, 'screen': {}, 'confirmation': {}}
    if getattr(args, 'sampling_policy', 'hop_support') == 'representative':
        for dataset, context in contexts.items():
            sampling = context['representative_sampling']
            indices['screen'][dataset] = sampling['screen_indices']
            indices['confirmation'][dataset] = sampling['confirmation_indices']
            if selection_policy(args) == 'hotpot_bridge':
                fixtures = BRIDGE_SMOKE_FIXTURES[dataset]
                if (any(index < 0 or index >= len(sampling['features']) for index in fixtures) or
                        set(fixtures) & (set(indices['screen'][dataset]) | set(indices['confirmation'][dataset]))):
                    raise ValueError(f'{dataset}: bridge smoke fixtures must exist and be excluded from development')
                indices['smoke'][dataset] = list(fixtures)
                continue
            features, pool = sampling['features'], sampling['screen_indices']
            first = min(pool, key=lambda i: (features[i]['hops'], i))
            # Two contrasting structures for integration, never chosen using
            # retrieval accuracy. This smoke is not the representative set.
            second = max((i for i in pool if i != first), key=lambda i: (
                int(features[i]['primary'] != features[first]['primary']),
                int('comparison' in features[i]['primary'] or 'join' in features[i]['primary']),
                features[i]['hops'], -i))
            indices['smoke'][dataset] = sorted([first, second])
        return indices
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
    summary_name, temp_name = namespace(getattr(args, 'run_tag', None))
    variants = selected_variants(args)
    baseline = baseline_variant(args)
    selection_policy(args)
    temp = out / temp_name
    if temp.is_symlink():
        raise ValueError('Refusing a symlinked round2 temporary directory')
    temp.mkdir(parents=True, exist_ok=True)
    summary_file = out / 'metadata' / summary_name / 'selection.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('Existing vLLM service is unavailable')
    if not Path(args.vllm_log).is_file():
        raise ValueError('vLLM request log does not exist')
    before_code = previous.code_hashes(Path(args.hippo_root))
    contexts = {dataset: previous.context_for(out, dataset, temp) for dataset in DATASETS}
    if selection_policy(args) == 'hotpot_bridge':
        for dataset, context in contexts.items():
            reference = out / 'cases' / dataset / 'exp4_dependency_binding_plan_prune'
            previous.verify_trial_report(reference)
            rows, _ = previous.experiments.validate_result(
                read_json(reference / 'result.json'), context['manifest'], 'exp4_dependency_binding',
                context['data'], context['corpus'], expected_stage=4)
            context['historical_plan_prune_measurements'] = {row['query_index']: row for row in rows}
    representative = getattr(args, 'sampling_policy', 'hop_support') == 'representative'
    if representative:
        from .representative_sampling import collect_prior_exclusions, make_representative_split
        exclusions = collect_prior_exclusions(out, additional_run_tags=exclusion_tags(args))
        if selection_policy(args) == 'hotpot_bridge':
            exclusions = deepcopy(exclusions)
            exclusions['integration_fixture_exclusions'] = deepcopy(BRIDGE_SMOKE_FIXTURES)
            for dataset, fixtures in BRIDGE_SMOKE_FIXTURES.items():
                exclusions['indices'][dataset] = sorted(set(exclusions['indices'][dataset]) | set(fixtures))
        args.previous_round_exclusions = dict(exclusions, excluded_indices=exclusions['indices'])
        for dataset, context in contexts.items():
            context['representative_sampling'] = make_representative_split(context['data'], dataset,
                excluded_indices=exclusions['indices'][dataset], screen_size=args.screen_size,
                confirmation_size=args.confirm_size, seed=args.screen_seed,
                confirmation_seed=args.confirm_seed, hops=context['hops'])
    else:
        args.previous_round_exclusions = prior_exclusions(out, getattr(args, 'run_tag', None))
    protocol = protocol_for(args, contexts)
    if summary_file.is_file():
        summary = read_json(summary_file)
        if summary.get('protocol') != protocol or summary.get('protocol_sha256') != signature(protocol):
            raise ValueError('Round2 algorithm, runner, runtime, source indexes, cache or target protocol changed')
        if representative:
            for dataset, saved in summary['sampling_reports'].items():
                if (previous.experiments.sha256(Path(saved['path'])) != saved['sha256'] or
                        read_json(saved['path']) != contexts[dataset]['representative_sampling']):
                    raise ValueError(f'{dataset}: representative sampling report changed')
            if summary['indices'] != development_indices(args, contexts, args.previous_round_exclusions):
                raise ValueError('Representative question indices changed')
    else:
        indices = development_indices(args, contexts, args.previous_round_exclusions)
        summary = {'schema_version': 1, 'round': 2, 'protocol': protocol,
                   'run_tag': getattr(args, 'run_tag', None),
                   'sampling_policy': getattr(args, 'sampling_policy', 'hop_support'),
                   'maximum_possible_groups': (21 if representative else 24) + 3 * len(variants),
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
        if representative:
            summary['sampling_reports'] = {}
            (summary_file.parent / 'sampling').mkdir(parents=True, exist_ok=True)
            for dataset, context in contexts.items():
                sampling_path = summary_file.parent / 'sampling' / (dataset + '.json')
                write_json(sampling_path, context['representative_sampling'])
                summary['sampling_reports'][dataset] = {
                    'path': str(sampling_path), 'sha256': previous.experiments.sha256(sampling_path)}
        if selection_policy(args) == 'hotpot_bridge':
            summary.update(baseline_profile='plan_prune', selection_policy='hotpot_bridge',
                component_selection_reference='fresh_paired_plan_prune',
                historical_original_exp4_is_diagnostic_only=True,
                llm_attribution_warning='Package reuses same-phase parent cached plan/repair/verify outputs; '
                    'validate per-question parent traces and request diagnostics before attributing AB differences.')
            summary['cache_policy'] = ('A and B start from the same fresh plan_prune phase cache; '
                'AB starts from validated B outputs in that same phase. All caches are private snapshots; '
                'warm-cache times cannot establish efficiency gains.')
        register_groups(summary, summary_file, 'smoke', [baseline, baseline_variant(args, 'baseline_replay'),
                                                      smoke_combination(args)], 2)
        register_groups(summary, summary_file, 'screen', [baseline, *variants], args.screen_size)
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
    parser.add_argument('--run-tag', help='Independent output namespace using letters, digits or underscores')
    parser.add_argument('--variants', help='Comma-separated subset of named screen profiles; default is the original eight')
    parser.add_argument('--baseline-profile', choices=('original_exp4', 'plan_prune'), default='original_exp4',
                        help='Fresh paired control; plan_prune retains previously observed 2Wiki planning benefits')
    parser.add_argument('--selection-policy', choices=('original', 'hotpot_bridge'), default='original',
                        help='Hotpot policy requires new raw/calibrated Hotpot gain plus all six regression guards')
    parser.add_argument('--sampling-policy', choices=('hop_support', 'representative'), default='hop_support')
    parser.add_argument('--exclude-run-tags', default='',
                        help='Explicit prior round2 namespaces to exclude, in addition to the fixed three rounds')
    parser.add_argument('--screen-size', type=int, default=48)
    parser.add_argument('--confirm-size', type=int, default=24)
    parser.add_argument('--screen-seed', type=int, default=142)
    parser.add_argument('--confirm-seed', type=int, default=242)
    parser.add_argument('--gain-unit', choices=('absolute_pp', 'relative_percent', 'error_reduction'), default='absolute_pp')
    parser.add_argument('--target-gain', type=float, default=4.0)
    parser.add_argument('--max-regression', type=float, default=1.0, help='Absolute percentage points')
    parser.add_argument('--protected-metrics', default=','.join(METRICS),
                        help='Comma-separated Recall metrics protected by the paired regression limit')
    parser.add_argument('--require-raw-guard', action='store_true',
                        help='Require both calibrated and raw sample metrics to meet the regression limit')
    parser.add_argument('--allowed-exceptions', type=int, default=2)
    parser.add_argument('--reference', choices=('matched_exp4',), default='matched_exp4')
    parser.add_argument('--out-root', '--output-root', default=str(previous.DEFAULT_OUT))
    parser.add_argument('--runtime-deps', default=str(previous.DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=str(previous.DEFAULT_OUT / 'logs/vllm.log'))
    args = parser.parse_args(argv)
    try:
        namespace(args.run_tag)
        selected_variants(args)
        exclusion_tags(args)
        protected_metrics(args)
        baseline_variant(args)
        selection_policy(args)
    except ValueError as error:
        parser.error(str(error))
    if args.sampling_policy == 'representative' and not args.run_tag:
        parser.error('Representative sampling needs an independent run tag')
    if args.exclude_run_tags and args.sampling_policy != 'representative':
        parser.error('Additional run exclusions are supported by representative sampling')
    if (args.target_gain <= 0 or args.max_regression < 0 or not 0 <= args.allowed_exceptions <= 5 or
            args.screen_size < 2 or args.confirm_size < 1):
        parser.error('Invalid target, regression limit, exception count or sample size')
    return run(args)
