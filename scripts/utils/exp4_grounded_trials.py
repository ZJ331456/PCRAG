"""Paired retrieval trials with frozen historical difficulty strata.

Historical gold ranks are used only by the offline sampler. Existing frozen
indexes are cloned privately; no full retrieval, QA or server restart is run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import urllib.request

from . import exp4_improvements as previous
from . import exp4_round2 as paired
from .common import read_json, write_json
from .difficulty_sampling import make_difficulty_split
from .representative_sampling import collect_prior_exclusions, poststratified_metrics


ROOT, DATASETS = previous.ROOT, previous.DATASETS
RUN_TAG = 'grounded_difficulty'
TEMP_NAME = '_exp4_grounded_trials'
SUMMARY_NAME = 'exp4_grounded_selection'
RECALLS = paired.EXPORTED_RECALLS
BASE = paired.Variant('baseline', ('planning', 'plan_prune'),
                      plan_validation='canonical_refs', plan_routing='question_structure')
DAG = replace(BASE, name='dag_control', flags=BASE.flags + ('dag_package',))
SOURCE = replace(DAG, name='source_witness', flags=DAG.flags + ('source_witness',))
RECOVERY = replace(DAG, name='failure_recovery', flags=DAG.flags + ('failure_recovery',))
COMBINED = replace(DAG, name='witness_recovery',
                   flags=DAG.flags + ('source_witness', 'failure_recovery'))
PROFILES = (BASE, DAG, SOURCE, RECOVERY, COMBINED)
SMOKE_PROFILES = (BASE, replace(BASE, name='baseline_replay'), replace(COMBINED, name='combined_smoke'))
SMOKE_INDICES = {'hotpotqa': [0, 942], '2wikimultihopqa': [742, 876], 'musique': [445, 793]}
EXCLUDE_TAGS = ('structure_representative', 'failure_focused', 'bridge_support', 'structural_support')
PARENT_FIELDS = ('plan', 'bindings', 'branch_scores', 'routes', 'search_count',
                 'llm_plan_calls', 'llm_verification_calls', 'planning_outputs', 'verification_outputs')


def validate_recovery_parent(parent, result):
    """Hold the dependency-search outputs before recovery fixed."""
    before = {row['query_index']: row for row in parent['results']}
    after = {row['query_index']: row for row in result['results']}
    if (not before or before.keys() != after.keys() or len(before) != len(parent['results'])
            or len(after) != len(result['results'])):
        raise ValueError('Recovery parent question subset changed')
    for index in before:
        old = before[index]['retrieval_trace']['evidence']
        new = after[index]['retrieval_trace']['evidence'].get('improvement_failure_recovery', {})
        snapshot = new.get('original_parent_fields')
        if (not isinstance(snapshot, dict) or not set(PARENT_FIELDS).issubset(snapshot)
                or any(old.get(key) != snapshot[key] for key in PARENT_FIELDS)):
            raise ValueError(f'Recovery initial parent outputs differ on question {index}')
    return {'validated': True, 'question_count': len(before), 'equal_fields': list(PARENT_FIELDS),
            'scope': 'Before recovery; additional repair requests and resulting inputs may differ.'}


def progress(summary, path):
    completed = sum(bool(item.get('validated')) for item in summary['group_plan'])
    summary['progress'] = {'planned_groups': len(summary['group_plan']), 'completed_groups': completed,
        'remaining_groups': len(summary['group_plan']) - completed, 'maximum_possible_groups': 36}
    write_json(path, summary)
    print('[grounded-progress]', json.dumps(summary['progress']), flush=True)


def register(summary, path, phase, profiles, n):
    known = {item['id'] for item in summary['group_plan']}
    for profile in profiles:
        for dataset in DATASETS:
            name = f'{phase}/{dataset}/{profile.name}'
            if name not in known:
                summary['group_plan'].append({'id': name, 'phase': phase, 'dataset': dataset,
                    'variant': profile.name, 'samples': n, 'validated': False})
    progress(summary, path)


def mark(summary, path, phase, dataset, profile):
    name = f'{phase}/{dataset}/{profile.name}'
    found = next((item for item in summary['group_plan'] if item['id'] == name), None)
    if found is None:
        raise ValueError(f'Unplanned group: {name}')
    found['validated'] = True
    progress(summary, path)


def decorate_report(context, case, phase, report, parent=None):
    """Add calibrated measurements and re-sign the report, without changing rankings."""
    if phase != 'smoke':
        sampling = context['difficulty_sampling']
        indices = [row['query_index'] for row in report['per_question']]
        estimates = {}
        for name, target in (('initial_available_population', sampling['available_indices']),
                             ('full_population_diagnostic', None)):
            try:
                estimate = poststratified_metrics(sampling['features'], indices, report['per_question'],
                    target_indices=target, metrics=RECALLS)
                estimate['weight_definition'] = ('Historical difficulty × structure calibration; '
                    'screening uses stratified SRS in each available cell. Confirmation weights '
                    'target the original available population; conditional weights are in sampling report.')
                estimate['uncertainty_warning'] = ('Development set with historical difficulty labels and '
                    'prior exclusions; calibration is not an independent final-test claim.')
                estimates[name] = estimate
            except ValueError as error:
                if name == 'initial_available_population':
                    raise
                estimates[name] = {'unavailable': str(error)}
        report['difficulty_evaluation'] = estimates
        report['difficulty_sampling_sha256'] = context['difficulty_sampling_sha256']
    if parent:
        source = Path(parent['source_result_path'])
        if previous.experiments.sha256(source) != parent['source_result_sha256']:
            raise ValueError('Recovery parent result changed')
        report['recovery_parent'] = parent
        report['recovery_parent_output_validation'] = validate_recovery_parent(
            read_json(source), read_json(case / 'result.json'))
    write_json(case / 'report.json', report)
    marker = read_json(case / 'validated.ok')
    marker['report_sha256'] = previous.experiments.sha256(case / 'report.json')
    write_json(case / 'validated.ok', marker)
    return report


def parent_record(case, cache, saved=None):
    if not cache.is_dir():
        if not (case / 'index/llm_cache').is_dir():
            raise ValueError('Required parent cache was removed before its snapshot')
        previous.snapshot_cache(case / 'index/llm_cache', cache)
    record = {'path': str(cache), 'sha256': previous.experiments.cache_hashes(cache),
              'source_result_path': str(case / 'result.json'),
              'source_result_sha256': previous.experiments.sha256(case / 'result.json')}
    if saved is not None and saved != record:
        raise ValueError('Frozen phase parent cache or result changed')
    return record


def run_group(args, context, temp, env, summary, path, phase, profile, parent=None, snapshot=False):
    indices = summary['indices'][phase][context['dataset']]
    case = temp / phase / context['dataset'] / profile.name
    if parent:
        if previous.experiments.cache_hashes(Path(parent['path'])) != parent['sha256']:
            raise ValueError('Frozen initial cache changed')
        if previous.experiments.sha256(Path(parent['source_result_path'])) != parent['source_result_sha256']:
            raise ValueError('Frozen parent result changed')
        context = dict(context, cache=Path(parent['path']), cache_hashes=parent['sha256'])
    report = paired.run_case(args, context, case, indices, profile, env, cleanup=False)
    gate_parent = parent if phase != 'smoke' and 'failure_recovery' in profile.flags else None
    try:
        report = decorate_report(context, case, phase, report, gate_parent)
        if profile.name == DAG.name and parent:
            report['dag_control_parent_validation'] = paired.validate_package_parent_outputs(
                read_json(parent['source_result_path']), read_json(case / 'result.json'))
            write_json(case / 'report.json', report)
            marker = read_json(case / 'validated.ok')
            marker['report_sha256'] = previous.experiments.sha256(case / 'report.json')
            write_json(case / 'validated.ok', marker)
        saved_parent = None
        if snapshot:
            records = summary.setdefault('parent_caches', {}).setdefault(phase, {}).setdefault(profile.name, {})
            cache = temp / 'parent_caches' / phase / context['dataset'] / profile.name
            saved_parent = parent_record(case, cache, records.get(context['dataset']))
            records[context['dataset']] = saved_parent
            write_json(path, summary)
    except Exception:
        (case / 'validated.ok').unlink(missing_ok=True)
        raise
    if not read_json(case / 'validated.ok').get('cleaned_private_index'):
        paired.cleanup_private_index(context, case)
    summary.setdefault(phase, {}).setdefault('profiles', {}).setdefault(profile.name, {
        'config': asdict(profile), 'reports': {}})['reports'][context['dataset']] = report
    mark(summary, path, phase, context['dataset'], profile)
    return report, saved_parent


def compare(args, reports, reference):
    weighted = lambda records: {dataset: dict(report, retrieval_metrics=
        report['difficulty_evaluation']['initial_available_population']['metrics'])
        for dataset, report in records.items()}
    kwargs = dict(target_gain=args.target_gain, gain_unit='absolute_pp',
                  max_regression=args.max_regression, allowed_exceptions=2, protected_metrics=RECALLS)
    raw = paired.target_assessment(reports, reference, **kwargs)
    calibrated = paired.target_assessment(weighted(reports), weighted(reference), **kwargs)
    gains = {dataset: {metric: {
        'raw_delta': reports[dataset]['retrieval_metrics'][metric] - reference[dataset]['retrieval_metrics'][metric],
        'calibrated_delta': weighted(reports)[dataset]['retrieval_metrics'][metric] -
                            weighted(reference)[dataset]['retrieval_metrics'][metric]}
        for metric in RECALLS} for dataset in DATASETS}
    positives = [f'{dataset}/{metric}' for dataset in DATASETS for metric in ('Recall@5', 'Recall@10')
                 if gains[dataset][metric]['raw_delta'] > 1e-10 and
                    gains[dataset][metric]['calibrated_delta'] > 1e-10]
    guard = raw['regression_limit_met'] and calibrated['regression_limit_met']
    return {'raw': raw, 'calibrated': calibrated, 'gains': gains, 'regression_limit_met': guard,
            'positive_metrics': positives, 'positive_signal': guard and bool(positives),
            'large_target_met': raw['strong_target_met'] and calibrated['strong_target_met']}


def assessments(args, summary, path, phase):
    records = summary[phase]['profiles']
    baseline = records['baseline']['reports']
    control = records['dag_control']['reports']
    for name, entry in records.items():
        if name not in ('baseline',):
            entry['vs_plan_prune'] = compare(args, entry['reports'], baseline)
        if name not in ('baseline', 'dag_control'):
            entry['vs_dag_control'] = compare(args, entry['reports'], control)
    comparisons = {}
    for label, new, old in (('source_minus_dag', SOURCE.name, DAG.name),
                            ('recovery_minus_dag', RECOVERY.name, DAG.name),
                            ('combined_minus_source', COMBINED.name, SOURCE.name),
                            ('combined_minus_recovery', COMBINED.name, RECOVERY.name)):
        comparisons[label] = (compare(args, records[new]['reports'], records[old]['reports'])
            if new in records and old in records else {'unavailable': 'Both profiles not run in this phase.'})
    summary[phase]['component_comparisons'] = comparisons
    write_json(path, summary)


def phase_run(args, contexts, temp, env, summary, path, phase, profiles):
    for profile in profiles:
        for dataset, context in contexts.items():
            records = summary.get('parent_caches', {}).get(phase, {})
            parent_name = SOURCE.name if profile.name == COMBINED.name else BASE.name
            parent = None if profile.name == BASE.name else records.get(parent_name, {}).get(dataset)
            if profile.name != BASE.name and not parent:
                raise ValueError(f'{phase}/{dataset}: required {parent_name} cache is missing')
            snapshot = profile.name in (BASE.name, SOURCE.name)
            run_group(args, context, temp, env, summary, path, phase, profile, parent, snapshot)
    summary.setdefault(phase, {})['complete'] = True
    if phase == 'smoke':
        for dataset in DATASETS:
            base = read_json(temp / phase / dataset / 'baseline/result.json')
            replay = read_json(temp / phase / dataset / 'baseline_replay/result.json')
            fields = ('docs', 'doc_scores', 'candidate_docs', 'candidate_doc_scores', 'retrieval_metrics')
            before = {row['query_index']: row for row in base['results']}
            after = {row['query_index']: row for row in replay['results']}
            if (before.keys() != after.keys() or len(before) != len(base['results'])
                    or len(after) != len(replay['results'])):
                raise ValueError(f'{dataset}: baseline replay question subset differs')
            if any(before[index].get(key) != after[index].get(key) for index in before for key in fields):
                raise ValueError(f'{dataset}: baseline replay differs')
    else:
        assessments(args, summary, path, phase)
    write_json(path, summary)


def candidate_order(entry):
    assessment = entry.get('vs_dag_control', entry['vs_plan_prune'])
    priorities = sum(name.startswith(('hotpotqa/', 'musique/')) for name in assessment['positive_metrics'])
    gain = sum(assessment['gains'][dataset][metric]['calibrated_delta']
               for dataset in DATASETS for metric in ('Recall@5', 'Recall@10'))
    return (-priorities, -int(assessment['large_target_met']), -gain,
            len(entry['config']['flags']), entry['config']['name'])


def confirmation_profiles(summary):
    entries = summary['screen']['profiles']
    new = [entries[p.name] for p in (SOURCE, RECOVERY, COMBINED)
           if entries[p.name]['vs_dag_control']['positive_signal'] and
              entries[p.name]['vs_plan_prune']['regression_limit_met']]
    chosen = sorted(new, key=candidate_order)[:2]
    # An independently useful existing DAG component can be confirmed too,
    # but is never labelled a new witness/recovery contribution.
    if not chosen and entries[DAG.name]['vs_plan_prune']['positive_signal']:
        return (BASE, DAG), [DAG.name]
    if not chosen:
        return (), []
    names = [entry['config']['name'] for entry in chosen]
    if COMBINED.name in names:
        # A is the mandatory initial-parent control for AB. Keep at most two
        # upgraded profiles; do not silently omit the parent cache.
        names = [SOURCE.name, COMBINED.name]
    profiles = tuple(p for p in PROFILES if p.name in {'baseline', 'dag_control', *names})
    return profiles, [entry['config']['name'] for entry in chosen if entry['config']['name'] in names]


def build_contexts(args, out, temp):
    contexts = {dataset: previous.context_for(out, dataset, temp) for dataset in DATASETS}
    exclusions = collect_prior_exclusions(out, additional_run_tags=EXCLUDE_TAGS)
    for dataset, context in contexts.items():
        reference = out / 'cases' / dataset / 'exp4_dependency_binding_plan_prune'
        previous.verify_trial_report(reference)
        rows, _ = previous.experiments.validate_result(read_json(reference / 'result.json'),
            context['manifest'], 'exp4_dependency_binding', context['data'], context['corpus'], expected_stage=4)
        excluded = set(exclusions['indices'][dataset]) | set(SMOKE_INDICES[dataset])
        sampling = make_difficulty_split(context['data'], dataset, hops=context['hops'],
            gold_counts=[len(previous.experiments.gold_docs(sample, dataset)) for sample in context['data']],
            baseline_measurements={row['query_index']: row for row in rows}, excluded_indices=sorted(excluded),
            screen_size=args.screen_size, confirmation_size=args.confirm_size,
            seed=args.screen_seed, confirmation_seed=args.confirm_seed)
        context['difficulty_sampling'] = sampling
        context['difficulty_sampling_sha256'] = paired.signature(sampling)
        context['historical_plan_prune_sha256'] = previous.experiments.sha256(reference / 'result.json')
    return contexts, exclusions


def run(args):
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(ROOT / 'outputs') or out == ROOT / 'outputs':
        raise ValueError('Output root must be a dedicated project outputs directory')
    temp, path = out / TEMP_NAME, out / 'metadata' / SUMMARY_NAME / 'selection.json'
    if temp.is_symlink():
        raise ValueError('Refusing a symlinked trial directory')
    temp.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('Existing vLLM is unavailable')
    if not Path(args.vllm_log).is_file():
        raise ValueError('vLLM request log missing')
    hippo_before = previous.code_hashes(Path(args.hippo_root))
    contexts, exclusions = build_contexts(args, out, temp)
    protocol = previous.protocol_for(args, contexts)
    for relative in ('scripts/utils/exp4_grounded_trials.py', 'scripts/exp4_grounded_trials.py',
                     'scripts/run_exp4_grounded_trials.sh', 'scripts/utils/exp4_round2.py',
                     'scripts/utils/difficulty_sampling.py', 'scripts/utils/representative_sampling.py'):
        protocol['algorithm_code_sha256'][relative] = previous.experiments.sha256(ROOT / relative)
    protocol.update(run_tag=RUN_TAG, profiles=[asdict(p) for p in PROFILES],
        hipporag_source_sha256=hippo_before,
        smoke_indices=SMOKE_INDICES, exclusions=exclusions, screen_seed=args.screen_seed,
        confirmation_seed=args.confirm_seed, max_regression_pp=args.max_regression, target_gain_pp=args.target_gain,
        sampling={dataset: context['difficulty_sampling_sha256'] for dataset, context in contexts.items()},
        historical_baseline_sha256={dataset: context['historical_plan_prune_sha256'] for dataset, context in contexts.items()},
        calibration_target='Initial available pool, after previous development exclusions',
        parent_policy='A/B start from fresh phase baseline. AB starts from completed A cache; '
                      'before-recovery parent fields must match baseline/A. No universal zero-HTTP assertion.')
    protocol = json.loads(json.dumps(protocol))
    if path.is_file():
        summary = read_json(path)
        if summary.get('protocol') != protocol or summary.get('protocol_sha256') != paired.signature(protocol):
            raise ValueError('Frozen algorithm, sampling, sources or runtime changed')
        for dataset, context in contexts.items():
            sample_path = path.parent / 'sampling' / f'{dataset}.json'
            if read_json(sample_path) != context['difficulty_sampling']:
                raise ValueError('Frozen difficulty sampling report changed')
    else:
        indices = {'smoke': SMOKE_INDICES,
            'screen': {dataset: context['difficulty_sampling']['screen_indices'] for dataset, context in contexts.items()},
            'confirmation': {dataset: context['difficulty_sampling']['confirmation_indices'] for dataset, context in contexts.items()}}
        summary = {'schema_version': 1, 'run_tag': RUN_TAG, 'protocol': protocol,
            'protocol_sha256': paired.signature(protocol), 'indices': indices, 'group_plan': [],
            'sampling_policy': 'historical_difficulty_structure_srs', 'selection_status': 'pending_small_sample',
            'automatic_full_run_enabled': False,
            'warning': 'Historical ranks are offline sampling labels. Existing evaluation files have been used '
                       'for development; these are not independent final-test results.'}
        for dataset, context in contexts.items():
            write_json(path.parent / 'sampling' / f'{dataset}.json', context['difficulty_sampling'])
        register(summary, path, 'smoke', SMOKE_PROFILES, 2)
        register(summary, path, 'screen', PROFILES, args.screen_size)
    env = previous.environment(args)
    env['PYTHONHASHSEED'] = '42'
    if args.mode in ('smoke', 'all'):
        phase_run(args, contexts, temp, env, summary, path, 'smoke', SMOKE_PROFILES)
    if args.mode in ('screen', 'all'):
        if not summary.get('smoke', {}).get('complete'):
            raise ValueError('A completed smoke is required before screening')
        phase_run(args, contexts, temp, env, summary, path, 'screen', PROFILES)
    if args.mode in ('confirm', 'all'):
        if not summary.get('screen', {}).get('complete'):
            raise ValueError('Completed screening is required')
        profiles, candidate_names = confirmation_profiles(summary)
        if summary.get('confirmation_candidates') is not None and summary['confirmation_candidates'] != candidate_names:
            raise ValueError('Confirmation candidates changed on resume')
        summary['confirmation_candidates'] = candidate_names
        if profiles:
            register(summary, path, 'confirmation', profiles, args.confirm_size)
            phase_run(args, contexts, temp, env, summary, path, 'confirmation', profiles)
            confirmed = summary['confirmation']['profiles']
            summary['replicated_positive_metrics'] = {
                name: sorted(set(summary['screen']['profiles'][name][
                    'vs_plan_prune' if name == DAG.name else 'vs_dag_control']['positive_metrics']) &
                    set(confirmed[name][
                    'vs_plan_prune' if name == DAG.name else 'vs_dag_control']['positive_metrics']))
                for name in candidate_names}
            summary['confirmed_new_profiles'] = [name for name in candidate_names if name != DAG.name
                and confirmed[name]['vs_dag_control']['positive_signal']
                and confirmed[name]['vs_plan_prune']['regression_limit_met']
                and summary['replicated_positive_metrics'][name]]
            summary['confirmed_existing_dag'] = (
                confirmed[DAG.name]['vs_plan_prune']['positive_signal'] and bool(
                    set(summary['screen']['profiles'][DAG.name]['vs_plan_prune']['positive_metrics']) &
                    set(confirmed[DAG.name]['vs_plan_prune']['positive_metrics'])))
            summary['selection_status'] = ('small_sample_new_signal_confirmed' if summary['confirmed_new_profiles']
                else 'small_sample_existing_dag_only' if summary['confirmed_existing_dag'] else 'small_sample_no_confirmed_signal')
        else:
            summary['confirmation'] = {'complete': True, 'skipped': True, 'reason': 'No guarded paired positive signal.'}
            summary['selection_status'] = 'small_sample_no_guarded_signal'
        summary['automatic_full_run_enabled'] = False
        write_json(path, summary)
    if previous.code_hashes(Path(args.hippo_root)) != hippo_before:
        raise ValueError('Original HippoRAG source changed')
    for context in contexts.values():
        if previous.experiments.asset_hashes(context['source'], context['manifest']['model_dir']) != context['manifest']['source_asset_sha256']:
            raise ValueError('Public source index changed')
    progress(summary, path)
    print('[grounded-done]', args.mode, summary['selection_status'], str(path), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'screen', 'confirm', 'all'), default='all')
    parser.add_argument('--out-root', default=str(previous.DEFAULT_OUT))
    parser.add_argument('--screen-size', type=int, default=90)
    parser.add_argument('--confirm-size', type=int, default=45)
    parser.add_argument('--screen-seed', type=int, default=1142)
    parser.add_argument('--confirm-seed', type=int, default=1242)
    parser.add_argument('--max-regression', type=float, default=1.0)
    parser.add_argument('--target-gain', type=float, default=4.0)
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--runtime-deps', default=str(previous.DEFAULT_RUNTIME))
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=str(previous.DEFAULT_OUT / 'logs/vllm.log'))
    args = parser.parse_args(argv)
    if args.screen_size < 2 or args.confirm_size < 1 or args.max_regression < 0 or args.target_gain <= 0:
        parser.error('Invalid sample budgets or gain limits')
    # run_case's older representative policy would calibrate only structural
    # classes. This runner owns joint difficulty estimates and their hashes.
    args.sampling_policy = 'difficulty'
    return run(args)
