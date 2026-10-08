"""Screen five opt-in retrieval changes and reuse frozen indexes for a final run.

Development and confirmation questions are disjoint from each other, but both
come from the final 1,000-question files. These results are exploratory tuning,
not an untouched test-set claim. Cache warmth is identical across variants.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
from types import SimpleNamespace
import urllib.request

from . import improvement_experiments as experiments
from .common import http_status_counts, read_json, write_json
from .new_index_compare import PC_ARGUMENTS, ROOT, code_hashes, environment, execute


DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique')
FLAGS = ('planning', 'selection', 'binding', 'closure', 'adaptive')
DEFAULT_OUT = ROOT / 'outputs/3multi_hop_datasets_results_10_5'
DEFAULT_RUNTIME = Path('/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513')
FINAL_CASE = 'exp4_dependency_binding_improvement'
TEMP_NAME = '_exp4_improvement_trials'
LOG = logging.getLogger('exp4_improvements')


def flags_name(flags):
    return '+'.join(flag for flag in FLAGS if flag in flags) or 'baseline'


def stratified_indices(data, hops, dataset, count, seed=42, excluded=()):
    """Use labels only to balance offline evaluation; never pass them to planning."""
    available = [i for i in range(len(data)) if i not in set(excluded)]
    if count < 1 or count > len(available):
        raise ValueError(f'{dataset}: sample count {count} exceeds available questions')
    groups = defaultdict(list)
    for index in available:
        group = (int(hops[index]), len(experiments.gold_docs(data[index], dataset)))
        groups[group].append(index)
    rng = random.Random(seed)
    for group in sorted(groups):
        rng.shuffle(groups[group])
    if count < len(groups):
        # Bounded smoke covers the smallest and largest structural groups.
        group_ids = [sorted(groups)[0], sorted(groups)[-1]][:count]
        return sorted(groups[group].pop() for group in group_ids)
    # Preserve population proportions while giving each structural group a
    # sample. A balanced 50/50 two-/four-support subset would overstate the
    # influence of 2Wiki's less frequent four-support questions.
    desired = {group: count * len(values) / len(available) for group, values in groups.items()}
    allocation = {group: max(1, min(len(values), int(desired[group])))
                  for group, values in groups.items()}
    while sum(allocation.values()) < count:
        candidates = [group for group in groups if allocation[group] < len(groups[group])]
        chosen = max(candidates, key=lambda group: (desired[group] - allocation[group], group))
        allocation[chosen] += 1
    while sum(allocation.values()) > count:
        candidates = [group for group in groups if allocation[group] > 1]
        chosen = min(candidates, key=lambda group: (desired[group] - allocation[group], group))
        allocation[chosen] -= 1
    return sorted(index for group in sorted(groups) for index in groups[group][:allocation[group]])


def subset_manifest(manifest, data, hops, indices):
    result = deepcopy(manifest)
    result.update(
        selected_indices=list(indices), sample_size_requested=len(indices),
        selected_sample_ids=[experiments.sample_identity(data[i], i) for i in indices],
        benchmark_hops=[hops[i] for i in indices],
        hop_distribution=dict(Counter(str(hops[i]) for i in indices)),
        gold_support_count_distribution=dict(Counter(
            str(len(experiments.gold_docs(data[i], manifest['dataset']))) for i in indices)),
    )
    return result


def aggregate_measurements(rows):
    keys = tuple(rows[0]['metrics'])
    return {
        'n_samples': len(rows),
        'retrieval_metrics': {key: sum(row['metrics'][key] for row in rows) / len(rows) for key in keys},
        'all_gold_top5': sum(row['all_gold_top5'] for row in rows) / len(rows),
        'all_gold_top10': sum(row['all_gold_top10'] for row in rows) / len(rows),
        'per_question': rows,
    }


def baseline_subset(context, indices):
    return aggregate_measurements([context['baseline_measurements'][i] for i in indices])


def compact(report):
    return {key: value for key, value in report.items() if key != 'per_question'}


def comparison_score(reports, baselines, tolerance=0.02):
    """Choose one configuration for all datasets using explicit nonregression gates."""
    gains = {}
    for dataset in DATASETS:
        new, old = reports[dataset], baselines[dataset]
        gains[dataset] = {
            'Recall@5': new['retrieval_metrics']['Recall@5'] - old['retrieval_metrics']['Recall@5'],
            'Recall@10': new['retrieval_metrics']['Recall@10'] - old['retrieval_metrics']['Recall@10'],
            'all_gold_top5': new['all_gold_top5'] - old['all_gold_top5'],
        }
    mean5 = sum(value['Recall@5'] for value in gains.values()) / len(DATASETS)
    mean10 = sum(value['Recall@10'] for value in gains.values()) / len(DATASETS)
    meanall = sum(value['all_gold_top5'] for value in gains.values()) / len(DATASETS)
    eligible = mean5 > 1e-8 and mean10 >= -1e-8 and all(
        value[key] >= -tolerance - 1e-8
        for value in gains.values() for key in ('Recall@5', 'Recall@10'))
    paired = {}
    for dataset in DATASETS:
        new_rows = {row['query_index']: row for row in reports[dataset].get('per_question', [])}
        old_rows = {row['query_index']: row for row in baselines[dataset].get('per_question', [])}
        if new_rows and new_rows.keys() == old_rows.keys():
            paired[dataset] = {}
            for metric in ('Recall@5', 'Recall@10'):
                differences = [new_rows[index]['metrics'][metric] - old_rows[index]['metrics'][metric]
                               for index in sorted(new_rows)]
                paired[dataset][metric] = {
                    'wins': sum(value > 1e-8 for value in differences),
                    'losses': sum(value < -1e-8 for value in differences),
                    'unchanged': sum(abs(value) <= 1e-8 for value in differences)}
    return {'eligible': eligible, 'mean_gain_r5': mean5, 'mean_gain_r10': mean10,
            'mean_gain_all5': meanall, 'score': .6 * mean5 + .25 * mean10 + .15 * meanall,
            'dataset_gains': gains, 'paired_changes': paired,
            'per_dataset_nonregression_tolerance': tolerance}


def snapshot_cache(source, destination):
    """SQLite backup captures committed WAL data without copying active locks."""
    destination.mkdir(parents=True)
    for original in sorted(source.rglob('*')):
        if not original.is_file() or original.name.endswith(('.lock', '-wal', '-shm')):
            continue
        target = destination / original.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.name.endswith('.sqlite'):
            with closing(sqlite3.connect(original.as_uri() + '?mode=ro', uri=True)) as reader:
                with closing(sqlite3.connect(target)) as writer:
                    reader.backup(writer)
                    writer.commit()
                    writer.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        else:
            shutil.copy2(original, target)


def context_for(out, dataset, temp):
    manifest = read_json(out / 'metadata' / dataset / 'manifest.json')
    source = out / 'shared_indexes' / dataset
    if Path(manifest['source_index']).resolve() != source.resolve():
        raise ValueError(f'{dataset}: source index is not the declared public index')
    data, corpus, hops = experiments.validated_dataset(
        manifest['data_path'], manifest['corpus_path'], dataset_name=dataset)
    if (experiments.sha256(manifest['data_path']) != manifest['data_sha256'] or
            experiments.sha256(manifest['corpus_path']) != manifest['corpus_sha256']):
        raise ValueError(f'{dataset}: data or corpus changed')
    if experiments.asset_hashes(source, manifest['model_dir']) != manifest['source_asset_sha256']:
        raise ValueError(f'{dataset}: frozen source index changed')
    baseline_case = out / 'cases' / dataset / 'exp4_dependency_binding'
    if not (baseline_case / 'validated.ok').is_file():
        raise ValueError(f'{dataset}: original exp4 is not validated')
    raw = read_json(baseline_case / 'result.json')
    measurements, _ = experiments.validate_result(
        raw, manifest, 'exp4_dependency_binding', data, corpus, expected_stage=4)
    by_index = {row['query_index']: row for row in measurements}
    prefixes = {row['query_index']: row['docs'] for row in raw['results']}
    del raw
    cache = temp / 'warm_cache' / dataset
    if not cache.is_dir():
        snapshot_cache(baseline_case / 'index/llm_cache', cache)
    return {'dataset': dataset, 'manifest': manifest, 'source': source,
            'clone_source': baseline_case / 'index',
            'data': data, 'corpus': corpus, 'hops': hops,
            'baseline_measurements': by_index, 'baseline_prefixes': prefixes,
            'cache': cache, 'cache_hashes': experiments.cache_hashes(cache)}


def initialize_case(context, case):
    if case.exists():
        raise ValueError(f'Unvalidated case already exists; inspect before retrying: {case}')
    case.mkdir(parents=True)
    source, manifest = context['clone_source'], context['manifest']
    if experiments.asset_hashes(source, manifest['model_dir']) != manifest['source_asset_sha256']:
        raise ValueError('Original exp4 assets differ from the public frozen index')
    linked = {source / manifest['model_dir'] / name for name in experiments.ASSETS}

    def copy_file(original, target):
        if Path(original) in linked:
            os.link(original, target)
            return target
        return shutil.copy2(original, target)

    shutil.copytree(source, case / 'index', copy_function=copy_file,
                    ignore=shutil.ignore_patterns('llm_cache', '*.lock', '*.sqlite*',
                                                 'metrics*.json', 'eval_results*', 'openie_progress*'))
    shutil.copytree(context['cache'], case / 'index/llm_cache')
    actual = experiments.asset_hashes(case / 'index', manifest['model_dir'])
    if actual != manifest['source_asset_sha256']:
        raise ValueError('Cloned graph/vector assets differ')
    if experiments.cache_hashes(case / 'index/llm_cache') != context['cache_hashes']:
        raise ValueError('Private starting cache differs')
    write_json(case / 'before.json', {'asset_sha256': actual,
                                     'initial_cache_sha256': context['cache_hashes']})


def command(args, context, case, indices, flags):
    manifest = context['manifest']
    runtime = manifest['runtime']
    return [args.python, '-B', '-u', str(ROOT / 'scripts/eval_dataset.py'),
        '--dataset', context['dataset'], '--sample_size', str(len(indices)),
        '--sample_seed', '42', '--sample_indices_file', str(case / 'selected_indices.json'),
        '--llm_name', 'qwen3-8b', '--llm_base_url', args.llm_base_url,
        '--embedding_batch_size', str(runtime['embedding_batch_size']),
        '--openie_max_workers', '8', '--llm_prefetch_workers', '8',
        '--eval_mode', 'retrieve', '--retrieval_top_k', '200', '--result_top_k', '10',
        '--candidate_output_top_k', '200', '--reuse_index',
        *PC_ARGUMENTS,
        '--data_path', manifest['data_path'], '--corpus_path', manifest['corpus_path'],
        '--corpus_mode', 'full', '--qa_top_k', '5', '--max_qa_steps', '1',
        '--max_new_tokens', '2048',
        '--embedding_model_name', runtime['embedding_model_name'],
        '--openie_strict', 'false', '--openie_prompt_version', 'optimized',
        '--openie_validation_mode', 'structural', '--improvement_stage', '4',
        '--evidence_improvements', ','.join(flag for flag in FLAGS if flag in flags),
        '--evidence_plan_node_budget', '6', '--evidence_plan_depth_budget', '4',
        '--evidence_selection_top_k', '10',
        '--stratified_eval', '--stratified_output', str(case / 'stratified.json'),
        '--save_dir', str(case / 'index'), '--output', str(case / 'result.json')]


def validate_case(args, context, case, manifest, flags, elapsed, start, end):
    result = read_json(case / 'result.json')
    measurements, metrics = experiments.validate_result(
        result, manifest, 'exp4_dependency_binding', context['data'], context['corpus'], expected_stage=4)
    config = result.get('runtime_config', {})
    value = config.get('evidence_improvements', '')
    actual_flags = set(value.split(',')) - {''} if isinstance(value, str) else set(value)
    if actual_flags != set(flags):
        raise ValueError(f'Configured improvements differ: {actual_flags} != {flags}')
    stats = result.get('llm_request_stats') or {}
    if stats.get('failures') != 0 or stats.get('max_in_flight') != 8:
        raise ValueError(f'LLM error accounting or concurrency differs: {stats}')
    if start < 0 or end < start:
        raise ValueError('vLLM request log rotated or truncated')
    status = http_status_counts(args.vllm_log, start, end)
    if any(code != '200' for code in status):
        raise ValueError(f'Non-200 API response: {status}')
    if stats.get('http_attempts') and not status:
        raise ValueError('HTTP attempts lack server request accounting')
    before = read_json(case / 'before.json')
    after = experiments.asset_hashes(case / 'index', manifest['model_dir'])
    source_after = experiments.asset_hashes(context['source'], manifest['model_dir'])
    if after != before['asset_sha256'] or source_after != manifest['source_asset_sha256']:
        raise ValueError('Private or public frozen index changed')
    report = aggregate_measurements(measurements)
    report.update(name=case.name, dataset=context['dataset'], improvement_stage=4,
        improvements=list(flag for flag in FLAGS if flag in flags), validated=True,
        seconds=elapsed, retrieval_seconds=result.get('retrieval_seconds'),
        llm_request_stats=stats, http_status_in_log=status,
        hop_distribution=manifest['hop_distribution'],
        vllm_log_slice={'path': args.vllm_log, 'start': start, 'end': end},
        asset_sha256_before=before['asset_sha256'], asset_sha256_after=after,
        initial_cache_sha256=before['initial_cache_sha256'],
        cache_policy='private identical snapshot of original exp4 completed cache; timings are warm-cache',
        benchmark_hop_policy=manifest['benchmark_hop_policy'])
    write_json(case / 'report.json', report)
    write_json(case / 'validated.ok', {
        'result_sha256': experiments.sha256(case / 'result.json'),
        'report_sha256': experiments.sha256(case / 'report.json'),
        'manifest_sha256': experiments.sha256(case / 'manifest.json')})
    print(f"[validated] {context['dataset']}/{case.name} n={len(measurements)} "
          f"R@5={metrics['Recall@5']:.4f} R@10={metrics['Recall@10']:.4f} sec={elapsed}", flush=True)
    return report, result


def run_case(args, context, case, indices, flags, env):
    manifest = subset_manifest(context['manifest'], context['data'], context['hops'], indices)
    if (case / 'validated.ok').is_file():
        if read_json(case / 'manifest.json') != manifest:
            raise ValueError(f'Resume question subset differs: {case}')
        report = read_json(case / 'report.json')
        if set(report.get('improvements', [])) != set(flags):
            raise ValueError(f'Resume configuration differs: {case}')
        marker = read_json(case / 'validated.ok')
        for field, filename in [('result_sha256', 'result.json'), ('report_sha256', 'report.json'),
                                ('manifest_sha256', 'manifest.json')]:
            if marker[field] != experiments.sha256(case / filename):
                raise ValueError(f'Validated {filename} changed: {case}')
        before = read_json(case / 'before.json')
        expected_assets = context['manifest']['source_asset_sha256']
        if (experiments.asset_hashes(case / 'index', manifest['model_dir']) != expected_assets or
                experiments.asset_hashes(context['source'], manifest['model_dir']) != expected_assets or
                before['asset_sha256'] != expected_assets or
                before['initial_cache_sha256'] != context['cache_hashes']):
            raise ValueError(f'Resumed index or initial cache contract changed: {case}')
        return report, None
    initialize_case(context, case)
    write_json(case / 'selected_indices.json', list(indices))
    write_json(case / 'manifest.json', manifest)
    start = Path(args.vllm_log).stat().st_size
    elapsed = execute(command(args, context, case, indices, flags), case / 'run.log', env)
    end = Path(args.vllm_log).stat().st_size
    return validate_case(args, context, case, manifest, flags, elapsed, start, end)


def fresh_phase_baseline(args, context, temp, phase, indices, env):
    """Match subset embedding batches before comparing any retrieval changes."""
    case = temp / phase / context['dataset'] / 'baseline'
    baseline, result = run_case(args, context, case, indices, frozenset(), env)
    if result is None:
        result = read_json(case / 'result.json')
    cache = temp / 'phase_start_cache' / phase / context['dataset']
    if not cache.is_dir():
        snapshot_cache(case / 'index/llm_cache', cache)
    phase_context = dict(context, cache=cache, cache_hashes=experiments.cache_hashes(cache))
    return phase_context, baseline, result


def smoke(args, contexts, temp, env, summary):
    phase = {'variants': {}, 'baseline_reproduction': {}, 'historical_prefix_matches': {}}
    for dataset, context in contexts.items():
        # Test rare 4-support/4-hop questions when present, plus another real question.
        indices = stratified_indices(context['data'], context['hops'], dataset, 2)
        phase.setdefault('indices', {})[dataset] = indices
        phase_context, baseline, baseline_result = fresh_phase_baseline(
            args, context, temp, 'smoke', indices, env)
        phase['variants'].setdefault('baseline', {})[dataset] = compact(baseline)
        prefixes = {row['query_index']: row['docs'] for row in baseline_result['results']}
        phase['historical_prefix_matches'][dataset] = {
            str(index): prefixes[index] == context['baseline_prefixes'][index] for index in indices}
        for name, flags in [('baseline_replay', frozenset()), ('all_five', frozenset(FLAGS))]:
            case = temp / 'smoke' / dataset / name
            report, result = run_case(args, phase_context, case, indices, flags, env)
            phase['variants'].setdefault(name, {})[dataset] = compact(report)
            if not flags:
                if result is None:
                    result = read_json(case / 'result.json')
                matches = [row['docs'] == prefixes[row['query_index']]
                           for row in result['results']]
                phase['baseline_reproduction'][dataset] = all(matches)
                if not all(matches):
                    raise ValueError(f'{dataset}: fresh subset baseline replay changed Top10; inspect smoke output')
    summary['smoke'] = phase
    return phase


def screen(args, contexts, temp, env, summary, summary_file):
    indices, confirm_indices, baselines, phase_contexts = {}, {}, {}, {}
    for dataset, context in contexts.items():
        indices[dataset] = stratified_indices(context['data'], context['hops'], dataset, args.screen_size)
        confirm_indices[dataset] = stratified_indices(
            context['data'], context['hops'], dataset, args.confirm_size, excluded=indices[dataset])
        phase_contexts[dataset], baselines[dataset], _ = fresh_phase_baseline(
            args, context, temp, 'screen', indices[dataset], env)
    phase = {'screen_indices': indices, 'confirmation_indices': confirm_indices,
             'baselines': baselines,
             'variants': {}, 'confirmation': {},
             'selection_policy': 'mean R@5 gain >0, mean R@10 >=0, every dataset R@5/R@10 >= baseline-.02; '
                                 'rank by .6*R5+.25*R10+.15*allgold5 gains; confirm on disjoint questions',
             'screen_size_per_dataset': args.screen_size, 'confirm_size_per_dataset': args.confirm_size}
    phase['historical_baselines_diagnostic_only'] = {
        dataset: baseline_subset(context, indices[dataset]) for dataset, context in contexts.items()}
    summary['screen'] = phase

    def evaluate(name, flags):
        reports = {}
        for dataset, context in phase_contexts.items():
            reports[dataset], _ = run_case(args, context,
                temp / 'screen' / dataset / name, indices[dataset], flags, env)
        judged = comparison_score(reports, baselines)
        phase['variants'][name] = {'flags': [flag for flag in FLAGS if flag in flags],
            'reports': reports, **judged}
        write_json(summary_file, summary)
        print(f"[screen] {name} R5gain={judged['mean_gain_r5']:+.5f} "
              f"R10gain={judged['mean_gain_r10']:+.5f} eligible={judged['eligible']}", flush=True)
        return judged

    seen = set()
    for flag in FLAGS:
        evaluate(flag, frozenset([flag]))
        seen.add(frozenset([flag]))
    for flags in [frozenset(['binding', 'closure']), frozenset(['selection', 'closure'])]:
        evaluate(flags_name(flags), flags)
        seen.add(flags)
    useful = frozenset(flag for flag in FLAGS if phase['variants'][flag]['eligible'])
    for variant in list(phase['variants'].values()):
        if variant['eligible']:
            useful = useful | frozenset(variant['flags'])
    for flags in [useful, frozenset(FLAGS)]:
        if flags and flags not in seen:
            evaluate(flags_name(flags), flags)
            seen.add(flags)
    ranked = sorted((item for item in phase['variants'].items() if item[1]['eligible']),
                    key=lambda item: (-item[1]['score'], len(item[1]['flags']), item[0]))
    confirmation_baselines, confirmation_contexts = {}, {}
    if ranked:
        for dataset, context in contexts.items():
            confirmation_contexts[dataset], confirmation_baselines[dataset], _ = fresh_phase_baseline(
                args, context, temp, 'confirmation', confirm_indices[dataset], env)
    phase['confirmation_baselines'] = confirmation_baselines
    chosen = frozenset()
    # At most two contenders are confirmed, bounding development cost.
    for name, candidate in ranked[:2]:
        flags = frozenset(candidate['flags'])
        reports = {}
        for dataset, context in confirmation_contexts.items():
            reports[dataset], _ = run_case(args, context,
                temp / 'confirmation' / dataset / name, confirm_indices[dataset], flags, env)
        judged = comparison_score(reports, confirmation_baselines)
        phase['confirmation'][name] = {'flags': candidate['flags'],
            'reports': reports, **judged}
        write_json(summary_file, summary)
        if judged['eligible']:
            chosen = flags
            break
    summary['chosen_flags'] = [flag for flag in FLAGS if flag in chosen]
    summary['selection_status'] = 'confirmed_exploratory' if chosen else 'no_confirmed_improvement'
    summary['chosen_name'] = flags_name(chosen)
    write_json(summary_file, summary)
    print(f"[selected] {summary['chosen_name']} status={summary['selection_status']}", flush=True)
    return chosen


def full(args, contexts, env, summary, summary_file):
    if 'chosen_flags' not in summary:
        raise ValueError('Run screen first: no reviewed configuration selection exists')
    flags = frozenset(summary['chosen_flags'])
    if not flags <= set(FLAGS):
        raise ValueError('Selection contains unknown flags')
    if not flags:
        summary['full_skipped_no_confirmed_improvement'] = True
        write_json(summary_file, summary)
        print('[full-skipped] No improvement passed both screening and confirmation; '
              'original full results preserved, no duplicate baseline labeled as improvement.', flush=True)
        return
    summary.setdefault('full', {})
    case_name = getattr(args, 'final_case_name', None) or FINAL_CASE
    for dataset, context in contexts.items():
        indices = list(context['manifest']['selected_indices'])
        print(f'[full-start] {dataset}/{case_name} n={len(indices)} flags={flags_name(flags)}', flush=True)
        report, _ = run_case(args, context, Path(args.out_root) / 'cases' / dataset / case_name,
                            indices, flags, env)
        summary['full'][dataset] = compact(report)
        write_json(summary_file, summary)
    summary['full_complete'] = True
    if args.mode == 'reviewed-full':
        summary['selection_status'] = '48_screen_positive_full_validation_completed'
    write_json(summary_file, summary)


def require_smoke(summary):
    phase = summary.get('smoke', {})
    if (phase.get('baseline_reproduction') != {dataset: True for dataset in DATASETS} or
            any(not phase.get('variants', {}).get(name, {}).get(dataset, {}).get('validated')
                for name in ('baseline', 'all_five') for dataset in DATASETS)):
        raise ValueError('Run a successful smoke first: original baseline and all-five variants must validate on all datasets')


def cleanup_trials(out, summary, summary_file):
    """Delete only our temporary tree after preserving the selection decision."""
    if 'chosen_flags' not in summary:
        raise ValueError('Refusing cleanup before the compact selection summary is saved')
    require_smoke(summary)
    temp = out / TEMP_NAME
    if temp.parent != out or temp.name != TEMP_NAME or temp.is_symlink():
        raise ValueError('Refusing to clean an unexpected temporary directory')
    if temp.exists():
        shutil.rmtree(temp)
    summary['temporary_trials_removed'] = True
    summary['temporary_trial_path'] = str(temp)
    write_json(summary_file, summary)
    print(f'[cleaned] {temp}; compact screening and confirmation measurements preserved', flush=True)


def cleanup_trial_results(out, summary, summary_file):
    """Remove test outputs before full runs, retaining only the common cache."""
    if 'chosen_flags' not in summary:
        raise ValueError('No durable selection exists; refusing trial cleanup')
    require_smoke(summary)
    temp = out / TEMP_NAME
    if temp.is_symlink() or temp.parent != out or temp.name != TEMP_NAME:
        raise ValueError('Refusing unexpected temporary trial path')
    if temp.is_dir():
        for path in temp.iterdir():
            if path.name == 'warm_cache':
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    summary['temporary_trial_results_removed_before_full'] = True
    write_json(summary_file, summary)
    print('[cleaned-before-full] smoke/screen/confirmation results removed; '
          'only common original cache remains until final runs finish.', flush=True)


def protocol_for(args, contexts):
    files = sorted((ROOT / 'src/pathcondrag').rglob('*.py')) + [
        ROOT / 'scripts/eval_dataset.py', ROOT / 'scripts/eval_utils.py',
        ROOT / 'scripts/utils/exp4_improvements.py', ROOT / 'scripts/utils/new_index_compare.py',
        ROOT / 'scripts/utils/common.py', ROOT / 'scripts/utils/improvement_experiments.py']
    return {'algorithm_code_sha256': {str(path.relative_to(ROOT)): experiments.sha256(path) for path in files},
            'runtime': {'embedding_batch_size': 4, 'llm_workers': 8, 'openie_workers': 8,
                        'max_new_tokens': 2048, 'enable_thinking': False,
                        'hop_source': 'benchmark', 'node_budget': 6, 'depth_budget': 4,
                        'selection_top_k': 10, 'runtime_deps': str(Path(args.runtime_deps).resolve()),
                        'python_hash_seed': 42,
                        'python': str(Path(args.python).resolve()), 'llm_base_url': args.llm_base_url},
            'sources': {dataset: {'data_sha256': context['manifest']['data_sha256'],
                'corpus_sha256': context['manifest']['corpus_sha256'],
                'asset_sha256': context['manifest']['source_asset_sha256'],
                'warm_cache_sha256': context['cache_hashes']}
                for dataset, context in contexts.items()},
            'screen_size': args.screen_size, 'confirm_size': args.confirm_size, 'seed': 42}


def require_screening_protocol(summary, protocol):
    """Reuse screening measurements only when the orchestration file changed."""
    old = summary.get('protocol')
    if not isinstance(old, dict):
        raise ValueError('A signed screening protocol is required')
    expected_hash = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    if summary.get('protocol_sha256') != expected_hash:
        raise ValueError('Original screening protocol signature differs')
    previous, current = deepcopy(old), deepcopy(protocol)
    runner_path = 'scripts/utils/exp4_improvements.py'
    for item in (previous, current):
        if runner_path not in item.get('algorithm_code_sha256', {}):
            raise ValueError('Screening protocol omits the orchestration code hash')
        item['algorithm_code_sha256'].pop(runner_path)
    if previous != current:
        raise ValueError('Resume requires unchanged core algorithm, inputs, cache and runtime; '
                         'only this orchestration file may change')


def require_reviewed_protocol(summary, protocol):
    """A user-reviewed full run may change this orchestration file, nothing else."""
    require_screening_protocol(summary, protocol)
    if summary.get('reviewed_full_protocol') and summary['reviewed_full_protocol'] != protocol:
        raise ValueError('Existing reviewed-full protocol changed; refusing incompatible resume')


def reviewed_case_name(suffix):
    if not isinstance(suffix, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,80}', suffix):
        raise ValueError('Case suffix must contain only letters, digits or underscores')
    return FINAL_CASE + '_' + suffix


def verify_trial_report(case):
    marker = read_json(case / 'validated.ok')
    for field, filename in [('result_sha256', 'result.json'), ('report_sha256', 'report.json'),
                            ('manifest_sha256', 'manifest.json')]:
        if marker.get(field) != experiments.sha256(case / filename):
            raise ValueError(f'Screening artifact changed before cleanup: {case / filename}')
    report, manifest = read_json(case / 'report.json'), read_json(case / 'manifest.json')
    rows = report.get('per_question') or []
    if (report.get('validated') is not True or len(rows) != report.get('n_samples') or
            [row['query_index'] for row in rows] != manifest['selected_indices']):
        raise ValueError(f'Incomplete screening measurements: {case}')
    return report


def prepare_reviewed_full(args, contexts, summary, summary_file, protocol):
    require_smoke(summary)
    require_reviewed_protocol(summary, protocol)
    tokens = [part.strip() for part in (args.flags or '').split(',') if part.strip()]
    flags = frozenset(tokens)
    if not flags or len(tokens) != len(flags) or not flags <= set(FLAGS):
        raise ValueError('Reviewed-full requires explicit, unique known --flags')
    args.final_case_name = reviewed_case_name(args.case_suffix)
    name = flags_name(flags)
    candidate = summary.get('screen', {}).get('variants', {}).get(name)
    if not candidate or candidate.get('eligible') is not True or set(candidate.get('flags', [])) != flags:
        raise ValueError('Explicit flags must identify a completed positive screening variant')
    reports = candidate.get('reports', {})
    if set(reports) != set(DATASETS) or any(report.get('validated') is not True for report in reports.values()):
        raise ValueError('Reviewed variant needs validated reports from all three datasets')
    temp = Path(args.out_root).resolve() / TEMP_NAME
    already_prepared = bool(summary.get('reviewed_full_decision'))
    if already_prepared:
        decision = summary['reviewed_full_decision']
        if decision['flags'] != [flag for flag in FLAGS if flag in flags] or decision['case_name'] != args.final_case_name:
            raise ValueError('Reviewed-full decision differs from the existing full run')
    else:
        # Recheck selected reports and their paired controls before deleting raw
        # trials. Hash markers cover exported results, reports and manifests.
        for dataset, context in contexts.items():
            for variant in ('baseline', name):
                case = temp / 'screen' / dataset / variant
                report = verify_trial_report(case)
                if experiments.asset_hashes(case / 'index', context['manifest']['model_dir']) != context['manifest']['source_asset_sha256']:
                    raise ValueError(f'Screened private index differs from public frozen index: {case}')
                saved = (summary['screen']['baselines'][dataset] if variant == 'baseline' else reports[dataset])
                if report != saved:
                    raise ValueError(f'Saved screening statistics differ from validated report: {case}')
        before_file = summary_file.with_name('selection_before_reviewed_full.json')
        if not before_file.exists():
            write_json(before_file, summary)
        archive = {}
        for phase in ('smoke', 'screen', 'confirmation'):
            for report_path in sorted((temp / phase).glob('*/*/report.json')):
                case = report_path.parent
                if not (case / 'validated.ok').is_file():
                    continue
                report = verify_trial_report(case)
                archive[str(case.relative_to(temp))] = report
        summary['reviewed_archived_trial_reports'] = archive
        summary['reviewed_full_decision'] = {
            'flags': [flag for flag in FLAGS if flag in flags], 'case_name': args.final_case_name,
            'basis': 'explicit user request to proceed from positive 48-question screening to full evaluation',
            'user_requested_skip_confirmation': True, 'confirmation_passed': False,
            'prior_selection_status': summary.get('selection_status'),
            'screening_variant': name,
            'original_selection_snapshot': str(before_file),
            'original_selection_snapshot_sha256': experiments.sha256(before_file),
            'known_disabled_component_issue': 'Closure fallback may exclude accepted documents; '
                                              'core algorithm is frozen and closure is not enabled for adaptive-only.',
            'closure_enabled': 'closure' in flags,
        }
    summary['reviewed_full_protocol'] = protocol
    summary['reviewed_full_protocol_sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    summary['chosen_flags'] = [flag for flag in FLAGS if flag in flags]
    summary['chosen_name'] = name
    summary['selection_status'] = '48_screen_positive_full_validation_pending'
    write_json(summary_file, summary)
    print(f'[reviewed-full] flags={name} case={args.final_case_name}; '
          '48-question screening reused, no new smoke/screen/24-question confirmation will run.', flush=True)


def prepare_confirmation(args, contexts, summary, summary_file, protocol):
    """Verify archived screening records without rerunning deleted raw trials."""
    require_smoke(summary)
    require_screening_protocol(summary, protocol)
    signed = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if summary.get('confirmation_protocol') and (
            summary['confirmation_protocol'] != protocol or
            summary.get('confirmation_protocol_sha256') != signed):
        raise ValueError('Confirmation protocol changed; refusing incompatible resume')
    decision = summary.get('reviewed_full_decision', {})
    snapshot_file = summary_file.with_name('selection_before_reviewed_full.json')
    if (Path(decision.get('original_selection_snapshot', '')).resolve() != snapshot_file.resolve() or
            not snapshot_file.is_file() or
            experiments.sha256(snapshot_file) != decision.get('original_selection_snapshot_sha256')):
        raise ValueError('Original screening snapshot is missing or its signature changed')
    original = read_json(snapshot_file)
    if (original.get('protocol') != summary.get('protocol') or
            original.get('protocol_sha256') != summary.get('protocol_sha256') or
            original.get('smoke') != summary.get('smoke')):
        raise ValueError('Original screening protocol or smoke measurements changed')
    phase = summary.get('screen', {})
    stable = lambda value: {key: item for key, item in value.items()
                            if key not in ('confirmation', 'confirmation_baselines')}
    if stable(phase) != stable(original.get('screen', {})):
        raise ValueError('Saved screening measurements or confirmation indices changed')
    archive = summary.get('reviewed_archived_trial_reports', {})
    for dataset, context in contexts.items():
        screened = phase['screen_indices'][dataset]
        indices = phase['confirmation_indices'][dataset]
        expected = stratified_indices(context['data'], context['hops'], dataset,
                                      args.confirm_size, excluded=screened)
        if (len(screened) != args.screen_size or len(set(screened)) != len(screened) or
                len(indices) != args.confirm_size or indices != expected or
                not set(screened).isdisjoint(indices)):
            raise ValueError(f'{dataset}: confirmation indices differ from the fixed disjoint subset')
        saved_reports = {'baseline': phase['baselines'][dataset], **{
            name: variant['reports'][dataset] for name, variant in phase['variants'].items()}}
        for name, report in saved_reports.items():
            rows = report.get('per_question', [])
            if (archive.get(f'screen/{dataset}/{name}') != report or
                    report.get('validated') is not True or report.get('n_samples') != args.screen_size or
                    [row['query_index'] for row in rows] != screened):
                raise ValueError(f'{dataset}/{name}: archived screening report differs')
    ranked = sorted((item for item in phase['variants'].items() if item[1]['eligible']),
                    key=lambda item: (-item[1]['score'], len(item[1]['flags']), item[0]))[:2]
    plan = [{'dataset': dataset, 'name': name}
            for name in (['baseline'] + [name for name, _ in ranked]) for dataset in DATASETS]
    if not ranked:
        raise ValueError('No positive screening configuration needs confirmation')
    if summary.get('confirmation_plan') and summary['confirmation_plan'] != plan:
        raise ValueError('Confirmation candidates changed during resume')
    summary['confirmation_plan'] = plan
    summary['confirmation_protocol'] = protocol
    summary['confirmation_protocol_sha256'] = signed
    summary['chosen_flags'] = []
    summary['chosen_name'] = 'pending_confirmation'
    summary['selection_status'] = 'confirmation_pending'
    write_json(summary_file, summary)
    print('[confirm-preflight] archived screening reports and signed snapshot verified; '
          'deleted raw screening results are not revalidated or rerun.', flush=True)
    return ranked


def confirm(args, contexts, temp, env, summary, summary_file, ranked):
    """Finish independent small-sample confirmation; never launch full retrieval."""
    phase = summary['screen']
    baselines = phase.setdefault('confirmation_baselines', {})
    judgments = phase.setdefault('confirmation', {})
    completed = set()
    total = len(summary['confirmation_plan'])

    def progress():
        summary['confirmation_progress'] = {
            'planned_groups': total, 'completed_groups': len(completed),
            'remaining_groups': total - len(completed), 'samples_per_group': args.confirm_size,
            'completed_cases': sorted(completed)}
        write_json(summary_file, summary)
        print(f'[confirm-progress] planned={total} completed={len(completed)} '
              f'remaining={total-len(completed)} samples_per_group={args.confirm_size}', flush=True)

    # Hash-check completed report exports before including them in the initial
    # progress count. run_case also rechecks frozen assets and starting cache.
    for item in summary['confirmation_plan']:
        case = temp / 'confirmation' / item['dataset'] / item['name']
        if (case / 'validated.ok').is_file():
            verify_trial_report(case)
            completed.add(f"{item['dataset']}/{item['name']}")
    progress()
    phase_contexts = {}
    for dataset, context in contexts.items():
        indices = phase['confirmation_indices'][dataset]
        phase_contexts[dataset], baselines[dataset], _ = fresh_phase_baseline(
            args, context, temp, 'confirmation', indices, env)
        completed.add(f'{dataset}/baseline')
        progress()
    chosen = frozenset()
    for name, candidate in ranked:
        flags = frozenset(candidate['flags'])
        record = judgments.setdefault(name, {'flags': candidate['flags'], 'reports': {}})
        for dataset, context in phase_contexts.items():
            indices = phase['confirmation_indices'][dataset]
            record['reports'][dataset], _ = run_case(args, context,
                temp / 'confirmation' / dataset / name, indices, flags, env)
            completed.add(f'{dataset}/{name}')
            progress()
        record.update(comparison_score(record['reports'], baselines))
        write_json(summary_file, summary)
        if record['eligible'] and not chosen:
            chosen = flags
    summary['chosen_flags'] = [flag for flag in FLAGS if flag in chosen]
    summary['chosen_name'] = flags_name(chosen)
    summary['selection_status'] = 'confirmed_exploratory' if chosen else 'no_confirmed_improvement'
    summary['confirmation_complete'] = True
    write_json(summary_file, summary)
    print(f"[confirmed-small-only] {summary['chosen_name']} status={summary['selection_status']}; "
          'full retrieval is not started; confirmation outputs retained.', flush=True)


def run(args):
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(ROOT / 'outputs') or out == ROOT / 'outputs':
        raise ValueError('Output root must be a dedicated PathCondRAG outputs directory')
    summary_dir = out / 'metadata/exp4_improvement_selection'
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / 'selection.json'
    if args.mode == 'clean':
        if not summary_file.is_file():
            raise ValueError('No preserved screening summary exists')
        cleanup_trials(out, read_json(summary_file), summary_file)
        return 0
    temp = out / TEMP_NAME
    if temp.is_symlink():
        raise ValueError('Refusing a symlinked temporary trial directory')
    temp.mkdir(exist_ok=True)
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('Existing vLLM service is unavailable')
    if not Path(args.vllm_log).is_file():
        raise ValueError('vLLM request log does not exist')
    before_code = code_hashes(Path(args.hippo_root))
    contexts = {dataset: context_for(out, dataset, temp) for dataset in DATASETS}
    env = environment(args)
    env['PYTHONHASHSEED'] = '42'
    summary = read_json(summary_file) if summary_file.is_file() else {
        'schema_version': 1, 'datasets': list(DATASETS), 'seed': 42,
        'candidate_flags': list(FLAGS), 'evaluation_mode': 'retrieve',
        'development_overlap_final_evaluation': True,
        'paper_claim_warning': 'Screen and confirmation questions are from final files. '
                               'A separate untouched evaluation is required for paper claims.',
        'cache_policy': 'All variants start from identical private copies of completed original exp4 cache. '
                        'Per development phase, first run an empty-flags baseline, then snapshot its '
                        'completed cache as the identical starting point for every variant. '
                        'Wall time is warm-cache and unsuitable as a fair efficiency comparison.',
        'historical_subset_warning': 'Original full-run query embedding batches and subset batches can differ '
                                     'numerically. Historical filtered rows are diagnostic only; fresh matched '
                                     'subset controls determine development and confirmation gains.',
        'sources': {dataset: {'index': str(context['source']),
            'asset_sha256': context['manifest']['source_asset_sha256'],
            'data_sha256': context['manifest']['data_sha256'],
            'corpus_sha256': context['manifest']['corpus_sha256'],
            'warm_cache_sha256': context['cache_hashes']}
            for dataset, context in contexts.items()},
        'runtime': {'embedding_batch_size': 4, 'llm_workers': 8, 'openie_workers': 8,
                    'max_new_tokens': 2048, 'enable_thinking': False, 'hop_source': 'benchmark',
                    'retrieval_top_k': 200, 'result_top_k': 10,
                    'node_budget': 6, 'depth_budget': 4, 'selection_top_k': 10},
    }
    protocol = protocol_for(args, contexts)
    confirmation_candidates = None
    if args.mode == 'confirm':
        confirmation_candidates = prepare_confirmation(args, contexts, summary, summary_file, protocol)
    elif args.mode == 'reviewed-full':
        print('[reviewed-full-preflight] checking frozen indexes, existing reports and unchanged algorithm; '
              'this is CPU validation, not a new 48-question retrieval.', flush=True)
        prepare_reviewed_full(args, contexts, summary, summary_file, protocol)
    elif summary.get('protocol') and summary['protocol'] != protocol:
        raise ValueError('Algorithm source, input indexes, cache or runtime protocol changed; do not resume incompatible trials')
    if args.mode not in ('reviewed-full', 'confirm'):
        summary['protocol'] = protocol
        summary['protocol_sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(summary_file, summary)
    if args.mode in ('smoke', 'all'):
        smoke(args, contexts, temp, env, summary)
        write_json(summary_file, summary)
    if args.mode in ('screen', 'all'):
        require_smoke(summary)
        screen(args, contexts, temp, env, summary, summary_file)
    if args.mode == 'confirm':
        confirm(args, contexts, temp, env, summary, summary_file, confirmation_candidates)
    if args.mode in ('full', 'all', 'reviewed-full'):
        require_smoke(summary)
        cleanup_trial_results(out, summary, summary_file)
        full(args, contexts, env, summary, summary_file)
        cleanup_trials(out, summary, summary_file)
    if code_hashes(Path(args.hippo_root)) != before_code:
        raise ValueError('Native HippoRAG source changed during experiments')
    for context in contexts.values():
        if experiments.asset_hashes(context['source'], context['manifest']['model_dir']) != context['manifest']['source_asset_sha256']:
            raise ValueError('Public frozen index changed during experiments')
    print(f'[done] mode={args.mode} summary={summary_file}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'screen', 'confirm', 'full', 'clean', 'all', 'reviewed-full'), default='all')
    parser.add_argument('--flags', help='Explicit positive screened flags for reviewed-full only')
    parser.add_argument('--case-suffix', help='Letters/digits/underscores suffix for reviewed-full output cases')
    parser.add_argument('--screen-size', type=int, default=48)
    parser.add_argument('--confirm-size', type=int, default=24)
    parser.add_argument('--out-root', '--output-root', default=str(DEFAULT_OUT))
    parser.add_argument('--runtime-deps', default=str(DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=str(DEFAULT_OUT / 'logs/vllm.log'))
    args = parser.parse_args(argv)
    if args.mode != 'reviewed-full' and (args.flags is not None or args.case_suffix is not None):
        parser.error('--flags and --case-suffix are available only with --mode reviewed-full')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)
