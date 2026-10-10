"""One paired NV2 subset experiment with frozen indexes and upstream caches.

Sampling uses the question text and the already authorized benchmark hop count,
never gold documents, historical scores or failure categories. Labels enter only
the independent result evaluation. This is exploratory tuning on existing test
files; paired confidence intervals do not turn it into an untouched test set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import copy
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import shutil

from .common import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique')
DEFAULT_INDEX = ROOT / 'outputs/3multi_hop_datasets_results_with_nv2_10_8'
DEFAULT_OUT = ROOT / 'outputs/nv2_dependency_scoring_subset_10_10'
FLAGS = ('planning', 'plan_prune', 'dag_package', 'support_semantic_veto')
CASES = (('baseline', 'legacy'), ('dependency_scoring', 'dependency'))
METRICS = tuple(f'Recall@{k}' for k in (1, 2, 5, 10, 20, 200))
ALL_METRICS = (*METRICS, 'all_gold_top5', 'all_gold_top10')
STRICT_UPSTREAM = ('plan', 'bindings', 'planning_outputs', 'verification_outputs',
                   'routes', 'search_count', 'llm_plan_calls', 'llm_verification_calls',
                   'branch_scores', 'candidate_count', 'semantic_failures', 'rejected_hypotheses')
STRICT_RETRIEVAL_UPSTREAM = ('fact_filter', 'dense_fallback', 'hops', 'seed_entities',
                             'fact_seed_distribution', 'static_sub_questions', 'pcqd_sub_questions',
                             'pcqd_validation', 'path_hints', 'hint_diagnostics', 'selected_path_candidates')
LOG = logging.getLogger('nv2_dependency_subset')


def signature(value):
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def question_structure(question):
    """Coarse query-visible patterns; no dataset type or decomposition content."""
    text = ' '.join(str(question).lower().split())
    if re.search(r'\b(?:both|either|which of the two|older|younger|earlier|later|larger|'
                 r'smaller|taller|longer|same|more|fewer)\b', text):
        return 'parallel_or_comparison'
    if re.search(r'\b(?:whose|that|which|who)\b.+\b(?:was|is|were|are|has|had|'
                 r'wrote|directed|born|located)\b', text):
        return 'nested_relation'
    if re.search(r'\b(?:when|year|date|month|day|how many|how old|how long)\b', text):
        return 'time_or_quantity'
    if re.search(r'\b(?:where|country|city|state|town|county|place)\b', text):
        return 'location'
    return 'other_relation'


def choose_subset(data, hops, count, seed=1031, *, smoke=False):
    if len(data) != len(hops) or count < 1 or count > len(data):
        raise ValueError('Invalid sample budget or benchmark hop alignment')
    features = [{'query_index': index, 'hops': int(hops[index]),
                 'question_structure': question_structure(sample['question'])}
                for index, sample in enumerate(data)]
    groups = defaultdict(list)
    for row in features:
        row['stratum'] = f"hop={row['hops']}|query={row['question_structure']}"
        groups[row['stratum']].append(row['query_index'])
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    if smoke:
        # Exercise minimum and maximum hops while preserving the complete
        # original corpus. A two-question smoke cannot cover all strata.
        ordered = sorted(groups, key=lambda key: (features[groups[key][0]]['hops'], key))
        first, last = ordered[0], ordered[-1]
        selected = [groups[first][0]]
        selected.append(groups[last][0] if last != first else groups[first][1])
    else:
        if count < len(groups):
            raise ValueError(f'{count} samples cannot cover all {len(groups)} query/hop strata')
        desired = {key: count * len(values) / len(data) for key, values in groups.items()}
        minimum = 2 if sum(min(2, len(values)) for values in groups.values()) <= count else 1
        allocation = {key: min(len(values), max(minimum, int(desired[key])))
                      for key, values in groups.items()}
        while sum(allocation.values()) < count:
            eligible = [key for key in groups if allocation[key] < len(groups[key])]
            chosen = max(eligible, key=lambda key: (desired[key] - allocation[key], key))
            allocation[chosen] += 1
        while sum(allocation.values()) > count:
            eligible = [key for key in groups if allocation[key] > min(minimum, len(groups[key]))]
            chosen = min(eligible, key=lambda key: (desired[key] - allocation[key], key))
            allocation[chosen] -= 1
        selected = [index for key in sorted(groups) for index in groups[key][:allocation[key]]]
    selected = sorted(selected)
    population = Counter(row['stratum'] for row in features)
    sample_counts = Counter(features[index]['stratum'] for index in selected)
    return {'indices': selected, 'seed': seed, 'population_size': len(data),
            'population_distribution': dict(sorted(population.items())),
            'sample_distribution': dict(sorted(sample_counts.items())),
            'all_strata_covered': set(sample_counts) == set(population),
            'features': features,
            'sampling_uses': ['question text', 'benchmark hop count'],
            'sampling_uses_gold_or_historical_results': False,
            'strata_with_fewer_than_two_sampled_queries': [key for key, value in sorted(sample_counts.items()) if value < 2],
            'warning': 'Exploratory subset from existing evaluation files; smoke is functional only.'}


def bootstrap_differences(rows, sampling, repetitions=2000, seed=1031):
    """Paired stratified bootstrap with shared question draws for all metrics."""
    if repetitions < 100:
        raise ValueError('At least 100 bootstrap repetitions are required')
    grouped = defaultdict(list)
    for row in rows:
        grouped[sampling['features'][row['query_index']]['stratum']].append(row)
    population = sampling['population_distribution']
    total = sampling['population_size']
    complete = set(grouped) == set(population)
    rng = random.Random(seed)
    samples = {metric: [] for metric in ALL_METRICS}
    calibrated = {metric: [] for metric in ALL_METRICS}
    for _ in range(repetitions):
        raw_sum = {metric: 0.0 for metric in ALL_METRICS}
        weighted_sum = dict(raw_sum)
        for key in sorted(grouped):
            group = grouped[key]
            draws = [group[rng.randrange(len(group))] for _ in group]
            for metric in ALL_METRICS:
                subtotal = sum(row['delta'][metric] for row in draws)
                raw_sum[metric] += subtotal
                weighted_sum[metric] += population[key] / total * subtotal / len(group)
        for metric in ALL_METRICS:
            samples[metric].append(raw_sum[metric] / len(rows))
            if complete:
                calibrated[metric].append(weighted_sum[metric])

    def interval(values):
        values = sorted(values)
        return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]

    result = {}
    for metric in ALL_METRICS:
        values = [row['delta'][metric] for row in rows]
        result[metric] = {'delta': sum(values) / len(values),
                          'delta_pp': 100 * sum(values) / len(values),
                          'wins': sum(value > 1e-12 for value in values),
                          'losses': sum(value < -1e-12 for value in values),
                          'ties': sum(abs(value) <= 1e-12 for value in values),
                          'paired_stratified_bootstrap_ci95': interval(samples[metric])}
        if complete:
            weighted = sum(population[key] / total * sum(row['delta'][metric] for row in group) / len(group)
                           for key, group in grouped.items())
            result[metric].update(population_calibrated_delta=weighted,
                                  population_calibrated_delta_pp=100 * weighted,
                                  population_calibrated_ci95=interval(calibrated[metric]))
    return result


def paired_comparison(baseline_result, candidate_result, baseline_report, candidate_report, sampling,
                      repetitions=2000, seed=1031):
    old = {row['query_index']: row for row in baseline_result['results']}
    new = {row['query_index']: row for row in candidate_result['results']}
    old_measurements = {row['query_index']: row for row in baseline_report['per_question']}
    new_measurements = {row['query_index']: row for row in candidate_report['per_question']}
    indices = sampling['indices']
    if set(old) != set(new) or set(old) != set(indices) or set(old_measurements) != set(indices) or set(new_measurements) != set(indices):
        raise ValueError('Paired question indices differ')
    rows, candidate_checks, upstream_checks = [], [], []
    for index in indices:
        a, b = old[index], new[index]
        if a['sample_id'] != b['sample_id'] or a['question'] != b['question']:
            raise ValueError(f'Question {index}: paired identity differs')
        pool_a, pool_b = a['candidate_docs'], b['candidate_docs']
        if len(pool_a) != len(set(pool_a)) or len(pool_b) != len(set(pool_b)) or set(pool_a) != set(pool_b):
            raise ValueError(f'Question {index}: Top200 candidate set differs; scoring isolation is invalid')
        candidate_checks.append({'query_index': index, 'count': len(pool_a),
                                 'sorted_document_set_sha256': signature(sorted(pool_a))})
        trace_a, trace_b = a['retrieval_trace']['evidence'], b['retrieval_trace']['evidence']
        retrieval_a, retrieval_b = a['retrieval_trace'], b['retrieval_trace']
        strict = []
        unavailable = []
        for field in STRICT_UPSTREAM:
            if field not in trace_a or field not in trace_b:
                unavailable.append(field)
            elif trace_a[field] != trace_b[field]:
                strict.append(field)
        for field in STRICT_RETRIEVAL_UPSTREAM:
            if field not in retrieval_a or field not in retrieval_b:
                unavailable.append('retrieval.' + field)
            elif retrieval_a[field] != retrieval_b[field]:
                strict.append('retrieval.' + field)
        before_hash, after_hash = trace_a.get('finalizer_input_hash'), trace_b.get('finalizer_input_hash')
        if not isinstance(before_hash, str) or not before_hash or before_hash != after_hash:
            strict.append('finalizer_input_hash')
        if strict:
            raise ValueError(f'Question {index}: cached upstream outputs differ: {strict}')
        upstream_checks.append({'query_index': index,
                                'equal_present_fields': [field for field in STRICT_UPSTREAM if field not in unavailable],
                                'unavailable_fields': unavailable,
                                'equal_present_retrieval_fields': [field for field in STRICT_RETRIEVAL_UPSTREAM
                                                                  if 'retrieval.' + field not in unavailable],
                                'finalizer_input_hash': before_hash,
                                'shared_upstream_sha256': signature({field: trace_a.get(field) for field in STRICT_UPSTREAM})})
        measured_a, measured_b = old_measurements[index], new_measurements[index]
        values_a = dict(measured_a['metrics'], all_gold_top5=int(measured_a['all_gold_top5']),
                        all_gold_top10=int(measured_a['all_gold_top10']))
        values_b = dict(measured_b['metrics'], all_gold_top5=int(measured_b['all_gold_top5']),
                        all_gold_top10=int(measured_b['all_gold_top10']))
        rows.append({'query_index': index, 'sample_id': a['sample_id'], 'question': a['question'],
                     'stratum': sampling['features'][index]['stratum'],
                     'baseline': values_a, 'dependency_scoring': values_b,
                     'delta': {metric: values_b[metric] - values_a[metric] for metric in ALL_METRICS},
                     'ranking_changed': pool_a != pool_b,
                     'baseline_top10_document_sha256': [signature(doc) for doc in pool_a[:10]],
                     'candidate_top10_document_sha256': [signature(doc) for doc in pool_b[:10]]})
    metrics = bootstrap_differences(rows, sampling, repetitions, seed)
    guard = all(value['delta'] >= -.01 - 1e-12 for key, value in metrics.items() if key in METRICS)
    calibrated_guard = (all(value.get('population_calibrated_delta', value['delta']) >= -.01 - 1e-12
                            for key, value in metrics.items() if key in METRICS))
    return {'validated': True, 'n_samples': len(rows), 'metrics': metrics,
            'raw_recall_regression_limit_0_01_met': guard,
            'calibrated_recall_regression_limit_0_01_met': calibrated_guard,
            'candidate_top200_sets_identical': True, 'candidate_set_checks': candidate_checks,
            'upstream_outputs': upstream_checks, 'per_question': rows,
            'bootstrap': {'paired': True, 'within_sampling_strata': True, 'repetitions': repetitions, 'seed': seed},
            'timing_comparison_valid': False,
            'cache_policy': 'Candidate inherits baseline committed LLM cache; ranking-dependent finalizer may call LLM anew.',
            'warning': 'Population calibration is descriptive. Bootstrap intervals on small strata are approximate; this is exploratory tuning, not independent test evidence.'}


def dependencies():
    # Import model-connected runners only when actually executing, not for
    # --help, sampling or the CPU statistical tests.
    global previous, round2, veto, nv_environment, locate
    from . import exp4_improvements as previous
    from . import exp4_round2 as round2
    from . import multi_dataset_retrieval_support_semantic_veto as veto
    from .multi_dataset_retrieval import environment as nv_environment
    from .exp4_http_audit import locate


def frozen_hashes(source, model_dir):
    names = list(previous.experiments.ASSETS) + ['index_manifest.json', 'chunk_metadata.json', 'openie_state.json']
    result = {str(Path(model_dir) / name): previous.experiments.sha256(source / model_dir / name)
              for name in names if (source / model_dir / name).is_file()}
    result.update({str(path.relative_to(source)): previous.experiments.sha256(path)
                   for path in sorted(source.glob('openie_results*.json'))})
    return result


def prepare_context(index_root, work, dataset):
    manifest = read_json(index_root / 'metadata' / dataset / 'manifest.json')
    source = index_root / 'shared_indexes' / dataset
    if Path(manifest['source_index']).resolve() != source.resolve():
        raise ValueError(f'{dataset}: source index declaration differs')
    if manifest['runtime']['embedding_model_name'] != '/root/models/NV-Embed-v2':
        raise ValueError(f'{dataset}: this control requires the NV-Embed-v2 index')
    required = {'embedding_batch_size': 4, 'llm_prefetch_workers': 8, 'openie_max_workers': 8,
                'max_new_tokens': 2048, 'enable_thinking': False}
    if any(manifest['runtime'].get(key) != value for key, value in required.items()):
        raise ValueError(f'{dataset}: frozen runtime differs from the requested control')
    if (previous.experiments.sha256(manifest['data_path']) != manifest['data_sha256']
            or previous.experiments.sha256(manifest['corpus_path']) != manifest['corpus_sha256']
            or previous.experiments.asset_hashes(source, manifest['model_dir']) != manifest['source_asset_sha256']):
        raise ValueError(f'{dataset}: source data/corpus/index SHA changed')
    data, corpus, hops = previous.experiments.validated_dataset(manifest['data_path'], manifest['corpus_path'], dataset)
    warm = index_root / 'cases' / dataset / 'exp4_dependency_binding' / 'index' / 'llm_cache'
    if not warm.is_dir():
        warm = source / 'llm_cache'
    if not warm.is_dir():
        raise ValueError(f'{dataset}: no existing cache to snapshot')
    cache = work / 'cache_snapshots' / dataset / 'baseline'
    if not cache.is_dir():
        cache.parent.mkdir(parents=True, exist_ok=True)
        previous.snapshot_cache(warm, cache)
    return {'dataset': dataset, 'manifest': manifest, 'source': source, 'clone_source': source,
            'data': data, 'corpus': corpus, 'hops': hops, 'cache': cache,
            'cache_hashes': previous.experiments.cache_hashes(cache),
            'frozen_sha256': frozen_hashes(source, manifest['model_dir']), 'cache_origin': str(warm)}


def run_case(args, context, case, indices, scoring_mode, env):
    manifest = previous.subset_manifest(context['manifest'], context['data'], context['hops'], indices)
    variant = round2.Variant(case.name, FLAGS, plan_validation='canonical_refs', plan_routing='question_structure')
    configuration = {'scoring_mode': scoring_mode, 'flags': list(FLAGS), 'sample_seed': args.seed}
    if (case / 'validated.ok').is_file():
        report = previous.verify_trial_report(case)
        if read_json(case / 'manifest.json') != manifest or read_json(case / 'scoring_config.json') != configuration:
            raise ValueError(f'{case}: saved subset/configuration differs')
        if read_json(case / 'before.json')['initial_cache_sha256'] != context['cache_hashes']:
            raise ValueError(f'{case}: starting upstream cache differs')
        if previous.experiments.asset_hashes(case / 'index', manifest['model_dir']) != manifest['source_asset_sha256']:
            raise ValueError(f'{case}: retained graph/vector assets changed')
        return report
    previous.initialize_case(context, case)
    write_json(case / 'selected_indices.json', list(indices))
    write_json(case / 'manifest.json', manifest)
    write_json(case / 'scoring_config.json', configuration)
    command = round2.command(args, context, case, indices, variant)
    command[command.index('--sample_seed') + 1] = str(args.seed)
    command += ['--evidence_scoring_mode', scoring_mode]
    start = Path(args.vllm_log).stat().st_size
    elapsed = previous.execute(command, case / 'run.log', env)
    end = Path(args.vllm_log).stat().st_size
    try:
        report, result = previous.validate_case(args, context, case, manifest, FLAGS, elapsed, start, end)
        runtime = result['runtime_config']
        if runtime.get('evidence_scoring_mode') != scoring_mode:
            raise ValueError(f'{case}: scoring mode did not reach the runtime configuration')
        if runtime.get('evidence_plan_validation') != 'canonical_refs' or runtime.get('evidence_plan_routing') != 'question_structure':
            raise ValueError(f'{case}: original plan settings changed')
        finalizer = veto.validate_finalizer(case, result)
        write_json(case / 'support_semantic_veto_validation.json', finalizer)
        report.update(scoring_mode=scoring_mode, improvements=list(FLAGS),
                      cache_policy='Baseline warm snapshot; candidate inherits this baseline completed cache. Not comparable timings.',
                      support_semantic_veto_validation=finalizer, timing_comparison_valid=False)
        write_json(case / 'report.json', report)
        marker = read_json(case / 'validated.ok')
        marker.update(report_sha256=previous.experiments.sha256(case / 'report.json'),
                      scoring_config_sha256=previous.experiments.sha256(case / 'scoring_config.json'))
        write_json(case / 'validated.ok', marker)
        veto.retain_http_audit(case, report)
        if frozen_hashes(context['source'], manifest['model_dir']) != context['frozen_sha256']:
            raise ValueError(f'{case}: shared OpenIE/index metadata changed')
        return report
    except Exception:
        (case / 'validated.ok').unlink(missing_ok=True)
        raise


def publish_summary(work, reports, comparisons, statuses):
    lines = ['| Dataset | Case | R@1 | R@2 | R@5 | R@10 | R@20 | R@200 | All gold@5 | All gold@10 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        for name, _ in CASES:
            report = reports.get(dataset, {}).get(name)
            if report:
                values = [f'{report["retrieval_metrics"][metric]:.4f}' for metric in METRICS]
                values += [f'{report[metric]:.4f}' for metric in ('all_gold_top5', 'all_gold_top10')]
                lines.append(f'| {dataset} | {name} | ' + ' | '.join(values) + ' |')
        if dataset in comparisons:
            values = [f'{comparisons[dataset]["metrics"][metric]["delta_pp"]:+.2f} pp' for metric in ALL_METRICS]
            lines.append(f'| {dataset} | dependency − baseline | ' + ' | '.join(values) + ' |')
    (work / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    compact = {dataset: {name: previous.compact(report) for name, report in cases.items()}
               for dataset, cases in reports.items()}
    assessment = None
    if set(comparisons) == set(DATASETS):
        guard = all(row['raw_recall_regression_limit_0_01_met']
                    and row['calibrated_recall_regression_limit_0_01_met'] for row in comparisons.values())
        mean5 = sum(row['metrics']['Recall@5']['delta'] for row in comparisons.values()) / len(DATASETS)
        assessment = {'all_six_recalls_regress_no_more_than_0_01_per_dataset': guard,
                      'mean_recall5_delta': mean5, 'screening_promising': guard and mean5 > 1e-12,
                      'full_run_automatically_launched': False,
                      'warning': 'Subset screening only. A promising mean does not prove full-dataset improvement.'}
    write_json(work / 'summary.json', {'datasets': compact, 'stages': statuses,
                                     'comparisons': {name: {key: value for key, value in row.items()
                                                           if key not in ('per_question', 'candidate_set_checks', 'upstream_outputs')}
                                                     for name, row in comparisons.items()},
                                     'assessment': assessment})
    print('\n'.join(lines), flush=True)


def run(args):
    dependencies()
    outputs = ROOT / 'outputs'
    source = Path(args.index_root).resolve()
    work = Path(args.out_root).resolve()
    if args.smoke and work == DEFAULT_OUT:
        work = work.with_name(work.name + '_smoke')
    if work == outputs or not work.is_relative_to(outputs) or source == work or source.is_relative_to(work) or work.is_relative_to(source):
        raise ValueError('Output must be a separate dedicated directory under outputs, outside the frozen index experiment')
    if work.is_symlink() or shutil.disk_usage(outputs).free < 3 * 1024 ** 3:
        raise ValueError('Output cannot be a symlink and at least 3 GiB free disk is required')
    (work / 'metadata').mkdir(parents=True, exist_ok=True)
    args.vllm_log = args.vllm_log or locate(args.llm_base_url)
    if not Path(args.vllm_log).is_file():
        raise ValueError('A readable vLLM HTTP log is required')
    service_file = source / 'metadata/nv2_memory_tuning/active_service.json'
    if service_file.is_file():
        service = read_json(service_file)
        if abs(float(service.get('gpu_memory_utilization', -1)) - .52) > 1e-8:
            raise ValueError('Recorded vLLM service must use GPU memory utilization 0.52 for this experiment')
        if service.get('max_model_len') != 8192:
            raise ValueError('Recorded vLLM context length must remain 8192')
        write_json(work / 'metadata/vllm_service_snapshot.json',
                   {'source': str(service_file), 'source_sha256': previous.experiments.sha256(service_file),
                    'service': service, 'http_log': args.vllm_log,
                    'verification_scope': 'Existing service launch metadata; this runner never starts/reconfigures the service.'})
    contexts = {dataset: prepare_context(source, work, dataset) for dataset in DATASETS}
    samplings = {dataset: choose_subset(context['data'], context['hops'], args.sample_size, args.seed, smoke=args.smoke)
                 for dataset, context in contexts.items()}
    protocol = {'datasets': list(DATASETS), 'cases': [list(case) for case in CASES], 'smoke': args.smoke,
                'index_root': str(source), 'build_indexes': False, 'qa': False,
                'embedding_model': '/root/models/NV-Embed-v2', 'embedding_provider': 'nvembed',
                'embedding_batch_size': 4, 'nv_embedding_oom_split': False,
                'llm_prefetch_workers': 8, 'openie_max_workers': 8, 'max_new_tokens': 2048,
                'thinking': False, 'hop_source': 'benchmark', 'flags': list(FLAGS),
                'vllm_gpu_memory_utilization': .52, 'vllm_max_model_len': 8192,
                'sample_seed': args.seed, 'sample_size_per_dataset': 2 if args.smoke else args.sample_size,
                'selected_indices': {dataset: sampling['indices'] for dataset, sampling in samplings.items()},
                'cache_policy': 'Baseline completed cache copied into candidate before execution; never copied rankings/results.',
                'timing_comparison_valid': False,
                'source_sha256': {dataset: context['frozen_sha256'] for dataset, context in contexts.items()}}
    path = work / 'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Saved protocol differs; inspect before restarting')
    write_json(path, protocol)
    for dataset, sampling in samplings.items():
        write_json(work / 'metadata' / f'{dataset}_sampling.json', sampling)
    nv_args = copy(args)
    nv_args.embedding_model, nv_args.embedding_provider = '/root/models/NV-Embed-v2', 'nvembed'
    env = nv_environment(nv_args)
    env['PATHCONDRAG_NVEMBED_OOM_SPLIT'] = 'false'
    before_code = previous.code_hashes(Path(args.hippo_root))
    reports, comparisons, statuses = {}, {}, {}
    for dataset, context in contexts.items():
        sampling = samplings[dataset]
        indices = sampling['indices']
        reports[dataset] = {}
        try:
            for name, mode in CASES:
                case = work / 'cases' / dataset / name
                current = context
                if name != 'baseline':
                    cache = work / 'cache_snapshots' / dataset / 'dependency_scoring'
                    if not cache.exists():
                        previous.snapshot_cache(work / 'cases' / dataset / 'baseline' / 'index/llm_cache', cache)
                    current = dict(context, cache=cache, cache_hashes=previous.experiments.cache_hashes(cache))
                print(f'[start] {dataset}/{name} n={len(indices)} scoring={mode}', flush=True)
                reports[dataset][name] = run_case(args, current, case, indices, mode, env)
                statuses[f'{dataset}/{name}'] = {'state': 'ready', 'n_samples': len(indices)}
                publish_summary(work, reports, comparisons, statuses)
            baseline = work / 'cases' / dataset / 'baseline/result.json'
            candidate = work / 'cases' / dataset / 'dependency_scoring/result.json'
            comparison = paired_comparison(read_json(baseline), read_json(candidate), reports[dataset]['baseline'],
                                           reports[dataset]['dependency_scoring'], sampling, args.bootstrap_repetitions, args.seed)
            comparison.update(baseline_result_sha256=previous.experiments.sha256(baseline),
                              candidate_result_sha256=previous.experiments.sha256(candidate))
            comparisons[dataset] = comparison
            write_json(work / 'metadata' / f'{dataset}_paired_comparison.json', comparison)
            statuses[f'{dataset}/paired_validation'] = {'state': 'ready', 'n_samples': len(indices)}
            print(f'[paired] {dataset} R@5 delta={comparison["metrics"]["Recall@5"]["delta_pp"]:+.2f} pp '
                  f'R@10 delta={comparison["metrics"]["Recall@10"]["delta_pp"]:+.2f} pp', flush=True)
        except Exception as error:
            LOG.exception('[failed] %s', dataset)
            statuses[f'{dataset}/paired_validation'] = {'state': 'failed', 'error': f'{type(error).__name__}: {error}'}
        write_json(work / 'stage_status.json', statuses)
        publish_summary(work, reports, comparisons, statuses)
    if previous.code_hashes(Path(args.hippo_root)) != before_code:
        raise ValueError('Original HippoRAG source changed during the experiment')
    for context in contexts.values():
        if frozen_hashes(context['source'], context['manifest']['model_dir']) != context['frozen_sha256']:
            raise ValueError('Frozen graph, vectors or extraction source changed')
    if set(comparisons) != set(DATASETS):
        return 1
    write_json(work / 'completed.ok', {'datasets': list(DATASETS), 'case_count': 6, 'smoke': args.smoke,
                                     'sample_size_per_dataset': 2 if args.smoke else args.sample_size})
    print(f'[done] all 6 paired cases validated: {work}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-root', default=str(DEFAULT_INDEX))
    parser.add_argument('--out-root', default=str(DEFAULT_OUT))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--runtime-deps', default='/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log')
    parser.add_argument('--sample-size', type=int, default=96)
    parser.add_argument('--seed', type=int, default=1031)
    parser.add_argument('--bootstrap-repetitions', type=int, default=2000)
    parser.add_argument('--smoke', action='store_true', help='Run 2 queries per dataset using the complete frozen corpus')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)
