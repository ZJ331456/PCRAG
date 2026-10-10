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
import os
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


def case_specs(candidate_mode='dependency'):
    if candidate_mode == 'dependency':
        return CASES
    if candidate_mode == 'dependency_joint':
        return (('baseline', 'legacy'), ('dependency_joint', 'dependency_joint'))
    raise ValueError(f'Unknown candidate scoring mode: {candidate_mode}')


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
                      repetitions=2000, seed=1031, candidate_name='dependency_scoring'):
    old = {row['query_index']: row for row in baseline_result['results']}
    new = {row['query_index']: row for row in candidate_result['results']}
    old_measurements = {row['query_index']: row for row in baseline_report['per_question']}
    new_measurements = {row['query_index']: row for row in candidate_report['per_question']}
    indices = sampling['indices']
    if set(old) != set(new) or set(old) != set(indices) or set(old_measurements) != set(indices) or set(new_measurements) != set(indices):
        raise ValueError('Paired question indices differ')
    rows, candidate_checks, upstream_checks, inactive_checks = [], [], [], []
    for index in indices:
        a, b = old[index], new[index]
        if a['sample_id'] != b['sample_id'] or a['question'] != b['question']:
            raise ValueError(f'Question {index}: paired identity differs')
        pool_a, pool_b = a['candidate_docs'], b['candidate_docs']
        if len(pool_a) != len(set(pool_a)) or len(pool_b) != len(set(pool_b)) or set(pool_a) != set(pool_b):
            raise ValueError(f'Question {index}: Top200 candidate set differs; scoring isolation is invalid')
        if candidate_name == 'dependency_joint' and pool_a[:2] != pool_b[:2]:
            raise ValueError(f'Question {index}: dependency joint selector changed the protected Top2')
        candidate_checks.append({'query_index': index, 'count': len(pool_a),
                                 'sorted_document_set_sha256': signature(sorted(pool_a))})
        trace_a, trace_b = a['retrieval_trace']['evidence'], b['retrieval_trace']['evidence']
        retrieval_a, retrieval_b = a['retrieval_trace'], b['retrieval_trace']
        strict = []
        unavailable = []
        inactive = {}
        for field in STRICT_UPSTREAM:
            if field not in trace_a or field not in trace_b:
                unavailable.append(field)
                if field in trace_a or field in trace_b:
                    strict.append(field + '.availability')
                elif field == 'verification_outputs' and all(
                        trace.get('llm_verification_calls') == 0 for trace in (trace_a, trace_b)):
                    inactive[field] = 'Both branches made zero verification calls; no outputs exist.'
            elif trace_a[field] is None or trace_b[field] is None:
                strict.append(field + '.null')
            elif trace_a[field] != trace_b[field]:
                strict.append(field)
        for field in STRICT_RETRIEVAL_UPSTREAM:
            if field not in retrieval_a or field not in retrieval_b:
                unavailable.append('retrieval.' + field)
                if field in retrieval_a or field in retrieval_b:
                    strict.append('retrieval.' + field + '.availability')
                elif field not in ('fact_filter', 'dense_fallback') and all(
                        trace.get('dense_fallback') is True for trace in (retrieval_a, retrieval_b)):
                    inactive['retrieval.' + field] = 'Both used dense fallback; graph/QD/path branch did not run.'
            elif retrieval_a[field] is None or retrieval_b[field] is None:
                strict.append('retrieval.' + field + '.null')
            elif retrieval_a[field] != retrieval_b[field]:
                strict.append('retrieval.' + field)
        before_hash, after_hash = trace_a.get('finalizer_input_hash'), trace_b.get('finalizer_input_hash')
        if not isinstance(before_hash, str) or not before_hash or before_hash != after_hash:
            strict.append('finalizer_input_hash')
        if strict:
            raise ValueError(f'Question {index}: cached upstream outputs differ: {strict}')
        active_missing = [field for field in unavailable if field not in inactive]
        if active_missing:
            raise ValueError(f'Question {index}: required upstream audit fields unavailable: {active_missing}')
        if inactive:
            inactive_checks.append({'query_index': index, 'inactive_fields': inactive})
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
                     'baseline': values_a, candidate_name: values_b,
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
            'upstream_outputs': upstream_checks, 'inactive_upstream_fields': inactive_checks, 'per_question': rows,
            'bootstrap': {'paired': True, 'within_sampling_strata': True, 'repetitions': repetitions, 'seed': seed},
            'timing_comparison_valid': False,
            'cache_policy': 'Candidate inherits baseline committed LLM cache; ranking-dependent finalizer may call LLM anew.',
            'warning': 'Population calibration is descriptive. Bootstrap intervals on small strata are approximate; this is exploratory tuning, not independent test evidence.'}


def validate_joint_selection(result):
    """Prove the joint hook ran and retained its no-request ranking invariants."""
    totals = Counter()
    fallback = Counter()
    per_question = []
    prefix_metrics = defaultdict(list)
    for row in result['results']:
        evidence = row.get('retrieval_trace', {}).get('evidence', {})
        diagnostic = evidence.get('dependency_joint_selection')
        if not isinstance(diagnostic, dict):
            raise ValueError(f'Query {row["query_index"]}: joint selector hook did not export diagnostics')
        required_true = ('enabled', 'top200_set_preserved', 'document_set_preserved',
                         'unique_documents', 'top2_preserved')
        if (diagnostic.get('mode') != 'dependency_joint'
                or any(diagnostic.get(key) is not True for key in required_true)
                or diagnostic.get('gold_labels_used') is not False
                or any(diagnostic.get(key) != 0 for key in
                       ('extra_requests', 'extra_llm_requests', 'extra_embedding_calls'))):
            raise ValueError(f'Query {row["query_index"]}: joint selector runtime invariants failed')
        choices = diagnostic.get('eligible_proofs')
        prefixes = diagnostic.get('prefix_optimization')
        old_choices = evidence.get('improvement_dag_package', {}).get('eligible_proofs', {})
        if not isinstance(choices, dict) or not isinstance(prefixes, list):
            raise ValueError(f'Query {row["query_index"]}: joint proof/coverage audit is missing')
        old_pairs = {(nid, proof['doc_id']) for nid, alternatives in old_choices.items() for proof in alternatives}
        new_pairs = {(nid, proof['doc_id']) for nid, alternatives in choices.items() for proof in alternatives}
        additions = sorted(new_pairs - old_pairs)
        totals['hook_validated_questions'] += 1
        totals['queries_with_eligible_proofs'] += bool(new_pairs)
        totals['eligible_proof_alternatives'] += len(new_pairs)
        totals['queries_with_new_proofs'] += bool(additions)
        totals['new_proof_alternatives'] += len(additions)
        totals['queries_with_optimization_attempts'] += bool(prefixes)
        totals['queries_changed_top5'] += any(p['top_k'] == 5 and p['set_changed'] for p in prefixes)
        totals['queries_changed_top10'] += any(p['top_k'] == 10 and p['set_changed'] for p in prefixes)
        fallback[str(diagnostic.get('fallback'))] += 1
        for prefix in prefixes:
            needed = ('top_k', 'set_changed', 'coverage_before', 'coverage_after',
                      'complete_terminals_before', 'complete_terminals_after', 'node_coverage_before',
                      'node_coverage_after', 'terminal_coverage_before', 'terminal_coverage_after',
                      'added_doc_ids', 'removed_doc_ids')
            if any(key not in prefix for key in needed):
                raise ValueError(f'Query {row["query_index"]}: joint prefix coverage audit is incomplete')
            if not set(prefix['coverage_before']) <= set(prefix['coverage_after']):
                raise ValueError(f'Query {row["query_index"]}: joint selector lost supported ancestor nodes')
            if prefix['set_changed'] and len(prefix['complete_terminals_after']) <= len(prefix['complete_terminals_before']):
                raise ValueError(f'Query {row["query_index"]}: changed prefix lacks strict terminal coverage gain')
            for metric in ('node_coverage_before', 'node_coverage_after',
                           'terminal_coverage_before', 'terminal_coverage_after'):
                prefix_metrics[f'top{prefix["top_k"]}_{metric}'].append(prefix[metric])
        per_question.append({'query_index': row['query_index'], 'eligible_proof_count': len(new_pairs),
                             'new_proofs': [{'node': nid, 'doc_id': doc_id} for nid, doc_id in additions],
                             'fallback': diagnostic.get('fallback'),
                             'node_denominator': diagnostic.get('node_denominator'),
                             'terminal_denominator': diagnostic.get('terminal_denominator'),
                             'prefix_optimization': prefixes})
    return {'validated': True, 'n_samples': len(result['results']), 'counts': dict(totals),
            'fallback_counts': dict(fallback),
            'coverage_means_among_optimization_attempts': {key: sum(values) / len(values)
                                                          for key, values in prefix_metrics.items()},
            'per_question': per_question,
            'warning': 'Proof coverage and passage changes are model-free diagnostics, not gold recall gains.'}


def validate_reuse_protocol(saved, current):
    """Reject reuse across samples, model budgets or changed frozen sources."""
    keys = ('datasets', 'smoke', 'index_root', 'build_indexes', 'qa', 'embedding_model',
            'embedding_provider', 'embedding_batch_size', 'nv_embedding_oom_split',
            'llm_prefetch_workers', 'openie_max_workers', 'max_new_tokens', 'thinking',
            'hop_source', 'flags', 'vllm_gpu_memory_utilization', 'vllm_max_model_len',
            'sample_seed', 'sample_size_per_dataset', 'selected_indices', 'source_sha256')
    changed = [key for key in keys if key not in saved or saved[key] != current[key]]
    if ('baseline', 'legacy') not in [tuple(case) for case in saved.get('cases', [])]:
        changed.append('baseline legacy case')
    if changed:
        raise ValueError(f'Baseline reuse protocol differs: {changed}')


def copy_reused_baseline(source, destination, model_dir):
    """Copy mutable artifacts; only immutable graph/vector files share inodes.

    SQLite backup includes committed WAL rows and omits live lock files. This
    prevents the candidate or a resumed run from writing to the old cache.
    """
    if source.is_symlink() or destination.exists():
        raise ValueError(f'Unsafe or existing baseline copy: {destination}')
    linked = {source / 'index' / model_dir / name for name in previous.experiments.ASSETS}

    def copy_file(original, target):
        if Path(original).is_symlink():
            raise ValueError(f'Symlink in retained baseline: {original}')
        if Path(original) in linked:
            os.link(original, target)
            return target
        return shutil.copy2(original, target)

    shutil.copytree(source, destination, copy_function=copy_file,
                    ignore=shutil.ignore_patterns('llm_cache', '*.lock', '*-wal', '*-shm'))
    previous.snapshot_cache(source / 'index/llm_cache', destination / 'index/llm_cache')


def reuse_baseline(args, context, case, sampling, protocol):
    """Validate old measurements and their upstream fingerprints before reuse."""
    origin = Path(args.baseline_results_dir).resolve()
    source_case = origin / 'cases' / context['dataset'] / 'baseline'
    validate_reuse_protocol(read_json(origin / 'protocol.json'), protocol)
    if not (origin / 'completed.ok').is_file():
        raise ValueError(f'Baseline experiment is not complete: {origin}')
    report = previous.verify_trial_report(source_case)
    manifest = previous.subset_manifest(context['manifest'], context['data'], context['hops'], sampling['indices'])
    if (read_json(source_case / 'manifest.json') != manifest
            or read_json(source_case / 'selected_indices.json') != sampling['indices']):
        raise ValueError(f'{source_case}: saved baseline sample/manifest differs')
    configuration = {'scoring_mode': 'legacy', 'flags': list(FLAGS), 'sample_seed': args.seed}
    if read_json(source_case / 'scoring_config.json') != configuration:
        raise ValueError(f'{source_case}: baseline scoring configuration differs')
    marker = read_json(source_case / 'validated.ok')
    if marker.get('scoring_config_sha256') != previous.experiments.sha256(source_case / 'scoring_config.json'):
        raise ValueError(f'{source_case}: scoring configuration hash changed')
    if frozen_hashes(source_case / 'index', manifest['model_dir']) != context['frozen_sha256']:
        raise ValueError(f'{source_case}: retained baseline extraction/index source SHA changed')
    result = read_json(source_case / 'result.json')
    required = {'evidence_scoring_mode': 'legacy', 'improvement_stage': 4,
                'llm_name': 'qwen3-8b', 'llm_base_url': args.llm_base_url,
                'embedding_model_name': '/root/models/NV-Embed-v2',
                'embedding_batch_size': 4, 'embedding_max_seq_len': 2048,
                'llm_prefetch_workers': 8, 'openie_max_workers': 8, 'max_new_tokens': 2048,
                'evidence_plan_validation': 'canonical_refs', 'evidence_plan_routing': 'question_structure',
                'temperature': 0, 'retrieval_top_k': 200, 'evidence_selection_top_k': 10}
    config = result.get('runtime_config', {})
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError(f'{source_case}: baseline runtime contract differs')
    flags = config.get('evidence_improvements', '')
    if set(flags.split(',') if isinstance(flags, str) else flags) != set(FLAGS):
        raise ValueError(f'{source_case}: baseline runtime improvements differ')
    if (report.get('scoring_mode') != 'legacy' or report.get('llm_request_stats', {}).get('failures') != 0
            or report.get('llm_request_stats', {}).get('max_in_flight') != 8):
        raise ValueError(f'{source_case}: baseline request accounting differs')
    measurements, _ = previous.experiments.validate_result(
        result, manifest, 'exp4_dependency_binding', context['data'], context['corpus'], expected_stage=4)
    if measurements != report['per_question']:
        raise ValueError(f'{source_case}: baseline measurements no longer match the result')
    # A self-pair validates identity, all required upstream fields and scorer
    # fingerprints without sampling bootstrap values or calling either model.
    checked = paired_comparison(result, result, report, report, sampling, 100, args.seed)
    prior = read_json(origin / 'metadata' / f"{context['dataset']}_paired_comparison.json")
    if prior.get('validated') is not True or prior.get('baseline_result_sha256') != marker['result_sha256']:
        raise ValueError(f'{source_case}: original paired baseline audit differs')
    expected_upstream = {row['query_index']: row for row in prior.get('upstream_outputs', [])}
    if any(expected_upstream.get(row['query_index']) != row for row in checked['upstream_outputs']):
        raise ValueError(f'{source_case}: retained upstream hashes no longer match the original audit')
    cache_source = source_case / 'index/llm_cache'
    source_cache_sha = previous.experiments.cache_hashes(cache_source)
    provenance = {'source_experiment': str(origin), 'source_case': str(source_case),
                  'protocol_sha256': previous.experiments.sha256(origin / 'protocol.json'),
                  'result_sha256': marker['result_sha256'], 'report_sha256': marker['report_sha256'],
                  'manifest_sha256': marker['manifest_sha256'], 'source_cache_sha256': source_cache_sha,
                  'upstream_sha256': signature(checked['upstream_outputs']),
                  'frozen_sha256': context['frozen_sha256'], 'executed_again': False,
                  'cache_copy_method': 'independent SQLite backup; graph/vector hardlinks are read-only'}
    if case.exists():
        previous.verify_trial_report(case)
        if read_json(case / 'baseline_reuse.json') != provenance:
            raise ValueError(f'{case}: reused baseline provenance changed')
    else:
        copy_reused_baseline(source_case, case, manifest['model_dir'])
        write_json(case / 'baseline_reuse.json', provenance)
    previous.verify_trial_report(case)
    if previous.experiments.cache_hashes(cache_source) != source_cache_sha:
        raise ValueError(f'{source_case}: original cache changed during backup')
    if frozen_hashes(case / 'index', manifest['model_dir']) != context['frozen_sha256']:
        raise ValueError(f'{case}: cloned reused source assets changed')
    print(f'[reuse] {context["dataset"]}/baseline n={len(sampling["indices"])} source={source_case}', flush=True)
    return report


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
        if scoring_mode == 'dependency_joint':
            joint = validate_joint_selection(read_json(case / 'result.json'))
            if read_json(case / 'dependency_joint_selection_validation.json') != joint:
                raise ValueError(f'{case}: joint selector audit changed')
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
        if scoring_mode == 'dependency_joint':
            joint = validate_joint_selection(result)
            write_json(case / 'dependency_joint_selection_validation.json', joint)
            report['dependency_joint_selection_validation'] = {key: value for key, value in joint.items()
                                                              if key != 'per_question'}
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


def publish_summary(work, reports, comparisons, statuses, cases=CASES):
    lines = ['| Dataset | Case | R@1 | R@2 | R@5 | R@10 | R@20 | R@200 | All gold@5 | All gold@10 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        for name, _ in cases:
            report = reports.get(dataset, {}).get(name)
            if report:
                values = [f'{report["retrieval_metrics"][metric]:.4f}' for metric in METRICS]
                values += [f'{report[metric]:.4f}' for metric in ('all_gold_top5', 'all_gold_top10')]
                lines.append(f'| {dataset} | {name} | ' + ' | '.join(values) + ' |')
        if dataset in comparisons:
            values = [f'{comparisons[dataset]["metrics"][metric]["delta_pp"]:+.2f} pp' for metric in ALL_METRICS]
            label = 'dependency' if cases == CASES else cases[1][0]
            lines.append(f'| {dataset} | {label} − baseline | ' + ' | '.join(values) + ' |')
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
    cases = case_specs(args.candidate_mode)
    candidate_name = cases[1][0]
    outputs = ROOT / 'outputs'
    source = Path(args.index_root).resolve()
    work = Path(args.out_root).resolve()
    if args.smoke and work == DEFAULT_OUT:
        work = work.with_name(work.name + '_smoke')
    if work == outputs or not work.is_relative_to(outputs) or source == work or source.is_relative_to(work) or work.is_relative_to(source):
        raise ValueError('Output must be a separate dedicated directory under outputs, outside the frozen index experiment')
    if work.is_symlink() or shutil.disk_usage(outputs).free < 3 * 1024 ** 3:
        raise ValueError('Output cannot be a symlink and at least 3 GiB free disk is required')
    if args.baseline_results_dir:
        retained = Path(args.baseline_results_dir).resolve()
        if retained == work or retained.is_relative_to(work) or work.is_relative_to(retained):
            raise ValueError('Reused baseline must be outside the new output directory')
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
    protocol = {'datasets': list(DATASETS), 'cases': [list(case) for case in cases], 'smoke': args.smoke,
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
    if args.baseline_results_dir:
        protocol['baseline_results_dir'] = str(Path(args.baseline_results_dir).resolve())
        validate_reuse_protocol(read_json(Path(args.baseline_results_dir) / 'protocol.json'), protocol)
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
            for name, mode in cases:
                case = work / 'cases' / dataset / name
                current = context
                if name != 'baseline':
                    cache = work / 'cache_snapshots' / dataset / candidate_name
                    if not cache.exists():
                        previous.snapshot_cache(work / 'cases' / dataset / 'baseline' / 'index/llm_cache', cache)
                    current = dict(context, cache=cache, cache_hashes=previous.experiments.cache_hashes(cache))
                print(f'[start] {dataset}/{name} n={len(indices)} scoring={mode}', flush=True)
                if name == 'baseline' and args.baseline_results_dir:
                    reports[dataset][name] = reuse_baseline(args, current, case, sampling, protocol)
                else:
                    reports[dataset][name] = run_case(args, current, case, indices, mode, env)
                statuses[f'{dataset}/{name}'] = {'state': 'ready', 'n_samples': len(indices)}
                publish_summary(work, reports, comparisons, statuses, cases)
            baseline = work / 'cases' / dataset / 'baseline/result.json'
            candidate = work / 'cases' / dataset / candidate_name / 'result.json'
            comparison = paired_comparison(read_json(baseline), read_json(candidate), reports[dataset]['baseline'],
                                           reports[dataset][candidate_name], sampling, args.bootstrap_repetitions,
                                           args.seed, candidate_name=candidate_name)
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
        publish_summary(work, reports, comparisons, statuses, cases)
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
    parser.add_argument('--candidate-mode', choices=('dependency', 'dependency_joint'), default='dependency')
    parser.add_argument('--baseline-results-dir',
                        help='Reuse a completed baseline experiment with identical indices, source hashes and runtime')
    parser.add_argument('--bootstrap-repetitions', type=int, default=2000)
    parser.add_argument('--smoke', action='store_true', help='Run 2 queries per dataset using the complete frozen corpus')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)
