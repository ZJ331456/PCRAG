"""Offline random sampling by structure and frozen historical retrieval difficulty.

Historical scores determine strata, not which proposed module wins. Only the
selected indices may be passed to retrieval. Exact sampling weights concern the
remaining development pool; calibration to the full, previously used dataset
does not create an independent test set.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import random

from . import representative_sampling as representative


METRICS = ('Recall@1', 'Recall@2', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@200')
DIFFICULTIES = ('complete5', 'missing5_complete10', 'missing10_complete200', 'missing200')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _measurements(rows, count):
    rows = list(rows.values()) if isinstance(rows, dict) else list(rows)
    if len(rows) != count or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Historical baseline needs exactly one metric row for every question')
    by_index = {}
    for row in rows:
        index = row.get('query_index')
        if type(index) is not int or not 0 <= index < count or index in by_index:
            raise ValueError('Historical baseline question indices are missing, duplicated or out of range')
        metrics = row.get('metrics')
        if not isinstance(metrics, dict):
            raise ValueError('Historical baseline metric rows require metrics')
        values = []
        for metric in METRICS:
            value = metrics.get(metric)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'Historical baseline has missing or invalid {metric}')
            values.append(float(value))
        if any(a > b + 1e-12 for a, b in zip(values, values[1:])):
            raise ValueError('Historical baseline Recall values must be monotone')
        for flag, metric in (('all_gold_top5', 'Recall@5'), ('all_gold_top10', 'Recall@10')):
            if flag in row and (type(row[flag]) is not bool or
                                row[flag] != (metrics[metric] >= 1 - 1e-12)):
                raise ValueError('Historical all-gold flags disagree with completeness Recall')
        by_index[index] = dict(row, metrics=dict(zip(METRICS, values)))
    return by_index


def historical_difficulty(metrics):
    if metrics['Recall@5'] >= 1 - 1e-12:
        return DIFFICULTIES[0]
    if metrics['Recall@10'] >= 1 - 1e-12:
        return DIFFICULTIES[1]
    if metrics['Recall@200'] >= 1 - 1e-12:
        return DIFFICULTIES[2]
    return DIFFICULTIES[3]


def _partitions(full_counts, pool_counts):
    """All contiguous difficulty merges that preserve both phase controls."""
    candidates = []
    for keep in itertools.product((False, True), repeat=3):
        edges = (0, *(i + 1 for i, yes in enumerate(keep) if yes), 4)
        bins = tuple(tuple(range(a, b)) for a, b in zip(edges, edges[1:])
                     if sum(full_counts[i] for i in range(a, b)))
        if all(sum(pool_counts[i] for i in group) >= 2 for group in bins):
            candidates.append(bins)
    return sorted(set(candidates), key=lambda bins: (-len(bins), bins))


def _strata(features, available, maximum_cells):
    full, pool = defaultdict(Counter), defaultdict(Counter)
    for row in features:
        full[row['structure_primary']][DIFFICULTIES.index(row['historical_difficulty'])] += 1
    for index in available:
        row = features[index]
        pool[row['structure_primary']][DIFFICULTIES.index(row['historical_difficulty'])] += 1
    choices, partitions = {}, {}
    for structure in sorted(full):
        choices[structure] = _partitions(full[structure], pool[structure])
        if not choices[structure]:
            raise ValueError(f'Structure {structure} has fewer than two available questions; '
                             'cannot preserve disjoint screening/confirmation coverage')
        partitions[structure] = choices[structure][0]
    if len(partitions) > maximum_cells:
        raise ValueError('Sample budget cannot cover every structural class in both phases')
    # Merge within a structure before drawing any questions. No class silently
    # disappears because screening consumed its sole remaining example.
    while sum(len(bins) for bins in partitions.values()) > maximum_cells:
        options = []
        for structure, bins in partitions.items():
            for candidate in choices[structure]:
                if len(candidate) == len(bins) - 1 and all(
                        any(set(group) <= set(merged) for merged in candidate) for group in bins):
                    cost = sum(sum(full[structure][i] for i in group) * (len(group) - 1)
                               for group in candidate)
                    options.append((cost, structure, candidate))
        if not options:
            raise ValueError('Cannot merge difficulty cells enough to satisfy both phase budgets')
        _, structure, candidate = min(options)
        partitions[structure] = candidate
    mapping, audit = {}, []
    for structure, bins in partitions.items():
        for group in bins:
            labels = [DIFFICULTIES[i] for i in group]
            primary = structure + '|difficulty=' + '+'.join(labels)
            for label in labels:
                mapping[structure, label] = primary
            audit.append({'structure_primary': structure, 'primary': primary,
                          'difficulty_categories': labels,
                          'full_counts_by_difficulty': {DIFFICULTIES[i]: full[structure][i] for i in group},
                          'available_counts_by_difficulty': {DIFFICULTIES[i]: pool[structure][i] for i in group},
                          'full_population_count': sum(full[structure][i] for i in group),
                          'available_count': sum(pool[structure][i] for i in group),
                          'merged': len(group) > 1})
    return mapping, audit


def _quotas(counts, size, capacities):
    if size < len(counts) or size > sum(capacities.values()):
        raise ValueError('Sample budget cannot cover all cells without exhausting the next phase')
    result = {cell: 1 for cell in counts}
    targets = {cell: size * count / sum(counts.values()) for cell, count in counts.items()}
    while sum(result.values()) < size:
        eligible = [cell for cell in counts if result[cell] < capacities[cell]]
        if not eligible:
            raise ValueError('Insufficient stratum capacities')
        cell = max(eligible, key=lambda name: (targets[name] - result[name], name))
        result[cell] += 1
    return result


def _draw(groups, quotas, seed):
    rng = random.Random(seed)
    return sorted(index for cell in sorted(groups)
                  for index in rng.sample(sorted(groups[cell]), quotas[cell]))


def _metric_audit(features, rows, selected, available, remaining=None):
    subset = [rows[index] for index in selected]
    result = {'raw_metrics': {metric: sum(row['metrics'][metric] for row in subset) / len(subset)
                              for metric in METRICS},
              'available_pool_weighted': representative.poststratified_metrics(
                  features, selected, subset, target_indices=available, metrics=METRICS),
              'full_population_calibrated': representative.poststratified_metrics(
                  features, selected, subset, metrics=METRICS)}
    if remaining is not None:
        result['conditional_remaining_pool_weighted'] = representative.poststratified_metrics(
            features, selected, subset, target_indices=remaining, metrics=METRICS)
    result['available_pool_weighted'].update(
        weight_definition='Exact available-pool normalized inverse inclusion weight N_h/(N*n_h).',
        uncertainty_warning='Random stratified development sampling; does not create an independent test.')
    result['full_population_calibrated'].update(
        weight_definition='Historical full-population primary calibration N_h/(N*n_h).',
        uncertainty_warning='Excluded questions have zero inclusion probability; full calibration is descriptive.')
    return result


def _coverage(features, selected, available, initial_pool, quotas, phase):
    result = representative._coverage(features, selected, available, quotas, 1)
    initial_counts = Counter(features[index]['primary'] for index in initial_pool)
    conditional_counts = Counter(features[index]['primary'] for index in available)
    marginal_weights = representative.poststratification_weights(features, selected, target_indices=initial_pool)
    result.update(
        initial_available_pool_poststratification_weights=marginal_weights,
        inclusion_probabilities=[{'query_index': index, 'primary': features[index]['primary'],
            'marginal_probability': quotas[features[index]['primary']] / initial_counts[features[index]['primary']],
            'conditional_probability': quotas[features[index]['primary']] / conditional_counts[features[index]['primary']]}
            for index in selected],
        inclusion_probability_scope='Uniform sampling without replacement within frozen primary cells; '
            'confirmation marginal probabilities refer to the initial pool, conditional probabilities '
            'refer to the realized screening complement.',
        exact_design_weight_definition='For the initial available pool: pi_h=n_h/N_h; '
            'normalized inverse inclusion weight=1/(N*pi_h)=N_h/(N*n_h).',
        phase=phase,
        original_structure_distribution={
            'full': dict(Counter(row['structure_primary'] for row in features)),
            'available': dict(Counter(features[i]['structure_primary'] for i in available)),
            'selected': dict(Counter(features[i]['structure_primary'] for i in selected))},
        missing_original_difficulty_categories=sorted(
            {features[i]['historical_difficulty'] for i in available} -
            {features[i]['historical_difficulty'] for i in selected}))
    return result


def make_difficulty_split(samples, dataset, hops, gold_counts, baseline_measurements, excluded_indices,
                          screen_size=90, confirmation_size=45, seed=1142, confirmation_seed=1242):
    """Freeze a random, difficulty-stratified exploratory development split.

    A cell needs at least two available questions. Adjacent historical difficulty
    categories are merged within the same original structure before either
    draw, with an explicit audit. Structural classes are never silently dropped.
    """
    if (type(screen_size) is not int or type(confirmation_size) is not int or
            screen_size < 1 or confirmation_size < 1 or type(seed) is not int or
            type(confirmation_seed) is not int):
        raise ValueError('Sample sizes must be positive integers and seeds must be integers')
    features = representative.extract_features(samples, dataset, hops=hops, gold_counts=gold_counts)
    rows = _measurements(baseline_measurements, len(samples))
    excluded = set(excluded_indices)
    if any(type(index) is not int or not 0 <= index < len(samples) for index in excluded):
        raise ValueError('Excluded indices are outside the dataset')
    available = [index for index in range(len(samples)) if index not in excluded]
    if screen_size + confirmation_size > len(available):
        raise ValueError('Not enough questions for disjoint screening and confirmation')
    for row in features:
        row['structure_primary'] = row['primary']
        row['historical_difficulty'] = historical_difficulty(rows[row['query_index']]['metrics'])
    mapping, merges = _strata(features, available, min(screen_size, confirmation_size))
    groups = defaultdict(list)
    for row in features:
        row['primary'] = mapping[row['structure_primary'], row['historical_difficulty']]
        if row['query_index'] not in excluded:
            groups[row['primary']].append(row['query_index'])
    counts = {cell: len(values) for cell, values in groups.items()}
    screen_quotas = _quotas(counts, screen_size, {cell: count - 1 for cell, count in counts.items()})
    screened = _draw(groups, screen_quotas, seed)
    selected_set = set(screened)
    confirmation_pool = [index for index in available if index not in selected_set]
    remaining = {cell: [index for index in values if index not in selected_set] for cell, values in groups.items()}
    confirm_quotas = _quotas(counts, confirmation_size, {cell: len(values) for cell, values in remaining.items()})
    confirmed = _draw(remaining, confirm_quotas, confirmation_seed)
    full_indices = list(range(len(features)))
    full_metrics = {metric: sum(rows[index]['metrics'][metric] for index in full_indices) / len(full_indices)
                    for metric in METRICS}
    available_metrics = {metric: sum(rows[index]['metrics'][metric] for index in available) / len(available)
                         for metric in METRICS}
    audit = {phase: _metric_audit(features, rows, selected, available,
                                confirmation_pool if phase == 'confirmation' else None)
             for phase, selected in (('screen', screened), ('confirmation', confirmed))}
    for report in audit.values():
        report['raw_minus_available_pp'] = {metric: 100 * (report['raw_metrics'][metric] - available_metrics[metric])
                                           for metric in METRICS}
        report['weighted_minus_available_pp'] = {metric: 100 * (
            report['available_pool_weighted']['metrics'][metric] - available_metrics[metric]) for metric in METRICS}
        report['raw_minus_full_pp'] = {metric: 100 * (report['raw_metrics'][metric] - full_metrics[metric])
                                      for metric in METRICS}
        report['calibrated_minus_full_pp'] = {metric: 100 * (
            report['full_population_calibrated']['metrics'][metric] - full_metrics[metric]) for metric in METRICS}
    parameters = dict(dataset=dataset, screen_size=screen_size, confirmation_size=confirmation_size,
                      seed=seed, confirmation_seed=confirmation_seed, excluded_indices=sorted(excluded),
                      difficulty_categories=list(DIFFICULTIES), minimum_available_per_cell=2,
                      minimum_per_phase_cell=1, merge_policy='Contiguous within original structure before draws; '
                      'preserve phase coverage, then satisfy the smaller phase cell budget.')
    sources = {'samples_sha256': _hash(samples), 'hops_sha256': _hash(hops),
               'gold_counts_sha256': _hash(gold_counts),
               'historical_baseline_measurements_sha256': _hash([rows[i] for i in full_indices]),
               'parameters_sha256': _hash(parameters)}
    return {'schema_version': 2, 'dataset': dataset, 'seed': seed, 'confirmation_seed': confirmation_seed,
            'features': features, 'feature_sha256': _hash(features),
            'population_size': len(samples), 'available_size': len(available),
            'excluded_indices': sorted(excluded), 'available_indices': available,
            'confirmation_available_indices': confirmation_pool,
            'screen_indices': screened, 'confirmation_indices': confirmed,
            'population_distribution': representative.distributions(features, full_indices),
            'available_distribution': representative.distributions(features, available),
            'screen': _coverage(features, screened, available, available, screen_quotas, 'screen'),
            'confirmation': _coverage(features, confirmed, confirmation_pool, available,
                                      confirm_quotas, 'confirmation'),
            'difficulty_stratum_merge_audit': merges,
            'historical_difficulty_distribution': {
                'full': dict(Counter(row['historical_difficulty'] for row in features)),
                'available': dict(Counter(features[i]['historical_difficulty'] for i in available)),
                'screen': dict(Counter(features[i]['historical_difficulty'] for i in screened)),
                'confirmation': dict(Counter(features[i]['historical_difficulty'] for i in confirmed))},
            'historical_baseline_audit': {'full_population_metrics': full_metrics,
                'available_pool_metrics': available_metrics, **audit},
            'source_hashes': sources, 'sampling_parameters': parameters,
            'sampling_protocol_sha256': _hash(dict(source_hashes=sources, parameters=parameters,
                feature_sha256=_hash(features), merge_audit=merges,
                screen_indices=screened, confirmation_indices=confirmed)),
            'sampling_policy': 'Fixed structure × historical-difficulty quotas; strict random sampling '
                'without replacement within every cell. No secondary greedy balancing.',
            'offline_only_features': ['type/topology', '#n dependency DAG', 'benchmark hops',
                'gold supporting-document count', 'question terminal attribute',
                'frozen historical plan_prune Recall@5/@10/@200 difficulty'],
            'limits': ['Exploratory development on previously evaluated data; no independent test claim.',
                'Difficulty merges are explicit and may reduce resolution for rare failures.',
                'Structure and difficulty coverage does not guarantee every semantic or crossed category.',
                'Exact inclusion probabilities concern the remaining pool; excluded questions have pi=0.',
                'Full-population calibration is descriptive and cannot undo previous tuning.',
                'Difficulty uses historical baseline only; new module outcomes never select questions.',
                'Only selected indices enter retrieval; historical scores and offline labels stay in evaluation.']}
