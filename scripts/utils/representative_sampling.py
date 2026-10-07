"""Offline, score-independent sampling and poststratified development reports.

Only question text and benchmark structure labels determine the sample. These
features must never be supplied to the retriever. Secondary balancing and prior
question exclusions mean primary-stratum weights are calibration weights, not
exact inverse inclusion probabilities or a claim of unbiased test performance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re


DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique')
SECONDARY = ('hops', 'gold_count', 'terminal_attribute', 'hop_gold',
             'hop_gold_terminal_attribute', 'topology', 'dag_shape')


def collect_prior_exclusions(output_root, additional_run_tags=()):
    """Exclude explicitly named past development runs, with source hashes.

    Fixed historical sources preserve the preceding round's sampling policy.
    Additional namespaces must be named by the caller: a directory scan would
    accidentally exclude this run's own questions when resuming it.
    """
    root = Path(output_root)
    sources, excluded = [], {dataset: set() for dataset in DATASETS}
    tags = tuple(additional_run_tags)
    if len(tags) != len(set(tags)) or any(not re.fullmatch(r'[A-Za-z0-9_]{1,80}', tag) for tag in tags):
        raise ValueError('Excluded run tags must be unique namespace names')
    directories = ('exp4_improvement_selection', 'exp4_round2_selection',
                   'exp4_round2_selection_relation_plan_fix') + tuple(
                       'exp4_round2_selection_' + tag for tag in tags)
    if len(directories) != len(set(directories)):
        raise ValueError('An excluded run is already a fixed historical source')
    for directory in directories:
        path = root / 'metadata' / directory / 'selection.json'
        record = json.loads(path.read_text())
        phases = record['screen'] if directory == 'exp4_improvement_selection' else record['indices']
        screen_key = 'screen_indices' if directory == 'exp4_improvement_selection' else 'screen'
        confirm_key = 'confirmation_indices' if directory == 'exp4_improvement_selection' else 'confirmation'
        counts = {}
        for dataset in DATASETS:
            indices = set(phases[screen_key][dataset]) | set(phases[confirm_key][dataset])
            excluded[dataset].update(indices)
            counts[dataset] = len(indices)
        sources.append({'path': str(path.resolve()),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'planned_question_counts': counts})
    return {'sources': sources, 'indices': {name: sorted(values) for name, values in excluded.items()},
            'scope': 'All previously planned development questions, whether or not each prior case executed.'}


def terminal_attribute(question):
    """Coarse text-only property categories, not semantic ground-truth labels."""
    text = ' '.join(str(question).lower().split())
    checks = (
        ('date_or_time', r'\bwhen\b|\b(?:what|which) (?:year|date|month|day)\b'),
        ('quantity', r'\bhow (?:many|much|old|long|far|often|tall)\b|\b(?:population|height|age)\b'),
        ('comparison', r'\b(?:older|younger|earlier|later|larger|smaller|taller|more|fewer|longer)\b'),
        ('location', r'\bwhere\b|\b(?:what|which) (?:city|country|state|town|place|continent|county)\b'),
        ('creator', r'\b(?:author|writer|director|composer)\b|\bwho (?:wrote|directed|composed|created)\b'),
        ('affiliation', r'\b(?:team|club|university|college|school|company|party|organization|employer)\b'),
        ('work_or_object', r'\b(?:film|movie|book|novel|song|album|show|series|building|award)\b'),
        ('reason', r'\bwhy\b'),
        ('manner', r'^how\b'),
        ('identity', r'\b(?:who|whom|whose)\b'),
        ('property', r'^\s*(?:what|which)\b'),
    )
    return next((label for label, pattern in checks if re.search(pattern, text)), 'other')


def decomposition_shape(sample):
    """Canonicalize #n references; reference ordering does not create new strata."""
    nodes = sample.get('question_decomposition') or []
    parents = [sorted(set(int(value) for value in re.findall(r'#(\d+)', node.get('question', ''))))
               for node in nodes]
    valid = bool(nodes) and all(1 <= parent < child
                                for child, refs in enumerate(parents, 1) for parent in refs)
    edges = ';'.join(f'{child}<-{",".join(map(str, refs)) or "root"}'
                     for child, refs in enumerate(parents, 1))
    return edges or 'none', valid


def _gold_count(sample, dataset):
    if dataset == 'musique':
        # MuSiQue can contain two distinct supporting paragraphs with the same
        # title. Deduplicating titles would incorrectly collapse its gold set.
        paragraphs = sample.get('paragraphs') or []
        return len({(str(row.get('title', '')), str(row.get('paragraph_text', '')))
                    for row in paragraphs if row.get('is_supporting')})
    return len({str(item[0]) for item in sample.get('supporting_facts', []) if item})


def extract_features(samples, dataset, *, hops=None, gold_counts=None):
    if dataset not in DATASETS:
        raise ValueError('Unsupported dataset')
    if any(values is not None and len(values) != len(samples) for values in (hops, gold_counts)):
        raise ValueError('Offline hop/gold labels must align with every sample')
    result = []
    for index, sample in enumerate(samples):
        shape, valid = decomposition_shape(sample) if dataset == 'musique' else ('not_available', True)
        topology = str(sample.get('id', '')).split('__')[0] if dataset == 'musique' else 'not_available'
        if dataset == 'musique':
            match = re.fullmatch(r'([234])hop(?:[123])?', topology)
            inferred_hops = int(match.group(1)) if match else len(sample.get('question_decomposition') or [])
            primary = f'topology={topology or "unknown"}|dag={shape}'
            decomposition = sample.get('question_decomposition') or []
            tail = decomposition[-1].get('question', sample.get('question', '')) if decomposition else sample.get('question', '')
        else:
            inferred_hops = 2  # Dataset prior, never inferred from gold count.
            primary = f'type={sample.get("type") or "unknown"}'
            tail = sample.get('question', '')
        hop = int(hops[index]) if hops is not None else inferred_hops
        gold = int(gold_counts[index]) if gold_counts is not None else _gold_count(sample, dataset)
        if hop < 1 or gold < 1:
            raise ValueError(f'{dataset}/{index}: missing offline hop or supporting-document labels')
        attribute = terminal_attribute(tail)
        result.append({'query_index': index, 'sample_id': str(sample.get('id') or sample.get('_id') or index),
                       'primary': primary, 'hops': hop, 'gold_count': gold,
                       'terminal_attribute': attribute, 'hop_gold': f'hop={hop}|gold={gold}',
                       'hop_gold_terminal_attribute': f'hop={hop}|gold={gold}|attribute={attribute}',
                       'topology': topology,
                       'dag_shape': shape, 'dag_valid': valid})
    return result


def distributions(features, indices):
    return {key: dict(sorted(Counter(str(features[index][key]) for index in indices).items()))
            for key in ('primary', *SECONDARY)}


def _quotas(counts, size, minimum):
    if size < len(counts) or size > sum(counts.values()):
        raise ValueError('Sample budget cannot cover every available primary stratum')
    effective_minimum = minimum if sum(min(minimum, n) for n in counts.values()) <= size else 1
    result = {name: min(effective_minimum, n) for name, n in counts.items()}
    wanted = {name: size * n / sum(counts.values()) for name, n in counts.items()}
    while sum(result.values()) < size:
        candidates = [name for name in counts if result[name] < counts[name]]
        chosen = max(candidates, key=lambda name: (wanted[name] - result[name], name))
        result[chosen] += 1
    return result, effective_minimum


def _draw(features, available, size, seed, minimum):
    groups = defaultdict(list)
    for index in available:
        groups[features[index]['primary']].append(index)
    counts = {key: len(values) for key, values in groups.items()}
    quotas, effective_minimum = _quotas(counts, size, minimum)
    rng = random.Random(seed)
    tie_breakers = {index: rng.random() for index in sorted(available)}
    targets = {key: {value: size * count / len(available) for value, count in
                     Counter(str(features[index][key]) for index in available).items()}
               for key in SECONDARY}
    seen = {key: Counter() for key in SECONDARY}
    selected, selected_by_class = [], Counter()
    while len(selected) < size:
        primary = max((key for key in groups if selected_by_class[key] < quotas[key]),
                      key=lambda key: ((quotas[key] - selected_by_class[key]) / quotas[key],
                                       quotas[key], key))

        def utility(index):
            score = 0.0
            for key in SECONDARY:
                value = str(features[index][key])
                wanted = targets[key][value]
                score += (wanted - seen[key][value]) / (wanted + 1.0)
                if not seen[key][value]:
                    score += .25
            return score, tie_breakers[index]

        chosen = max(groups[primary], key=utility)
        groups[primary].remove(chosen)
        selected.append(chosen)
        selected_by_class[primary] += 1
        for key in SECONDARY:
            seen[key][str(features[chosen][key])] += 1
    return sorted(selected), quotas, effective_minimum


def poststratification_weights(features, indices, *, target_indices=None):
    """Return normalized primary-stratum calibration weights N_h/(N*n_h)."""
    target = list(range(len(features))) if target_indices is None else list(target_indices)
    subset = list(indices)
    if not target or not subset or len(set(target)) != len(target) or len(set(subset)) != len(subset):
        raise ValueError('Target and subset indices must be nonempty and unique')
    if not set(subset) <= set(target):
        raise ValueError('Subset must belong to the weighting target population')
    counts = Counter(features[index]['primary'] for index in target)
    selected = Counter(features[index]['primary'] for index in subset)
    missing = set(counts) - set(selected)
    if missing:
        raise ValueError(f'Cannot poststratify missing primary strata: {sorted(missing)}')
    return [{'query_index': index, 'primary': features[index]['primary'],
             'weight': counts[features[index]['primary']] / len(target) / selected[features[index]['primary']]}
            for index in subset]


def poststratified_metrics(features, indices, per_question, *, target_indices=None,
                          metrics=('Recall@5', 'Recall@10')):
    weights = poststratification_weights(features, indices, target_indices=target_indices)
    rows = {row['query_index']: row for row in per_question}
    if len(rows) != len(per_question) or set(rows) != set(indices):
        raise ValueError('Metric rows must match the sample indices exactly')
    weight_sum = sum(item['weight'] for item in weights)
    values = {}
    for metric in metrics:
        measured = []
        for item in weights:
            row = rows[item['query_index']]
            value = (row.get('metrics') or row.get('retrieval_metrics') or {}).get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Missing or invalid {metric} in sampled metric rows')
            measured.append((item['weight'], value))
        # Weights sum to one mathematically. Normalizing by their floating-point
        # sum keeps a perfect recall exactly 1 without changing the gain target.
        values[metric] = sum(weight * value for weight, value in measured) / weight_sum
    return {'metrics': values, 'weight_sum': weight_sum,
            'effective_sample_size': 1 / sum(item['weight'] ** 2 for item in weights),
            'weight_definition': 'Primary poststratification N_h/(N*n_h); not exact inverse inclusion probabilities',
            'uncertainty_warning': 'Small, secondary-balanced development sample; no unbiasedness or significance claim.'}


def _coverage(features, selected, available, quotas, minimum):
    full = distributions(features, range(len(features)))
    pool = distributions(features, available)
    actual = distributions(features, selected)
    missing = {key: sorted(set(pool[key]) - set(actual[key])) for key in pool}
    oversampled = []
    for primary, count in actual['primary'].items():
        pool_rate = pool['primary'][primary] / len(available)
        sample_rate = count / len(selected)
        if sample_rate > pool_rate + 1e-10:
            oversampled.append({'primary': primary, 'sample_count': count,
                                'pool_count': pool['primary'][primary],
                                'sample_fraction': sample_rate, 'pool_fraction': pool_rate,
                                'oversampling_factor': sample_rate / pool_rate})
    try:
        population_weights = poststratification_weights(features, selected)
        population_error = None
    except ValueError as error:
        population_weights, population_error = None, str(error)
    return {'n_samples': len(selected), 'selected_indices': selected,
            'primary_quotas': quotas, 'effective_minimum_per_primary': minimum,
            'population_distribution': full, 'available_distribution': pool, 'subset_distribution': actual,
            'missing_available_categories': missing,
            'unavailable_population_primary_strata': sorted(set(full['primary']) - set(pool['primary'])),
            'primary_coverage_complete': not missing['primary'], 'oversampled_primary_strata': oversampled,
            'population_poststratification_weights': population_weights,
            'population_weighting_error': population_error,
            'available_pool_poststratification_weights': poststratification_weights(
                features, selected, target_indices=available)}


def make_representative_split(samples, dataset, *, excluded_indices=(), screen_size=60,
                              confirmation_size=30, seed=342, min_per_primary=2,
                              hops=None, gold_counts=None, confirmation_seed=None):
    """Build a reproducible, label-balanced split without consulting retrieval."""
    features = extract_features(samples, dataset, hops=hops, gold_counts=gold_counts)
    excluded = set(excluded_indices)
    if any(not isinstance(index, int) or index < 0 or index >= len(samples) for index in excluded):
        raise ValueError('Excluded indices are outside the dataset')
    if min_per_primary < 1 or screen_size < 1 or confirmation_size < 1:
        raise ValueError('Sample sizes and stratum minimum must be positive')
    available = [index for index in range(len(samples)) if index not in excluded]
    if screen_size + confirmation_size > len(available):
        raise ValueError('Not enough available questions for disjoint screening and confirmation')
    screened, screen_quotas, screen_min = _draw(features, available, screen_size, seed, min_per_primary)
    confirmation_pool = [index for index in available if index not in set(screened)]
    confirmation_seed = seed + 1 if confirmation_seed is None else confirmation_seed
    confirmed, confirm_quotas, confirm_min = _draw(
        features, confirmation_pool, confirmation_size, confirmation_seed, min_per_primary)
    return {'schema_version': 1, 'dataset': dataset, 'seed': seed,
            'confirmation_seed': confirmation_seed, 'features': features,
            'feature_sha256': hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest(),
            'population_size': len(samples), 'available_size': len(available),
            'excluded_indices': sorted(excluded), 'available_indices': available,
            'confirmation_available_indices': confirmation_pool,
            'screen_indices': screened, 'confirmation_indices': confirmed,
            'population_distribution': distributions(features, range(len(features))),
            'available_distribution': distributions(features, available),
            'screen': _coverage(features, screened, available, screen_quotas, screen_min),
            'confirmation': _coverage(features, confirmed, confirmation_pool, confirm_quotas, confirm_min),
            'sampling_policy': 'Primary proportional quotas with at least 1/2 per available class; '
                               'greedy secondary marginal balance with randomized tie-breaking.',
            'offline_only_features': ['type/topology', '#n dependency DAG', 'benchmark hops',
                                      'gold supporting-document count', 'question terminal attribute'],
            'limits': ['60/30 questions do not cover all semantics or rare crossed feature combinations.',
                       'Question attributes are coarse text heuristics, not gold semantic classes.',
                       'Exclusions and secondary balancing change inclusion probabilities; '
                       'poststratification is a descriptive calibration, not an unbiased test estimator.',
                       'Only indices enter retrieval; offline labels and gold evidence remain in sampling/evaluation.']}
