"""Frozen-candidate paired development tests without embedding, indexing or QA.

Replay only the new pure finalizer on completed DAG candidates. This does not
replay the original upstream retriever or claim an independent benchmark test.
Historical difficulty is used by the offline sampler, never by the selector.
The joint-selector family adds one bounded local LLM request per eligible query.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import pyarrow.parquet as pq

from . import exp4_improvements as previous
from . import difficulty_sampling as difficulty
from .common import http_status_counts, read_json, write_json
from .difficulty_sampling import make_difficulty_split
from .representative_sampling import collect_prior_exclusions, poststratified_metrics

CASE = 'exp4_dependency_binding_plan_prune_dag_package'
META = 'exp4_anchor_companion_selection'
TEMP = '_exp4_anchor_companion_trials'
RECALLS = ('Recall@1', 'Recall@2', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@200')
PROFILES = {'baseline': None, 'anchor_completion': 'anchor',
            'bridge_companion': 'companion', 'anchor_companion': 'combined'}
EXCLUDE_TAGS = ('structure_representative', 'failure_focused', 'bridge_support', 'structural_support')


def trial_config(args):
    if args.trial_family == 'typed_joint_selector':
        return ('exp4_typed_joint_selector_selection', '_exp4_typed_joint_selector_trials',
                {'baseline': None, 'typed_joint_selector': 'typed_joint'})
    if args.trial_family == 'joint_selector':
        return ('exp4_joint_selector_selection', '_exp4_joint_selector_trials',
                {'baseline': None, 'joint_selector': 'joint'})
    if args.trial_family == 'condition_companion':
        return ('exp4_condition_companion_selection', '_exp4_condition_companion_trials',
                {'baseline': None, 'condition_companion': 'condition'})
    return META, TEMP, PROFILES


def _fresh_confirmation_sampling(parent, size=30, seed=2358):
    """One fresh stratified phase; a remaining singleton is always included.

    The two-phase sampler requires two questions in each original structure.
    Here screening is already observed, so one available question per fresh
    cell suffices. Difficulty bins remain contiguous within each structure.
    """
    features = deepcopy(parent['features'])
    prior_questions = set(parent['screen_indices']) | set(parent['confirmation_indices'])
    available = [index for index in parent['available_indices'] if index not in prior_questions]
    full_counts, pool_counts = {}, {}
    labels = difficulty.DIFFICULTIES
    for row in features:
        structure = row['structure_primary']
        full_counts.setdefault(structure, Counter())[labels.index(row['historical_difficulty'])] += 1
    for index in available:
        row = features[index]
        pool_counts.setdefault(row['structure_primary'], Counter())[labels.index(row['historical_difficulty'])] += 1
    mapping, audit = {}, []
    for structure in sorted(full_counts):
        full, pool = full_counts[structure], pool_counts.get(structure, Counter())
        partitions = []
        for keep in product((False, True), repeat=3):
            edges = (0, *(i + 1 for i, yes in enumerate(keep) if yes), 4)
            bins = tuple(tuple(range(a, b)) for a, b in zip(edges, edges[1:])
                         if sum(full[i] for i in range(a, b)))
            if all(sum(pool[i] for i in group) >= 1 for group in bins):
                partitions.append(bins)
        if not partitions:
            raise ValueError(f'No unused confirmation question remains for structure {structure}')
        bins = min(partitions, key=lambda groups: (-len(groups), groups))
        for group in bins:
            names = [labels[i] for i in group]
            primary = structure + '|difficulty=' + '+'.join(names)
            for name in names:
                mapping[structure, name] = primary
            audit.append({'structure_primary': structure, 'primary': primary,
                          'difficulty_categories': names, 'merged': len(group) > 1,
                          'full_population_count': sum(full[i] for i in group),
                          'available_count': sum(pool[i] for i in group),
                          'available_counts_by_difficulty': {labels[i]: pool[i] for i in group}})
    groups = {}
    for row in features:
        row['primary'] = mapping[row['structure_primary'], row['historical_difficulty']]
    for index in available:
        groups.setdefault(features[index]['primary'], []).append(index)
    counts = {cell: len(values) for cell, values in groups.items()}
    quotas = difficulty._quotas(counts, size, counts)
    confirmed = difficulty._draw(groups, quotas, seed)
    if set(confirmed) & prior_questions:
        raise ValueError('Fresh confirmation overlaps previously observed development questions')
    return {'schema_version': 3, 'dataset': parent['dataset'], 'features': features,
            'feature_sha256': difficulty._hash(features),
            'available_size': len(available), 'available_indices': available,
            'excluded_indices': sorted(set(range(len(features))) - set(available)),
            'confirmation_indices': confirmed, 'seed': seed,
            'confirmation': difficulty._coverage(features, confirmed, available, available,
                                                 quotas, 'confirmation'),
            'difficulty_stratum_merge_audit': audit,
            'sampling_protocol_sha256': difficulty._hash(
                {'seed': seed, 'size': size, 'available_indices': available,
                 'selected_indices': confirmed, 'quota': quotas, 'stratum_merge_audit': audit}),
            'historical_difficulty_distribution': {
                'available': dict(Counter(features[i]['historical_difficulty'] for i in available)),
                'selected': dict(Counter(features[i]['historical_difficulty'] for i in confirmed))},
            'sampling_policy': 'Single-phase strict random sampling without replacement within '
                               'structure × historical-difficulty cells; minimum one per available cell. '
                               'Contiguous difficulty merges preserve every original structure; singletons have pi=1.',
            'weight_definition': 'For the remaining fresh pool: pi_h=n_h/N_h; weight=N_h/(N*n_h).',
            'limits': ['Old screening and confirmation questions are excluded.',
                       'Fresh means unused for this selector, not an untouched benchmark test.',
                       'Question structure and merged difficulty coverage do not guarantee every semantic case.']}


def prepare_typed(args, metadata, profiles):
    """Keep observed screening fixed and reserve a new, unused confirmation draw."""
    out = Path(args.out_root).resolve()
    parent_path = out / 'metadata/exp4_joint_selector_selection/selection.json'
    parent = read_json(parent_path)
    if any(parent['protocol'][key] != getattr(args, key)
           for key in ('screen_size', 'confirmation_size', 'seed')):
        raise ValueError('Typed trials must retain the frozen original screening protocol')
    core_hash = previous.experiments.sha256(previous.ROOT / 'src/pathcondrag/evidence_joint_selector.py')
    if parent.get('selector_sha256') != core_hash:
        raise ValueError('Typed selector must delegate to the exact previously evaluated core selector')
    summary = deepcopy(parent)
    summary['phase_results'] = {}
    summary.pop('selector_sha256', None)
    summary['core_selector_sha256'] = core_hash
    summary['protocol'].update(
        profiles=profiles, confirmation_seed=2358,
        scope='Typed gate developed on observed old screening; old confirmation is also observed '
              'development data and is excluded. Fresh confirmation is a new draw from the remaining pool.',
        cache_save_dir=str(out / 'metadata/exp4_joint_selector_selection'),
        regression_metrics='Six Recall plus All gold@5/@10; both raw and remaining-pool weighted deltas',
        confirmation_policy='Single fresh phase, draw seed 2358, no dummy screening draw; '
                            'minimum one per structure/difficulty cell, explicit rare-cell merging',
        screening_policy='Reuse all old 60 screening indices and their original remaining-289-pool weighting')
    summary['sampling_parent'] = {'path': str(parent_path), 'sha256': previous.experiments.sha256(parent_path),
                                 'old_confirmation_scope': 'Observed development; never replayed as fresh confirmation'}
    for dataset, old_record in parent['datasets'].items():
        original_sampling = read_json(old_record['sampling_file'])
        if previous.experiments.sha256(old_record['sampling_file']) != old_record['sampling_sha256']:
            raise ValueError('Original frozen screening sampling changed')
        fresh = _fresh_confirmation_sampling(original_sampling, args.confirmation_size, seed=2358)
        sampling = deepcopy(original_sampling)
        sampling.update(schema_version=3, confirmation_indices=fresh['confirmation_indices'],
                        fresh_confirmation_sampling=fresh,
                        old_confirmation_indices=original_sampling['confirmation_indices'],
                        old_confirmation_scope='Already observed development, excluded from fresh pool',
                        phase_weighting={'screen': 'Original frozen screening strata and 289-question pool',
                                         'confirmation': 'Fresh single-phase strata and remaining 199-question pool'})
        sampling['confirmation'] = fresh['confirmation']
        sampling['confirmation_available_indices'] = fresh['available_indices']
        sampling['confirmation_seed'] = 2358
        sampling['sampling_policy'] = 'Observed original screen plus separately stratified unused confirmation; '
        sampling['sampling_policy'] += 'use fresh_confirmation_sampling features/weights for confirmation only.'
        sampling_path = metadata / f'{dataset}_sampling.json'
        write_json(sampling_path, sampling)
        summary['datasets'][dataset] = dict(old_record, sampling_file=str(sampling_path),
            sampling_sha256=previous.experiments.sha256(sampling_path),
            confirmation_indices=fresh['confirmation_indices'], confirmation_available_size=fresh['available_size'],
            original_sampling_file=old_record['sampling_file'], original_sampling_sha256=old_record['sampling_sha256'])
    write_json(metadata / 'selection.json', summary)
    print('[typed prepared]', {dataset: {'screen_pool': record['available_size'],
                                         'fresh_confirmation_pool': record['confirmation_available_size']}
                              for dataset, record in summary['datasets'].items()}, flush=True)
    return summary


def prepare(args):
    out = Path(args.out_root).resolve()
    meta_name, _, profiles = trial_config(args)
    metadata = out / 'metadata' / meta_name
    metadata.mkdir(parents=True, exist_ok=True)
    summary_path = metadata / 'selection.json'
    if summary_path.exists():
        summary = read_json(summary_path)
        if (summary['protocol']['screen_size'] != args.screen_size or
                summary['protocol']['confirmation_size'] != args.confirmation_size or
                summary['protocol']['seed'] != args.seed):
            raise ValueError('Requested sampling differs from the frozen protocol')
        return summary
    if args.trial_family == 'typed_joint_selector':
        return prepare_typed(args, metadata, profiles)
    if args.trial_family in ('condition_companion', 'joint_selector'):
        parent_path = out / 'metadata' / META / 'selection.json'
        parent = read_json(parent_path)
        if any(parent['protocol'][key] != getattr(args, key)
               for key in ('screen_size', 'confirmation_size', 'seed')):
            raise ValueError('Follow-up trials must reuse the exact frozen parent sampling protocol')
        summary = dict(parent, phase_results={})
        summary.pop('selector_sha256', None)
        summary['protocol'] = dict(parent['protocol'], profiles=profiles,
            scope='Follow-up on the same development screen; original confirmation has not been evaluated')
        if args.trial_family == 'joint_selector':
            summary['protocol'].update(
                evaluation_scope='Frozen DAG candidates plus one LLM joint-selection proposal per eligible query',
                new_http_requests='At most one per eligible query; exact cache reuse across pilot and screening',
                generation={'model': 'qwen3-8b', 'temperature': 0., 'seed': 0,
                            'max_new_tokens': 2048, 'num_gen_choices': 1,
                            'response_format': {'type': 'json_object'}, 'enable_thinking': False},
                workers=8, llm_base_url=args.llm_base_url,
                max_model_len=8192, tokenizer_path='/root/models/Qwen3-8B',
                pilot_policy='First N frozen screening indices; separate pilot exports, no confirmation access')
        summary['sampling_parent'] = {'path': str(parent_path),
                                     'sha256': previous.experiments.sha256(parent_path)}
        write_json(summary_path, summary)
        return summary
    exclusions = collect_prior_exclusions(out, EXCLUDE_TAGS)
    grounded_path = out / 'metadata/exp4_grounded_selection/selection.json'
    grounded = read_json(grounded_path)
    sources = exclusions['sources'] + [{'path': str(grounded_path),
                                        'sha256': previous.experiments.sha256(grounded_path)}]
    summary = {
        'protocol': {'screen_size': args.screen_size, 'confirmation_size': args.confirmation_size,
            'seed': args.seed, 'confirmation_seed': args.seed + 100,
            'profiles': profiles, 'evaluation_scope': 'exact pure-finalizer replay of frozen DAG candidates',
            'candidate_policy': 'Top2 fixed; Top10 set and suffix from rank11 fixed; no new retrieval',
            'regression_limit': .01, 'new_http_requests': 0,
            'warning': 'Exploratory development using previously evaluated datasets, not untouched final tests'},
        'source_exclusions': sources, 'datasets': {}, 'phase_results': {},
    }
    for dataset in previous.DATASETS:
        manifest = read_json(out / 'metadata' / dataset / 'manifest.json')
        data, _, hops = previous.experiments.validated_dataset(
            manifest['data_path'], manifest['corpus_path'], dataset_name=dataset)
        case = out / 'cases' / dataset / CASE
        baseline = previous.verify_trial_report(case)
        excluded = set(exclusions['indices'][dataset])
        for phase in ('smoke', 'screen', 'confirmation'):
            excluded.update(grounded.get('indices', {}).get(phase, {}).get(dataset, []))
        split = make_difficulty_split(
            data, dataset, hops, [len(previous.experiments.gold_docs(row, dataset)) for row in data],
            baseline['per_question'], sorted(excluded), args.screen_size, args.confirmation_size,
            args.seed, args.seed + 100)
        write_json(metadata / f'{dataset}_sampling.json', split)
        summary['datasets'][dataset] = {
            'source_result': str(case / 'result.json'),
            'source_result_sha256': previous.experiments.sha256(case / 'result.json'),
            'source_report_sha256': previous.experiments.sha256(case / 'report.json'),
            'sampling_file': str(metadata / f'{dataset}_sampling.json'),
            'sampling_sha256': previous.experiments.sha256(metadata / f'{dataset}_sampling.json'),
            'available_size': split['available_size'],
            'screen_indices': split['screen_indices'], 'confirmation_indices': split['confirmation_indices'],
        }
    write_json(summary_path, summary)
    print('[prepared]', {key: value['available_size'] for key, value in summary['datasets'].items()}, flush=True)
    return summary


def aggregate(rows, sampling, *, partial=False, include_all_gold=False):
    report = previous.aggregate_measurements(rows)
    if partial:
        report['available_pool_weighted'] = None
        if include_all_gold:
            report['available_pool_weighted_all_gold'] = None
        report['weighting_warning'] = 'Pilot uses the first frozen screening indices without full stratum coverage; '
        report['weighting_warning'] += 'no remaining-pool weighted estimate or generalization claim.'
        return report
    report['available_pool_weighted'] = poststratified_metrics(
        sampling['features'], [row['query_index'] for row in rows], rows,
        target_indices=sampling['available_indices'], metrics=RECALLS)
    report['available_pool_weighted'].update(
        weight_definition='Remaining-pool normalized inverse inclusion weights N_h/(N*n_h)',
        uncertainty_warning='Random stratified development sample estimates the remaining pool only; '
                            'prior exclusions and tuning prevent an untouched full-test claim')
    if include_all_gold:
        all_gold_rows = [dict(row, metrics={key: float(row[key])
                                            for key in ('all_gold_top5', 'all_gold_top10')}) for row in rows]
        report['available_pool_weighted_all_gold'] = poststratified_metrics(
            sampling['features'], [row['query_index'] for row in rows], all_gold_rows,
            target_indices=sampling['available_indices'], metrics=('all_gold_top5', 'all_gold_top10'))
        report['available_pool_weighted_all_gold'].update(
            weight_definition='Remaining-pool normalized inverse inclusion weights N_h/(N*n_h)',
            uncertainty_warning='Development-pool complete-evidence rates; no untouched full-test claim')
    return report


def evaluate(samples, row, candidates):
    gold = previous.experiments.gold_docs(samples[row['query_index']], row['dataset'])
    metrics = {f'Recall@{k}': len(gold & set(candidates[:k])) / len(gold)
               for k in (1, 2, 5, 10, 20, 200)}
    return {'query_index': row['query_index'], 'metrics': metrics,
            'all_gold_top5': gold <= set(candidates[:5]), 'all_gold_top10': gold <= set(candidates[:10])}


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _joint_stats_delta(before, after):
    return {key: (after[key] if key == 'max_in_flight' else after[key] - before[key])
            for key in after}


def _joint_log_offset(path):
    if not path.is_file():
        raise FileNotFoundError(f'Joint-selector HTTP audit requires the running vLLM log: {path}')
    return path.stat().st_size


def _joint_http_audit(path, start, attempts):
    """Require a nontruncated server log and successful observed completion statuses."""
    deadline = time.monotonic() + 2.
    while True:
        end = _joint_log_offset(path)
        if end < start:
            raise ValueError('vLLM log was truncated during joint selection')
        statuses = http_status_counts(path, start, end, clamp=False)
        if any(code != '200' for code in statuses):
            raise RuntimeError(f'Non-200 completion status during joint selection: {statuses}')
        if sum(statuses.values()) >= attempts:
            return end, statuses
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Incomplete vLLM HTTP log audit: {attempts} attempts, statuses={statuses}')
        time.sleep(.1)


def _joint_prefetch(llm, contexts, metadata, dataset, phase, generation, source_hash, selector_hash,
                    core_selector_hash=None):
    """Call only already built, label-free prompts, sharing the isolated SQLite cache."""
    response_dir = metadata / 'responses' / dataset
    response_dir.mkdir(parents=True, exist_ok=True)
    records = {}

    def request(index, messages):
        manifest = {'messages_sha256': _json_sha(messages), 'generation': generation,
                    'llm_base_url': llm.llm_base_url, 'source_result_sha256': source_hash,
                    'selector_sha256': selector_hash}
        if core_selector_hash is not None:
            manifest['core_selector_sha256'] = core_selector_hash
        manifest_hash = _json_sha(manifest)
        path = response_dir / f'{index}.json'
        old_record = read_json(path) if path.is_file() else None
        if old_record and old_record['prompt_manifest_sha256'] != manifest_hash:
            raise ValueError(f'Frozen joint prompt changed: {dataset}/{index}')
        response, info, cached = llm.infer(messages, response_format={'type': 'json_object'})
        record = {'dataset': dataset, 'query_index': index, 'messages': messages,
                  'prompt_manifest': manifest, 'prompt_manifest_sha256': manifest_hash,
                  'response': response, 'metadata': info, 'cache_hit': bool(cached),
                  'prompt_tokens_preflight': contexts[index]['prompt_tokens'],
                  'observations': (old_record or {}).get('observations', []) + [
                      {'phase': phase, 'cache_hit': bool(cached), 'metadata': info,
                       'response_sha256': _json_sha(response)}]}
        write_json(path, record)
        return index, record

    eligible = {index: context['messages'] for index, context in contexts.items()
                if context['messages'] is not None}
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(request, index, messages): index
                   for index, messages in eligible.items()}
        try:
            for future in as_completed(pending):
                index, record = future.result()  # HTTP/client failures propagate; never become abstentions.
                records[index] = record
                print(f'[joint_llm] {dataset}/{phase} {len(records)}/{len(eligible)} '
                      f'index={index} cache={record["cache_hit"]}', flush=True)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return records


def replay_joint(args, summary, phase):
    """Reuse frozen candidates; only the joint semantic proposal needs the local API."""
    from pathcondrag.evidence_joint_selector import apply_joint_selection, build_joint_prompt
    from pathcondrag.llm.openai_gpt import CacheOpenAI
    from pathcondrag.utils.config_utils import BaseConfig
    from transformers import AutoTokenizer

    typed = args.trial_family == 'typed_joint_selector'
    if typed:
        from pathcondrag.evidence_typed_joint_selector import apply_joint_selection, build_joint_prompt

    out = Path(args.out_root).resolve()
    meta_name, temp_name, profiles = trial_config(args)
    metadata = out / 'metadata' / meta_name
    server_log = Path(args.vllm_log or out / 'logs/vllm_joint_selector.log').resolve()
    _joint_log_offset(server_log)
    selector = previous.ROOT / 'src/pathcondrag' / (
        'evidence_typed_joint_selector.py' if typed else 'evidence_joint_selector.py')
    selector_hash = previous.experiments.sha256(selector)
    if summary.get('selector_sha256', selector_hash) != selector_hash:
        raise ValueError('Joint selector changed after prompts were frozen; use a new experiment namespace')
    if summary['protocol']['llm_base_url'] != args.llm_base_url:
        raise ValueError('Joint-selector endpoint differs from the frozen generation protocol')
    summary['selector_sha256'] = selector_hash
    core_hash = None
    if typed:
        core_hash = previous.experiments.sha256(previous.ROOT / 'src/pathcondrag/evidence_joint_selector.py')
        if summary.get('core_selector_sha256', core_hash) != core_hash:
            raise ValueError('Core joint selector changed after typed prompts were frozen')
        summary['core_selector_sha256'] = core_hash
    write_json(metadata / 'selection.json', summary)
    generation = summary['protocol']['generation']
    tokenizer = AutoTokenizer.from_pretrained(summary['protocol']['tokenizer_path'],
                                             trust_remote_code=True, local_files_only=True)
    if urlsplit(args.llm_base_url).hostname in ('127.0.0.1', 'localhost', '::1'):
        os.environ.setdefault('OPENAI_API_KEY', 'sk-local')
    config = BaseConfig(llm_name='qwen3-8b', llm_base_url=args.llm_base_url,
                        temperature=0., seed=0, max_new_tokens=2048, num_gen_choices=1,
                        max_retry_attempts=1,
                        save_dir=summary['protocol'].get('cache_save_dir', str(metadata)))
    llm = CacheOpenAI.from_experiment_config(config)
    # Keep the existing client, semaphore, cache and timeout implementation.
    # A transport/API failure invalidates this trial instead of silently retrying into a successful report.
    llm.max_retries = 1
    try:
        for dataset, record in summary['datasets'].items():
            source = Path(record['source_result'])
            if previous.experiments.sha256(source) != record['source_result_sha256']:
                raise ValueError('Frozen DAG result changed')
            if previous.experiments.sha256(record['sampling_file']) != record['sampling_sha256']:
                raise ValueError('Frozen sampling protocol changed')
            original = read_json(source)
            original_rows = {row['query_index']: row for row in original['results']}
            manifest = read_json(out / 'metadata' / dataset / 'manifest.json')
            sampling = read_json(record['sampling_file'])
            if typed and phase == 'confirmation':
                sampling = sampling['fresh_confirmation_sampling']
            parquet = out / 'shared_indexes' / dataset / manifest['model_dir'] / 'chunk_embeddings/vdb_chunk.parquet'
            if previous.experiments.sha256(parquet) != manifest['source_asset_sha256']['chunk_embeddings/vdb_chunk.parquet']:
                raise ValueError('Frozen passage index changed')
            documents = pq.read_table(parquet, columns=['content']).column('content').to_pylist()
            text_to_id = {text: index for index, text in enumerate(documents)}
            if len(text_to_id) != len(documents):
                raise ValueError('Public passage index contains duplicate texts')
            indices = (record['screen_indices'][:args.pilot_size] if phase == 'pilot'
                       else record[f'{phase}_indices'])
            contexts = {}
            for index in indices:
                parent = original_rows[index]
                old = [text_to_id[text] for text in parent['candidate_docs']]
                trace = deepcopy(parent['retrieval_trace']['evidence'])
                state = {'evidence_trace': trace, '_evidence_plan': trace.get('plan', []),
                         '_evidence_winning_bindings': trace.get('bindings', {}),
                         '_evidence_beams': trace.get('branch_scores', [])}
                before = json.dumps(state, sort_keys=True)
                messages = build_joint_prompt(parent['question'], old, state, documents.__getitem__)
                if json.dumps(state, sort_keys=True) != before:
                    raise ValueError('Joint prompt builder mutated upstream state')
                prompt_tokens = (len(tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
                    if messages is not None else 0)
                if prompt_tokens + 2048 > summary['protocol']['max_model_len']:
                    raise ValueError(f'Joint prompt exceeds the unchanged server context budget before HTTP: '
                                     f'{dataset}/{index} {prompt_tokens}+2048 > '
                                     f'{summary["protocol"]["max_model_len"]}')
                contexts[index] = {'old': old, 'state': state, 'messages': messages,
                                   'prompt_tokens': prompt_tokens}
            # No labels are supplied to prompt construction or API calls. They enter evaluation only.
            samples = read_json(manifest['data_path'])
            baseline_rows = [evaluate(samples, dict(original_rows[index], dataset=dataset),
                                      original_rows[index]['candidate_docs']) for index in indices]
            expected_by_index = {row['query_index']: row for row in baseline_rows}
            baseline = aggregate(baseline_rows, sampling, partial=phase == 'pilot', include_all_gold=True)
            for profile, mode in profiles.items():
                started = time.monotonic()
                case = out / temp_name / phase / dataset / profile
                case.mkdir(parents=True, exist_ok=True)
                (case / 'validated.ok').unlink(missing_ok=True)
                before_stats = llm.get_request_stats()
                log_start = _joint_log_offset(server_log)
                responses = (_joint_prefetch(llm, contexts, metadata, dataset, phase, generation,
                                             record['source_result_sha256'], selector_hash, core_hash)
                             if mode in ('joint', 'typed_joint') else {})
                stats = _joint_stats_delta(before_stats, llm.get_request_stats())
                if stats['failures'] or stats['retries']:
                    raise RuntimeError(f'Joint-selector request failure/retry invalidates the trial: {stats}')
                log_end, statuses = _joint_http_audit(server_log, log_start, stats['http_attempts'])
                output, measurements, changes, abstentions = [], [], [], Counter()
                for index in indices:
                    parent, context = original_rows[index], contexts[index]
                    old, state = context['old'], context['state']
                    before = json.dumps(state, sort_keys=True)
                    if mode is None:
                        new, diagnostic = old[:], {'enabled': False, 'promotions': []}
                    else:
                        response = responses.get(index)
                        new, diagnostic = apply_joint_selection(
                            parent['question'], old, state, documents.__getitem__,
                            response['response'] if response else '{"swap": null}',
                            response['metadata'].get('finish_reason') if response else 'stop')
                        diagnostic.update(extra_requests=0 if response is None or response['cache_hit'] else 1,
                                          cache_hit=response['cache_hit'] if response else None,
                                          prompt_manifest_sha256=response['prompt_manifest_sha256'] if response else None)
                        abstentions[diagnostic.get('abstain_reason') or 'accepted'] += 1
                    if (new[:2] != old[:2] or set(new[:10]) != set(old[:10]) or new[10:] != old[10:]
                            or len(new) != len(set(new)) or json.dumps(state, sort_keys=True) != before):
                        raise ValueError(f'Joint-selector invariant failed: {dataset}/{profile}/{index}')
                    candidates = [documents[doc_id] for doc_id in new]
                    measured = evaluate(samples, dict(parent, dataset=dataset), candidates)
                    expected = expected_by_index[index]
                    if any(measured['metrics'][key] != expected['metrics'][key]
                           for key in RECALLS if key != 'Recall@5'):
                        raise ValueError('Protected Recall changed despite fixed Top2/Top10')
                    row = deepcopy(parent)
                    row.update(docs=candidates[:10], candidate_docs=candidates,
                               retrieval_metrics=measured['metrics'], all_gold_in_top5=measured['all_gold_top5'],
                               all_gold_in_top10=measured['all_gold_top10'],
                               gold_document_ranks=[{'doc': text, 'rank': candidates.index(text) + 1
                                                    if text in candidates else None} for text in parent['gold_docs']])
                    if new != old:
                        row['candidate_doc_scores'] = [(len(new) - rank) / len(new) for rank in range(len(new))]
                        row['doc_scores'] = row['candidate_doc_scores'][:10]
                        previous_prefix = {item['doc_id']: item for item in state['evidence_trace'].get('selected_prefix', [])}
                        row['retrieval_trace']['evidence']['selected_prefix'] = [
                            dict(previous_prefix.get(doc_id, {'doc_id': doc_id, 'selection_source': profile}),
                                 original_greedy_rank=old.index(doc_id) + 1) for doc_id in new[:5]]
                        changes.append({'query_index': index, 'question': parent['question'],
                                        'delta_r5': measured['metrics']['Recall@5'] - expected['metrics']['Recall@5'],
                                        'diagnostic': diagnostic})
                    row['retrieval_trace']['evidence'][
                        'improvement_typed_joint_selector' if typed else 'improvement_joint_selector'] = diagnostic
                    output.append(row)
                    measurements.append(measured)
                report = aggregate(measurements, sampling, partial=phase == 'pilot', include_all_gold=True)
                raw_delta = {key: report['retrieval_metrics'][key] - baseline['retrieval_metrics'][key] for key in RECALLS}
                raw_delta.update({key: report[key] - baseline[key]
                                  for key in ('all_gold_top5', 'all_gold_top10')})
                weighted_delta = (None if phase == 'pilot' else {
                    key: report['available_pool_weighted']['metrics'][key] - baseline['available_pool_weighted']['metrics'][key]
                    for key in RECALLS})
                if weighted_delta is not None:
                    weighted_delta.update({key: report['available_pool_weighted_all_gold']['metrics'][key]
                                          - baseline['available_pool_weighted_all_gold']['metrics'][key]
                                          for key in ('all_gold_top5', 'all_gold_top10')})
                report.update(profile=profile, dataset=dataset, evaluation_scope='frozen_dag_joint_selection',
                              selector_seconds=time.monotonic() - started, new_llm_requests=stats['http_attempts'],
                              llm_request_stats=stats, http_status_in_log=statuses,
                              http_log_segment={'path': str(server_log), 'start': log_start, 'end': log_end},
                              eligible_questions=sum(c['messages'] is not None for c in contexts.values()),
                              abstention_reasons=dict(abstentions), changed_questions=len(changes),
                              wins=sum(row['delta_r5'] > 0 for row in changes),
                              losses=sum(row['delta_r5'] < 0 for row in changes), changes=changes,
                              raw_delta=raw_delta, weighted_delta=weighted_delta,
                              generation=generation, selector_sha256=selector_hash,
                              core_selector_sha256=core_hash,
                              sampling_scope=('Previously observed development screening questions' if typed and phase != 'confirmation'
                                              else 'Fresh unused development questions from the remaining pool' if typed
                                              else 'Frozen exploratory development questions'))
                report['regression_limit_met'] = all(value >= -.01 - 1e-12
                    for values in (raw_delta, weighted_delta or {}) for value in values.values())
                report['regression_check_scope'] = ('Pilot raw sample only; no weighting or generalization'
                                                     if phase == 'pilot' else 'Raw and weighted paired deltas for six Recall and All gold@5/@10')
                safe_fields = ('dataset', 'eval_mode', 'indexed_docs', 'n_docs', 'result_top_k',
                               'candidate_output_top_k', 'hop_source', 'benchmark_hop_policy')
                result = {key: original[key] for key in safe_fields if key in original}
                result.update(results=output, selected_indices=indices, sample_size_effective=len(indices),
                              sample_size_requested=len(indices), retrieval_metrics=report['retrieval_metrics'],
                              evaluation_scope='frozen_dag_joint_selection', source_result=str(source),
                              source_result_sha256=record['source_result_sha256'], retrieval_seconds=None,
                              selector_seconds=report['selector_seconds'], upstream_runtime_config=original.get('runtime_config'),
                              postprocessing_config={'profile': profile, 'mode': mode,
                                                     'selector_sha256': selector_hash, 'core_selector_sha256': core_hash,
                                                     'generation': generation, 'workers': 8},
                              llm_request_stats=stats,
                              hop_distribution=dict(Counter(str(row['benchmark_hops']) for row in output)))
                write_json(case / 'result.json', result)
                write_json(case / 'report.json', report)
                write_json(case / 'validated.ok', {'result_sha256': previous.experiments.sha256(case / 'result.json'),
                                                 'report_sha256': previous.experiments.sha256(case / 'report.json')})
                summary['phase_results'].setdefault(phase, {}).setdefault(profile, {})[dataset] = {
                    key: value for key, value in report.items() if key not in ('per_question', 'changes')}
                write_json(metadata / 'selection.json', summary)
                print(f'[{phase}] {dataset}/{profile} n={len(indices)} '
                      f'R5={report["retrieval_metrics"]["Recall@5"]:.5f} '
                      f'delta={raw_delta["Recall@5"]:+.5f} wins/losses={report["wins"]}/{report["losses"]} '
                      f'http={stats["http_attempts"]} cache={stats["cache_hits"]} '
                      f'guard={report["regression_limit_met"]}', flush=True)
    finally:
        llm.openai_client.close()


def replay(args, summary, phase):
    if args.trial_family in ('joint_selector', 'typed_joint_selector'):
        return replay_joint(args, summary, phase)
    from pathcondrag.evidence_anchor_companion import rerank_anchor_companions

    out = Path(args.out_root).resolve()
    meta_name, temp_name, profiles = trial_config(args)
    condition_selector = None
    if args.trial_family == 'condition_companion':
        from pathcondrag.evidence_condition_companion import rerank_condition_companions
        condition_selector = rerank_condition_companions
    selector_name = ('evidence_condition_companion.py' if condition_selector
                     else 'evidence_anchor_companion.py')
    selector = previous.ROOT / 'src/pathcondrag' / selector_name
    selector_hash = previous.experiments.sha256(selector)
    if summary.get('selector_sha256', selector_hash) != selector_hash:
        raise ValueError('Selector changed after trials were frozen; use a new experiment namespace')
    summary['selector_sha256'] = selector_hash
    write_json(out / 'metadata' / meta_name / 'selection.json', summary)
    for dataset, record in summary['datasets'].items():
        source = Path(record['source_result'])
        if previous.experiments.sha256(source) != record['source_result_sha256']:
            raise ValueError('Frozen DAG result changed')
        if previous.experiments.sha256(record['sampling_file']) != record['sampling_sha256']:
            raise ValueError('Frozen sampling protocol changed')
        original = read_json(source)
        original_rows = {row['query_index']: row for row in original['results']}
        manifest = read_json(out / 'metadata' / dataset / 'manifest.json')
        samples = read_json(manifest['data_path'])
        sampling = read_json(record['sampling_file'])
        parquet = out / 'shared_indexes' / dataset / manifest['model_dir'] / 'chunk_embeddings/vdb_chunk.parquet'
        if previous.experiments.sha256(parquet) != manifest['source_asset_sha256']['chunk_embeddings/vdb_chunk.parquet']:
            raise ValueError('Frozen passage index changed')
        documents = pq.read_table(parquet, columns=['content']).column('content').to_pylist()
        text_to_id = {text: index for index, text in enumerate(documents)}
        if len(text_to_id) != len(documents):
            raise ValueError('Public passage index contains duplicate texts')
        indices = record[f'{phase}_indices']
        baseline_rows = []
        for index in indices:
            baseline_rows.append(evaluate(samples, dict(original_rows[index], dataset=dataset),
                                          original_rows[index]['candidate_docs']))
        baseline = aggregate(baseline_rows, sampling)
        for profile, mode in profiles.items():
            started = time.monotonic()
            case = out / temp_name / phase / dataset / profile
            case.mkdir(parents=True, exist_ok=True)
            output, measurements, changes = [], [], []
            for index in indices:
                parent = original_rows[index]
                old = [text_to_id[text] for text in parent['candidate_docs']]
                trace = deepcopy(parent['retrieval_trace']['evidence'])
                state = {'evidence_trace': trace, '_evidence_plan': trace.get('plan', []),
                         '_evidence_winning_bindings': trace.get('bindings', {}),
                         '_evidence_beams': trace.get('branch_scores', [])}
                before = json.dumps(state, sort_keys=True)
                if mode is None:
                    new, diagnostic = old[:], {'enabled': False, 'promotions': []}
                elif mode == 'condition':
                    new, diagnostic = condition_selector(parent['question'], old, state, documents.__getitem__)
                else:
                    new, diagnostic = rerank_anchor_companions(
                        parent['question'], old, state, documents.__getitem__, mode=mode)
                if (new[:2] != old[:2] or set(new[:10]) != set(old[:10]) or new[10:] != old[10:]
                        or len(new) != len(set(new)) or json.dumps(state, sort_keys=True) != before):
                    raise ValueError(f'Selector invariant failed: {dataset}/{profile}/{index}')
                candidates = [documents[doc_id] for doc_id in new]
                measured = evaluate(samples, dict(parent, dataset=dataset), candidates)
                expected = baseline_rows[indices.index(index)]
                for key in RECALLS:
                    if key != 'Recall@5' and measured['metrics'][key] != expected['metrics'][key]:
                        raise ValueError('Protected Recall changed despite fixed Top2/Top10')
                row = deepcopy(parent)
                row.update(docs=candidates[:10], candidate_docs=candidates,
                    retrieval_metrics=measured['metrics'], all_gold_in_top5=measured['all_gold_top5'],
                    all_gold_in_top10=measured['all_gold_top10'],
                    gold_document_ranks=[{'doc': text, 'rank': candidates.index(text) + 1
                                         if text in candidates else None} for text in parent['gold_docs']])
                if new != old:
                    row['candidate_doc_scores'] = [(len(new) - rank) / len(new) for rank in range(len(new))]
                    row['doc_scores'] = row['candidate_doc_scores'][:10]
                    previous_prefix = {item['doc_id']: item for item in trace.get('selected_prefix', [])}
                    row['retrieval_trace']['evidence']['selected_prefix'] = [
                        dict(previous_prefix.get(doc_id, {'doc_id': doc_id, 'selection_source': profile}),
                             original_greedy_rank=old.index(doc_id) + 1) for doc_id in new[:5]]
                diagnostic_key = ('improvement_condition_companion' if mode == 'condition'
                                  else 'improvement_anchor_companion')
                row['retrieval_trace']['evidence'][diagnostic_key] = diagnostic
                output.append(row)
                measurements.append(measured)
                delta = measured['metrics']['Recall@5'] - expected['metrics']['Recall@5']
                if new != old:
                    changes.append({'query_index': index, 'question': parent['question'],
                                    'delta_r5': delta, 'diagnostic': diagnostic})
            report = aggregate(measurements, sampling)
            report.update(profile=profile, dataset=dataset, evaluation_scope='frozen_dag_candidate_replay',
                          selector_seconds=time.monotonic() - started, new_llm_requests=0,
                          changed_questions=len(changes), wins=sum(row['delta_r5'] > 0 for row in changes),
                          losses=sum(row['delta_r5'] < 0 for row in changes), changes=changes,
                          raw_delta={key: report['retrieval_metrics'][key] - baseline['retrieval_metrics'][key]
                                     for key in RECALLS},
                          weighted_delta={key: report['available_pool_weighted']['metrics'][key]
                              - baseline['available_pool_weighted']['metrics'][key] for key in RECALLS})
            report['regression_limit_met'] = all(value >= -.01 - 1e-12
                for key in ('raw_delta', 'weighted_delta') for value in report[key].values())
            safe_fields = ('dataset', 'eval_mode', 'indexed_docs', 'n_docs', 'runtime_config',
                           'result_top_k', 'candidate_output_top_k', 'hop_source', 'benchmark_hop_policy')
            result = {key: original[key] for key in safe_fields if key in original}
            result.update(results=output, selected_indices=indices, sample_size_effective=len(indices),
                          sample_size_requested=len(indices), retrieval_metrics=report['retrieval_metrics'],
                          evaluation_scope='frozen_dag_candidate_replay', source_result=str(source),
                          source_result_sha256=record['source_result_sha256'],
                          retrieval_seconds=None, selector_seconds=report['selector_seconds'],
                          upstream_runtime_config=original['runtime_config'],
                          postprocessing_config={'profile': profile, 'mode': mode,
                                                 'selector_sha256': selector_hash},
                          llm_request_stats={'http_attempts': 0, 'failures': 0, 'retries': 0,
                                             'max_in_flight': 8},
                          hop_distribution=dict(Counter(str(row['benchmark_hops']) for row in output)))
            write_json(case / 'result.json', result)
            write_json(case / 'report.json', report)
            write_json(case / 'validated.ok', {'result_sha256': previous.experiments.sha256(case / 'result.json'),
                                             'report_sha256': previous.experiments.sha256(case / 'report.json')})
            summary['phase_results'].setdefault(phase, {}).setdefault(profile, {})[dataset] = {
                key: value for key, value in report.items() if key not in ('per_question', 'changes')}
            write_json(out / 'metadata' / meta_name / 'selection.json', summary)
            print(f"[{phase}] {dataset}/{profile} n={len(indices)} "
                  f"R5={report['retrieval_metrics']['Recall@5']:.5f} "
                  f"delta={report['raw_delta']['Recall@5']:+.5f} "
                  f"wins/losses={report['wins']}/{report['losses']} guard={report['regression_limit_met']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', default=str(previous.DEFAULT_OUT))
    parser.add_argument('--phase', choices=('prepare', 'pilot', 'screen', 'confirmation', 'all'), default='all')
    parser.add_argument('--trial-family', choices=('anchor_companion', 'condition_companion', 'joint_selector',
                                                  'typed_joint_selector'),
                        default='anchor_companion')
    parser.add_argument('--screen-size', type=int, default=60)
    parser.add_argument('--confirmation-size', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2158)
    parser.add_argument('--pilot-size', type=int, default=0,
                        help='Joint-selector pilot: use only the first N frozen screening indices per dataset')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default=None,
                        help='Joint-selector server HTTP log; defaults to OUT_ROOT/logs/vllm_joint_selector.log')
    args = parser.parse_args(argv)
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(previous.ROOT / 'outputs') or out == previous.ROOT / 'outputs':
        raise ValueError('Output must be under PathCondRAG/outputs')
    if not 0 <= args.pilot_size <= args.screen_size:
        raise ValueError('pilot-size must be between zero and the frozen screening size')
    if (args.pilot_size or args.phase == 'pilot') and args.trial_family not in ('joint_selector', 'typed_joint_selector'):
        raise ValueError('Pilot mode is only available for the joint-selector family')
    if args.phase == 'pilot' and not args.pilot_size:
        raise ValueError('Pilot phase requires a positive pilot-size')
    if args.pilot_size and args.phase == 'confirmation':
        raise ValueError('A pilot cannot consume reserved confirmation questions')
    summary = prepare(args)
    if args.pilot_size and args.phase != 'prepare':
        replay(args, summary, 'pilot')
        return 0
    if args.phase in ('screen', 'all'):
        replay(args, summary, 'screen')
    if args.phase in ('confirmation', 'all'):
        replay(args, summary, 'confirmation')
    return 0
