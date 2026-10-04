"""Checkpoint every extraction stage and retry only unresolved source rows.

An unresolved row is a diagnostic checkpoint, never an approved empty result.
The caller still owns the strict gate before graph publication. Both pools are
bounded to eight workers and run sequentially; callbacks run on the main thread.
"""

import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import logging

from .misc_utils import NerRawOutput, TripleRawOutput
from .openie_quality import validate_triples


logger = logging.getLogger(__name__)
QUEUE_VERSION = 'pathcondrag_pending_openie_queue_v1'
MAX_PENDING_ROUNDS = 3
MAX_WORKERS = 8


def _stage_failed(metadata):
    return (metadata.get('error') or metadata.get('openie_skipped')
            or metadata.get('quality_status') in ('failed', 'partial', 'pending')
            or metadata.get('complete') is False
            or metadata.get('finish_reason') == 'length')


def _ner_complete(result):
    return (isinstance(result, NerRawOutput) and isinstance(result.metadata, dict)
            and not _stage_failed(result.metadata)
            and isinstance(result.unique_entities, list)
            and all(isinstance(entity, str) and entity.strip()
                    for entity in result.unique_entities)
            and result.metadata.get('finish_reason') == 'stop')


def _triple_complete(extractor, result):
    if not isinstance(result, TripleRawOutput) or not isinstance(result.metadata, dict):
        return False
    metadata = result.metadata
    if _stage_failed(metadata):
        return False
    try:
        if validate_triples(result.triples).invalid_triples:
            return False
    except (TypeError, ValueError):
        return False
    if not result.triples and not metadata.get('source_no_supported_relations'):
        return False
    checker = getattr(extractor, 'is_verified_complete', None)
    if checker is not None:
        return bool(checker(result))
    return (metadata.get('complete') is True and metadata.get('semantic_verified') is True
            and metadata.get('finish_reason') == 'stop'
            and metadata.get('quality_status') in ('success', 'empty_valid'))


def _auditable_prior(result):
    if not isinstance(result, TripleRawOutput) or _stage_failed(result.metadata or {}):
        return False
    try:
        return bool(result.triples) and not validate_triples(result.triples).invalid_triples
    except (TypeError, ValueError):
        return False


def openie_row_is_verified_complete(extractor, row):
    """One completion rule for queue replay and both graph publishers."""
    metadata = row.get('openie_metadata') or {}
    if not isinstance(row.get('extracted_entities'), list) or not isinstance(row.get('extracted_triples'), list):
        return False
    ner = NerRawOutput(row['idx'], '', row['extracted_entities'], metadata.get('ner') or {})
    triples = TripleRawOutput(row['idx'], '', row['extracted_triples'], metadata.get('triples') or {})
    return _ner_complete(ner) and _triple_complete(extractor, triples)


def _exception_output(stage, key, error):
    metadata = {'quality_status': 'failed', 'complete': False,
                'error': f'{type(error).__name__}: {error}',
                'build_queue_exception': True}
    if stage == 'ner':
        return NerRawOutput(key, '', [], metadata)
    metadata.update({'openie_skipped': True, 'semantic_verified': False,
                     'requires_semantic_verification': True})
    return TripleRawOutput(key, '', [], metadata)


def _pending_triples(key, reason='Triple extraction awaits successful NER'):
    return TripleRawOutput(key, '', [], {
        'quality_status': 'pending', 'complete': False, 'openie_skipped': True,
        'openie_skip_reason': reason, 'semantic_verified': False,
        'requires_semantic_verification': True,
    })


def _restore_rows(chunks, initial_rows):
    if initial_rows is None:
        return {}, {}
    rows = initial_rows.values() if isinstance(initial_rows, dict) else initial_rows
    ner_results, triple_results = {}, {}
    seen = set()
    for row in rows:
        key = row['idx']
        if key in seen:
            raise ValueError(f'Duplicate initial OpenIE row: {key}')
        seen.add(key)
        if key not in chunks:
            continue
        if row.get('passage') != chunks[key]['content']:
            raise ValueError(f'Checkpoint passage does not match current source: {key}')
        metadata, responses = row.get('openie_metadata') or {}, row.get('openie_responses') or {}
        ner_results[key] = NerRawOutput(
            key, responses.get('ner', ''), copy.deepcopy(row.get('extracted_entities', [])),
            copy.deepcopy(metadata.get('ner') or {}))
        triple_results[key] = TripleRawOutput(
            key, responses.get('triples', ''), copy.deepcopy(row.get('extracted_triples', [])),
            copy.deepcopy(metadata.get('triples') or {}))
    return ner_results, triple_results


def _checkpoint_row(key, passage, ner, triples):
    ner_metadata = copy.deepcopy(ner.metadata) if ner is not None else {
        'quality_status': 'pending', 'complete': False}
    if triples is None:
        triples = _pending_triples(key)
    return {'idx': key, 'passage': passage,
            'extracted_entities': copy.deepcopy(ner.unique_entities) if ner is not None else [],
            'extracted_triples': copy.deepcopy(triples.triples),
            'openie_metadata': {'ner': ner_metadata, 'triples': copy.deepcopy(triples.metadata)},
            'openie_responses': {'ner': ner.response if ner is not None else '',
                                 'triples': triples.response}}


def _record_history(stage, key, previous, result, round_number, operation):
    """Retain prior failed raw output without recursively embedding the history."""
    history = copy.deepcopy((previous.metadata or {}).get('build_queue_history', [])) if previous else []
    if previous is not None and not history:
        prior_metadata = copy.deepcopy(previous.metadata or {})
        prior_metadata.pop('build_queue_history', None)
        history.append({'round': 0, 'operation': 'checkpoint', 'response': previous.response,
                        'metadata': prior_metadata,
                        'values': copy.deepcopy(previous.unique_entities if stage == 'ner'
                                                else previous.triples)})
    metadata = copy.deepcopy(result.metadata or {})
    metadata.pop('build_queue_history', None)
    history.append({'round': round_number, 'operation': operation,
                    'response': result.response, 'metadata': metadata,
                    'values': copy.deepcopy(result.unique_entities if stage == 'ner'
                                            else result.triples)})
    result.metadata = dict(result.metadata or {})
    result.metadata.update({'build_queue_contract': QUEUE_VERSION,
                            'build_queue_round': round_number,
                            'build_queue_history': history})
    return result


def _bounded_results(keys, workers, operation, stage):
    """Keep only a worker-sized set of futures alive, including long corpora."""
    iterator = iter(keys)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def submit_next():
            try:
                key = next(iterator)
            except StopIteration:
                return False
            pending[pool.submit(operation, key)] = key
            return True

        for _ in range(workers):
            if not submit_next():
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                key = pending.pop(future)
                try:
                    result = future.result()
                    expected = NerRawOutput if stage == 'ner' else TripleRawOutput
                    if not isinstance(result, expected) or result.chunk_id != key:
                        raise ValueError(f'{stage} output does not match requested chunk {key}')
                    if not isinstance(result.metadata, dict):
                        raise ValueError(f'{stage} returned non-object metadata for {key}')
                except Exception as error:
                    result = _exception_output(stage, key, error)
                yield key, result
                submit_next()


def run_openie_queue(extractor, chunks, initial_rows=None, checkpoint=None):
    """Return complete or explicitly pending outputs after three bounded rounds.

    ``audit_existing_triples`` approves older successful candidates under the
    current source-verification contract. ``recover_pending_triples`` changes
    strategy for unresolved candidates in later rounds. An optional
    ``recover_pending_ner`` similarly changes a failed NER request. Persisting a
    diagnostic does not approve it: the publication caller must reject pending
    rows. ``checkpoint`` receives one standard OpenIE row after every result.
    """
    ner_results, triple_results = _restore_rows(chunks, initial_rows)
    ner_workers, triple_workers = extractor.worker_limits()
    ner_workers = min(MAX_WORKERS, int(ner_workers))
    triple_workers = min(MAX_WORKERS, int(triple_workers))
    if ner_workers < 1 or triple_workers < 1:
        raise ValueError('OpenIE queue worker limits must be positive')
    triple_attempts = {key: 0 for key in chunks}
    logger.info('OpenIE pending queue: NER workers=%d, triple workers=%d, rounds=%d',
                ner_workers, triple_workers, MAX_PENDING_ROUNDS)

    def save(key):
        if checkpoint is not None:
            checkpoint(_checkpoint_row(key, chunks[key]['content'],
                                       ner_results.get(key), triple_results.get(key)))

    for round_number in range(1, MAX_PENDING_ROUNDS + 1):
        ner_pending = [key for key in chunks if not _ner_complete(ner_results.get(key))]
        logger.info('OpenIE queue round %d NER stage: pending=%d', round_number, len(ner_pending))

        def extract_ner(key):
            previous = ner_results.get(key)
            recover = getattr(extractor, 'recover_pending_ner', None)
            if previous is not None and round_number > 1 and recover is not None:
                return recover(key, chunks[key]['content'], previous, round_number)
            return extractor.ner(key, chunks[key]['content'])

        for processed, (key, result) in enumerate(
                _bounded_results(ner_pending, ner_workers, extract_ner, 'ner'), 1):
            ner_results[key] = _record_history(
                'ner', key, ner_results.get(key), result, round_number,
                'recover_pending_ner' if round_number > 1 else 'ner')
            if not _ner_complete(result) and key not in triple_results:
                # NER failure blocks publication through its own metadata. Do
                # not discard an independently completed triple stage on resume.
                triple_results[key] = _record_history(
                    'triples', key, None, _pending_triples(key), round_number, 'await_ner')
            save(key)
            if processed % 100 == 0 or processed == len(ner_pending):
                logger.info('OpenIE queue round %d NER checkpointed: %d/%d',
                            round_number, processed, len(ner_pending))

        triple_pending = [key for key in chunks
                          if _ner_complete(ner_results.get(key))
                          and not _triple_complete(extractor, triple_results.get(key))]
        operations = {}

        def extract_triples(key):
            passage = chunks[key]['content']
            entities = ner_results[key].unique_entities
            previous = triple_results.get(key)
            if triple_attempts[key] == 0:
                if _auditable_prior(previous):
                    operations[key] = 'audit_existing_triples'
                    return extractor.audit_existing_triples(key, passage, entities, previous)
                if previous is not None and (previous.metadata or {}).get('fresh_extraction_history'):
                    operations[key] = 'recover_pending_triples'
                    return extractor.recover_pending_triples(key, passage, entities, previous, round_number)
                operations[key] = 'triple_extraction'
                return extractor.triple_extraction(key, passage, entities)
            operations[key] = 'recover_pending_triples'
            return extractor.recover_pending_triples(
                key, passage, entities, previous, round_number)

        logger.info('OpenIE queue round %d: NER pending=%d, triple pending=%d',
                    round_number, sum(not _ner_complete(ner_results.get(key)) for key in chunks),
                    len(triple_pending))
        round_failed = 0
        for processed, (key, result) in enumerate(
                _bounded_results(triple_pending, triple_workers, extract_triples, 'triples'), 1):
            triple_attempts[key] += 1
            previous = triple_results.get(key)
            result = _record_history('triples', key, previous, result, round_number,
                                     operations.get(key, 'worker_exception'))
            if not _triple_complete(extractor, result):
                round_failed += 1
                result.metadata.update({'complete': False, 'quality_status': 'failed',
                                        'openie_skipped': True,
                                        'requires_semantic_verification': True})
                result.metadata.setdefault('openie_skip_reason',
                                           'Current source-verification contract is incomplete')
            triple_results[key] = result
            save(key)
            if processed % 100 == 0 or processed == len(triple_pending):
                logger.info('OpenIE queue round %d triples checkpointed: %d/%d, pending_in_processed=%d',
                            round_number, processed, len(triple_pending), round_failed)

        pending = [key for key in chunks
                   if not _ner_complete(ner_results.get(key))
                   or not _triple_complete(extractor, triple_results.get(key))]
        logger.info('OpenIE queue round %d finished: remaining pending=%d', round_number, len(pending))
        if not pending:
            break

    # All requested keys are represented even when NER never succeeds.
    for key in chunks:
        if key not in triple_results:
            triple_results[key] = _pending_triples(key)
    return ({key: ner_results[key] for key in chunks},
            {key: triple_results[key] for key in chunks})
