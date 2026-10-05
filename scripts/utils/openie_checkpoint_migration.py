"""Explicitly migrate source-verified extraction into structural-only progress.

Only the original, unaudited extraction can become a structural result. Facts
selected by semantic auditing, compact recovery, or atomic recovery are never
substituted for that initial extraction. The source journal remains read-only.
"""

import copy
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from pathcondrag.index.openie_checkpoint import OpenIECheckpoint
from pathcondrag.index.openie.openie_quality import validate_triples


MIGRATION_VERSION = 'pathcondrag_source_to_structural_checkpoint_v1'
STRUCTURAL_VERSION = 'pathcondrag_structural_openie_v1'
RECOVERY_BUDGET = 3


def validate_migration_identity(source_identity, target_identity):
    """Require an explicit matching corpus, LLM producer, and NER contract.

    A changed triple prompt permits NER reuse only. This never relabels triples
    produced by one prompt as the extraction of a different prompt.
    """
    if not isinstance(source_identity, dict) or not isinstance(target_identity, dict):
        raise ValueError('Migration requires both explicit checkpoint identities')
    old, new = source_identity.get('provenance'), target_identity.get('provenance')
    if not isinstance(old, dict) or not isinstance(new, dict):
        raise ValueError('Migration identities require extraction provenance')
    corpus = source_identity.get('corpus_ids_sha256')
    if not isinstance(corpus, str) or not corpus or corpus != target_identity.get('corpus_ids_sha256'):
        raise ValueError('Migration corpus identity differs')
    for name in ('identity', 'producer'):
        if not isinstance(old.get(name), dict) or old[name] != new.get(name):
            raise ValueError(f'Migration extraction {name} differs')
    old_profile, new_profile = old.get('quality_profile'), new.get('quality_profile')
    if not isinstance(old_profile, dict) or not isinstance(new_profile, dict):
        raise ValueError('Migration requires explicit source and target quality profiles')
    if old_profile.get('semantic_scope') != 'all_final_relations':
        raise ValueError('Migration source is not an all-relation source-verified checkpoint')
    if (new_profile.get('validation_mode') != 'structural'
            or new_profile.get('semantic_scope') != 'none'
            or new_profile.get('structural_schema') != STRUCTURAL_VERSION):
        raise ValueError('Migration target is not the explicit structural extraction contract')
    for name in ('ner_prompt_sha256', 'ner_max_tokens', 'ner_recovery'):
        if old_profile.get(name) is None or old_profile[name] != new_profile.get(name):
            raise ValueError(f'Migration NER contract differs: {name}')
    same_triple_prompt = (old_profile.get('prompt_sha256') is not None
                         and old_profile['prompt_sha256'] == new_profile.get('prompt_sha256')
                         and old_profile.get('triple_max_tokens') == new_profile.get('triple_max_tokens'))
    return {'ner_reusable': True, 'initial_triples_reusable': same_triple_prompt}


def _first_original_extraction(metadata):
    """Find the earliest normal extraction, including older queue snapshots.

    A resumed audit can wrap an already verified candidate as its ``initial``
    entry. Descend that metadata instead of treating its filtered facts as a
    new original extraction. Recovery-stage entries are never selected.
    """
    if not isinstance(metadata, dict):
        return None
    for previous in metadata.get('build_queue_history') or []:
        found = _first_original_extraction(previous.get('metadata'))
        if found is not None:
            return found
    for entry in metadata.get('fresh_extraction_history') or []:
        if entry.get('stage') != 'initial':
            continue
        original = entry.get('metadata') or {}
        if (original.get('fresh_extraction_history') or original.get('source_verified_schema')
                or original.get('resumed_candidate_stage')
                or original.get('atomic_recovery_contract') or original.get('recovery_version')):
            found = _first_original_extraction(original)
            if found is not None:
                return found
            continue
        return copy.deepcopy(entry)
    return None


def _ner_complete(row):
    metadata = (row.get('openie_metadata') or {}).get('ner') or {}
    values = row.get('extracted_entities')
    return (isinstance(values, list)
            and all(isinstance(value, str) and value.strip() for value in values)
            and metadata.get('finish_reason') == 'stop'
            and not metadata.get('error') and not metadata.get('openie_skipped')
            and metadata.get('complete') is not False
            and metadata.get('quality_status') not in ('failed', 'partial', 'pending'))


def _structural_row(row, triple_prompt_matches):
    original = _first_original_extraction((row.get('openie_metadata') or {}).get('triples') or {})
    original_metadata = copy.deepcopy((original or {}).get('metadata') or {})
    candidates = copy.deepcopy((original or {}).get('triples') or [])
    calls = original_metadata.get('openie_attempt_count', 0)
    calls = calls if type(calls) is int and calls >= 0 else 0
    calls += (original_metadata.get('window_recovery_attempt_count', 0)
              if type(original_metadata.get('window_recovery_attempt_count', 0)) is int else 0)
    try:
        report = validate_triples(candidates)
        valid = bool(report.valid_triples) and not report.invalid_triples
    except (TypeError, ValueError):
        valid = False
    complete = (triple_prompt_matches and _ner_complete(row) and valid
                and original_metadata.get('finish_reason') == 'stop'
                and not original_metadata.get('error') and not original_metadata.get('openie_skipped')
                and original_metadata.get('quality_status') not in ('failed', 'partial', 'pending'))
    metadata = {
        'structural_schema': STRUCTURAL_VERSION, 'validation_scope': 'structural',
        'semantic_verified': False, 'requires_semantic_verification': False, 'semantic_scope': 'none',
        'complete': complete, 'quality_status': 'success' if complete else 'pending',
        'finish_reason': 'stop' if complete else None,
        # Migration issues no new inference. Historical cost remains explicit
        # below; the new structural request ceiling starts at zero.
        'structural_infer_calls': 0, 'attempt_count': 0, 'openie_attempt_count': 0,
        'migrated_initial': complete, 'structural_retry_budget': RECOVERY_BUDGET,
        'max_completion_tokens': 2048, 'thinking': False,
        'checkpoint_migration': {
            'version': MIGRATION_VERSION, 'selected_stage': 'initial' if original else None,
            'triple_prompt_matches': triple_prompt_matches,
            'original_metadata': original_metadata, 'original_candidates': candidates,
            'historical_extraction_calls': calls, 'initial_call_count': calls,
            'semantic_audit_results_reused': False,
        },
    }
    if not complete:
        metadata.update(openie_skipped=True,
                        openie_skip_reason='Initial extraction is absent or incomplete for the structural contract')
    responses = row.get('openie_responses') or {}
    migrated = {
        'idx': row['idx'], 'passage': row['passage'],
        'extracted_entities': copy.deepcopy(row.get('extracted_entities', [])),
        'extracted_triples': report.valid_triples if complete else [],
        'openie_metadata': {'ner': copy.deepcopy((row.get('openie_metadata') or {}).get('ner') or {}),
                            'triples': metadata},
        'openie_responses': {'ner': copy.deepcopy(responses.get('ner', '')),
                             'triples': copy.deepcopy((original or {}).get('response', ''))},
    }
    return migrated


def migrate_source_verified_checkpoint(source_path, target_path, *, source_identity,
                                       target_identity, chunks):
    """Write a new journal after validating the explicit source and target.

    ``chunks`` maps chunk IDs to rows containing the unchanged ``content``.
    Target must not exist; callers own the visible backup/migration decision.
    Failed/absent original extraction remains pending without invented facts.
    """
    source, target = Path(source_path).resolve(), Path(target_path).resolve()
    if source == target or target.exists():
        raise ValueError('Checkpoint migration requires a new target path')
    compatible = validate_migration_identity(source_identity, target_identity)
    corpus_digest = hashlib.sha256('\n'.join(sorted(chunks)).encode()).hexdigest()
    if corpus_digest != target_identity['corpus_ids_sha256']:
        raise ValueError('Migration chunks do not match the declared corpus identity')
    rows = []
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as connection:
        connection.execute('BEGIN')
        identity = connection.execute("SELECT value FROM identity WHERE key='contract'").fetchone()
        if identity is None or json.loads(identity[0]) != source_identity:
            raise RuntimeError('Source checkpoint identity differs from the explicitly approved source')
        for key, digest, serialized in connection.execute(
                'SELECT chunk_id, passage_sha256, payload FROM progress'):
            if key not in chunks:
                raise RuntimeError('Migration source checkpoint contains an out-of-corpus chunk')
            passage = chunks[key]['content']
            row = json.loads(serialized)
            if (digest != hashlib.sha256(passage.encode()).hexdigest()
                    or row.get('idx') != key or row.get('passage') != passage):
                raise RuntimeError('Migration source checkpoint passage or row identity differs')
            rows.append(_structural_row(row, compatible['initial_triples_reusable']))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + '.migration-', suffix='.sqlite', dir=target.parent)
    os.close(fd)
    staging = Path(name)
    progress = None
    try:
        progress = OpenIECheckpoint(staging, target_identity)
        # The journal is unpublished until the exclusive final link. A single
        # transaction preserves durability while avoiding thousands of fsyncs
        # for records already persisted in the read-only source journal.
        with progress.connection:
            progress.connection.executemany('INSERT OR REPLACE INTO progress VALUES (?, ?, ?)', (
                (row['idx'], hashlib.sha256(row['passage'].encode()).hexdigest(),
                 json.dumps(row, ensure_ascii=False)) for row in rows))
        progress.close()
        progress = None
        # Exclusive link avoids overwriting a journal created by another owner.
        os.link(staging, target)
    finally:
        if progress is not None:
            progress.close()
        for path in (staging, Path(str(staging) + '-wal'), Path(str(staging) + '-shm')):
            path.unlink(missing_ok=True)
    completed = sum(row['openie_metadata']['triples']['complete'] for row in rows)
    return {
        'version': MIGRATION_VERSION, 'source_path': str(source), 'target_path': str(target),
        'source_identity': copy.deepcopy(source_identity), 'target_identity': copy.deepcopy(target_identity),
        'corpus_chunk_count': len(chunks), 'migrated_row_count': len(rows),
        'reused_ner_count': sum(_ner_complete(row) for row in rows),
        'reused_initial_triple_count': completed,
        'pending_triple_count': len(rows) - completed,
        'missing_row_count': len(chunks) - len(rows),
        'semantic_audit_results_reused': False, **compatible,
    }
