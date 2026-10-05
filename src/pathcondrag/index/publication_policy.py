"""Explicit failure handling at graph publication; extraction stays unchanged."""

import argparse
import copy
import json
import os
from pathlib import Path


STRICT_FAILURE_POLICY = 'checkpoint_then_abort_before_graph_publication'
LENIENT_FAILURE_POLICY = 'checkpoint_then_skip_failed_relations_and_continue'


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ('true', '1', 'yes', 'on'):
        return True
    if lowered in ('false', '0', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def apply_publication_policy(rows, failed_ids, *, strict=True, report_path=None):
    """Keep every passage and failed diagnostic, excluding failed graph facts.

    The returned report records the decision, not a claim that graph building
    has succeeded. No failed stage becomes verified or an approved empty row.
    """
    failed = list(dict.fromkeys(failed_ids))
    failed_set = set(failed)
    excluded_count = 0
    if not strict:
        for row in rows:
            if row['idx'] not in failed_set:
                continue
            metadata = row.setdefault('openie_metadata', {})
            publication = metadata.setdefault('publication', {})
            candidates = publication.get('excluded_triples', row.get('extracted_triples', []))
            publication.update(openie_strict=False, skipped_from_graph=True,
                               excluded_triples=copy.deepcopy(candidates),
                               reason='Final extraction quality checks did not complete')
            excluded_count += len(candidates)
            row['extracted_triples'] = []
            row['extracted_entities'] = [entity for entity in row.get('extracted_entities', [])
                                         if isinstance(entity, str) and entity.strip()]
    report = {
        'schema': 'pathcondrag_openie_publication_policy_v1',
        'openie_strict': strict,
        'failure_policy': STRICT_FAILURE_POLICY if strict else LENIENT_FAILURE_POLICY,
        'document_count': len(rows), 'failed_document_count': len(failed),
        'failed_chunk_ids': failed, 'excluded_relation_count': excluded_count,
        'action': 'abort' if strict and failed else 'continue',
    }
    if report_path is not None:
        target = Path(report_path)
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, target)
    return report
