"""Bounded flat-triple fallback for failed OpenIE repairs.

Compact extraction is not semantic verification. Publication must additionally
pass the separate verifier against the entire unchanged original passage.
"""

import copy
import json

from pathcondrag.utils.misc_utils import TripleRawOutput
from pathcondrag.utils.openie_quality import TRIPLE_JSON_SCHEMA, merge_triples, validate_triples

from .openie_semantic_validation import CONTEXT_TOKENS, MAX_COMPLETION_TOKENS, _prompt_tokens


RECOVERY_VERSION = 'pathcondrag_compact_source_recovery_v3'
SCHEMA = {
    'type': 'object',
    'properties': {
        'triples': TRIPLE_JSON_SCHEMA['properties']['triples'],
        'status': {'type': 'string', 'enum': ['success', 'no_supported_relations']},
    },
    'required': ['triples', 'status'], 'additionalProperties': False,
}
SYSTEM = '''Extract only relationships supported by the supplied SOURCE; no outside knowledge.
Return exactly {"triples": [["subject", "predicate", "object"]], "status": "success"}.
Every triple has exactly three non-empty strings. Do not emit support quotes or explanations.
Named entities are hints, not a requirement to place all of them in the graph.
Split shared predicates over coordinated subjects. For explicit shared parents each child has
each stated parent; separately assigned relationships retain their specific assignments.
Use meaningful predicates, not and/or. Put relation words in predicates and named entities
in their proper argument positions. A supported property can be ["person","trained in",
"classical theater"]; do not fill its object with an unrelated film/title from nearby text.
Empty quotes, missing names and absent table labels remain unknown. Never invent an unspecified
name, use a neighboring but/which clause to fill a missing argument, or guess a table header.
Respect source rows, units, dates, qualification and negation. Keep compact complete assertions.
Never repeat an identical triple. Stop after extracting the supported facts once.
For an explicit list of dates, numbers or colors sharing one subject and predicate, retain
the exact list as one object string, including markers and qualifiers; do not expand it
into repetitive per-item triples. Repeated titles are headings, not additional facts.
If the requested relationships cannot be supported, return
{"triples": [], "status": "no_supported_relations"}.'''


def _windows(passage, chars=450, overlap=60, maximum=8, keep_title=True):
    newline = passage.find('\n')
    title = passage[:newline] if keep_title and 0 < newline <= 200 else ''
    start = newline + 1 if title else 0
    windows = []
    while start < len(passage):
        end = min(len(passage), start + chars)
        if end < len(passage):
            boundary = passage.rfind(' ', start + chars // 2, end)
            if boundary > start:
                end = boundary
        text = (title + '\n' if title else '') + passage[start:end]
        windows.append((start, end, text))
        if end == len(passage):
            return windows
        if len(windows) >= maximum:
            return []
        start = max(start + 1, end - overlap)
        while start < end and start > 0 and not passage[start - 1].isspace():
            start += 1
    return windows


def _child_windows(passage, start, end):
    """Cover one failed parent span, retaining its original absolute offsets."""
    children = _windows(passage[start:end], chars=225, overlap=40, maximum=3,
                        keep_title=False)
    newline = passage.find('\n')
    title = passage[:newline] if 0 < newline <= 200 else ''
    return [(start + child_start, start + child_end,
             (title + '\n' if title else '') + text)
            for child_start, child_end, text in children]


def _recover_failed_window(llm, passage, start, end, entities, context, raw, metadata):
    """Spend at most three additional calls; never accept incomplete coverage."""
    children = _child_windows(passage, start, end)
    parent_metadata = copy.deepcopy(metadata)
    accepted, child_results, child_responses = [], [], []
    for child_start, child_end, text in children:
        found, child_raw, diagnostic = _extract(llm, text, entities, context, 1)
        accepted = merge_triples(accepted, found)
        child_results.append({'source_start': child_start, 'source_end': child_end,
                              'metadata': diagnostic})
        child_responses.append(child_raw)
    complete = bool(children) and all(item['metadata']['complete'] for item in child_results)
    metadata.update({'parent_attempt_metadata': parent_metadata,
                     'child_window_recovery': child_results,
                     'child_window_coverage_complete': bool(children),
                     'child_window_recovery_complete': complete,
                     'complete': complete,
                     'openie_attempt_count': parent_metadata['openie_attempt_count']
                     + sum(item['metadata']['openie_attempt_count'] for item in child_results),
                     'valid_triple_count': len(accepted),
                     'quality_status': ('success' if complete and accepted else 'empty_valid'
                                        if complete else 'partial' if accepted else 'failed')})
    if complete:
        metadata.update({'finish_reason': 'stop', 'aggregated_from_complete_children': True,
                         'repair_status': 'success' if accepted else 'no_supported_relations'})
        metadata.pop('openie_skipped', None)
        metadata.pop('openie_skip_reason', None)
    else:
        metadata['openie_skipped'] = True
        metadata['openie_skip_reason'] = 'Failed parent span was not completely recovered by child windows'
    response = json.dumps({'parent_response': raw, 'child_responses': child_responses},
                          ensure_ascii=False)
    return accepted, response, metadata


def _extract(llm, passage, entities, context, max_calls):
    hints = [entity for entity in entities if isinstance(entity, str)
             and entity.casefold() in passage.casefold()][:24]
    base = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': json.dumps(
            {'SOURCE': passage, 'named_entity_hints': hints,
             'repair_scope': context[:1600]}, ensure_ascii=False)},
    ]
    configured = getattr(getattr(llm, 'llm_config', None), 'generate_params', {}) or {}
    extra = copy.deepcopy(configured.get('extra_body') or {})
    extra['guided_json'] = SCHEMA
    chat = dict(extra.get('chat_template_kwargs') or {})
    chat['enable_thinking'] = False
    extra['chat_template_kwargs'] = chat
    settings = {'max_completion_tokens': MAX_COMPLETION_TOKENS,
                'temperature': 0.0, 'extra_body': extra}
    attempts, error, response, calls = [], None, '', 0
    metadata = {}
    for index in range(max_calls):
        messages = copy.deepcopy(base)
        if error:
            messages.append({'role': 'user', 'content': (
                'Correct the failed JSON extraction: ' + error[:500]
                + '\nReturn only complete three-string arrays and status; no quotes or filler.'
                + '\nPrevious output (diagnostic, not evidence): ' + response[:700]
            )})
        record = {'attempt': index + 1, 'raw_response': None}
        attempts.append(record)
        try:
            count = _prompt_tokens(llm, messages)
            record['prompt_token_count'] = count
            if count is not None and count + MAX_COMPLETION_TOKENS > CONTEXT_TOKENS:
                raise ValueError(f'Input needs {count} tokens; input+2048 exceeds 8192')
            calls += 1
            response, raw_metadata, cache_hit = llm.infer(messages=messages, **copy.deepcopy(settings))
            metadata = dict(raw_metadata or {})
            record.update({'raw_response': response, 'metadata': copy.deepcopy(metadata),
                           'cache_hit': cache_hit})
        except Exception as exception:
            error = f'{type(exception).__name__}: {exception}'[:500]
            record['request_error'] = error
            break  # transport retry belongs to the client, not this fallback
        try:
            if metadata.get('error') or metadata.get('finish_reason') != 'stop':
                raise ValueError(f"Incomplete response: finish_reason={metadata.get('finish_reason')!r}")
            payload = json.loads(response)
            if not isinstance(payload, dict) or set(payload) != {'triples', 'status'}:
                raise ValueError('JSON must contain exactly triples and status')
            report = validate_triples(payload['triples'])
            if report.invalid_triples:
                raise ValueError('; '.join(report.issues[:3]))
            status = payload['status']
            if status not in ('success', 'no_supported_relations'):
                raise ValueError('Invalid extraction status')
            if bool(report.valid_triples) != (status == 'success'):
                raise ValueError('success needs triples; no_supported_relations needs an empty array')
            metadata.update({'quality_status': 'success' if report.valid_triples else 'empty_valid',
                             'repair_status': status, 'openie_attempt_count': calls,
                             'raw_triple_count': report.raw_count,
                             'valid_triple_count': len(report.valid_triples),
                             'attempts': attempts, 'complete': True})
            return report.valid_triples, response, metadata
        except Exception as exception:
            error = f'{type(exception).__name__}: {exception}'[:500]
            record['validation_error'] = error
    metadata.update({'quality_status': 'failed', 'complete': False,
                     'openie_attempt_count': calls, 'valid_triple_count': 0,
                     'attempts': attempts, 'openie_skipped': True,
                     'openie_skip_reason': error or 'Extraction did not complete'})
    return [], response, metadata


def compact_recovery(llm, chunk_key, passage, named_entities, repair_context):
    """Try the whole source, bounded windows, then smaller failed spans only."""
    triples, response, metadata = _extract(llm, passage, named_entities, repair_context, 1)
    metadata.update({'recovery_version': RECOVERY_VERSION, 'recovery_strategy': 'compact_flat_triples',
                     'requires_semantic_verification': True, 'semantic_verified': False,
                     'max_completion_tokens': MAX_COMPLETION_TOKENS,
                     'temperature': 0.0, 'thinking': False})
    if metadata['complete']:
        return TripleRawOutput(chunk_key, response, triples, metadata)
    windows = _windows(passage)
    if not windows:
        metadata['window_coverage_complete'] = False
        metadata['openie_skip_reason'] += '; source cannot be covered within eight bounded windows'
        return TripleRawOutput(chunk_key, response, [], metadata)
    window_results, responses, accepted = [], [], []
    for start, end, text in windows:
        found, raw, diagnostic = _extract(llm, text, named_entities, repair_context, 2)
        if not diagnostic['complete']:
            found, raw, diagnostic = _recover_failed_window(
                llm, passage, start, end, named_entities, repair_context, raw, diagnostic)
        accepted = merge_triples(accepted, found)
        window_results.append({'source_start': start, 'source_end': end, 'metadata': diagnostic})
        responses.append(raw)
    complete = all(item['metadata']['complete'] for item in window_results)
    metadata.update({'window_recovery': window_results, 'window_coverage_complete': True,
                     'window_recovery_complete': complete, 'complete': complete,
                     'window_recovery_attempt_count': sum(item['metadata']['openie_attempt_count']
                                                          for item in window_results),
                     'valid_triple_count': len(accepted),
                     'quality_status': ('success' if accepted and complete else 'empty_valid' if complete
                                        else 'partial' if accepted else 'failed')})
    if complete:
        metadata.pop('openie_skipped', None)
        metadata.pop('openie_skip_reason', None)
        metadata['repair_status'] = 'success' if accepted else 'no_supported_relations'
    raw = json.dumps({'whole_response': response, 'window_responses': responses}, ensure_ascii=False)
    return TripleRawOutput(chunk_key, raw, accepted, metadata)
