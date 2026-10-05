"""NER stages for source-verified index construction."""

import copy
import hashlib
import json
from collections import deque

from ...utils.misc_utils import NerRawOutput
from ..openie.openie_structured_output import guided_json_parameters


NER_RECOVERY_VERSION = 'pathcondrag_bounded_source_span_ner_v1'
MAX_NER_RECOVERY_CALLS = 64
MAX_NER_SPLIT_DEPTH = 12


class SourceVerifiedNERMixin:
    """NER requests and source-unit recovery with the fixed output budget."""

    def ner(self, chunk_key, passage):
        return self._ner_attempts(chunk_key, passage)

    def _ner_attempts(self, chunk_key, passage, context=''):
        """Retain the fixed 512-token NER budget on every quality retry."""
        from ..openie.openie_semantic_validation import _prompt_tokens
        messages = self.prompt_template_manager.render(name='ner', passage=passage)
        schema = {'type': 'object', 'properties': {'named_entities': {
            'type': 'array', 'items': {'type': 'string', 'minLength': 1}}},
            'required': ['named_entities'], 'additionalProperties': False}
        attempts, error, response = [], None, ''
        if context:
            messages.append({'role': 'user', 'content': context[:1000]})
        for index in range(3):
            task = copy.deepcopy(messages)
            if error:
                task.append({'role': 'user', 'content': (
                    f'NER recovery attempt {index + 1}/3. Previous named-entity extraction was invalid: ' + error[:500]
                    + '\nReturn unique source names once in {"named_entities":[...]}; no reasoning. '
                    'The output budget is 512 tokens. Do not repeat names or invent missing ones.')})
            record = {'attempt': index + 1}
            attempts.append(record)
            try:
                count = _prompt_tokens(self.llm_model, task)
                if count is not None and count + 512 > 8192:
                    raise ValueError('Whole NER source exceeds the fixed context allowance')
                response, metadata, hit = self.llm_model.infer(
                    messages=task, max_completion_tokens=512, temperature=0.0,
                    extra_body=guided_json_parameters(schema))
                metadata = dict(metadata or {})
                record.update(raw_response=response, metadata=copy.deepcopy(metadata), cache_hit=hit,
                              prompt_token_count=count)
                if metadata.get('error') or metadata.get('finish_reason') != 'stop':
                    raise ValueError('NER response did not finish with an error-free stop')
                payload = json.loads(response)
                values = payload.get('named_entities') if isinstance(payload, dict) else None
                if (set(payload) != {'named_entities'} or not isinstance(values, list)
                        or any(not isinstance(value, str) or not value.strip() for value in values)):
                    raise ValueError('NER must return exactly a list of nonempty named-entity strings')
                metadata.update(complete=True, quality_status='success', cache_hit=hit,
                                ner_max_tokens_used=512, ner_attempts=attempts, thinking=False)
                return NerRawOutput(chunk_key, response, list(dict.fromkeys(value.strip() for value in values)), metadata)
            except Exception as exception:
                error = f'{type(exception).__name__}: {exception}'
                record['validation_error'] = error
        return NerRawOutput(chunk_key, response, [], {'quality_status': 'failed', 'complete': False,
            'error': error, 'openie_skipped': True, 'ner_max_tokens_used': 512,
            'ner_attempts': attempts, 'thinking': False})

    def recover_pending_ner(self, chunk_key, passage, previous, round_number):
        """Split failed spans, retain completed leaves and cover every character.

        The call ceiling includes earlier attempts for this chunk. Successes
        from a matching checkpoint are replayed; only uncovered source spans
        issue new fixed-budget requests. Parents are diagnostics, never leaves.
        """
        from ..openie.openie_atomic_recovery import source_units, _split_span
        digest = hashlib.sha256(passage.encode('utf-8')).hexdigest()
        prior = previous.metadata or {}
        calls = prior.get('ner_recovery_calls')
        if type(calls) is not int or calls < 0:
            calls = len(prior.get('ner_attempts') or [])
        units, failed_history = [], []
        if (prior.get('ner_recovery_version') == NER_RECOVERY_VERSION
                and prior.get('ner_recovery_source_sha256') == digest):
            failed_history = copy.deepcopy(prior.get('ner_recovery_history_failedunits') or [])
            cursor = 0
            for unit in sorted(prior.get('ner_units') or [], key=lambda row: row.get('source_start', -1)):
                start, end = unit.get('source_start'), unit.get('source_end')
                metadata = unit.get('metadata') or {}
                values = unit.get('entities')
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(passage) or start < cursor
                        or unit.get('source_text') != passage[start:end]
                        or metadata.get('complete') is not True
                        or metadata.get('finish_reason') != 'stop' or metadata.get('error')
                        or metadata.get('openie_skipped')
                        or metadata.get('quality_status') in ('failed', 'partial', 'pending')
                        or not isinstance(values, list)
                        or any(not isinstance(value, str) or not value.strip() for value in values)):
                    continue
                units.append(copy.deepcopy(unit))
                cursor = end

        # Subtract approved leaves from each original sentence/newline unit.
        # This preserves the original partitions and every unprocessed tail.
        prior_depths = {
            (record.get('source_start'), record.get('source_end')): record.get('split_depth', 0)
            for record in prior.get('ner_pending_spans') or []
            if (prior.get('ner_recovery_source_sha256') == digest
                and type(record.get('split_depth', 0)) is int
                and 0 <= record.get('split_depth', 0) <= MAX_NER_SPLIT_DEPTH)
        }
        pending = deque()
        for start, end, _ in source_units(passage):
            cursor = start
            for unit in units:
                left, right = unit['source_start'], unit['source_end']
                if right <= cursor or left >= end:
                    continue
                if left > cursor:
                    right_edge = min(left, end)
                    pending.append((cursor, right_edge, prior_depths.get((cursor, right_edge), 0)))
                cursor = min(end, max(cursor, right))
            if cursor < end:
                pending.append((cursor, end, prior_depths.get((cursor, end), 0)))

        error = None
        while pending:
            start, end, depth = pending.popleft()
            # Each _ner_attempts call can issue at most three requests. Reserve
            # that full bound so the hard ceiling cannot be crossed mid-call.
            if calls + 3 > MAX_NER_RECOVERY_CALLS:
                pending.appendleft((start, end, depth))
                error = 'NER recovery call budget exhausted with unprocessed original spans'
                break
            text = passage[start:end]
            result = self._ner_attempts(chunk_key, text, (
                f'Recovery round {round_number}: extract names only from this original source span. '
                'List each name once; an unnamed pronoun is not a new person.'))
            calls += max(1, len((result.metadata or {}).get('ner_attempts') or []))
            record = {'source_start': start, 'source_end': end, 'source_text': text,
                      'split_depth': depth, 'metadata': copy.deepcopy(result.metadata),
                      'response': result.response, 'entities': list(result.unique_entities)}
            if (not result.metadata.get('error') and result.metadata.get('complete') is True
                    and result.metadata.get('finish_reason') == 'stop'
                    and not result.metadata.get('openie_skipped')
                    and result.metadata.get('quality_status') not in ('failed', 'partial', 'pending')):
                record['status'] = 'success'
                units.append(record)
                continue
            children = _split_span(passage, start, end) if depth < MAX_NER_SPLIT_DEPTH else []
            if children:
                record.update(status='split_parent', children=[list(span) for span in children])
                failed_history.append(record)
                for left, right in reversed(children):
                    pending.appendleft((left, right, depth + 1))
            else:
                record['status'] = 'failed_leaf'
                failed_history.append(record)
                pending.appendleft((start, end, depth))
                error = 'An original NER span remains incomplete and cannot be split further'
                break

        units.sort(key=lambda unit: unit['source_start'])
        cursor = 0
        for unit in units:
            if unit['source_start'] != cursor:
                break
            cursor = unit['source_end']
        complete = not pending and error is None and cursor == len(passage)
        metadata = {
            'quality_status': 'success' if complete else 'failed', 'complete': complete,
            'ner_units': units, 'ner_recovery_history_failedunits': failed_history,
            'ner_pending_spans': [{'source_start': start, 'source_end': end, 'split_depth': depth}
                                  for start, end, depth in pending],
            'ner_recovery_version': NER_RECOVERY_VERSION, 'ner_recovery_source_sha256': digest,
            'ner_recovery_calls': calls, 'ner_recovery_max_calls': MAX_NER_RECOVERY_CALLS,
            'ner_recovery_max_split_depth': MAX_NER_SPLIT_DEPTH,
            'ner_source_coverage_complete': complete,
            'ner_max_tokens_used': 512, 'thinking': False,
        }
        if complete:
            metadata['finish_reason'] = 'stop'
            entities = list(dict.fromkeys(entity for unit in units for entity in unit['entities']))
        else:
            metadata.update(error=error or 'NER original source coverage is incomplete', openie_skipped=True)
            entities = []
        return NerRawOutput(chunk_key, json.dumps(units, ensure_ascii=False), entities, metadata)
