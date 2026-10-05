"""NER stages for source-verified index construction."""

import copy
import json

from ...utils.misc_utils import NerRawOutput
from ..openie.openie_structured_output import guided_json_parameters


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
                    'Previous named-entity extraction was invalid: ' + error[:500]
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
        from ..openie.openie_atomic_recovery import source_units
        entities, units = [], []
        for start, end, text in source_units(passage):
            result = self._ner_attempts(chunk_key, text, (
                f'Recovery round {round_number}: extract names only from this original source unit. '
                'List each name once; an unnamed pronoun is not a new person.'))
            units.append({'source_start': start, 'source_end': end, 'metadata': result.metadata,
                          'response': result.response})
            if result.metadata.get('error') or result.metadata.get('complete') is not True:
                return NerRawOutput(chunk_key, json.dumps(units, ensure_ascii=False), [], {
                    'quality_status': 'failed', 'complete': False, 'openie_skipped': True,
                    'error': 'An original NER unit remains incomplete', 'ner_units': units,
                    'ner_max_tokens_used': 512, 'thinking': False})
            entities.extend(result.unique_entities)
        return NerRawOutput(chunk_key, json.dumps(units, ensure_ascii=False), list(dict.fromkeys(entities)), {
            'quality_status': 'success', 'complete': True, 'finish_reason': 'stop',
            'ner_units': units, 'ner_max_tokens_used': 512, 'thinking': False})
