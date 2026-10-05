"""Ordinary-cost OpenIE with local validation and bounded failure feedback.

Normal passages require one triple-generation request. This mode does not run
source entailment audits, compact extraction, atomic recovery or source windows.
The three-call budget concerns triples only, separately from NER/HTTP retries.
"""

import copy
import json

from .openie_openai import OpenIE
from .openie_quality import TRIPLE_JSON_SCHEMA, validate_triples
from .openie_structured_output import guided_json_parameters
from ..extraction_utils import _base_generate_seed
from ..ner.source_verified import SourceVerifiedNERMixin
from ...utils.misc_utils import TripleRawOutput


STRUCTURAL_VERSION = 'pathcondrag_structural_openie_v1'
RECOVERY_BUDGET = 3


class StructuralOpenIE(SourceVerifiedNERMixin, OpenIE):
    """Keep fixed-budget NER, validate triples locally, retry failures only."""

    def __init__(self, *args, ner_max_tokens=512, triple_max_tokens=2048, **kwargs):
        if ner_max_tokens != 512 or triple_max_tokens != 2048:
            raise ValueError('Structural OpenIE requires NER=512 and triples=2048 tokens')
        kwargs.setdefault('guided_recovery', True)
        super().__init__(*args, **kwargs)
        self.ner_max_tokens = ner_max_tokens
        self.triple_max_tokens = triple_max_tokens
        self.structured_initial_triples = True
        self.bounded_structured_output = True
        self.initial_rows = {}
        self.checkpoint = None

    @staticmethod
    def _prior_calls(previous):
        metadata = (previous.metadata or {}) if previous is not None else {}
        for field in ('structural_infer_calls', 'openie_attempt_count', 'attempt_count'):
            count = metadata.get(field)
            if type(count) is int and count >= 0:
                return count
        return 0

    @staticmethod
    def is_verified_complete(result):
        """Compatibility interface: complete structure is not semantic proof."""
        metadata = result.metadata or {}
        count = metadata.get('structural_infer_calls')
        if (metadata.get('structural_schema') != STRUCTURAL_VERSION
                or metadata.get('validation_scope') != 'structural'
                or metadata.get('semantic_verified') is not False
                or metadata.get('complete') is not True
                or metadata.get('finish_reason') != 'stop'
                or metadata.get('quality_status') not in ('success', 'empty_valid')
                or metadata.get('error') or metadata.get('openie_skipped')
                or type(count) is not int or not 0 <= count <= RECOVERY_BUDGET):
            return False
        try:
            validation = validate_triples(result.triples)
        except (TypeError, ValueError):
            return False
        if validation.invalid_triples or validation.raw_count != len(validation.valid_triples):
            return False
        return bool(result.triples) or (
            metadata.get('deterministically_empty_source') is True
            and metadata.get('source_no_supported_relations') is True)

    def batch_openie(self, chunks):
        from ..openie_build_queue import run_openie_queue
        return run_openie_queue(self, chunks, initial_rows=self.initial_rows, checkpoint=self.checkpoint)

    @staticmethod
    def _metadata(base, calls, history, *, complete, error=None):
        metadata = copy.deepcopy(base or {})
        # Preserve extraction diagnostics, never inherited semantic approval.
        historical = copy.deepcopy(metadata.get('historical_semantic_metadata') or {})
        for field in ('source_verified_schema', 'semantic_verification_history',
                      'semantic_verifier_contract', 'fresh_extraction_history',
                      'atomic_empty_focus_audits'):
            if field in metadata:
                historical[field] = metadata.pop(field)
        if historical:
            metadata['historical_semantic_metadata'] = historical
        metadata.update(structural_schema=STRUCTURAL_VERSION, validation_scope='structural',
                        semantic_verified=False, requires_semantic_verification=False,
                        semantic_scope='none', structural_infer_calls=calls, attempt_count=calls,
                        openie_attempt_count=calls, structural_retry_budget=RECOVERY_BUDGET,
                        structural_attempts=copy.deepcopy(history), complete=complete,
                        quality_status='success' if complete else 'failed',
                        max_completion_tokens=2048, thinking=False)
        if complete:
            for field in ('error', 'openie_skipped', 'openie_skip_reason'):
                metadata.pop(field, None)
        else:
            metadata.update(error=error or 'Structural extraction incomplete', openie_skipped=True,
                            openie_skip_reason=error or 'Structural extraction incomplete')
        return metadata

    def audit_existing_triples(self, chunk_key, passage, named_entities, previous):
        """Reuse a completed raw extraction without an additional LLM request."""
        calls = self._prior_calls(previous)
        try:
            if calls > RECOVERY_BUDGET:
                raise ValueError('Historical triple request count exceeds structural budget')
            if (previous.metadata or {}).get('finish_reason') != 'stop':
                raise ValueError('Cached triple generation did not finish with stop')
            validation = validate_triples(previous.triples)
            if validation.invalid_triples:
                raise ValueError('; '.join(validation.issues[:3]))
            if not validation.valid_triples:
                raise ValueError('Nonempty source has no locally completed relations')
            metadata = self._metadata(previous.metadata, calls, [], complete=True)
            metadata.update(valid_triple_count=len(validation.valid_triples), invalid_triple_count=0,
                            source_no_supported_relations=False, reused_completed_extraction=True)
            return TripleRawOutput(chunk_key, previous.response, validation.valid_triples, metadata)
        except (TypeError, ValueError) as error:
            return TripleRawOutput(chunk_key, previous.response, [],
                                   self._metadata(previous.metadata, calls, [], complete=False, error=str(error)))

    def recover_pending_triples(self, chunk_key, passage, named_entities, previous, round_number):
        return self.triple_extraction(chunk_key, passage, named_entities,
                                      _previous=previous, _round=round_number)

    def triple_extraction(self, chunk_key, passage, named_entities, repair_context='',
                          _allow_window_recovery=True, _support_source=None,
                          *, _previous=None, _round=0):
        from .openie_semantic_validation import _prompt_tokens
        if not isinstance(passage, str):
            raise TypeError('Structural OpenIE requires the unchanged source string')
        calls = self._prior_calls(_previous)
        history = copy.deepcopy((_previous.metadata or {}).get('structural_attempts') or []) if _previous else []
        response = _previous.response if _previous else ''
        base_metadata = copy.deepcopy(_previous.metadata or {}) if _previous else {}
        error = str(base_metadata.get('error') or base_metadata.get('openie_skip_reason') or '')
        if not passage.strip():
            metadata = self._metadata({}, calls, history, complete=True)
            metadata.update(finish_reason='stop', quality_status='empty_valid',
                            deterministically_empty_source=True, source_no_supported_relations=True,
                            empty_result_decision='original_source_contains_only_whitespace')
            return TripleRawOutput(chunk_key, '{"triples":[]}', [], metadata)

        configured = getattr(getattr(self.llm_model, 'llm_config', None), 'generate_params', {}) or {}
        parameters = {'max_completion_tokens': 2048, 'temperature': 0.0,
                      'extra_body': guided_json_parameters(TRIPLE_JSON_SCHEMA, configured.get('extra_body'))}
        while calls < RECOVERY_BUDGET:
            messages = self._triple_messages(passage, named_entities)
            if calls or error:
                messages.append({'role': 'user', 'content': (
                    f'Structural extraction attempt {calls + 1}/{RECOVERY_BUDGET}. '
                    'Previous extraction was incomplete: ' + error[:500]
                    + '\nRe-extract the explicit source relations as unique complete '
                    'three-string arrays. Return only {"triples":[...]}; no reasoning. '
                    'Use compact property/list values, do not repeat facts or invent missing arguments. '
                    'An empty result is incomplete for this nonempty source. '
                    'Previous response is diagnostic only: ' + str(response or '')[:700])})
            settings = copy.deepcopy(parameters)
            if calls:
                settings['seed'] = _base_generate_seed(self.llm_model) + 100 + calls
            record = {'attempt': calls + 1, 'pending_round': _round, 'request_sent': False}
            history.append(record)
            try:
                count = _prompt_tokens(self.llm_model, messages)
                record['prompt_token_count'] = count
                if count is not None and count + 2048 > 8192:
                    raise ValueError('Unchanged source plus 2048 exceeds the fixed context allowance')
                calls += 1
                record['request_sent'] = True
                response, generated, hit = self.llm_model.infer(messages=messages, **settings)
                base_metadata = dict(generated or {})
                record.update(raw_response=response, metadata=copy.deepcopy(base_metadata), cache_hit=hit)
                if base_metadata.get('error') or base_metadata.get('finish_reason') != 'stop':
                    raise ValueError('Triple response did not finish with error-free stop')
                payload = json.loads(response)
                if not isinstance(payload, dict) or set(payload) != {'triples'}:
                    raise ValueError('Return exactly one JSON object with the triples array')
                validation = validate_triples(payload['triples'])
                record.update(raw_triple_count=validation.raw_count,
                              invalid_triple_count=len(validation.invalid_triples))
                if validation.invalid_triples:
                    raise ValueError('; '.join(validation.issues[:3]))
                if not validation.valid_triples:
                    raise ValueError('Nonempty source returned no complete relations')
                metadata = self._metadata(base_metadata, calls, history, complete=True)
                metadata.update(valid_triple_count=len(validation.valid_triples), invalid_triple_count=0,
                                raw_triple_count=validation.raw_count, cache_hit=hit,
                                source_no_supported_relations=False)
                return TripleRawOutput(chunk_key, response, validation.valid_triples, metadata)
            except Exception as failure:
                error = f'{type(failure).__name__}: {failure}'
                record['validation_error'] = error
                if not record['request_sent']:
                    break  # Repeating the whole context cannot repair an oversized source.

        metadata = self._metadata(base_metadata, calls, history, complete=False,
                                  error=error or 'Triple inference budget exhausted')
        metadata.update(empty_result_decision='failed_extraction_keep_passage_without_graph_facts',
                        source_no_supported_relations=False, valid_triple_count=0)
        return TripleRawOutput(chunk_key, response, [], metadata)
