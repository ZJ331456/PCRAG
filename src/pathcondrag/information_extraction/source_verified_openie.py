"""Recover defective fresh extraction before publishing graph facts.

Every published relation receives whole-source evidence and role checks. Failed
extractions enter bounded alternative strategies; no path sees benchmark data.
"""

import copy
import json
import logging

from .openie_openai import OpenIE
from ..utils.misc_utils import NerRawOutput, TripleRawOutput
from ..utils.openie_compact_recovery import compact_recovery, RECOVERY_VERSION
from ..utils.openie_quality import validate_triples
from ..utils.openie_structured_output import guided_json_parameters
from ..utils.openie_source_evidence import (
    SourceEvidenceError as SemanticVerificationError, VERIFIER_VERSION,
    verify_source_relations as verify_repaired_triples,
    verify_source_empty_focus,
)


SOURCE_VERIFIED_VERSION = 'pathcondrag_fresh_source_verified_openie_v2'
MAX_COMPACT_ATTEMPTS = 2
LOG = logging.getLogger(__name__)


class SourceVerifiedOpenIE(OpenIE):
    """Strict evidence checks, diverse recovery and a durable pending queue."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('guided_recovery', True)
        super().__init__(*args, **kwargs)
        self.ner_max_tokens = 512
        self.triple_max_tokens = 2048
        self.initial_rows = {}
        self.checkpoint = None
        self.bounded_structured_output = True

    @staticmethod
    def is_verified_complete(result):
        metadata = result.metadata or {}
        if (metadata.get('source_verified_schema') != SOURCE_VERIFIED_VERSION
                or metadata.get('semantic_verifier_contract') != VERIFIER_VERSION
                or metadata.get('semantic_verified') is not True
                or metadata.get('complete') is not True
                or metadata.get('finish_reason') != 'stop'
                or metadata.get('error') or metadata.get('openie_skipped')
                or validate_triples(result.triples).invalid_triples):
            return False
        audits = metadata.get('semantic_verification_history') or []
        if not audits or audits[-1].get('complete') is not True or audits[-1].get('n_unverified') != 0:
            return False
        if any(check.get('supported') and check.get('deterministic_rejection')
               for check in audits[-1].get('checks', [])):
            return False
        accepted = [check['triple'] for check in audits[-1].get('checks', []) if check.get('supported') is True]
        return (accepted == result.triples and (bool(result.triples)
                or metadata.get('source_no_supported_relations') is True
                and audits[-1].get('source_no_supported_relations') is True))

    def batch_openie(self, chunks):
        from ..utils.openie_build_queue import run_openie_queue
        return run_openie_queue(self, chunks, initial_rows=self.initial_rows,
                                checkpoint=self.checkpoint)

    def ner(self, chunk_key, passage):
        return self._ner_attempts(chunk_key, passage)

    def _ner_attempts(self, chunk_key, passage, context=''):
        """Retain the fixed 512-token NER budget on every quality retry."""
        from ..utils.openie_semantic_validation import _prompt_tokens
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
        from ..utils.openie_atomic_recovery import source_units
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

    def audit_existing_triples(self, chunk_key, passage, named_entities, previous):
        return self._verify_and_recover(chunk_key, passage, named_entities, previous)

    def recover_pending_triples(self, chunk_key, passage, named_entities, previous, round_number):
        # A failed evidence request does not invalidate the extraction response.
        # Resume its last complete candidate set with fresh audit feedback before
        # asking the extractor to generate the same facts again.
        candidates = previous
        for entry in reversed((previous.metadata or {}).get('fresh_extraction_history', [])):
            metadata = copy.deepcopy(entry.get('metadata') or {})
            values = entry.get('triples')
            if (isinstance(values, list) and not validate_triples(values).invalid_triples
                    and not metadata.get('error') and not metadata.get('openie_skipped')
                    and metadata.get('quality_status') not in ('failed', 'partial')
                    and (metadata.get('complete') is True or metadata.get('finish_reason') == 'stop')
                    and (values or metadata.get('repair_status') == 'no_supported_relations')):
                metadata['openie_skip_reason'] = (previous.metadata or {}).get('openie_skip_reason', '')
                metadata['resumed_candidate_stage'] = entry.get('stage')
                candidates = TripleRawOutput(chunk_key, entry.get('response', ''), values, metadata)
                break
        return self._verify_and_recover(chunk_key, passage, named_entities, candidates,
                                       pending_round=round_number)

    def triple_extraction(self, chunk_key, passage, named_entities, repair_context='',
                          _allow_window_recovery=True, _support_source=None):
        initial = super().triple_extraction(
            chunk_key, passage, named_entities, repair_context=repair_context,
            # Fresh strict extraction uses complete source units below. Avoid
            # repeating the older character-window recovery before that path.
            _allow_window_recovery=False, _support_source=_support_source,
        )
        # Dedicated index repair owns its own verification/checkpoint pipeline.
        # Recursive windows are aggregated and audited once at their outer call.
        if repair_context or not _allow_window_recovery:
            return initial
        return self._verify_and_recover(chunk_key, passage, named_entities, initial)

    def _verify_and_recover(self, chunk_key, passage, named_entities, initial, pending_round=0):
        initial_metadata = initial.metadata or {}
        invalid = validate_triples(initial.triples).invalid_triples
        complete = (not invalid and not initial_metadata.get('openie_skipped')
                    and not initial_metadata.get('error')
                    and initial_metadata.get('quality_status') not in ('failed', 'partial')
                    and (initial_metadata.get('finish_reason') == 'stop'
                         or initial_metadata.get('window_recovery_complete')))
        history = [{'stage': 'initial', 'response': initial.response,
                    'metadata': copy.deepcopy(initial_metadata),
                    'triples': copy.deepcopy(initial.triples)}]
        audits = []
        candidates = initial if complete and initial.triples else None
        compact_attempts = 0
        atomic_used = False
        context = ('Fresh extraction was empty or incomplete. Extract all explicit '
                   'relationships from the original source. An empty heading or text '
                   'without supported relationships may return no_supported_relations. '
                   'Do not invent facts merely to make the result non-empty.')
        while True:
            if candidates is None:
                if compact_attempts >= MAX_COMPACT_ATTEMPTS:
                    if atomic_used:
                        return self._failed(chunk_key, initial, history, audits,
                                            'No supported relation after bounded source-unit recovery')
                    from ..utils.openie_atomic_recovery import atomic_recovery
                    atomic_used = True
                    candidates = atomic_recovery(self.llm_model, chunk_key, passage, named_entities,
                                                 context + f' Pending recovery round: {pending_round}.')
                    history.append({'stage': 'atomic', 'response': candidates.response,
                                    'metadata': copy.deepcopy(candidates.metadata),
                                    'triples': copy.deepcopy(candidates.triples)})
                    if not candidates.metadata.get('complete') or candidates.metadata.get('openie_skipped'):
                        return self._failed(chunk_key, initial, history, audits,
                                            'Source-unit recovery remains incomplete')
                else:
                    compact_attempts += 1
                    candidates = compact_recovery(
                        self.llm_model, chunk_key, passage, named_entities, context)
                    history.append({'stage': 'compact', 'attempt': compact_attempts,
                                    'response': candidates.response,
                                    'metadata': copy.deepcopy(candidates.metadata),
                                    'triples': copy.deepcopy(candidates.triples)})
                metadata = candidates.metadata or {}
                if (not metadata.get('complete') or metadata.get('openie_skipped')
                        or validate_triples(candidates.triples).invalid_triples):
                    context = self._incomplete_recovery_feedback(metadata)
                    candidates = None
                    continue
            try:
                empty_focus_audits = []
                for focus in candidates.metadata.get('unverified_empty_focuses', []):
                    start, end = ((focus['source_start'], focus['source_end'])
                                  if isinstance(focus, dict) else focus)
                    focus_audit = verify_source_empty_focus(
                        self.llm_model, passage, start, end,
                        retry_context=f'Original source-unit coverage check, pending round {pending_round}.')
                    empty_focus_audits.append(focus_audit)
                    if focus_audit.get('source_no_supported_relations') is not True:
                        return self._failed(chunk_key, initial, history, audits,
                                            'A source unit declared empty still contains supported relations')
                if empty_focus_audits:
                    candidates.metadata['atomic_empty_focus_audits'] = empty_focus_audits
                accepted, audit = verify_repaired_triples(
                    self.llm_model, passage, candidates.triples,
                    retry_context=(f'Pending source audit round {pending_round}: '
                                   + str(initial_metadata.get('openie_skip_reason', ''))
                                   if pending_round else ''))
            except SemanticVerificationError as error:
                audits.append(copy.deepcopy(error.audit_metadata))
                return self._failed(chunk_key, initial, history, audits, str(error))
            except Exception as error:
                return self._failed(chunk_key, initial, history, audits,
                                    f'Semantic verification failed: {type(error).__name__}: {error}')
            audits.append(copy.deepcopy(audit))
            legitimate_empty = (not candidates.triples
                                and candidates.metadata.get('repair_status') == 'no_supported_relations'
                                and audit.get('source_no_supported_relations') is True)
            from ..utils.openie_source_evidence import _subject_members
            scope_errors = any(
                check.get('rejection_kind') == 'attribution'
                or check.get('rejection_kind') == 'subject_scope'
                and len(_subject_members(check['triple'][0])) > 1
                for check in audit.get('checks', []) if not check.get('supported'))
            if accepted and scope_errors and not atomic_used:
                # Recover the correctly attributed relations as well as reject
                # the wrong joint-subject assertion. The whole source stays fixed.
                compact_attempts = MAX_COMPACT_ATTEMPTS
                context = ('Re-extract every source-supported relation by its actual subject. '
                           'The earlier candidate has an unresolved subject/attribution scope. '
                           'Do not copy a property of one named person onto a group.')
                candidates = None
                continue
            if accepted or legitimate_empty:
                metadata = copy.deepcopy(candidates.metadata)
                if metadata.get('finish_reason') != 'stop':
                    metadata['whole_chunk_finish_reason'] = metadata.get('finish_reason')
                metadata.update(self._diagnostics(history, audits))
                metadata.update({'finish_reason': 'stop', 'quality_status': 'success' if accepted else 'empty_valid',
                                 'quality_recovered': bool(initial_metadata.get('quality_recovered')
                                                           or compact_attempts or atomic_used),
                                 'semantic_verified': True,
                                 'requires_semantic_verification': False,
                                 'complete': True, 'valid_triple_count': len(accepted),
                                 'invalid_triple_count': 0,
                                 'source_no_supported_relations': legitimate_empty})
                metadata.pop('error', None)
                metadata.pop('openie_skipped', None)
                metadata.pop('openie_skip_reason', None)
                return TripleRawOutput(chunk_key, json.dumps(
                    {'extraction_responses': [entry['response'] for entry in history],
                     'verification_responses': [entry.get('raw_response') for entry in audits]},
                    ensure_ascii=False), accepted, metadata)
            context = ('All candidate relationships were rejected by whole-source entailment checks. '
                       'Re-extract supported relationships from the source only. Preserve grammatical '
                       'roles, qualifiers and unknown/missing names. Rejected assertions (diagnostic '
                       'only, not factual evidence): '
                       + json.dumps(audit.get('checks', []), ensure_ascii=False)[:1000])
            candidates = None

    @staticmethod
    def _incomplete_recovery_feedback(metadata):
        """Change a bounded retry using failed-span errors, never approve partial output."""
        errors = []

        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('validation_error', 'request_error') and isinstance(item, str):
                        if item not in errors:
                            errors.append(item)
                    elif key in ('attempts', 'window_recovery', 'child_window_recovery',
                                 'metadata', 'parent_attempt_metadata'):
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(metadata)
        return ('Previous compact extraction did not completely cover and validate the source. '
                'Re-extract all supported facts once, without duplicate triples. '
                'A coordinating word such as and/or is not a predicate; use the actual '
                'source relationship separately for each coordinated subject. '
                'Preserve grammatical roles, qualifiers and attribution. '
                'Diagnostics are not factual evidence: '
                + '; '.join(errors[-3:])[:750])

    @staticmethod
    def _diagnostics(history, audits):
        return {'source_verified_schema': SOURCE_VERIFIED_VERSION,
                'compact_recovery_contract': RECOVERY_VERSION,
                'semantic_verifier_contract': VERIFIER_VERSION,
                'fresh_extraction_history': copy.deepcopy(history),
                'semantic_verification_history': copy.deepcopy(audits),
                'recovery_max_compact_attempts': MAX_COMPACT_ATTEMPTS,
                'fresh_recovery_implementation': 'retry_incomplete_with_feedback_v2',
                'semantic_scope': 'all_final_relations',
                'max_completion_tokens': 2048, 'thinking': False,
                'response_is_aggregate': True}

    def _failed(self, chunk_key, initial, history, audits, reason):
        metadata = copy.deepcopy(initial.metadata or {})
        metadata.update(self._diagnostics(history, audits))
        metadata.update({'quality_status': 'failed', 'openie_skipped': True,
                         'openie_skip_reason': reason, 'complete': False,
                         'semantic_verified': False, 'requires_semantic_verification': True})
        # Return an explicit incomplete row so callers persist all diagnostics
        # before aborting publication. No unverified candidate enters the graph.
        metadata.pop('error', None)
        return TripleRawOutput(chunk_key, json.dumps(
            {'extraction_responses': [entry['response'] for entry in history],
             'verification_responses': [entry.get('raw_response') for entry in audits]},
            ensure_ascii=False), [], metadata)
