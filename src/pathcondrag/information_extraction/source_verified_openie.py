"""Recover defective fresh extraction before publishing graph facts.

Ordinary successful initial extraction retains the existing contract. Failed,
empty and recovered extractions receive bounded compact recovery and/or whole
source semantic filtering. Neither path sees benchmark questions or answers.
"""

import copy
import json

from .openie_openai import OpenIE
from ..utils.misc_utils import TripleRawOutput
from ..utils.openie_compact_recovery import compact_recovery, RECOVERY_VERSION
from ..utils.openie_quality import validate_triples
from ..utils.openie_semantic_validation import (
    SemanticVerificationError, VERIFIER_VERSION, verify_repaired_triples,
)


SOURCE_VERIFIED_VERSION = 'pathcondrag_fresh_source_verified_openie_v1'
MAX_COMPACT_ATTEMPTS = 2


class SourceVerifiedOpenIE(OpenIE):
    """A publication gate for fresh OpenIE, with at most two compact passes."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('guided_recovery', True)
        super().__init__(*args, **kwargs)
        self.ner_max_tokens = 512
        self.triple_max_tokens = 2048

    def triple_extraction(self, chunk_key, passage, named_entities, repair_context='',
                          _allow_window_recovery=True, _support_source=None):
        initial = super().triple_extraction(
            chunk_key, passage, named_entities, repair_context=repair_context,
            _allow_window_recovery=_allow_window_recovery, _support_source=_support_source,
        )
        # Dedicated index repair owns its own verification/checkpoint pipeline.
        # Recursive windows are aggregated and audited once at their outer call.
        if repair_context or not _allow_window_recovery:
            return initial
        initial_metadata = initial.metadata or {}
        invalid = validate_triples(initial.triples).invalid_triples
        complete = (not invalid and not initial_metadata.get('openie_skipped')
                    and not initial_metadata.get('error')
                    and initial_metadata.get('quality_status') not in ('failed', 'partial')
                    and (initial_metadata.get('finish_reason') == 'stop'
                         or initial_metadata.get('window_recovery_complete')))
        if complete and initial.triples and not initial_metadata.get('quality_recovered'):
            return initial

        history = [{'stage': 'initial', 'response': initial.response,
                    'metadata': copy.deepcopy(initial_metadata),
                    'triples': copy.deepcopy(initial.triples)}]
        audits = []
        candidates = initial if complete and initial.triples else None
        compact_attempts = 0
        all_rejected = False
        context = ('Fresh extraction was empty or incomplete. Extract all explicit '
                   'relationships from the original source. An empty heading or text '
                   'without supported relationships may return no_supported_relations. '
                   'Do not invent facts merely to make the result non-empty.')
        while True:
            if candidates is None:
                if compact_attempts >= MAX_COMPACT_ATTEMPTS:
                    return self._failed(chunk_key, initial, history, audits,
                                        'No supported recovered relation after bounded recovery')
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
                    return self._failed(chunk_key, initial, history, audits,
                                        'Compact recovery did not completely validate the source')
            try:
                accepted, audit = verify_repaired_triples(
                    self.llm_model, passage, candidates.triples)
            except SemanticVerificationError as error:
                audits.append(copy.deepcopy(error.audit_metadata))
                return self._failed(chunk_key, initial, history, audits, str(error))
            except Exception as error:
                return self._failed(chunk_key, initial, history, audits,
                                    f'Semantic verification failed: {type(error).__name__}: {error}')
            audits.append(copy.deepcopy(audit))
            legitimate_empty = (not candidates.triples
                                and candidates.metadata.get('repair_status') == 'no_supported_relations'
                                and not all_rejected)
            if accepted or legitimate_empty:
                metadata = copy.deepcopy(candidates.metadata)
                if metadata.get('finish_reason') != 'stop':
                    metadata['whole_chunk_finish_reason'] = metadata.get('finish_reason')
                metadata.update(self._diagnostics(history, audits))
                metadata.update({'finish_reason': 'stop', 'quality_status': 'success' if accepted else 'empty_valid',
                                 'quality_recovered': True, 'semantic_verified': True,
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
            all_rejected = True
            context = ('All candidate relationships were rejected by whole-source entailment checks. '
                       'Re-extract supported relationships from the source only. Preserve grammatical '
                       'roles, qualifiers and unknown/missing names. Rejected assertions (diagnostic '
                       'only, not factual evidence): '
                       + json.dumps(audit.get('checks', []), ensure_ascii=False)[:1000])
            candidates = None

    @staticmethod
    def _diagnostics(history, audits):
        return {'source_verified_schema': SOURCE_VERIFIED_VERSION,
                'compact_recovery_contract': RECOVERY_VERSION,
                'semantic_verifier_contract': VERIFIER_VERSION,
                'fresh_extraction_history': copy.deepcopy(history),
                'semantic_verification_history': copy.deepcopy(audits),
                'recovery_max_compact_attempts': MAX_COMPACT_ATTEMPTS,
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
