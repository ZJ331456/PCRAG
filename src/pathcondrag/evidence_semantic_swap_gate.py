"""Decision-only review control on an already source-validated bounded proposal.

This intentionally measures the model's semantic veto separately from a second
proof-construction task. Reviewer quotations are never accepted as evidence.
The proposed ranking retains the original selector's literal source checks;
semantic approval remains a fallible judgment, not an entailment certificate.
"""
from __future__ import annotations

import json

from .evidence_swap_verifier import _CHECKS, _IDENTIFIER, _proposal
from .evidence_swap_verifier_v3 import build_verifier_prompt


def validate_verification(query, order, state, document, proposal_diag, response, finish_reason='stop'):
    diag = {'enabled': True, 'policy': 'decision_only_semantic_veto_on_source_validated_proposal',
        'gold_labels_used': False, 'generator_rationale_visible': False, 'approved': False,
        'entailment_guaranteed': False, 'reviewer_quotes_used_as_evidence': False,
        'source_validation': 'Original bounded proposal retains its literal visible-source quotation checks',
        'finish_reason': finish_reason, 'checks': {}, 'rejection_reason': None}

    def reject(reason):
        diag['rejection_reason'] = reason
        return False, diag

    proposal, reason = _proposal([int(value) for value in order], state, proposal_diag)
    if reason:
        return reject(reason)
    diag['proposal'] = proposal
    if finish_reason != 'stop':
        return reject('review_response_not_completed')
    try:
        payload = json.loads(response)
    except (ValueError, TypeError):
        return reject('invalid_review_json')
    if not isinstance(payload, dict):
        return reject('review_payload_not_object')
    for key, doc_id in proposal.items():
        value = payload.get(key)
        if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or int(value[1:]) != doc_id:
            return reject('review_document_id_missing_or_mismatch:' + key)
    diag['checks'] = {key: payload.get(key) for key in _CHECKS}
    if any(type(payload.get(key)) is not bool for key in _CHECKS):
        return reject('review_checks_not_boolean')
    failed = [key for key in _CHECKS if not payload[key]]
    if failed:
        return reject('review_conditions_not_met:' + ','.join(failed))
    for key in ('old_missing_fact', 'candidate_supplied_fact', 'victim_reason', 'reason'):
        value = payload.get(key)
        if not isinstance(value, str) or len(value.strip()) < 8:
            return reject('missing_semantic_review_explanation:' + key)
        diag[key] = value
    diag['approved'] = True
    return True, diag
