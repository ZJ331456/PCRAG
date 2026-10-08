"""Retain baseline binding sources and explicit question anchors before a swap.

Source retention is deliberately broader than semantic promotion eligibility:
an existing winning proof may be preserved even when a later DAG checker cannot
recognize its relation. Preserving that passage makes no new truth claim about
its binding. This guard never repairs, promotes, or modifies a proof.

A collapsed single-node plan cannot certify unmodeled ancestor completeness.
Rather than inventing hops from relative clauses, the guard retains the actual
observed support and question anchors and reports this remaining limitation.
"""
from __future__ import annotations

from .evidence_anchor_companion import _literal, _norm, _protected, _specific, _title
from .evidence_joint_selector import _quote_span
from .evidence_retrieval import branch_matches
from .evidence_selection import _ancestors


PREFIX_K, FIXED_K, TOP_K = 5, 2, 10


def _doc_id(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value) if isinstance(value, (int, str)) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _winning_sources(state, trace, selected, document, diagnostic):
    bindings = state.get("_evidence_winning_bindings", trace.get("bindings", {}))
    branches = state.get("_evidence_beams") or trace.get("branch_scores", [])
    retained = set()
    if not isinstance(bindings, dict) or not isinstance(branches, list):
        diagnostic["source_trace_error"] = "invalid_bindings_or_branches"
        return retained
    for branch in branches:
        if not isinstance(branch, dict) or not isinstance(branch.get("bindings"), dict):
            continue
        if (set(branch["bindings"]) != set(bindings)
                or not branch_matches(branch["bindings"], bindings)):
            continue
        proofs = branch.get("proofs", {})
        if not isinstance(proofs, dict):
            continue
        for node, proof in proofs.items():
            if not isinstance(proof, dict):
                continue
            doc_id = _doc_id(proof.get("doc_id"))
            check = {"node": str(node), "doc_id": doc_id, "retained": False}
            reason = None
            if doc_id not in selected:
                reason = "source_outside_original_selected_prefix"
            elif node not in bindings or _norm(proof.get("answer", "")) != _norm(bindings[node]):
                reason = "proof_not_consistent_with_winning_binding"
            elif not isinstance(proof.get("requirements", {}), dict) or not branch_matches(
                    proof.get("requirements", {}), bindings):
                reason = "proof_not_consistent_with_winning_branch"
            else:
                # Citation grounding, not new confidence/type/relation tests.
                # Existing accepted confidence is intentionally not thresholded.
                span = _quote_span(proof.get("evidence"), str(document(doc_id)))
                if span is None:
                    reason = "existing_source_quote_not_literal_grounded"
                else:
                    retained.add(doc_id)
                    check.update(retained=True, source_quote=span,
                                 retention_basis="existing_winning_literal_source")
            if reason:
                check["reason"] = reason
            diagnostic["source_checks"].append(check)
    return retained


def support_guard(query, order, state, document, proposal_diagnostic):
    """Return whether one accepted proposal preserves observed baseline support."""
    original = [int(d) for d in order]
    trace = state.get("evidence_trace", {})
    selected = set(original[:PREFIX_K])
    plan = state.get("_evidence_plan", trace.get("plan", []))
    _, _, plan_error = _ancestors(plan) if isinstance(plan, list) else ({}, [], "invalid_plan")
    diag = {"enabled": True, "policy": "retain_existing_winning_support_and_question_anchors",
            "gold_labels_used": False, "extra_requests": 0, "semantic_relation_upgrade": False,
            "binding_correctness_guaranteed": False, "unmodeled_ancestor_completeness_guaranteed": False,
            "plan_node_count": len(plan) if isinstance(plan, list) else None, "plan_error": plan_error,
            "collapsed_plan_policy": "preserve_observed_support_without_inferred_hops",
            "source_checks": [], "anchor_checks": [], "allow": False, "veto_reason": None}
    sources = _winning_sources(state, trace, selected, document, diag)
    anchors = set()
    for doc_id in original[:PREFIX_K]:
        title = _title(str(document(doc_id)))
        if _specific(title) and _literal(title, query):
            # Equal titles never collapse distinct passage IDs: retain each
            # selected source that names the original question's document topic.
            anchors.add(doc_id)
            diag["anchor_checks"].append({"doc_id": doc_id, "title": title,
                                          "check": "literal_specific_question_title"})
    dag = _protected(trace, original)
    protected = sources | anchors | dag | set(original[:FIXED_K])
    diag.update(protected_source_doc_ids=sorted(sources), protected_anchor_doc_ids=sorted(anchors),
                protected_dag_doc_ids=sorted(dag), protected_doc_ids=sorted(protected))

    def veto(reason):
        diag["veto_reason"] = reason
        return False, diag

    if len(original) != len(set(original)):
        return veto("duplicate_ranking")
    promotions = proposal_diagnostic.get("promotions", []) if isinstance(proposal_diagnostic, dict) else []
    if not isinstance(promotions, list) or len(promotions) != 1 or not isinstance(promotions[0], dict):
        return veto("no_single_accepted_proposal")
    proposal = promotions[0]
    candidate, victim, root = (_doc_id(proposal.get(key)) for key in ("doc_id", "victim", "root_doc_id"))
    diag.update(candidate_doc_id=candidate, victim_doc_id=victim, root_doc_id=root)
    if candidate not in original[PREFIX_K:TOP_K] or victim not in original[FIXED_K:PREFIX_K]:
        return veto("proposal_outside_bounded_swap_scope")
    if root not in selected or root == victim:
        return veto("root_not_retained")
    if victim in sources:
        return veto("victim_is_existing_winning_support")
    if victim in anchors:
        return veto("victim_is_original_question_anchor")
    if victim in protected:
        return veto("victim_is_protected_document")
    diag["allow"] = True
    return True, diag
