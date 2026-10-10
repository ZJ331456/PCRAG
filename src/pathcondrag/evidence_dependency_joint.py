"""Select existing evidence sets with source-grounded dependency closure.

The legacy DAG selector remains the exact fallback.  This opt-in calculation
considers the same Top200 documents, makes no requests, and changes a prefix
only when it completes another terminal while preserving existing support.
"""
from __future__ import annotations

from functools import lru_cache
import math

from .evidence_dag_package import _collect_proofs, _proof_relation, rerank_dag_packages
from .evidence_dependency_scoring import _root_anchor
from .evidence_planning import relation_faithfulness_error
from .evidence_retrieval import branch_matches
from .evidence_selection import _ancestors


def _finite(value):
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _all_proof_alternatives(query, nodes, bindings, winning, candidates,
                            rank, document, diagnostic, proof_check):
    """Reuse the source checker without its three-alternative search cap.

    Each call sees at most three documents, so the existing collector cannot
    discard a fourth valid document.  Merging all batches retains every checked
    alternative; the selector below uses at most 64 node masks rather than a
    Cartesian product of passage alternatives.
    """
    def anchored_check(node, question, proof, source, answers):
        if not node.get("depends_on") and not _root_anchor(question, query):
            return None, "root_subject_not_in_original_question"
        return (proof_check or _proof_relation)(node, question, proof, source, answers)

    merged = {nid: {} for nid in nodes}
    candidate_ids = list(candidates)
    winning_proofs = winning.get("proofs", {})
    for nid, proof in winning_proofs.items():
        if not isinstance(proof, dict):
            continue
        try:
            doc_id = int(proof.get("doc_id"))
        except (ValueError, TypeError, OverflowError):
            diagnostic["rejected_proofs"].append({"node": nid, "reason": "invalid_proof_fields"})
            continue
        if doc_id not in candidates:
            diagnostic["rejected_proofs"].append({
                "node": nid, "doc_id": doc_id, "reason": "proof_outside_existing_top200_pool"})
    for offset in range(0, len(candidate_ids), 3):
        batch_ids = set(candidate_ids[offset:offset + 3])
        batch_candidates = {doc_id: candidates[doc_id] for doc_id in batch_ids}
        batch_proofs = {}
        for nid, proof in winning_proofs.items():
            if not isinstance(proof, dict):
                continue
            try:
                doc_id = int(proof.get("doc_id"))
            except (ValueError, TypeError, OverflowError):
                continue
            if doc_id in batch_ids:
                batch_proofs[nid] = proof
        choices = _collect_proofs(
            nodes, bindings, dict(winning, proofs=batch_proofs), batch_candidates,
            rank, document, diagnostic, anchored_check)
        for nid, alternatives in choices.items():
            for proof in alternatives:
                merged[nid][proof["doc_id"]] = proof
    return {nid: sorted(alternatives.values(), key=lambda p: (rank[p["doc_id"]], p["doc_id"]))
            for nid, alternatives in merged.items() if alternatives}


def _node_masks(node_order, ancestors, choices):
    bit = {nid: 1 << index for index, nid in enumerate(node_order)}
    ancestor_masks = {nid: sum(bit[parent] for parent in ancestors[nid]) for nid in node_order}
    by_document = {}
    for nid, alternatives in choices.items():
        for proof in alternatives:
            doc_id = proof["doc_id"]
            by_document[doc_id] = by_document.get(doc_id, 0) | bit[nid]

    @lru_cache(maxsize=64)
    def closed(raw_mask):
        covered = 0
        for nid in node_order:
            if raw_mask & bit[nid] and ancestor_masks[nid] & covered == ancestor_masks[nid]:
                covered |= bit[nid]
        return covered

    def mask_for(documents):
        mask = 0
        for doc_id in documents:
            mask |= by_document.get(doc_id, 0)
        return mask

    return bit, by_document, closed, mask_for


def _optimize_set(order, size, fixed_count, node_order, roots, leaves, choices,
                  bit, by_document, closed, mask_for, main_signal):
    """Exact bounded DP over document count and the plan's raw support mask.

    A plan has at most six nodes, hence at most 64 masks.  Alternatives with
    identical masks are interchangeable for closure; additive stability costs
    choose their representative without dropping any potentially better mask.
    """
    size = min(size, len(order))
    rank = {doc_id: index + 1 for index, doc_id in enumerate(order)}
    old_set = set(order[:size])
    supported = set(by_document)
    protected = old_set & supported
    fixed = order[:min(fixed_count, size)]
    required = set(fixed) | protected
    before = closed(mask_for(old_set))
    leaf_mask = sum(bit[nid] for nid in leaves)
    root_mask = sum(bit[nid] for nid in roots)
    baseline_terminals = before & leaf_mask

    # Each record is (added IDs, changed-page count, rank sum, main signal sum).
    initial = ((), 0, 0, 0.0)
    states = {(len(required), mask_for(required)): initial}

    def stability(record):
        selected, changes, ranks, signal = record
        return (-changes, -ranks, signal, tuple(-rank[d] for d in selected))

    for doc_id in order[:200]:
        if doc_id in required:
            continue
        for (count, mask), record in list(states.items()):
            if count >= size:
                continue
            selected, changes, ranks, signal = record
            key = (count + 1, mask | by_document.get(doc_id, 0))
            candidate = (selected + (doc_id,), changes + (doc_id not in old_set),
                         ranks + rank[doc_id], signal + main_signal.get(doc_id, 0.0))
            previous = states.get(key)
            if previous is None or stability(candidate) > stability(previous):
                states[key] = candidate

    best = None
    qualifying = 0
    for (count, mask), record in states.items():
        if count != size:
            continue
        covered = closed(mask)
        terminals = covered & leaf_mask
        # Existing closed nodes and terminals remain supported.  A root or an
        # unfinished chain alone cannot justify any new passage promotion.
        if covered & before != before or terminals.bit_count() <= baseline_terminals.bit_count():
            continue
        qualifying += 1
        objective = (terminals.bit_count() / max(1, len(leaves)),
                     covered.bit_count() / max(1, len(node_order)), *stability(record))
        if best is None or objective > best[0]:
            best = (objective, required | set(record[0]), covered)

    after, selected_set = before, old_set
    if best is not None:
        _, selected_set, after = best
    prefix = fixed + [doc_id for doc_id in order if doc_id in selected_set and doc_id not in set(fixed)]
    final = prefix + [doc_id for doc_id in order if doc_id not in selected_set]
    changed = selected_set != old_set
    if not changed:
        final = list(order)

    def names(mask):
        return [nid for nid in node_order if mask & bit[nid]]

    diagnostic = {
        "top_k": size, "fixed_doc_ids": fixed, "protected_doc_ids": sorted(protected),
        "states_examined": len(states), "qualifying_terminal_improvements": qualifying,
        "coverage_before": names(before), "coverage_after": names(after),
        "root_support_before": names(before & root_mask), "root_support_after": names(after & root_mask),
        "complete_terminals_before": names(baseline_terminals),
        "complete_terminals_after": names(after & leaf_mask),
        "node_coverage_before": before.bit_count() / max(1, len(node_order)),
        "node_coverage_after": after.bit_count() / max(1, len(node_order)),
        "terminal_coverage_before": baseline_terminals.bit_count() / max(1, len(leaves)),
        "terminal_coverage_after": (after & leaf_mask).bit_count() / max(1, len(leaves)),
        "set_changed": changed,
        "added_doc_ids": sorted(selected_set - old_set), "removed_doc_ids": sorted(old_set - selected_set),
        "fallback": None if changed else "no_strict_terminal_coverage_gain",
    }
    return final, diagnostic


def rerank_dependency_joint(query, order, state, document, signal_ids=None,
                            signal_scores=None, proof_check=None, legacy_proof_check=None):
    """Return a ranking, the exact legacy DAG diagnostic, and joint audit data."""
    original = [int(doc_id) for doc_id in order]
    cached_document = lru_cache(maxsize=None)(document)
    # Malformed collections cannot provide any proof.  A shallow fallback copy
    # lets the unchanged legacy selector report an empty/missing proof state;
    # all normal states are passed to it unchanged.
    trace = state.get("evidence_trace", {})
    trace = trace if isinstance(trace, dict) else {}
    branches = state.get("_evidence_beams") or trace.get("branch_scores", [])
    invalid_collection = None
    fallback_state = state
    if not isinstance(branches, list):
        invalid_collection = "invalid_winning_branches"
        fallback_state = dict(state, _evidence_beams=[],
                              evidence_trace=dict(trace, branch_scores=[]))
    if not isinstance(state.get("evidence_candidates", {}), dict):
        invalid_collection = "invalid_evidence_candidates"
        fallback_state = dict(fallback_state, evidence_candidates={})
    legacy, legacy_diagnostic = rerank_dag_packages(
        original, fallback_state, cached_document, legacy_proof_check)
    diagnostic = {
        "enabled": True, "mode": "dependency_joint",
        "policy": "complete_terminal_closure_with_fixed_plan_denominators",
        "gold_labels_used": False, "extra_requests": 0, "extra_llm_requests": 0,
        "extra_embedding_calls": 0, "entailment_guaranteed": False,
        "fallback": None, "plan_error": None, "rejected_proofs": [],
        "eligible_proofs": {}, "proof_alternative_count": 0,
        "prefix_optimization": [], "protected_top5_doc_ids": [],
        "node_denominator": 0, "terminal_denominator": 0,
        "legacy_top10": legacy[:10], "scored_top10": legacy[:10],
        "top2_preserved": True, "top200_set_preserved": True,
        "top5_set_preserved": True, "top10_set_preserved": True,
        "document_set_preserved": True, "unique_documents": True,
        "tail_relative_order_preserved": True,
    }
    if invalid_collection:
        diagnostic.update(plan_error=invalid_collection, fallback=invalid_collection)
        return legacy, legacy_diagnostic, diagnostic
    plan = state.get("_evidence_plan", trace.get("plan", []))
    bindings = state.get("_evidence_winning_bindings", trace.get("bindings", {}))
    if (not isinstance(plan, list) or not isinstance(bindings, dict)
            or any(not isinstance(n, dict) or not isinstance(n.get("question"), str)
                   or not n["question"].strip() for n in plan)):
        diagnostic.update(plan_error="invalid_plan_or_bindings", fallback="invalid_plan_or_bindings")
        return legacy, legacy_diagnostic, diagnostic
    ancestors, node_order, error = _ancestors(plan)
    if error or not node_order or len(node_order) > 6:
        diagnostic.update(plan_error=error or "empty_or_out_of_budget_plan",
                          fallback="invalid_plan_or_bindings")
        return legacy, legacy_diagnostic, diagnostic
    depth = {}
    for nid in node_order:
        deps = next(node.get("depends_on", []) for node in plan if node["id"] == nid)
        depth[nid] = 1 + max((depth[parent] for parent in deps), default=0)
    if max(depth.values(), default=0) > 4:
        diagnostic.update(plan_error="out_of_budget_plan_depth", fallback="invalid_plan_or_bindings")
        return legacy, legacy_diagnostic, diagnostic
    error = relation_faithfulness_error(query, plan)
    if error:
        diagnostic.update(plan_error=error, fallback="unfaithful_plan")
        return legacy, legacy_diagnostic, diagnostic
    nodes = {node["id"]: node for node in plan}
    roots = {nid for nid in node_order if not nodes[nid].get("depends_on")}
    parents = {parent for node in plan for parent in node.get("depends_on", [])}
    leaves = set(node_order) - parents
    diagnostic.update(node_denominator=len(node_order), terminal_denominator=len(leaves),
                      root_nodes=sorted(roots), terminal_nodes=sorted(leaves))
    winning = next((branch for branch in branches if isinstance(branch, dict)
                    and isinstance(branch.get("bindings"), dict)
                    and set(branch["bindings"]) == set(bindings)
                    and branch_matches(branch["bindings"], bindings)), None)
    if not winning or not isinstance(winning.get("proofs"), dict):
        diagnostic.update(plan_error="missing_exact_winning_branch", fallback="missing_winning_branch")
        return legacy, legacy_diagnostic, diagnostic
    candidates = state.get("evidence_candidates", {})
    choices = _all_proof_alternatives(
        query, nodes, bindings, winning, candidates,
        {doc_id: index + 1 for index, doc_id in enumerate(legacy)},
        cached_document, diagnostic, proof_check)
    diagnostic["eligible_proofs"] = choices
    diagnostic["proof_alternative_count"] = sum(len(alternatives) for alternatives in choices.values())
    if not choices:
        diagnostic["fallback"] = "no_locally_supported_dependency_goals"
        return legacy, legacy_diagnostic, diagnostic
    bit, by_document, closed, mask_for = _node_masks(node_order, ancestors, choices)
    signal_ids = legacy if signal_ids is None else signal_ids
    signal_scores = [0.0] * len(signal_ids) if signal_scores is None else signal_scores
    values = [_finite(value) for value in signal_scores]
    low = min(values, default=0.0)
    span = max(values, default=0.0) - low
    main_signal = {int(doc_id): (value - low) / span if span else 0.0
                   for doc_id, value in zip(signal_ids, values)}
    final = legacy
    for size, fixed in ((5, 2), (10, 5)):
        final, summary = _optimize_set(
            final, size, fixed, node_order, roots, leaves, choices,
            bit, by_document, closed, mask_for, main_signal)
        diagnostic["prefix_optimization"].append(summary)
    covered = closed(mask_for(final[:5]))
    for nid in node_order:
        if covered & bit[nid]:
            selected = next((proof["doc_id"] for proof in choices[nid] if proof["doc_id"] in final[:5]), None)
            if selected is not None:
                diagnostic["protected_top5_doc_ids"].append(selected)
    diagnostic["protected_top5_doc_ids"] = sorted(set(diagnostic["protected_top5_doc_ids"]))
    diagnostic.update(
        scored_top10=final[:10], top2_preserved=final[:2] == legacy[:2],
        top200_set_preserved=set(final[:200]) == set(legacy[:200]),
        top5_set_preserved=set(final[:5]) == set(legacy[:5]),
        top10_set_preserved=set(final[:10]) == set(legacy[:10]),
        document_set_preserved=set(final) == set(legacy), unique_documents=len(final) == len(set(final)),
        tail_relative_order_preserved=final[10:] == [doc_id for doc_id in legacy if doc_id not in set(final[:10])])
    if not all(diagnostic[key] for key in ("top2_preserved", "top200_set_preserved",
                                           "document_set_preserved", "unique_documents",
                                           "tail_relative_order_preserved")):
        diagnostic["fallback"] = "ranking_invariant_failure"
        return legacy, legacy_diagnostic, diagnostic
    if final == legacy:
        diagnostic["fallback"] = "no_strict_terminal_coverage_gain"
    return final, legacy_diagnostic, diagnostic
