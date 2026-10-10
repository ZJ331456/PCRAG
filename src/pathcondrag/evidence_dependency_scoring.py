"""Dependency-aware calibration of the existing evidence selection objective.

This is an alternative scoring calculation, not another retrieval stage. It
uses the existing plan, winning citations and ranking; it issues no model or
search requests. Unrecognized relations retain the incoming retrieval order.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

from .evidence_binding import _typed_answer_reason
from .evidence_dag_package import _proof_relation
from .evidence_planning import relation_faithfulness_error
from .evidence_relation_guard import relation_spec
from .evidence_retrieval import bind_question, branch_matches, normalized_text
from .evidence_selection import _ancestors


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def _literal(value, text):
    value = normalized_text(value)
    return bool(value) and f" {value} " in f" {normalized_text(text)} "


def scoring_input_sha256(query, ids, scores, state):
    """Fingerprint only scorer inputs, excluding results and benchmark labels."""
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    trace = state.get("evidence_trace", {})
    payload = {"query": query, "ids": ids, "scores": scores,
               "candidates": state.get("evidence_candidates", {}),
               "plan": state.get("_evidence_plan", trace.get("plan", [])),
               "bindings": state.get("_evidence_winning_bindings", trace.get("bindings", {})),
               "branches": state.get("_evidence_beams", trace.get("branch_scores", []))}
    encoded = json.dumps(clean(payload), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _goal(question):
    """Merge only the same concrete subject and recognized relation.

    Unknown constructions merge only when their full bound questions match.
    Qualifiers inside the subject are retained; token similarity never merges
    different entities or predicates.
    """
    spec = relation_spec(question)
    if spec.get("subject") and spec["relation"] not in {"other", "comparison"}:
        return f"{spec['relation']}::{normalized_text(spec['subject'])}"
    return "question::" + normalized_text(question)


def _root_anchor(question, query):
    subject = relation_spec(question).get("subject")
    if subject:
        return _literal(subject, query)
    names = re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*", question)
    ignored = {"Who", "What", "Which", "Where", "When", "How", "Is", "Was", "The", "A", "At", "In"}
    return any(_literal(name, query) for name in names if name not in ignored)


def _proofs(query, state, candidates, allowed, document, diagnostic):
    """Recheck winning literals and existing local predicates, then ancestors."""
    trace = state.get("evidence_trace", {})
    plan = state.get("_evidence_plan", trace.get("plan", []))
    if not isinstance(plan, list):
        diagnostic["plan_error"] = "invalid_plan"
        return {}, {}, [], {}
    if any(not isinstance(node, dict)
           or not isinstance(node.get("question"), str)
           or not node["question"].strip() for node in plan):
        diagnostic["plan_error"] = "invalid_plan_question"
        return {}, {}, [], {}
    ancestors, ordered, error = _ancestors(plan)
    error = error or relation_faithfulness_error(query, plan)
    diagnostic["plan_error"] = error
    if error or not ordered:
        return {}, {}, [], {}
    nodes = {node["id"]: node for node in plan}
    bindings = state.get("_evidence_winning_bindings", trace.get("bindings", {}))
    if not isinstance(bindings, dict):
        diagnostic["plan_error"] = "invalid_winning_bindings"
        return {}, {}, [], {}
    choices = {nid: [] for nid in ordered}
    branches = state.get("_evidence_beams") or trace.get("branch_scores", [])
    if not isinstance(branches, list):
        diagnostic["plan_error"] = "invalid_winning_branches"
        return {}, {}, [], {}
    for branch in branches:
        if (not isinstance(branch, dict) or not isinstance(branch.get("bindings"), dict)
                or set(branch["bindings"]) != set(bindings)
                or not branch_matches(branch["bindings"], bindings)):
            continue
        branch_proofs = branch.get("proofs")
        if not isinstance(branch_proofs, dict):
            diagnostic["plan_error"] = "invalid_winning_proofs"
            return {}, {}, [], {}
        for nid, proof in branch_proofs.items():
            if nid in choices and isinstance(proof, dict):
                choices[nid].append(proof)
        break
    for doc_id, candidate in candidates.items():
        if doc_id not in allowed or not isinstance(candidate, dict):
            continue
        for proof in candidate.get("verified", []):
            if not isinstance(proof, dict):
                continue
            goal = str(proof.get("goal", ""))
            if goal.startswith("dag:") and goal[4:] in choices:
                choices[goal[4:]].append(dict(proof, doc_id=doc_id))

    valid = {}
    for nid in ordered:
        node, seen, supported = nodes[nid], set(), []
        question = bind_question(node.get("question", ""), bindings)
        for proof in choices[nid]:
            try:
                doc_id = int(proof.get("doc_id"))
            except (ValueError, TypeError, OverflowError):
                diagnostic["rejected_proofs"].append({"node": nid, "reason": "invalid_document_id"})
                continue
            answer, quote = str(proof.get("answer", "")), str(proof.get("evidence", ""))
            if (doc_id, answer, quote) in seen:
                continue
            seen.add((doc_id, answer, quote))
            reason, check = None, None
            requirements = proof.get("requirements", {})
            if doc_id not in allowed or doc_id not in candidates:
                reason = "proof_outside_existing_top200_candidates"
            elif nid not in bindings or normalized_text(answer) != normalized_text(bindings[nid]):
                reason = "proof_conflicts_with_winning_binding"
            elif not isinstance(requirements, dict) or not branch_matches(requirements, bindings):
                reason = "proof_conflicts_with_winning_branch"
            elif question is None:
                reason = "unbound_node_question"
            elif not node.get("depends_on") and not _root_anchor(question, query):
                reason = "root_subject_not_in_original_question"
            else:
                source = str(document(doc_id))
                if len(quote) < 8 or len(quote) > 1500 or quote not in source:
                    reason = "proof_quote_not_literal_grounded"
                elif not _literal(answer, quote):
                    reason = "answer_not_literal_in_quote"
                elif _typed_answer_reason(answer, node.get("answer_type", "")):
                    reason = "answer_type_conflict"
                elif re.search(r"\b(?:not|never|without)\b", quote, re.I):
                    reason = "negated_quote_not_supported"
                elif len(node.get("depends_on", [])) > 1 and not all(
                        _literal(bindings.get(dep, ""), quote) for dep in node["depends_on"]):
                    reason = "multi_parent_inputs_not_grounded"
                else:
                    check, reason = _proof_relation(node, question, proof, source, bindings)
            if reason or not check:
                diagnostic["rejected_proofs"].append({"node": nid, "doc_id": doc_id,
                                                       "reason": reason or "unknown_local_relation"})
            else:
                supported.append({"doc_id": doc_id, "goal": _goal(question), "check": check,
                                  "question": question})
        if supported:
            # Original retrieval order breaks ties, never self-reported confidence.
            valid[nid] = min(supported, key=lambda item: (allowed[item["doc_id"]], item["doc_id"]))
    complete = {}
    for nid in ordered:
        if nid not in valid:
            continue
        missing = ancestors[nid] - set(complete)
        if missing:
            diagnostic["rejected_proofs"].append({"node": nid, "doc_id": valid[nid]["doc_id"],
                                                   "reason": "missing_reliable_ancestor",
                                                   "missing_ancestors": sorted(missing)})
        else:
            complete[nid] = valid[nid]
    diagnostic["reliable_proofs"] = complete
    return complete, ancestors, ordered, bindings


def dependency_scored_prefix(query, ids, scores, state, config, document,
                             signal_ids=None, signal_scores=None):
    """Choose a prefix inside the frozen legacy output and its Top200 set.

    Coverage is credited only when the selected documents actually contain a
    locally supported node and all its ancestors. Unverified route rankings
    provide at most a small tie preference, and cannot create coverage goals.
    """
    ids, scores = np.asarray(ids, dtype=int), np.asarray(scores, dtype=float)
    original = ids.tolist()
    diagnostic = {"mode": "dependency", "extra_llm_requests": 0, "extra_embedding_calls": 0,
                  "gold_labels_used": False, "plan_error": None, "reliable_proofs": {},
                  "rejected_proofs": [], "weak_route_documents": 0,
                  "merged_goal_count": 0, "input_count": len(original), "fallback": None,
                  "top200_set_preserved": True, "tail_relative_order_preserved": True}
    if len(original) != len(set(original)) or len(scores) != len(original):
        diagnostic["fallback"] = "invalid_incoming_ranking"
        return ids, scores, [], diagnostic
    candidates = state.get("evidence_candidates", {})
    allowed = {doc_id: rank for rank, doc_id in enumerate(original[:200])}
    proofs, ancestors, node_order, bindings = _proofs(
        query, state, candidates, allowed, document, diagnostic)
    if not proofs:
        diagnostic["fallback"] = "no_locally_supported_dependency_goals"
        return ids, scores, [], diagnostic
    trusted_goals = {item["goal"] for item in proofs.values()}
    diagnostic["merged_goal_count"] = len(trusted_goals)
    signal_ids = np.asarray(ids if signal_ids is None else signal_ids, dtype=int)
    signal_scores = np.asarray(scores if signal_scores is None else signal_scores, dtype=float)
    if len(signal_ids) != len(signal_scores):
        diagnostic["fallback"] = "invalid_main_query_signal"
        return ids, scores, [], diagnostic
    finite = np.asarray([_finite(value) for value in signal_scores])
    span = float(finite.max() - finite.min()) if len(finite) else 0.0
    norm = (finite - finite.min()) / span if span > 0 else np.ones(len(finite))
    main_base = dict(zip(signal_ids.tolist(), norm.tolist()))
    base = {doc_id: main_base.get(doc_id, 0.0) for doc_id in original}
    diagnostic["base_signal_scope"] = "original_main_ranking_before_legacy_evidence_selection"
    weak, goals_by_doc, proof_goals_by_doc = {}, {}, {}
    for doc_id, candidate in candidates.items():
        if doc_id not in allowed:
            continue
        for route in candidate.get("sources", []):
            requirements = route.get("requirements", {})
            if not isinstance(requirements, dict) or not branch_matches(requirements, bindings):
                continue
            goal = _goal(str(route.get("question", "")))
            if goal not in trusted_goals:
                continue
            rank = max(1.0, _finite(route.get("rank"), 100.0))
            # Raw cosine and min-max scores are not comparable across routes.
            weak[doc_id] = max(weak.get(doc_id, 0.0), 0.04 / (1.0 + 0.25 * rank))
            goals_by_doc.setdefault(doc_id, set()).add(goal)
    for item in proofs.values():
        goals_by_doc.setdefault(item["doc_id"], set()).add(item["goal"])
        proof_goals_by_doc.setdefault(item["doc_id"], set()).add(item["goal"])
    diagnostic["weak_route_documents"] = len(weak)
    base_weight = max(0.6, _finite(getattr(config, "evidence_base_weight", 0.4), 0.4))
    relation_weight = max(0.0, _finite(getattr(config, "evidence_relation_weight", 0.25), 0.25))
    coverage_weight = max(0.0, _finite(getattr(config, "evidence_coverage_weight", 0.45), 0.45))
    redundancy_weight = max(0.0, _finite(getattr(config, "evidence_redundancy_weight", 0.15), 0.15))
    diagnostic["weights"] = {"base": base_weight, "relation": relation_weight,
                              "coverage": coverage_weight, "same_goal_redundancy": redundancy_weight,
                              "unverified_route_maximum": 0.032}

    def covered(selected):
        nodes = set()
        for nid in node_order:
            if (nid in proofs and proofs[nid]["doc_id"] in selected
                    and ancestors[nid] <= nodes):
                nodes.add(nid)
        return nodes, {proofs[nid]["goal"] for nid in nodes}

    tokens = {}

    def redundancy(doc_id, selected):
        # A passage may be retrieved for both parent and child questions. Route
        # co-occurrence does not prove that it duplicates either relation.
        shared = [previous for previous in selected if proof_goals_by_doc.get(doc_id, set())
                  & proof_goals_by_doc.get(previous, set())]
        if not shared:
            return 0.0
        for item in [doc_id] + shared:
            if item not in tokens:
                tokens[item] = set(normalized_text(document(item)).split())
        return max(len(tokens[doc_id] & tokens[p]) / max(1, len(tokens[doc_id] | tokens[p]))
                   for p in shared)

    selected, details = [], []
    remaining = set(allowed)
    budget = min(max(1, int(getattr(config, "evidence_budget", 5))), len(remaining))
    for position in range(budget):
        old_nodes, old_goals = covered(set(selected))
        ranked, positive = [], False
        for doc_id in remaining:
            new_nodes, new_goals = covered(set(selected) | {doc_id})
            gain = len(new_goals - old_goals) / max(1, len(trusted_goals))
            positive |= gain > 0
            relation = 1.0 if gain > 0 else 0.0
            duplicate = redundancy(doc_id, selected)
            utility = (base_weight * base[doc_id] + relation_weight * relation
                       + coverage_weight * gain + weak.get(doc_id, 0.0)
                       - redundancy_weight * duplicate)
            ranked.append((utility, -allowed[doc_id], doc_id, gain, duplicate,
                           sorted(new_nodes - old_nodes)))
        if positive:
            utility, _, chosen, gain, duplicate, new_nodes = max(ranked)
        else:
            chosen = min(remaining, key=allowed.get)
            utility, _, _, gain, duplicate, new_nodes = next(row for row in ranked if row[2] == chosen)
        selected.append(chosen)
        remaining.remove(chosen)
        details.append({"doc_id": chosen, "utility": float(utility), "coverage_gain": gain,
                        "redundancy": duplicate, "original_rank": allowed[chosen] + 1,
                        "goals": {goal: 1.0 for goal in sorted(goals_by_doc.get(chosen, set()))},
                        "new_supported_nodes": new_nodes,
                        "selection_basis": "dependency_objective" if positive else "base_order_fallback",
                        "verified": candidates.get(chosen, {}).get("verified", [])})
    selected_set = set(selected)
    final = selected + [doc_id for doc_id in original if doc_id not in selected_set]
    diagnostic["top200_set_preserved"] = set(final[:200]) == set(original[:200])
    diagnostic["tail_relative_order_preserved"] = final[len(selected):] == [
        doc_id for doc_id in original if doc_id not in selected_set]
    if not diagnostic["top200_set_preserved"]:
        diagnostic["fallback"] = "ranking_invariant_failure"
        return ids, scores, [], diagnostic
    diagnostic["covered_nodes"], final_goals = covered(selected_set)
    diagnostic["covered_nodes"] = sorted(diagnostic["covered_nodes"])
    diagnostic["covered_goals"] = sorted(final_goals)
    if final == original:
        return ids, scores, details, diagnostic
    # Encode the unchanged legacy tail order even if its scores use a different
    # scale from the incoming main-query retrieval signal.
    final_scores = [2.0 + (len(selected) - rank) / max(1, len(selected))
                    for rank in range(len(selected))] + [
                        (len(final) - rank) / max(1, len(final))
                        for rank in range(len(selected), len(final))]
    return np.asarray(final, dtype=int), np.asarray(final_scores, dtype=float), details, diagnostic
