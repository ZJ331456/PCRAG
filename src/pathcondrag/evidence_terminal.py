"""Bounded terminal citation selection inside an existing winning evidence chain.

This is a local postprocessor: no planner, search, embedding, or LLM calls are
issued. Supported English constructions are checked again against their quoted
passage. Unknown constructions require an explicit subject, answer, and matched
relation cue in one clause; none of these checks imply general entailment.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .evidence_binding import (
    _attribute_reason, _same_entity, _typed_answer_reason, supported_entity_surface,
)
from .evidence_relation_guard import relation_spec, strict_relation_reason
from .evidence_retrieval import bind_question, branch_matches, normalized_text
from .evidence_selection import _ancestors


def _plain_entity(value):
    return re.sub(r"^(?:the|a|an)\s+", "", str(value).strip(), flags=re.I).rstrip("?. ")


def _literal(value, text):
    value = normalized_text(value)
    return bool(value) and " " + value + " " in " " + normalized_text(text) + " "


def _same_subject(left, right, document):
    return _same_entity(_plain_entity(left), _plain_entity(right), document)


def _subject_surface(entity, text, document):
    return supported_entity_surface(_plain_entity(entity), text, document)


def _question_entities(question):
    """Literal proper-name anchors; category descriptions are not new aliases."""
    names = re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*", question)
    ignored = {"Who", "What", "Which", "Where", "When", "How", "Is", "Was", "The", "A", "At"}
    return [n.rstrip("?.") for n in names if n.rstrip("?.") not in ignored]


def _relation_check(node, question, proof, document, bindings):
    """Return a check name, or a rejection reason with no semantic upgrade."""
    answer, quote = str(proof["answer"]), str(proof["evidence"])
    answer_type = node.get("answer_type", "")
    typed_reason = _typed_answer_reason(answer, answer_type)
    if typed_reason:
        return None, typed_reason
    # A present-but-wrong date is different from an absent date. Keep this
    # bounded type test for the common planner label "time" as well.
    if normalized_text(answer_type) in {"time", "year", "date"} and re.search(
            r"\b(?:born|birth|died|death|year|date)\b", question, re.I) and not re.search(r"\d", answer):
        return None, "answer_type_mismatch"
    checked_question = re.sub(r"\s+on\s+the\s+world\s+map\s*[?.]*$", "?", question, flags=re.I)
    spec = relation_spec(checked_question)
    if spec["relation"] == "comparison":
        return None, "comparison_requires_atomic_evidence"
    deps = node.get("depends_on", [])
    entity = spec.get("subject")
    if entity:
        entity = _plain_entity(entity)
    elif len(deps) == 1 and deps[0] in bindings:
        entity = _plain_entity(bindings[deps[0]])
    else:
        names = _question_entities(question)
        grounded = [n for n in names if _subject_surface(n, quote, document)[0]]
        entity = grounded[0] if len(grounded) == 1 else None
    if not entity:
        return None, "unresolved_quote_subject"
    subject = _subject_surface(entity, quote, document)[0]
    # A document topic licenses pronouns only inside that same document. The
    # existing personal attribute guard performs the actual pronoun check.
    if not subject and spec["relation"] not in {"nationality", "birth_date", "birth_place", "death_date"}:
        return None, "relationship_subject_not_in_quote"
    if spec["relation"] in {"nationality", "birth_date", "birth_place", "death_date"}:
        reason = _attribute_reason(spec["relation"], answer, entity, quote, quote, document)
        return ("local_personal_attribute", None) if reason is None else (None, reason)
    if spec["relation"] != "other":
        reason = strict_relation_reason(spec, question, answer, answer_type, entity, quote,
                                        document, _subject_surface, _same_subject)
        if reason is None:
            return "local_relation_construction", None
        if reason in {"relationship_subject_mismatch", "linked_subject_relationship_not_supported",
                      "answer_type_mismatch", "relationship_direction_conflict"}:
            return None, reason
        # Passive creator leads containing abbreviated titles (e.g. O.K.) can
        # be split by generic sentence heuristics. Match the complete quoted
        # owner before the passive marker, never an unrelated co-occurrence.
        markers = {"director": r"directed\s+by", "author": r"(?:written|authored)\s+by",
                   "composer": r"composed\s+by"}
        if spec["relation"] in markers:
            marker = re.search(markers[spec["relation"]], quote, re.I)
            if marker and _literal(entity, quote[:marker.start()]) and _literal(answer, quote[marker.end():]):
                owner = re.search(re.escape(entity), quote[:marker.start()], re.I)
                prefix = quote[:owner.start()].strip() if owner else ""
                if owner and not prefix and not re.search(r"\b(?:not|never)\b", quote, re.I):
                    return "local_passive_creator", None
        if spec["relation"] == "location" and subject:
            match = re.search(re.escape(subject) + r"\s+(?:is|was|lies)\s+(?:at|in|on)\b", quote, re.I)
            # The answer may include the owning preposition ("at the center"),
            # so retain it in the checked predicate rather than trimming it.
            if match and _literal(answer, quote[match.start() + len(subject):]) and not re.search(r"\b(?:not|never)\b", quote, re.I):
                return "local_location", None
        return None, reason
    # Unknown natural-language relations are not trusted on confidence alone.
    # Admit only explicit matched bounded cue families and one owning clause.
    families = [
        (r"\b(?:live|lives|reside|resides|home)\b", r"\b(?:live[sd]?|resides?|home)\b"),
        (r"\b(?:invented|invent|inventor)\b", r"\b(?:invented|inventor)\b"),
        (r"\b(?:married|spouse|wife|husband)\b", r"\b(?:married|spouse|wife|husband)\b"),
        (r"\b(?:played|portrayed|play|portray)\b", r"\b(?:played|portrayed)\b"),
        (r"\b(?:founded|established|created|built|opened)\b", r"\b(?:founded|established|created|built|opened)\b"),
    ]
    if re.search(r"\b(?:first|earlier|later|older|younger|same|longest|highest|largest)\b", question, re.I):
        return None, "non_atomic_or_comparative_unknown_relation"
    for query_cue, passage_cue in families:
        if not re.search(query_cue, question, re.I):
            continue
        for clause in re.split(r"(?<=[!?;])\s+|\n+|(?<=\.)\s+(?=[A-Z])", quote):
            if (len(clause) <= 450 and _subject_surface(entity, clause, document)[0]
                    and _literal(answer, clause) and re.search(passage_cue, clause, re.I)
                    and not re.search(r"\b(?:not|never|without)\b", clause, re.I)):
                surface = _subject_surface(entity, clause, document)[0]
                mention = re.search(re.escape(surface), clause, re.I)
                marker = re.search(passage_cue, clause, re.I)
                # Obvious relative ownership must not move a spouse's/father's
                # fact to the named entity merely because the names co-occur.
                if mention and (re.match(r"['’]s\b", clause[mention.end():]) or
                                re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\s+of\s*$",
                                          clause[:mention.start()], re.I)):
                    continue
                if mention and marker and mention.end() < marker.start() and re.search(
                        r"\b(?:father|mother|brother|sister|son|daughter)\b", clause[mention.end():marker.start()], re.I):
                    continue
                # All declared dependencies must occur in the quoted clause;
                # a multi-parent join cannot inherit an unrelated root's fact.
                if all(dep in bindings and _subject_surface(bindings[dep], clause, document)[0] for dep in deps):
                    return "matched_literal_relation_cue", None
        return None, "relation_cue_not_owned_by_quoted_subject"
    return None, "unknown_relation_not_promoted"


def rerank_terminal_evidence(order, state, document, mode="tail_only"):
    """Rerank existing IDs, returning the full order and auditable diagnostics.

    Prefix promotions are restricted to original ranks 6–10. Hence this local
    mode fixes Top2 and preserves the entire Top10/Top200 sets by construction.
    Tail mode can admit two cited documents from ranks 11–20, and exposes those
    set changes for empirical regression checks rather than claiming safety.
    """
    order = [int(d) for d in order]
    diag = {"enabled": True, "mode": mode, "policy": "complete_winning_terminal_citations",
            "max_prefix_promotions": 2 if mode == "prefix" else 0, "max_tail_promotions": 2,
            "gold_labels_used": False, "extra_requests": 0, "entailment_guaranteed": False,
            "promotions": [], "rejected_proofs": [], "deferred_chains": [],
            "top2_preserved": True, "top5_preserved": True, "top10_set_preserved": True,
            "top200_set_preserved": True, "document_set_preserved": True,
            "prefix_promotions": 0, "tail_promotions": 0}
    original = list(order)
    trace = state.get("evidence_trace", {})
    plan = state.get("_evidence_plan", trace.get("plan", []))
    bindings = state.get("_evidence_winning_bindings", trace.get("bindings", {}))
    ancestors, node_order, error = _ancestors(plan) if isinstance(plan, list) else ({}, [], "invalid_plan_type")
    if mode not in {"tail_only", "prefix"} or len(set(order)) != len(order):
        error = error or "invalid_mode_or_duplicate_ranking"
    if not isinstance(bindings, dict):
        error = error or "invalid_winning_bindings"
    diag["plan_error"] = error
    if error or not node_order:
        return original, diag
    nodes = {n["id"]: n for n in plan}
    candidates = state.get("evidence_candidates", {})
    branches = state.get("_evidence_beams") or trace.get("branch_scores", [])
    winning = next((b for b in branches if isinstance(b, dict) and isinstance(b.get("bindings"), dict)
                    and set(b["bindings"]) == set(bindings) and branch_matches(b["bindings"], bindings)), None)
    if not winning:
        diag["plan_error"] = "missing_exact_winning_branch"
        return original, diag
    rank = {d: i + 1 for i, d in enumerate(original)}
    choices = {nid: [] for nid in node_order}
    for nid, proof in (winning.get("proofs", {}).items() if isinstance(winning.get("proofs"), dict) else []):
        if nid in choices and isinstance(proof, dict):
            choices[nid].append(dict(proof))
    # Reuse an alternative citation only for the exact same winning answer and
    # compatible requirements. It never substitutes a losing answer binding.
    for doc_id, candidate in candidates.items():
        for proof in candidate.get("verified", []):
            goal = str(proof.get("goal", ""))
            if goal.startswith("dag:") and goal[4:] in choices:
                choices[goal[4:]].append(dict(proof, doc_id=doc_id))
    valid = {}
    for nid in node_order:
        checked, seen = [], set()
        for proof in choices[nid]:
            reason = None
            try:
                doc_id = int(proof.get("doc_id"))
                confidence = float(proof.get("confidence", 0.))
            except (TypeError, ValueError, OverflowError):
                diag["rejected_proofs"].append({"node": nid, "reason": "invalid_proof_fields"})
                continue
            answer, quote = str(proof.get("answer", "")), str(proof.get("evidence", ""))
            key = (doc_id, answer, quote)
            if key in seen:
                continue
            seen.add(key)
            requirements = proof.get("requirements", {})
            if doc_id not in candidates or doc_id not in rank:
                reason = "proof_document_outside_existing_pool"
            elif nid not in bindings or normalized_text(answer) != normalized_text(bindings[nid]):
                reason = "proof_conflicts_with_winning_binding"
            elif not isinstance(requirements, dict) or not branch_matches(requirements, bindings):
                reason = "proof_conflicts_with_winning_branch"
            elif not math.isfinite(confidence) or confidence < .85:
                reason = "low_or_invalid_confidence"
            elif len(quote) < 8 or quote not in document(doc_id):
                reason = "quote_not_grounded"
            elif not _literal(answer, quote):
                reason = "literal_answer_not_in_quote"
            bound_question = bind_question(nodes[nid].get("question", ""), bindings)
            check = None
            if not reason:
                if bound_question is None:
                    reason = "unbound_plan_question"
                else:
                    check, reason = _relation_check(nodes[nid], bound_question, proof, document(doc_id), bindings)
            if reason:
                diag["rejected_proofs"].append({"node": nid, "doc_id": doc_id, "reason": reason})
            else:
                checked.append({"doc_id": doc_id, "confidence": confidence, "check": check})
        if checked:
            valid[nid] = min(checked, key=lambda p: (rank[p["doc_id"]], -p["confidence"], p["doc_id"]))
    diag["reliable_proofs"] = valid
    children = {dep for n in plan for dep in n.get("depends_on", [])}
    leaves = [nid for nid in node_order if nid not in children and ancestors[nid]]
    protected = {p["doc_id"] for p in valid.values()}
    prefix_used = tail_used = 0
    for leaf in sorted(leaves, key=lambda nid: (rank.get(valid.get(nid, {}).get("doc_id"), len(order) + 1), nid)):
        required_nodes = [n for n in node_order if n in ancestors[leaf] or n == leaf]
        missing = [n for n in required_nodes if n not in valid]
        if missing:
            diag["deferred_chains"].append({"leaf": leaf, "reason": "missing_locally_supported_chain", "nodes": missing})
            continue
        terminal = valid[leaf]["doc_id"]
        bundle = list(dict.fromkeys(valid[n]["doc_id"] for n in required_nodes))
        already_selected = terminal in (order[:5] if mode == "prefix" else order[:6])
        if already_selected:
            continue
        if mode == "prefix":
            if rank[terminal] > 10 or not set(bundle) <= set(original[:10]):
                diag["deferred_chains"].append({"leaf": leaf, "reason": "prefix_preserves_existing_top10", "doc_ids": bundle})
                continue
            if prefix_used >= 2:
                diag["deferred_chains"].append({"leaf": leaf, "reason": "prefix_promotion_budget"})
                continue
            # Do not evict a locally supported node to replace it with another
            # node. Earlier winners and valid chain evidence remain protected.
            victim = next((d for d in reversed(order[2:5]) if d not in protected), None)
            if victim is None:
                diag["deferred_chains"].append({"leaf": leaf, "reason": "no_unprotected_top5_slot"})
                continue
            destination, source = order.index(victim), order.index(terminal)
            order[destination], order[source] = terminal, victim
            prefix_used += 1
        else:
            if rank[terminal] > 20 or not set(bundle) <= set(original[:10] + [terminal]):
                diag["deferred_chains"].append({"leaf": leaf, "reason": "tail_chain_or_rank_budget", "doc_ids": bundle})
                continue
            if tail_used >= 2:
                diag["deferred_chains"].append({"leaf": leaf, "reason": "tail_promotion_budget"})
                continue
            destination = 5 + tail_used
            victim = order[destination]
            # Moving a within-Top10 proof changes order only. An outside proof
            # may replace an unprotected tail document, never another valid node.
            if rank[terminal] > 10:
                victim = next((d for d in reversed(order[5:10]) if d not in protected), None)
                if victim is None or valid[leaf]["check"] == "matched_literal_relation_cue":
                    diag["deferred_chains"].append({"leaf": leaf, "reason": "outside_top10_requires_mechanical_relation_and_free_slot"})
                    continue
                # Keep every other old Top10 document before the displaced
                # victim; merely removing+inserting would evict rank10 blindly.
                source = order.index(terminal)
                order[order.index(victim)], order[source] = terminal, victim
            order.remove(terminal)
            order.insert(destination, terminal)
            tail_used += 1
        diag["promotions"].append({"leaf": leaf, "doc_id": terminal, "from_rank": rank[terminal],
                                    "to_rank": order.index(terminal) + 1, "victim": victim,
                                    "bundle_doc_ids": bundle, "check": valid[leaf]["check"],
                                    "reason": "missing_terminal_of_complete_winning_chain"})
    diag.update(top2_preserved=order[:2] == original[:2], top5_preserved=order[:5] == original[:5],
                top10_set_preserved=set(order[:10]) == set(original[:10]),
                top200_set_preserved=set(order[:200]) == set(original[:200]),
                document_set_preserved=set(order) == set(original), prefix_promotions=prefix_used,
                tail_promotions=tail_used)
    return order, diag


class TerminalEvidenceMixin:
    """Opt-in finalizer; an empty flag set preserves the exact parent result."""

    def finalize(self, query, ids, scores, ctx, state):
        result = super().finalize(query, ids, scores, ctx, state)
        if "terminal" not in self.improvements:
            return result
        original_ids, original_scores, original_trace = result
        final, diagnostic = rerank_terminal_evidence(original_ids, state, self._document,
                                                      getattr(self.cfg, "evidence_terminal_mode", "tail_only"))
        trace = dict(original_trace, improvement_terminal=diagnostic)
        if np.array_equal(np.asarray(final), original_ids):
            return original_ids, original_scores, trace
        old = {p["doc_id"]: p for p in trace.get("selected_prefix", [])}
        trace["selected_prefix"] = [dict(old.get(d, {"doc_id": d, "selection_source": "terminal_citation"}),
                                          original_greedy_rank=list(original_ids).index(d) + 1)
                                      for d in final[:5]]
        output_scores = np.arange(len(final), 0, -1, dtype=float) / max(1, len(final))
        return np.asarray(final, dtype=int), output_scores, trace
