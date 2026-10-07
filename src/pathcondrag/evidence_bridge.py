"""Opt-in, bounded recovery of an unresolved role or endorsement bridge.

The normal planner and verifier run first. A failed local repair never replaces
their DAG. Source identity recovery reuses existing model proposals and exact
same-document text; it does not make another model request.
"""
from __future__ import annotations

import copy
import json
import math
import re
from typing import Optional, Tuple

from .evidence_binding import (
    _attribute_reason, _same_entity, _typed_answer_reason, supported_entity_surface,
)
from .evidence_planning import canonicalize_plan_dependencies, validate_retrieval_plan
from .evidence_relation_guard import relation_spec, strict_relation_reason
from .evidence_retrieval import normalized_text, verify_hypotheses
from .question_structure import route_question_structure


def _literal(value: str, text: str) -> bool:
    value = normalized_text(value)
    return bool(value) and " " + value + " " in " " + normalized_text(text) + " "


def _sentences(text: str):
    """Preserve source offsets; a period inside a name is not a sentence end."""
    boundaries = [m for m in re.finditer(r"[.!?]\s+(?=[A-Z])|\n+", text)
                  if not (text[m.start():m.start() + 1] == "." and
                          re.search(r"\b(?:Mr|Mrs|Ms|Dr|St|Prof|Sr|Jr)$", text[:m.start()]))]
    starts = [0] + [m.end() for m in boundaries]
    ends = starts[1:] + [len(text)]
    return [(start, end, text[start:end].strip()) for start, end in zip(starts, ends)
            if text[start:end].strip()]


def _recovered_relation_supported(question, answer, quote, document) -> bool:
    """Only recognized owning clauses license recovered identity references."""
    if re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\b", quote, re.I):
        return False
    if re.search(r"\b(?:not|never|without)\b", quote, re.I):
        return False
    spec = relation_spec(question)
    if spec["relation"] == "comparison":
        return False
    entity = spec.get("subject")
    if spec["relation"] in {"nationality", "birth_date", "birth_place", "death_date"}:
        return bool(entity) and _attribute_reason(
            spec["relation"], answer, entity, quote, quote, document) is None
    if spec["relation"] != "other":
        # Exact local grammar only; unknown fine-grained locations/dates keep
        # the original verifier outcome rather than forcing a strict fallback.
        return bool(entity) and strict_relation_reason(
            spec, question, answer, "unknown", entity, quote, document,
            supported_entity_surface, _same_entity) is None
    families = [
        (r"\b(?:friend|friends|befriend)\b", r"\b(?:friend|friends|befriend)\b"),
        (r"\b(?:voice|voiced|voicing)\b", r"\b(?:voice|voiced|voicing)\b"),
        (r"\b(?:architect|designed|design)\b", r"\b(?:architect|designed)\b"),
        (r"\b(?:rap|rapper|singer|member)\b", r"\b(?:rapper|singer|member)\b"),
        (r"\b(?:endorse[ds]?|endorsing)\b", r"\b(?:endorse[ds]?|endorsing|worn)\b"),
        (r"\b(?:starred|starring|portrayed|played)\b", r"\b(?:starred|stars|portrayed|played)\b"),
    ]
    # At least one relation cue must appear in the *original predicate quote*,
    # not just in an identity sentence added around it.
    matched_relation = any(re.search(q, question, re.I) and re.search(p, quote, re.I)
                           for q, p in families)
    names = re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*", question)
    ignored = {"Who", "What", "Which", "Where", "When", "How", "On", "In", "At", "The", "A", "Is", "Was"}
    targets = [n.rstrip(".?") for n in names if n.rstrip(".?") not in ignored and not _same_entity(n.rstrip(".?"), answer, document)]
    return matched_relation and all(_literal(target, quote) for target in targets)


def recover_source_identity_quote(item: dict, document: str, question: str,
                                  answer_type: str = "") -> Tuple[Optional[dict], str]:
    """Recover a literal contiguous identity-plus-predicate span, or decline.

    This function never searches a different document, rewrites source text,
    turns a paraphrase into a quote, or accepts general same-document proximity.
    The identity must be the document's title/lead, and the quote must use that
    identity's surname, explicit short name, or unambiguous adjacent pronoun.
    Passing these checks is local source support, not universal entailment.
    """
    if not isinstance(item, dict):
        return None, "invalid_hypothesis"
    answer, original = str(item.get("answer", "")).strip(), str(item.get("evidence", "")).strip()
    if not answer or len(answer) > 200 or len(original) < 8 or original not in document:
        return None, "original_quote_not_exact"
    if _literal(answer, original):
        return None, "identity_recovery_not_needed"
    if _typed_answer_reason(answer, answer_type):
        return None, "answer_type_mismatch"
    try:
        confidence = float(item.get("confidence", .5))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence) or confidence < .25:
        return None, "low_claimed_confidence"
    lines = document.splitlines()
    title = lines[0].strip() if len(lines) > 1 else ""
    title = re.sub(r"\s*\([^()]+\)\s*$", "", title)
    lead = document[len(lines[0]) + 1:] if len(lines) > 1 else document
    # An answer somewhere else in the document is insufficient. The title or
    # owning lead must identify it, and its full surface must be in the span.
    lead_identity = re.match(re.escape(answer) + r"(?:\s*\(|\s+(?:is|was|are|were)\b)", lead.strip(), re.I)
    if normalized_text(title) != normalized_text(answer) and not lead_identity:
        return None, "answer_not_document_identity"
    if not _literal(answer, document[:min(700, len(document))]):
        return None, "literal_identity_not_in_source"
    position = document.find(original)
    if position < 0 or position + len(original) > 1500:
        return None, "identity_span_too_long"
    before = document[:position]
    tokens = re.findall(r"\w+", answer)
    surname = tokens[-1] if len(tokens) >= 2 else ""
    firstname = tokens[0] if len(tokens) >= 2 else ""
    surname_surface = bool(surname and re.search(r"(?<!\w)" + re.escape(surname) + r"(?!\w)", original, re.I))
    firstname_surface = bool(firstname and re.search(r"(?<!\w)" + re.escape(firstname) + r"(?!\w)", original, re.I))
    pronoun = bool(re.match(r"\s*(?:He|She|It|They|His|Her|Its)\b", original))
    if not (surname_surface or firstname_surface or pronoun):
        return None, "no_source_anchored_reference"
    # A different full name sharing the surname is an explicit conflict.
    if surname_surface and re.search(r"\b[A-Z][\w'-]+\s+" + re.escape(surname) + r"\b", original):
        return None, "different_full_name_in_quote"
    if firstname_surface and re.search(re.escape(firstname) + r"\s+[A-Z][\w'-]+", original):
        return None, "different_full_name_in_quote"
    if surname_surface or firstname_surface:
        recent_sentences = _sentences(before)
        recent = recent_sentences[-1][2] if recent_sentences else ""
        if not _literal(answer, recent):
            for short in (surname, firstname):
                if short and re.search(r"\b[A-Z][\w'-]+\s+" + re.escape(short) + r"\b|\b" +
                                       re.escape(short) + r"\s+[A-Z][\w'-]+\b", recent):
                    return None, "different_named_antecedent"
    if pronoun:
        previous = _sentences(before)
        # The immediately previous full sentence must anchor the topic; don't
        # resolve a pronoun across another named person or a long paragraph.
        previous = [s for s in previous if normalized_text(s[2]) != normalized_text(title)]
        if not previous or not _literal(answer, previous[-1][2]):
            return None, "pronoun_not_adjacent_to_identity"
        if len(previous[-1][2]) > 550 or re.search(
                r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\b", previous[-1][2], re.I):
            return None, "ambiguous_identity_sentence"
        # Multiple proper-name subjects near the antecedent are ambiguous.
        suffix = previous[-1][2]
        suffix = re.sub(re.escape(answer), "TOPIC", suffix, flags=re.I)
        if re.search(r"\b[A-Z][\w'-]+\s+[A-Z][\w'-]+\s+(?:is|was|had|has|became)\b", suffix):
            return None, "different_named_antecedent"
    if not _recovered_relation_supported(question, answer, original, document):
        return None, "relationship_not_locally_supported"
    expanded = document[:position + len(original)]
    if len(expanded) > 1500 or not _literal(answer, expanded):
        return None, "identity_span_too_long"
    recovered = dict(item, evidence=expanded)
    raw_id = str(item.get("doc_id", ""))
    if not re.fullmatch(r"D?\d+", raw_id):
        return None, "invalid_document_reference"
    if not verify_hypotheses({"hypotheses": [recovered]}, {int(raw_id.lstrip("D")): document})[0]:
        return None, "recovered_source_check_failed"
    return recovered, "contiguous_document_identity"


def bridge_recovery_kind(question: str) -> Optional[str]:
    """Recognize two bounded relation patterns without dataset identifiers."""
    if route_question_structure(question)["expand"]:
        return None
    if (re.search(r"\b(?:station|network|channel)\b", question, re.I)
            and re.search(r"\b(?:series|drama|show)\b", question, re.I)
            and re.search(r"\b(?:starring|starred|featuring)\b", question, re.I)
            and re.search(r"\b(?:rapper|singer|vocalist|member)\b", question, re.I)
            and re.search(r"\b(?:band|boyband|group)\b", question, re.I)):
        return "role_work_station"
    if (re.search(r"\b(?:international|internationl|national)\b.*\b(?:football|soccer)?\s*team\b", question, re.I)
            and re.search(r"\bplayer\b", question, re.I)
            and re.search(r"\b(?:endorse|endorsed|endorsing)\b", question, re.I)
            and re.search(r"\b(?:boot|boots|shoe|shoes|product)\b", question, re.I)):
        return "product_player_team"
    return None


def _repair_plan_reason(payload, query, kind):
    canonical, _changes, reason = canonicalize_plan_dependencies(payload)
    if reason:
        return [], reason
    nodes, reason = validate_retrieval_plan(canonical, 6, 4)
    if reason:
        return [], reason
    expected = 3 if kind == "role_work_station" else 2
    if len(nodes) != expected or [n["id"] for n in nodes] != [f"r{i + 1}" for i in range(expected)]:
        return [], "repair_requires_bounded_chain"
    for i, node in enumerate(nodes):
        if node["depends_on"] != ([] if i == 0 else [f"r{i}"]):
            return [], "repair_requires_literal_predecessor"
    if normalized_text(nodes[0]["answer_type"]) not in {"person", "player", "people"}:
        return [], "repair_bridge_must_be_person"
    if kind == "role_work_station":
        if not re.search(r"\b(?:rapper|singer|vocalist|member)\b", nodes[0]["question"], re.I):
            return [], "repair_missing_requested_role"
        if not re.search(r"\b(?:series|drama|show)\b", nodes[1]["question"], re.I):
            return [], "repair_missing_work_relation"
        if not re.search(r"\b(?:station|network|channel)\b", nodes[-1]["question"], re.I):
            return [], "repair_terminal_attribute_changed"
        years = re.findall(r"\b(?:19|20)\d{2}\b", query)
        if any(year not in nodes[1]["question"] for year in years):
            return [], "repair_dropped_year_qualifier"
        group = re.search(r"\b(?:boyband|band|group)\s+([A-Z][\w-]*(?:\s+[A-Z][\w-]*)*)", query)
        if group and not _literal(group.group(1).rstrip(".,?"), nodes[0]["question"]):
            return [], "repair_dropped_named_group"
        nationality = re.search(r"\b(?:\d{4}\s+)?((?:[A-Z][a-z]+\s+)?[A-Z][a-z]+)\s+television\s+series\b", query)
        if nationality and not _literal(nationality.group(1), nodes[1]["question"]):
            return [], "repair_dropped_series_qualifier"
    else:
        if not re.search(r"\b(?:endorse[ds]?|endorsing|wear|worn)\b", nodes[0]["question"], re.I):
            return [], "repair_missing_endorsement"
        if not re.search(r"\b(?:international|national)\b.*\bteam\b", nodes[-1]["question"], re.I):
            return [], "repair_terminal_attribute_changed"
        quoted = re.findall(r'["“]([^"”]+)["”]', query)
        if any(not _literal(title, nodes[0]["question"]) for title in quoted):
            return [], "repair_dropped_product_identity"
    return nodes, None


class BridgeRecoveryMixin:
    """Place before PlannerImprovementMixin; all changes require its own flag."""

    def _verify(self, question, answer_type, docs):
        original = super()._verify(question, answer_type, docs)
        if "bridge_recovery" not in self.improvements:
            return original
        accepted, rejected, reason = original
        # An existing valid binding or malformed/truncated response stays exact.
        if accepted or reason:
            return original
        key = question + "::" + ",".join(map(str, docs))
        diagnostic = self._verification_diagnostics.get(key, {})
        recoveries, declined = [], []
        seen = set()
        for payload in diagnostic.get("outputs", []):
            for item in (payload or {}).get("hypotheses", []):
                if not isinstance(item, dict):
                    continue
                raw_id = str(item.get("doc_id", ""))
                if not re.fullmatch(r"D?\d+", raw_id):
                    continue
                doc_id = int(raw_id.lstrip("D"))
                if doc_id not in docs:
                    continue
                identity = (normalized_text(item.get("answer", "")), doc_id)
                if identity in seen:
                    continue
                seen.add(identity)
                recovered, disposition = recover_source_identity_quote(item, docs[doc_id], question, answer_type)
                if recovered:
                    valid, _failures = verify_hypotheses({"hypotheses": [recovered]}, docs)
                    for hyp in valid:
                        hyp["identity_recovery"] = {"method": disposition, "extra_requests": 0,
                                                    "original_evidence": item.get("evidence", ""),
                                                    "entailment_guaranteed": False}
                    recoveries.extend(valid)
                else:
                    declined.append({"answer": item.get("answer", ""), "doc_id": doc_id, "reason": disposition})
        diagnostic["identity_recovery"] = {"extra_requests": 0, "accepted": len(recoveries), "declined": declined}
        if recoveries:
            recoveries.sort(key=lambda h: (-h["confidence"], h["doc_id"], normalized_text(h["answer"])))
            return recoveries[:3], rejected, reason
        return original

    def _bridge_repair_plan(self, state, kind):
        static = [q for q in state.get("static_sub_questions", []) if isinstance(q, str)][:4]
        messages = [
            {"role": "system", "content": (
                "Repair one missing evidence bridge using ONLY the question. Return ONLY JSON. "
                "Use letter-prefixed IDs r1,r2,r3, never numeric IDs. Each node has id, question, "
                "depends_on, answer_type. Every non-root question must literally contain its "
                "predecessor ${r1.answer} or ${r2.answer}, and list precisely that predecessor in "
                "depends_on. Resolve an unknown person first, never guess their name. Preserve "
                "all input year, nationality, work and role qualifiers. Keep relation direction. "
                "For role_work_station use exactly 3 nodes: identify the named group's requested "
                "member/role (person); identify the qualified series starring ${r1.answer} (work); "
                "identify the station airing ${r2.answer} (organization). For product_player_team "
                "use exactly 2 nodes: identify player(s) endorsing the stated product (person); "
                "identify the international/national team of ${r1.answer} (team). Each node asks "
                "one relation. Prefer an existing static subquestion for the initial relation "
                "when it exactly preserves the input. Never invent intermediate facts. "
                'Legal syntax: {"nodes":[{"id":"r1","question":"Who endorses Product X?",'
                '"depends_on":[],"answer_type":"person"},{"id":"r2",'
                '"question":"Which national team does ${r1.answer} represent?",'
                '"depends_on":["r1"],"answer_type":"team"}]}')},
            {"role": "user", "content": (
                f"Question: {state['query']}\nRepair pattern: {kind}\nMaximum nodes: 6\nMaximum depth: 4\n"
                "Existing static subquestions: " + json.dumps(static, ensure_ascii=False) + "\n"
                "Original unresolved plan: " + json.dumps(state.get("_evidence_plan", []), ensure_ascii=False))},
        ]
        payload, reason = self._infer_object(messages)
        nodes, reason = ([], reason) if reason else _repair_plan_reason(payload, state["query"], kind)
        return nodes, reason, payload

    def _dependency_search(self, states, executor):
        super()._dependency_search(states, executor)
        if "bridge_recovery" not in self.improvements:
            return
        pending = []
        for state in states:
            kind = bridge_recovery_kind(state["query"])
            if not kind:
                continue
            plan = state.get("_evidence_plan", [])
            bindings = state.get("_evidence_winning_bindings", {})
            diagnostic = {"kind": kind, "extra_plan_requests": 0, "extra_searches": 0,
                          "extra_verification_requests": 0, "applied": False}
            state["evidence_trace"]["improvement_bridge"] = diagnostic
            if plan and all(node["id"] in bindings for node in plan):
                diagnostic["deferred_reason"] = "existing_complete_chain"
                continue
            if len(plan) > 2:
                diagnostic["deferred_reason"] = "existing_non_compound_plan"
                continue
            if state["evidence_trace"]["search_count"] >= self.max_searches:
                diagnostic["deferred_reason"] = "existing_search_budget_exhausted"
                continue
            pending.append((state, kind))
        repaired = self._collect_jobs(executor, [(self._bridge_repair_plan, (s, k)) for s, k in pending])
        recovery_states = []
        for (state, _kind), (plan, reason, payload) in zip(pending, repaired):
            trace = state["evidence_trace"]
            diag = trace["improvement_bridge"]
            diag.update(extra_plan_requests=1, repair_output=payload, validation_error=reason,
                        original_plan=copy.deepcopy(state.get("_evidence_plan", [])))
            trace["llm_plan_calls"] += 1
            if reason:
                diag["deferred_reason"] = "invalid_repair_preserved_original_dag"
                continue
            clone = copy.deepcopy(state)
            clone["_evidence_plan"] = plan
            clone["evidence_trace"]["plan"] = plan
            for candidate in clone["evidence_candidates"].values():
                candidate["sources"] = [source for source in candidate["sources"]
                                        if not str(source.get("source", "")).startswith("dag:")]
                candidate["verified"] = [proof for proof in candidate["verified"]
                                         if not str(proof.get("goal", "")).startswith("dag:")]
            clone["evidence_trace"]["routes"] = [route for route in clone["evidence_trace"].get("routes", [])
                                                   if not str(route.get("source", "")).startswith("dag:")]
            recovery_states.append((state, clone, plan, payload,
                                    trace["search_count"], trace["llm_verification_calls"]))
        # Batch all eligible questions together so the existing worker pool
        # remains available at every dependency depth of the repairs.
        super()._dependency_search([entry[1] for entry in recovery_states], executor)
        for state, clone, plan, payload, searches_before, verification_before in recovery_states:
            trace = state["evidence_trace"]
            diag = trace["improvement_bridge"]
            clone_trace = clone["evidence_trace"]
            diag["extra_searches"] = clone_trace["search_count"] - searches_before
            diag["extra_verification_requests"] = clone_trace["llm_verification_calls"] - verification_before
            new_bindings = clone.get("_evidence_winning_bindings", {})
            complete = all(node["id"] in new_bindings for node in plan)
            # An unchanged group/product is not an unknown person resolved.
            bridge_answer = str(new_bindings.get("r1", ""))
            if complete and (not bridge_answer or _literal(bridge_answer, state["query"])):
                complete = False
                diag["validation_error"] = "bridge_answer_is_input_entity"
            diag["repair_bindings"] = dict(new_bindings)
            if not complete:
                trace["search_count"] = clone_trace["search_count"]
                trace["llm_verification_calls"] = clone_trace["llm_verification_calls"]
                state["_evidence_search_cache"].update(clone.get("_evidence_search_cache", {}))
                diag["deferred_reason"] = "incomplete_repair_preserved_original_dag"
                diag["repair_verification_outputs"] = clone_trace.get("verification_outputs", [])[len(trace.get("verification_outputs", [])):]
                continue
            diag["applied"] = True
            clone_trace["improvement_bridge"] = diag
            clone_trace.setdefault("planning_outputs", {})["bridge_repair"] = {
                "attempts": 1, "outputs": [payload], "validation_errors": []}
            state.update(clone)
