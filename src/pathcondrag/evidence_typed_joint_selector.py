"""Typed scope restriction for the frozen joint evidence selector.

Literal quotes cannot certify collection completeness, aggregate arithmetic,
comparisons or boolean conclusions. Recognized non-atomic target types therefore
abstain before any model request. Dates and years remain eligible. This is a
conservative risk gate, not proof that every remaining question is atomic.
Allowed questions delegate unchanged to the frozen prompt and validator, so
their exact model proposals can be reused from the existing cache.
"""
from __future__ import annotations

import re

from .evidence_joint_selector import apply_joint_selection as _apply
from .evidence_joint_selector import build_joint_prompt as _build


_NON_ATOMIC_TYPE = re.compile(
    r"\b(?:list|collection|array|set|multiple|plural|entities|people|persons|"
    r"locations|countries|number|count|quantity|percentage|percent|statistic|statistics|"
    r"numeric|numerical|int|integer|float|decimal|ratio|fraction|comparison|comparative|"
    r"boolean|bool|yesno|age|duration|population|measurement|distance|weight|height|length)\b",
    re.I,
)
_QUERY_RULES = (
    ("numeric_measure_target", re.compile(
        r"^\s*how\s+(?:many|much|old|long|far|tall|high|wide|heavy|large)\b", re.I)),
    ("aggregate_target", re.compile(
        r"^\s*(?:what|which)\s+(?:(?:is|are|was|were)\s+)?(?:the\s+)?"
        r"(?:number|count|quantity|percentage|percent|proportion|ratio|fraction|total|"
        r"sum|average|mean|median|difference)\b", re.I)),
    ("explicit_collection_target", re.compile(
        r"^\s*(?:list\b|enumerate\b|name\s+all\b|(?:what|which)\s+(?:two|three|four|all)\b)", re.I)),
    ("collection_noun_target", re.compile(
        r"^\s*(?:what|which)\s+(?:is|was|are|were)\s+(?:the\s+)?(?:list|collection|set)\b", re.I)),
    ("plural_attribute_target", re.compile(
        r"^\s*(?:what|which)\s+(?:are|were)\s+(?:the\s+)?(?:(?:two|three|four|all)\s+)?"
        r"(?:advantages|disadvantages|benefits|drawbacks|differences|similarities|types|"
        r"kinds|reasons|languages|dialects|names|countries|cities|actors|actresses)\b", re.I)),
    ("plural_entity_target", re.compile(
        r"^\s*(?:what|which)\s+(?:languages|dialects|countries|cities|actors|actresses|"
        r"advantages|disadvantages|benefits|drawbacks)\b", re.I)),
    ("comparison_target", re.compile(
        r"^\s*(?:which|who|what)\b.{0,160}\b(?:has|have|had|is|are|was|were)\s+"
        r"(?:the\s+)?(?:more|less|fewer|greater|higher|lower|maximum|minimum|most|least|"
        r"oldest|youngest|largest|smallest|longest|shortest|earliest|latest)\b", re.I)),
    ("boolean_target", re.compile(
        r"^\s*(?:is|are|was|were|do|does|did|can|could|has|have|had)\b", re.I)),
)


def _typed_reason(query, state):
    trace = state.get("evidence_trace", {})
    plan = state.get("_evidence_plan", trace.get("plan", []))
    for node in plan if isinstance(plan, list) else []:
        if not isinstance(node, dict):
            continue
        raw = str(node.get("answer_type", ""))
        answer_type = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw).replace("_", " ").replace("-", " ")
        match = _NON_ATOMIC_TYPE.search(answer_type)
        if match:
            return "plan_answer_type:" + match.group().casefold()
        if re.search(r"\b(?:yes\s*/\s*no|true\s*/\s*false)\b", answer_type, re.I):
            return "plan_answer_type:boolean_alias"
    for reason, pattern in _QUERY_RULES:
        if pattern.search(str(query)):
            return "original_query:" + reason
    return None


def build_joint_prompt(query, order, state, document):
    """Skip recognized unsupported targets; otherwise preserve exact core prompt."""
    return None if _typed_reason(query, state) else _build(query, order, state, document)


def apply_joint_selection(query, order, state, document, response, finish_reason="stop"):
    """Apply the same frozen proposal only when its target passes the typed gate."""
    reason = _typed_reason(query, state)
    if reason:
        return [int(d) for d in order], {
            "enabled": True, "policy": "typed_scope_joint_question_constraint_selection",
            "gold_labels_used": False, "entailment_guaranteed": False,
            "promotions": [], "abstain_reason": "typed_constraint_out_of_scope",
            "matched_reason": reason, "finish_reason": finish_reason, "max_promotions": 1,
            "extra_requests": 0, "top2_preserved": True, "top10_set_preserved": True,
            "suffix_preserved": True, "document_set_preserved": True,
            "protected_proofs_preserved": True,
        }
    final, diagnostic = _apply(query, order, state, document, response, finish_reason)
    diagnostic = dict(diagnostic, typed_scope_guard="recognized_non_atomic_targets",
                      matched_reason=None)
    return final, diagnostic
