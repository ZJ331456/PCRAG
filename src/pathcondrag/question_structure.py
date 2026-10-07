"""Conservative routing from the question text alone.

This router decides whether to increase planning capacity. It does not predict
answers, consume benchmark annotations, or rewrite the question. Ambiguous
conjunctions retain the original dependency planner.
"""
from __future__ import annotations

import re


_COMPARISON = re.compile(
    r"\b(?:earlier|later|older|younger|first|more|less|longer|shorter|larger|smaller|"
    r"higher|lower|taller|fewer|same|different|both|share|common|versus)\b", re.I)
_ALTERNATIVE = re.compile(r"\b(?:or|versus|vs\.?)\b", re.I)
_AND = re.compile(r"\band\b", re.I)
_SELECTION_QUESTION = re.compile(r"^\s*(?:which|who|whose|what|is|are|was|were|do|does|did|has|have)\b", re.I)
_SHARED_ATTRIBUTE = re.compile(r"\b(?:same|both|share|in\s+common)\b", re.I)
_SAME_AS = re.compile(r"\bsame\b(?:\s+\w+){0,4}\s+as\b", re.I)
_YES_NO = re.compile(r"^\s*(?:is|are|was|were|do|does|did|has|have)\b", re.I)
_ROLE = r"(?:directors?|producers?|writers?|authors?|composers?|parents?|fathers?|mothers?|spouses?|founders?|owners?)"
_NESTED_ROLE = re.compile(
    r"\b" + _ROLE + r"\s+(?:of|for)\s+(?:(?:the|both)\s+)?(?:films?|movies?|books?|novels?|works?)\b"
    r"|\b(?:films?|movies?|books?|novels?|works?)\b.*\b(?:has|have|whose)\b.*\b" + _ROLE + r"\b"
    r"|\bwhose\s+" + _ROLE + r"\b", re.I)
_NESTED_QUALIFIER = re.compile(r"\b(?:where|whose|that|which|who|whom)\b", re.I)
_MULTI_ATTRIBUTE_QUESTION = re.compile(
    r"^\s*(?:when|where|in\s+what\s+year|on\s+what\s+date|what\s+(?:year|date))\b", re.I)
_PLURAL_ATTRIBUTE_QUESTION = re.compile(
    r"^\s*(?:what|which)\s+(?:nationalities|birthdates|birthplaces|occupations|professions)\b", re.I)
_SINGULAR_DESCRIPTION = re.compile(
    r"^\s*(?:(?:did|does|do|was|were|is|has|have)\s+)?the\s+"
    r"(?:country|city|region|territory|province|state|company|person|author|director|writer|"
    r"birthplace|hometown|location|maker|manufacturer|owner|performer)\b", re.I)
_COUPLED_EVENT = re.compile(r"\b(?:become|became)\s+(?:allies|partners|friends|enemies)\b", re.I)
_SHARED_EVENT = re.compile(
    r"\b(?:open|opened|establish|established|found|founded|start|started|begin|began|"
    r"join|joined|graduate|graduated|die|died|born|become|became|release|released|"
    r"publish|published|live|lived|locate|located|work|worked)\b", re.I)
_NAME = re.compile(r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*")
_QUESTION_WORDS = {"Which", "Who", "What", "When", "Where", "Is", "Are", "Was", "Were",
                   "Do", "Does", "Did", "Has", "Have", "In", "On"}
_BRIDGE_ENTITY = (
    r"(?:writer|author|director|composer|performer|singer|actor|actress|maker|manufacturer|"
    r"creator|founder|owner|spouse|parent|father|mother|birthplace|hometown|city|country|"
    r"region|company|university|team|league|album|record\s+label)"
)
_EXPLICIT_UNKNOWN_DESCRIPTION = re.compile(
    r"\b(?:the|a|an)\s+" + _BRIDGE_ENTITY + r"\s+(?:of|for|behind|where|whose|that|which|who)\b"
    r"|\b(?:writer|author|director|composer|performer|maker|manufacturer|creator|owner)"
    r"\s+(?:of|for)\s+", re.I)
_POSSESSIVE_BRIDGE = re.compile(
    r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*['’]s\s+"
    r"(?:writer|author|director|composer|performer|maker|manufacturer|creator|owner|spouse)\b")


def _named_objects(text):
    """Count separate capitalized spans, not identities inferred from a model."""
    return [match.group(0) for match in _NAME.finditer(text)
            if match.group(0) not in _QUESTION_WORDS]


def route_question_structure(question):
    """Return an explainable capacity decision; the safe default is unchanged."""
    question = str(question or "").strip()
    nested_qualifier = bool(_NESTED_QUALIFIER.search(re.sub(r"^\w+\s*", "", question)))
    result = {"kind": "chain_or_unknown", "expand": False,
              "reason": "no_explicit_independent_branches", "qualifier_risk": nested_qualifier}
    names = _named_objects(question)
    if len(names) < 2:
        return result
    alternative = bool(_ALTERNATIVE.search(question))
    conjunction = bool(_AND.search(question))
    # A conjunction alone is not evidence of separate branches. Require an
    # alternative selection or an explicit shared/comparative attribute.
    comparative = bool(_COMPARISON.search(question))
    direct_alternative = bool(re.match(r"^\s*which\b", question, re.I) and alternative)
    if (_SELECTION_QUESTION.search(question)
            and ((alternative and (comparative or direct_alternative))
                 or (conjunction and not _SAME_AS.search(question) and (_SHARED_ATTRIBUTE.search(question)
                                      or (_YES_NO.search(question) and comparative))))):
        result.update(kind="bridge_comparison" if _NESTED_ROLE.search(question) else "parallel_comparison",
                      expand=True, reason="explicit_multi_object_comparison")
        return result
    if (_PLURAL_ATTRIBUTE_QUESTION.search(question) and conjunction
            and not _NESTED_QUALIFIER.search(re.sub(r"^\w+\s*", "", question))):
        result.update(kind="parallel_attribute", expand=True,
                      reason="explicit_multi_object_plural_attribute")
        return result
    if _MULTI_ATTRIBUTE_QUESTION.search(question) and conjunction and not _COUPLED_EVENT.search(question):
        and_match = _AND.search(question)
        event = next((match for match in _SHARED_EVENT.finditer(question)
                      if match.start() > and_match.end()), None)
        if event:
            subject = question[:event.start()]
            # Comma-separated three-way lists can contain descriptive subjects.
            # Otherwise a relative clause before the conjunction can hide a
            # single unknown entity (e.g. the country where A and B fought).
            explicit_list = subject.count(",") >= 2
            subject_body = re.sub(r"^\s*(?:when|where|in\s+what\s+year|on\s+what\s+date|what\s+(?:year|date))\b", "", subject, flags=re.I)
            if (explicit_list or (not _NESTED_QUALIFIER.search(subject_body)
                                  and not _SINGULAR_DESCRIPTION.search(subject_body))) and len(_named_objects(subject)) >= 2:
                result.update(kind="parallel_attribute", expand=True,
                              reason="explicit_multi_object_shared_attribute")
                return result
    return result


def route_dependency_depth(question, hops):
    """Extend text-only branch routing with an authorized dependency depth hint.

    The hint permits a different atomic-chain prompt; it is never a minimum
    number of nodes, and no decomposition or benchmark type is consulted.
    Simple two-hop questions and ambiguous conjunctions keep their old planner.
    """
    question = str(question or "").strip()
    result = route_question_structure(question)
    if result["expand"]:
        return result
    depth = max(1, int(hops))
    if depth >= 3:
        return dict(result, kind="dependency_chain", expand=True,
                    reason="authorized_dependency_depth_at_least_three", depth_hint=depth)
    if _EXPLICIT_UNKNOWN_DESCRIPTION.search(question) or _POSSESSIVE_BRIDGE.search(question):
        return dict(result, kind="nested_bridge", expand=True,
                    reason="explicit_unknown_entity_description", depth_hint=depth)
    return result
