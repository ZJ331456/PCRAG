"""Conservative local ownership checks for optional strict answer bindings.

These checks recognize a bounded set of English relation constructions.  They
are not a general entailment model: an unknown or ambiguous construction leaves
the answer unbound, while the caller retains its ordinary document candidates.
No dataset identifiers, labels, or extra model requests are used.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .evidence_retrieval import normalized_text


def _clean(value: str) -> str:
    return value.strip().rstrip("?.! ")


def relation_spec(question: str) -> dict:
    """Identify the requested relation and concrete subject when expressible."""
    q = _clean(question)
    lower = normalized_text(q)
    spec = {"relation": "other", "subject": None}
    if re.search(r"\b(?:born|died|director)\b.*\b(?:first|earlier|later|older|younger)\b", lower):
        return dict(spec, relation="comparison")
    patterns = [
        ("birth_date", r"(?:when|(?:in\s+)?what\s+year)\s+(?:was|is)\s+(.+?)\s+born$"),
        ("birth_place", r"where\s+(?:was|is)\s+(.+?)\s+born$"),
        ("birth_date", r"(?:birth\s+(?:date|year)|date\s+of\s+birth)\s+of\s+(.+)$"),
        ("birth_place", r"birthplace\s+of\s+(.+)$"),
        ("death_date", r"when\s+did\s+(.+?)\s+die$"),
        ("death_date", r"(?:in\s+)?what\s+year\s+did\s+(.+?)\s+die$"),
        ("nationality", r"what\s+nationality\s+(?:is|was|were)\s+(.+)$"),
        ("nationality", r"(?:nationality|citizenship)\s+of\s+(.+)$"),
        ("nationality", r"what\s+(?:is|was)\s+(.+?)['’]s\s+nationality$"),
        ("location", r"where\s+(?:is|was|are|were)\s+(.+?)(?:\s+(?:located|situated))?$"),
        ("location", r"(?:in\s+)?(?:what|which)\s+(?:region|country|city|continent|state|place)\s+(?:is|was)\s+(.+?)\s+(?:located|situated)$"),
        ("established_date", r"(?:when|(?:in\s+)?what\s+year)\s+(?:was|is|were)\s+(.+?)\s+(?:established|founded|formed|created|built|opened|incorporated)$"),
        ("director", r"who\s+(?:directed|directs)\s+(.+)$"),
        ("director", r"who\s+(?:is|was)\s+(?:the\s+)?director\s+of\s+(.+)$"),
        ("author", r"who\s+(?:wrote|writes|authored)\s+(.+)$"),
        ("author", r"who\s+(?:is|was)\s+(?:the\s+)?(?:author|writer)\s+of\s+(.+)$"),
        ("composer", r"who\s+composed\s+(.+)$"),
        ("composer", r"who\s+(?:is|was)\s+(?:the\s+)?composer\s+of\s+(.+)$"),
    ]
    for relation, pattern in patterns:
        match = re.search(pattern, q, re.I)
        if match:
            spec = {"relation": relation, "subject": _clean(match.group(1))}
            break
    direction = re.search(r"\b(north|south|east|west|northeast|northwest|southeast|southwest)\s+of\s+(.+)$", q, re.I)
    if direction:
        spec = {"relation": direction.group(1).casefold() + "_of",
                "subject": _clean(direction.group(2)), "direction": direction.group(1).casefold()}
    if spec["relation"] in {"birth_date", "birth_place", "death_date", "nationality"}:
        subject = spec["subject"] or ""
        owner = re.fullmatch(r"(?:the\s+)?(director|author|writer|composer)\s+of\s+(.+)", subject, re.I)
        reverse = re.fullmatch(r"(.+?)(?:['’]s)?\s+(director|author|writer|composer)", subject, re.I)
        if owner or reverse:
            role, work = ((owner.group(1), owner.group(2)) if owner else
                          (reverse.group(2), reverse.group(1)))
            spec.update(subject=None, owner_work=_clean(work), owner_role=role.casefold())
    # Award questions may specify their subject through descriptive qualifiers;
    # require the award's name in the same owning clause rather than treating a
    # bare year mentioned in a biography as proof.
    if spec["relation"] == "other" and re.search(r"\b(?:year|when)\b", lower) and re.search(r"\b(?:award|awarded|named|won|winner)\b", lower):
        spec["relation"] = "award_date"
    return spec


_RELATION_ALIASES = {
    "location": {"location", "located", "located_in", "situated_in", "is_in", "place"},
    "established_date": {"established_date", "established_year", "established", "founded", "founded_in", "founding_date", "founding_year", "formation_date", "formed", "built", "opened", "incorporated"},
    "director": {"director", "directed", "directed_by", "film_director"},
    "author": {"author", "writer", "wrote", "written_by", "authored_by"},
    "composer": {"composer", "composed", "composed_by"},
    "award_date": {"award_date", "award_year", "award", "awarded", "won", "named"},
}


def relation_name_matches(claimed: str, spec: dict) -> bool:
    relation = spec["relation"]
    if relation not in _RELATION_ALIASES and not spec.get("direction"):
        return True  # Existing typed checks handle personal attributes.
    value = normalized_text(claimed).replace(" ", "_")
    aliases = _RELATION_ALIASES.get(relation, {relation, "north_of", "south_of", "east_of", "west_of"})
    if spec.get("direction"):
        aliases = {relation, "border", "borders", "neighbor", "neighbor_to_" + spec["direction"]}
    return value in aliases


def _contains(value: str, text: str) -> bool:
    value = normalized_text(value)
    return bool(value) and " " + value + " " in " " + normalized_text(text) + " "


def _clauses(span: str):
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", span) if part.strip()]


def _topic_matches(entity: str, document: str, same_entity: Callable) -> bool:
    lines = document.splitlines()
    if len(lines) < 2:
        return False
    title = re.sub(r"\s*\([^()]+\)\s*$", "", lines[0].strip())
    return same_entity(entity, title, document)


def _owned_clause(entity: str, clause: str, document: str, surface: Callable,
                  same_entity: Callable) -> bool:
    # An unrelated named subject in this sentence cannot inherit the document's
    # topic merely because both names occur somewhere in the passage.
    literal = surface(entity, clause, document)[0]
    if literal:
        match = re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", clause, re.I)
        if match:
            # A possessive or prepositional mention is not the owning subject:
            # "X's port is located..." and "the port in X is located...".
            if re.match(r"['’]s\b", clause[match.end():]):
                return False
            if re.search(r"\b(?:of|by|for|in|on|from|with)\s*$", clause[:match.start()], re.I):
                return False
            prefix = clause[:match.start()].strip()
            # The named owner must lead the clause, possibly after an article or
            # a short role label. A name elsewhere in the subject phrase merely
            # co-occurs and does not establish ownership of this predicate.
            if prefix and not re.fullmatch(r"(?:(?:the|a|an)\s+)?(?:(?:film|movie|book|novel|person|actor|director|writer|composer|country|city|state)\s*)?", prefix, re.I):
                return False
        return True
    return (_topic_matches(entity, document, same_entity) and
            bool(re.match(r"(?:He|She|It|They|His|Her|Its)\b", clause, re.I)))


def _role_pair_supported(work: str, role: str, person: str, document: str,
                         surface: Callable, same_entity: Callable,
                         evidence: Optional[str] = None) -> bool:
    passive = {"director": r"directed\s+by", "author": r"(?:written|authored)\s+by",
               "writer": r"(?:written|authored)\s+by", "composer": r"composed\s+by"}[role]
    active = {"director": r"directed", "author": r"(?:wrote|authored)",
              "writer": r"(?:wrote|authored)", "composer": r"composed"}[role]
    for clause in _clauses(document if evidence is None else evidence):
        for marker in re.finditer(passive, clause, re.I):
            before, after = clause[:marker.start()], clause[marker.end():]
            if (_owned_clause(work, before, document, surface, same_entity) and
                    surface(person, after, document)[0]):
                return True
        for marker in re.finditer(active, clause, re.I):
            before, after = clause[:marker.start()], clause[marker.end():]
            if (_owned_clause(person, before, document, surface, same_entity) and
                    surface(work, after, document)[0]):
                return True
    return False


def strict_relation_reason(spec: dict, question: str, answer: str, answer_type: str,
                           entity: str, span: str, document: str,
                           surface: Callable, same_entity: Callable) -> Optional[str]:
    """Reject unsupported ownership; acceptance is bounded syntactic support."""
    relation, subject = spec["relation"], spec.get("subject")
    if relation == "comparison":
        return "comparison_requires_atomic_attribute_evidence"
    if relation == "other":
        return "unrecognized_requested_relationship"
    if subject:
        if re.search(r"\b(?:who|whose|that|which)\b", subject, re.I):
            return "unresolved_relationship_subject"
        if not same_entity(entity, subject, document):
            return "relationship_subject_mismatch"
    if spec.get("owner_work") and not _role_pair_supported(
            spec["owner_work"], spec["owner_role"], entity, document, surface, same_entity):
        return "linked_subject_relationship_not_supported"
    if relation in {"birth_date", "birth_place", "death_date", "nationality"}:
        # The original attribute checker now runs against a concrete subject or
        # the explicitly demonstrated creator, rather than an arbitrary person.
        return None
    answer_clauses = [clause for clause in _clauses(span) if _contains(answer, clause)]
    clauses = [clause for clause in answer_clauses
               if _owned_clause(entity, clause, document, surface, same_entity)]
    if relation == "location":
        for clause in clauses:
            marker = re.search(r"\b(?:is|was|are|were)\s+(?:(?:a|an|the)\s+[^.;!?]{0,90}?\s+)?(?:located|situated|in|on)\b|\b(?:lies|lie)\b", clause, re.I)
            if marker and _contains(answer, clause[marker.end():]):
                # Do not transfer a port's location to the country containing it.
                head = clause[:marker.start()]
                if _owned_clause(entity, head, document, surface, same_entity):
                    return None
        return "requested_location_not_supported_for_subject"
    if relation == "established_date":
        for clause in clauses:
            marker = re.search(r"\b(?:established|founded|formed|created|built|opened|incorporated)\b", clause, re.I)
            if marker and _contains(answer, clause[marker.end():]) and _owned_clause(entity, clause[:marker.start()], document, surface, same_entity):
                return None
        # Compact infobox entries belong only to their own document topic.
        if _topic_matches(entity, document, same_entity) and re.match(
                r"\s*(?:Established|Founded|Formed|Created|Built|Opened|Incorporated)\b", span, re.I):
            return None if re.search(r"\d", answer) else "answer_type_mismatch"
        return "requested_establishment_not_supported_for_subject"
    if spec.get("direction"):
        direction = re.escape(spec["direction"])
        for clause in answer_clauses:
            # Answer is north of Subject: both sides must be explicit.
            for marker in re.finditer(r"\b" + direction + r"\s+of\b", clause, re.I):
                if (_contains(answer, clause[:marker.start()]) and
                        surface(entity, clause[marker.end():], document)[0]):
                    return None
            # Subject is bordered to the north by Answer; the owner must occur
            # before the direction marker, not as another name later in a list.
            marker = re.search(r"\b(?:to|on)\s+the\s+" + direction + r"\b(?:\s+and\s+\w+)?\s+by\b", clause, re.I)
            if marker and _contains(answer, clause[marker.end():]) and _owned_clause(entity, clause[:marker.start()], document, surface, same_entity):
                return None
        return "requested_direction_not_supported_for_subject"
    if relation in {"director", "author", "composer"}:
        return (None if _role_pair_supported(entity, relation, answer, document,
                                            surface, same_entity, evidence=span) else
                "requested_creator_not_supported_for_subject")
    if relation == "award_date":
        if not re.search(r"\d", answer):
            return "answer_type_mismatch"
        award_words = {w for w in normalized_text(question).split()
                       if w in {"finals", "valuable", "nobel", "pulitzer", "oscar", "academy", "grammy"}}
        for clause in clauses:
            if (re.search(r"\b(?:won|awarded|named|received)\b", clause, re.I) and
                    award_words <= set(normalized_text(clause).split())):
                return None
        return "requested_award_not_supported_for_subject"
    return "unrecognized_requested_relationship"
