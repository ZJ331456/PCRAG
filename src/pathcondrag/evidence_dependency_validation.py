"""Opt-in local source checks for dependency evidence selection.

The existing checker is tried first. Additional bounded English constructions
use the same literal quotation and at most its immediately preceding sentence
to establish ownership. These checks issue no requests and are not a general
entailment model. Numeric bounds and time qualifiers are never silently dropped.
"""
from __future__ import annotations

import re

from .evidence_binding import _same_entity, _typed_answer_reason, supported_entity_surface
from .evidence_dag_package import _proof_relation
from .evidence_package import _title
from .evidence_relation_guard import clean_subject, relation_spec
from .evidence_retrieval import normalized_text
from .evidence_terminal import _literal


_NEGATION = re.compile(r"\b(?:not|never|without|neither|no longer)\b", re.I)
_KIN = re.compile(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband|partner)\b", re.I)


def _clean(value):
    return clean_subject(str(value).strip().strip('"“”').rstrip("?.! "))


def _sentences(value):
    # Do not split at ordinary initials. All ownership windows remain bounded.
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+(?=[A-Z])|\n+", value) if s.strip()]


def _identity_surfaces(entity, document):
    """Explicit title and lead identity, including a quoted given-name alias."""
    title = _title(document)
    if not title or not _same_entity(entity, title, document):
        return []
    body = " ".join(document.splitlines()[1:]).lstrip()
    lead = re.split(r"\s*\(|\s+(?:is|was|are|were)\b", body[:550], maxsplit=1, flags=re.I)[0].strip()
    surfaces = [entity, title]
    acronym = re.match(r"(?:The\s+)?([A-Z][A-Za-z'’ -]{2,100})\s*\(([^()]{1,25})\)\s+(?:is|was)\b", body)
    if acronym and normalized_text(acronym.group(2)) == normalized_text(entity):
        surfaces.append(acronym.group(1).strip())
    if _same_entity(entity, lead, document):
        surfaces.append(lead)
    else:
        # Shelton Jackson "Spike" Lee is an explicit nickname identity; a
        # coincidentally shared surname or arbitrary reordered names are not.
        names = normalized_text(entity).split()
        nicknames = re.findall(r'["“]([^"”]+)["”]', lead)
        plain = normalized_text(lead).split()
        if (2 <= len(names) <= 4 and plain and plain[-1] == names[-1]
                and any(normalized_text(nick) == " ".join(names[:-1]) for nick in nicknames)
                and not _KIN.search(lead)):
            surfaces.append(lead)
    return list(dict.fromkeys(s for s in surfaces if s))


def _leading_surface(entity, value, document):
    surfaces = [entity]
    surface, kind = supported_entity_surface(entity, value, document)
    # A surname alone is not an independently demonstrated clause owner.
    if surface and kind != "document_anchored_surname":
        surfaces.append(surface)
    surfaces.extend(_identity_surfaces(entity, document))
    for candidate in sorted(set(surfaces), key=len, reverse=True):
        match = re.match(r"\s*(?:The\s+)?" + re.escape(candidate) + r"(?!\w)", value, re.I)
        if match:
            return match
    return None


def _adjacent_topic(entity, clause, document):
    """A pronoun inherits only a uniquely adjacent, unambiguous named owner."""
    if not _identity_surfaces(entity, document) or document.count(clause) != 1:
        return False
    before = document[:document.index(clause)].rstrip()
    previous = _sentences(before)
    if not previous:
        return False
    last = previous[-1]
    if normalized_text(last) == normalized_text(_title(document)):
        # A bare title does not establish an owning identity for an arbitrary
        # pronoun lead. Require an explicit lead instead.
        return False
    if len(last) > 550 or _KIN.search(last) or _NEGATION.search(last):
        return False
    owner = _leading_surface(entity, last, document)
    if not owner:
        return False
    remainder = last[owner.end():].lstrip()
    if re.match(r"['’]s\b|\s*(?:and|or)\b", remainder, re.I):
        return False
    # Life dates may precede the identity predicate; nested or unclosed
    # parentheses are rejected rather than skipping arbitrary prose.
    remainder = re.sub(r"^\([^()]{1,100}\)\s*", "", remainder)
    if not re.match(r"(?:is|was|are|were|has|had|worked|served|became)\b", remainder, re.I):
        return False
    # An intervening other full name could be the antecedent. Conservative
    # abstention also covers family/partner clauses and coordinated people.
    return not re.search(r"\b[A-Z][a-z'’-]+\s+[A-Z][a-z'’-]+\b", remainder)


def _owned_clause(entity, clause, document, allow_company=False):
    """Return a named clause only after direct or adjacent ownership checks."""
    value = clause.strip()
    pronoun = re.match(r"(?:He|She|It|They|His|Her|Its|Their)\b", value, re.I)
    if pronoun:
        if not _adjacent_topic(entity, value, document):
            return None
        if pronoun.group().casefold() in {"his", "her", "its", "their"}:
            if not allow_company or not re.match(r"\s+production\s+company\b", value[pronoun.end():], re.I):
                return None
            return entity + "'s" + value[pronoun.end():]
        return entity + value[pronoun.end():]
    owner = _leading_surface(entity, value, document)
    if not owner:
        return None
    remainder = value[owner.end():].lstrip()
    if re.match(r"['’]s\b", remainder):
        return value if allow_company and re.match(r"['’]s\s+production\s+company\b", remainder, re.I) else None
    if _KIN.search(remainder.split(".", 1)[0].split(";", 1)[0]):
        return None
    return value


def _new_spec(question):
    q = _clean(question)
    patterns = [
        ("location", r"where\s+(?:is|was|are|were)\s+(.+?)\s+(?:based|headquartered)(?:\s+(?:in|at|out\s+of))?$"),
        ("location", r"(?:in\s+)?(?:what|which)\s+(?:country|city|state|region|continent)\s+(?:is|was)\s+(.+?)\s+(?:based|headquartered|located|situated)(?:\s+(?:in|at|out\s+of))?$"),
        ("headquarters", r"(?:what|where)\s+(?:is|was|are|were)\s+(?:the\s+)?headquarters\s+of\s+(.+)$"),
        ("headquarters", r"(?:what|where)\s+(?:is|was|are|were)\s+(.+?)['’]s\s+headquarters$"),
        ("capital", r"(?:what|which\s+city)\s+(?:is|was)\s+(?:the\s+)?capital(?:\s+city)?\s+of\s+(.+)$"),
        ("mother", r"who\s+(?:is|was)\s+(?:the\s+)?mother\s+of\s+(.+)$"),
        ("mother", r"who\s+(?:is|was)\s+(.+?)['’]s\s+mother$"),
        ("broadcast", r"(?:on\s+)?(?:what|which)\s+(?:television\s+)?(?:channel|network|station)\s+(?:is|was|did)\s+(.+?)\s+(?:broadcast|aired|air)$"),
        ("broadcast", r"(?:what|which)\s+(?:television\s+)?(?:channel|network|station)\s+(?:broadcast|aired)\s+(.+)$"),
        ("album_artist", r"who\s+(?:is|was)\s+(?:the\s+)?(?:artist|performer|musician)\s+(?:of|on)\s+(?:the\s+)?album\s+(.+)$"),
        ("album_artist", r"(?:who|which\s+(?:jazz\s+)?(?:pianist|musician|artist))\s+(?:recorded|performed|released)\s+(?:the\s+)?album\s+(.+)$"),
    ]
    for relation, pattern in patterns:
        match = re.fullmatch(pattern, q, re.I)
        if match:
            return relation, _clean(match.group(1))
    for relation, verb in [("producer", "produced"), ("publisher", "published"),
                           ("developer", "developed"), ("creator", "created")]:
        match = re.fullmatch(r"who\s+" + verb + r"\s+(.+)", q, re.I)
        if match:
            return relation, _clean(match.group(1))
    spec = relation_spec(q)
    return spec["relation"], spec.get("subject")


def _time_scope(question, clause):
    qualifiers = re.findall(r"\b(?:since|before|after|in|during|between)\s+\d{4}(?:\s+and\s+\d{4})?\b", question, re.I)
    # A later coordinated predicate's year cannot qualify this count. Keep
    # between1983and1990, but stop before andHisCompany/... another clause.
    scope = re.split(r"\s+(?:and|but|while|whereas)\s+(?=[A-Za-z])", clause, maxsplit=1, flags=re.I)[0]
    return all(_literal(value, scope) for value in qualifiers)


def _argument_matches(answer, value):
    """The answer must own the predicate argument, not occur later in prose."""
    expected = normalized_text(answer)
    value = re.sub(r"^(?:the|a|an)\s+", "", value.strip(), flags=re.I)
    # Keep original punctuation: normalization erases possessive boundaries
    # and would treat Alice's sister or StudioA's subsidiary as Alice/StudioA.
    tokens = list(re.finditer(r"\w+", value))
    answer_tokens = expected.split()
    if not answer_tokens or len(tokens) < len(answer_tokens):
        return False
    if normalized_text(value[:tokens[len(answer_tokens) - 1].end()]) != expected:
        return False
    tail = value[tokens[len(answer_tokens) - 1].end():].strip()
    if re.match(r"['’]s\b|['’]\s", tail, re.I):
        return False
    # Only explicit argument boundaries or a bounded location/appositional
    # continuation qualify. A longer name or an ownership noun is not an alias.
    return (not tail or bool(re.match(r"[.,;!?)]", tail))
            or bool(re.match(r"(?:in|at|on|with|and)\b", tail, re.I)))


def _strip_lead_parenthesis(value):
    return re.sub(r"^\([^()]{1,160}\)\s*", "", value)


def _album_argument_matches(answer, value):
    # Explicit artist role labels may precede a named album artist. Another
    # proper name or a later performer cannot be skipped as a role label.
    roles = r"(?:(?:American|British|Canadian|French|German|Italian|Australian)\s+)?(?:(?:bassist|pianist|musician|singer|singer-songwriter|composer|bandleader|artist|rock|jazz|and)[,\s]+)*"
    match = re.match(roles, value, re.I)
    return bool(match and _argument_matches(answer, value[match.end():]))


def _count_check(question, answer, quote, document):
    match = re.fullmatch(
        r"How\s+many\s+(films?|movies?|books?|novels?|songs?|albums?|episodes?)\s+"
        r"(?:has|have|did)\s+(.+?)\s+(produced|produce|published|publish|written|write)"
        r"(?:\s+((?:since|before|after|in|during|between)\s+\d{4}(?:\s+and\s+\d{4})?))?[?.!\s]*",
        question, re.I)
    if not match:
        return None, None
    unit, entity, verb, _ = match.groups()
    entity = _clean(entity)
    company = re.fullmatch(r"(.+?)['’]s\s+production\s+company", entity, re.I)
    owner = _clean(company.group(1)) if company else entity
    predicate = (r"produc(?:ed|es?)" if verb.casefold().startswith("produc") else
                 r"publish(?:ed|es)?" if verb.casefold().startswith("publish") else r"(?:written|wrote)")
    unit = unit.rstrip("s")
    # Explicit numeric bounds are part of the answer. An exact 35 never inherits
    # the support for over35/at least35/about35; answer text is not rewritten.
    count_pattern = re.compile(
        r"\b" + predicate + r"\s+(?:(over|more\s+than|at\s+least|at\s+most|about|approximately|exactly)\s+)?"
        r"(\d+)\s+(?:" + re.escape(unit) + r"s?)\b([^.;!?]*)", re.I)
    for clause in _sentences(quote):
        owned = _owned_clause(owner, clause, document, allow_company=True)
        if not owned:
            continue
        if company and not re.match(re.escape(owner) + r"['’]s\s+production\s+company\b", owned, re.I):
            continue
        for quantity in count_pattern.finditer(owned):
            # The production predicate must belong to the lead owner, not a
            # company/person mentioned as an object earlier in the sentence.
            head = owned[:quantity.start()]
            start = _leading_surface(owner, head, document)
            remainder = head[start.end():].strip() if start else ""
            if not start or not re.fullmatch(r"(?:['’]s\s+production\s+company(?:\s*,[^,]{1,100},)?\s*)?(?:has|have|had)?", remainder, re.I):
                continue
            modifier, number, scope = quantity.groups()
            expected = normalized_text((modifier or "") + " " + number)
            supplied = normalized_text(re.sub(r"\s+" + re.escape(unit) + r"s?[.! ]*$", "", answer, flags=re.I))
            if expected != supplied and not (modifier and modifier.casefold() == "exactly" and supplied == number):
                return None, "expanded_count_bound_mismatch"
            if not _time_scope(question, quantity.group()):
                return None, "expanded_count_time_scope_not_supported"
            if re.search(r"\bexact(?:ly)?\b", question, re.I) and modifier and modifier.casefold() != "exactly":
                return None, "expanded_count_bound_mismatch"
            return "expanded_owned_production_count", None
    return None, "expanded_count_owner_or_predicate_not_supported"


def _attribute_check(relation, entity, answer, quote, document):
    markers = {
        "location": r"(?:is|was|are|were)\s+(?:(?:a|an|the)\s+(?![^.;!?]{0,100}\b(?:of|by|from|for)\b)[A-Za-z -]{1,100}\s+)?(?:based|headquartered|located|situated)\s+(?:in|at|out\s+of)\s+",
        "headquarters": (r"(?:has|had)\s+(?:its\s+)?headquarters\s+(?:in|at)\s+|"
                         r"(?:is|was)\s+(?:headquartered|based)\s+(?:in|at)\s+|"
                         r"(?:is|was)\s+(?:a|an|the)\s+(?![^.;!?]{0,100}\b(?:of|by|from|for)\b)"
                         r"[A-Za-z -]{1,100}\s+with\s+its\s+headquarters\s+(?:in|at)\s+"),
        "broadcast": (r"(?:is|was|were|has\s+been)\s+(?:(?:first|originally)\s+)?(?:broadcast|aired|shown)\s+(?:by|on)\s+|"
                      r"(?:is|was)\s+(?:a|an|the)\s+[A-Za-z -]{0,45}television\s+series"
                      r"(?:\s+starring\s+[A-Za-z'’ -]{1,80})?\s+that\s+(?:aired|broadcast)\s+on\s+"),
        "album_artist": r"(?:is|was)\s+(?:an?\s+)?(?:[^.;!?]{0,50}\s+)?album\s+by\s+",
    }
    if relation in markers:
        for clause in _sentences(quote):
            owned = _owned_clause(entity, clause, document)
            if not owned:
                continue
            owner = _leading_surface(entity, owned, document)
            remainder = _strip_lead_parenthesis(owned[owner.end():].strip()) if owner else ""
            match = re.match(markers[relation], remainder, re.I)
            argument = remainder[match.end():] if match else ""
            matches = (_album_argument_matches(answer, argument) if relation == "album_artist"
                       else _argument_matches(answer, argument))
            if match and matches:
                return "expanded_owned_" + relation, None
        if relation != "headquarters":
            return None, "expanded_requested_attribute_not_supported"
    if relation in {"mother", "capital", "headquarters"}:
        attribute = {"mother": r"mother", "capital": r"capital(?:\s+city)?", "headquarters": r"headquarters"}[relation]
        for clause in _sentences(quote):
            owner = _leading_surface(entity, clause, document)
            if owner:
                rest = clause[owner.end():].strip()
                match = re.match(r"['’]s\s+" + attribute + r"\s+(?:is|was|are|were)\s+", rest, re.I)
                if match and _argument_matches(answer, rest[match.end():]):
                    return "expanded_owned_" + relation, None
            # Both ends are named: Mother is the mother of Child.
            subject = _leading_surface(answer, clause, document)
            if subject:
                rest = clause[subject.end():].strip()
                capital_extra = r"(?:\s+and\s+(?:(?:first|second|third)\s+)?largest\s+city)?" if relation == "capital" else ""
                match = re.match(r"(?:is|was|are|were)\s+(?:the\s+)?" + attribute + capital_extra + r"\s+of\s+", rest, re.I)
                tail = rest[match.end():].rstrip(". ") if match else ""
                if relation == "capital":
                    tail = re.sub(r"^(?:the\s+)?(?:(?:U\.S\.|United\s+States)\s+)?(?:state|country|region)\s+of\s+", "", tail, flags=re.I)
                if match and normalized_text(tail) == normalized_text(entity):
                    return "expanded_inverse_owned_" + relation, None
        return None, "expanded_requested_attribute_not_supported"
    return None, None


def _creator_check(relation, entity, answer, quote, document):
    verbs = {"producer": ("produced", "produced"), "publisher": ("published", "published"),
             "developer": ("developed", "developed"), "creator": ("created", "created"),
             "director": ("directed", "directed"), "author": ("written", "wrote"),
             "composer": ("composed", "composed")}
    if relation not in verbs:
        return None, None
    passive, active = verbs[relation]
    for clause in _sentences(quote):
        owned = _owned_clause(entity, clause, document)
        if owned:
            owner = _leading_surface(entity, owned, document)
            rest = owned[owner.end():].strip() if owner else ""
            match = re.match(r"(?:is|was|were|has\s+been)\s+" + passive + r"\s+by\s+", rest, re.I)
            if match and _argument_matches(answer, rest[match.end():]):
                return "expanded_owned_creator", None
        creator = _leading_surface(answer, clause, document)
        if creator:
            rest = clause[creator.end():].strip()
            match = re.match(r"(?:has\s+)?" + active + r"\s+", rest, re.I)
            if match and normalized_text(rest[match.end():].rstrip(". ")) == normalized_text(entity):
                return "expanded_active_creator", None
    return None, "expanded_creator_pair_not_supported"


def expanded_proof_relation(node, question, proof, document, bindings):
    """Retain legacy successes, expand only literal unambiguous source syntax."""
    if not isinstance(node, dict) or not isinstance(proof, dict) or not isinstance(bindings, dict):
        return None, "expanded_invalid_proof_fields"
    answer, quote = proof.get("answer"), proof.get("evidence")
    if not isinstance(answer, str) or not isinstance(quote, str) or not isinstance(question, str):
        return None, "expanded_invalid_proof_fields"
    if len(quote) < 8 or len(quote) > 1500 or quote not in document or not _literal(answer, quote):
        return None, "expanded_proof_not_literal_grounded"
    if _NEGATION.search(quote):
        return None, "expanded_negated_quote_not_supported"
    typed = _typed_answer_reason(answer, node.get("answer_type", ""))
    if typed:
        return None, typed
    check, reason = _proof_relation(node, question, proof, document, bindings)
    if check:
        return check, None
    # Do not reinterpret demonstrated conflicts as merely unknown grammar.
    if reason in {"biography_lead_identity_conflict", "relationship_direction_conflict",
                  "answer_type_mismatch", "answer_attribute_belongs_to_other_entity",
                  "relationship_subject_mismatch", "linked_subject_relationship_not_supported"}:
        return None, reason
    if len(node.get("depends_on", [])) > 1 and not all(
            _literal(bindings.get(dep, ""), quote) for dep in node["depends_on"]):
        return None, "expanded_multi_parent_inputs_not_grounded"
    check, expanded_reason = _count_check(question, answer, quote, document)
    if check or expanded_reason:
        return check, expanded_reason
    relation, entity = _new_spec(question)
    if not entity or re.search(r"\$\{|\b(?:who|whose|that|which)\b", entity, re.I):
        return None, reason or "expanded_unresolved_question_subject"
    check, expanded_reason = _attribute_check(relation, entity, answer, quote, document)
    if check:
        return check, None
    creator, creator_reason = _creator_check(relation, entity, answer, quote, document)
    if creator:
        return creator, None
    return None, expanded_reason or creator_reason or reason or "expanded_unknown_relation"
