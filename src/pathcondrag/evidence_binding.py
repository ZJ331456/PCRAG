"""Source-grounded aliases and typed answer bindings for evidence retrieval.

This verifier uses the same batched passage request as the original verifier.
Its local checks are deliberately limited: passing them is source support, not
a claim that an arbitrary natural-language relation has been formally proved.
No benchmark answers, support labels, or question decompositions are consulted.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from .evidence_retrieval import normalized_text, verify_hypotheses
from .evidence_relation_guard import (
    conservative_relation_reason, rejection_disposition, relation_name_matches,
    relation_spec, strict_relation_reason,
)


def _contains(phrase: str, text: str) -> bool:
    phrase, text = normalized_text(phrase), normalized_text(text)
    return bool(phrase) and f" {phrase} " in f" {text} "


def _name_tokens(name: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*", name)


def _name_extension(short: str, full: str) -> bool:
    a, b = _name_tokens(short), _name_tokens(full)
    return (2 <= len(a) <= len(b) <= 6 and a[0].casefold() == b[0].casefold()
            and a[-1].casefold() == b[-1].casefold()
            and all(token[:1].isupper() for token in b)
            and all(x.casefold() in [y.casefold() for y in b] for x in a))


def _identity_names(entity: str, document: str) -> List[str]:
    """Only the title and introductory identity may license shortened names."""
    lines = document.splitlines()
    title = lines[0].strip() if len(lines) > 1 else ""
    body = " ".join(lines[1:]) if title else document
    names: List[str] = []
    if normalized_text(title) == normalized_text(entity):
        names.append(entity)
    # Wikipedia leads commonly start with a full name, possibly a middle name.
    match = re.match(r"\s*([A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]*(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]*){1,5})(?=\s*\(|\s+(?:is|was)\b)", body)
    if match:
        full = match.group(1)
        if normalized_text(full) == normalized_text(entity) or _name_extension(entity, full):
            names.append(full)
    # Explicit alternative spellings/names are evidence, unlike fuzzy matching.
    lead = body[:700]
    if _contains(entity, lead):
        for alias in re.findall(r"(?:also known as|also called|known professionally as|born as)\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,5})", lead):
            names.append(alias)
    return list(dict.fromkeys(names))


def supported_entity_surface(entity: str, quote: str, document: str) -> Tuple[Optional[str], Optional[str]]:
    """Return a literal surface justified by this document's own identity."""
    if _contains(entity, quote):
        return entity, "literal"
    identities = _identity_names(entity, document)
    for name in identities:
        if _contains(name, quote):
            return name, "introductory_full_name"
    if not identities:
        return None, None
    tokens = _name_tokens(entity)
    if len(tokens) < 2:
        return None, None
    surname = tokens[-1]
    for occurrence in re.finditer(r"(?<!\w)" + re.escape(surname) + r"(?!\w)", quote, re.I):
        # A different full name sharing a surname never licenses the binding.
        prefix = quote[:occurrence.start()]
        previous = re.search(r"([A-Z][\w'-]*)\s+$", prefix)
        if previous and previous.group(1) not in {"Mr", "Mrs", "Ms", "Dr", "Sir", "President", "Director"}:
            continue
        return occurrence.group(0), "document_anchored_surname"
    return None, None


def requested_relation(question: str) -> str:
    q = normalized_text(question)
    if any(term in q for term in ("nationality", "citizenship", "nationalities")):
        return "nationality"
    if "born" in q or any(term in q for term in ("birthplace", "date of birth", "birth date", "birth year")):
        return "birth_place" if any(term in q for term in ("where", "birthplace", "place", "city", "country")) else "birth_date"
    if re.search(r"\b(?:die|died|death)\b", q):
        return "death_date" if any(term in q for term in ("when", "date", "year")) else "other"
    if re.search(r"\b(?:director|directed|direct)\b", q):
        return "director"
    if re.search(r"\b(?:wrote|writer|author)\b", q):
        return "author"
    return "other"


def _question_subject(question: str, relation: str) -> Optional[str]:
    patterns = {
        "nationality": [r"what\s+nationality\s+(?:is|was|were)\s+(.+?)[?.]*$",
                        r"(?:nationality|citizenship)\s+of\s+(.+?)[?.]*$",
                        r"what\s+(?:is|was)\s+(.+?)['’]s\s+nationality[?.]*$"],
        "birth_date": [r"(?:when|(?:in\s+)?what\s+year)\s+(?:was|is)\s+(.+?)\s+born[?.]*$",
                       r"(?:birth\s+(?:date|year)|date\s+of\s+birth)\s+of\s+(.+?)[?.]*$"],
        "birth_place": [r"where\s+(?:was|is)\s+(.+?)\s+born[?.]*$",
                        r"birthplace\s+of\s+(.+?)[?.]*$"],
        "death_date": [r"when\s+did\s+(.+?)\s+die[?.]*$"],
    }
    for pattern in patterns.get(relation, []):
        match = re.search(pattern, question.strip(), re.I)
        if match:
            subject = match.group(1).strip().rstrip("?. ")
            if not re.search(r"\b(?:of|who|that|whose)\b", subject, re.I) and len(subject) <= 160:
                return subject
    return None


def _relation_matches(value: str, requested: str) -> bool:
    value = normalized_text(value).replace(" ", "_")
    aliases = {"nationality": {"nationality", "citizenship", "is_nationality"},
               "birth_date": {"birth_date", "birth_year", "date_of_birth", "born", "born_on", "born_in"},
               "birth_place": {"birth_place", "birthplace", "place_of_birth", "born", "born_in"},
               "death_date": {"death_date", "death_year", "died", "died_on", "date_of_death"},
               "director": {"director", "directed", "directed_by", "film_director"},
               "author": {"author", "wrote", "written_by", "writer"}}
    return bool(value) and (requested == "other" or value in aliases.get(requested, {requested}))


def _typed_answer_reason(answer: str, answer_type: str) -> Optional[str]:
    kind = normalized_text(answer_type)
    if kind in {"year", "birth year", "death year"} and not re.search(r"\b\d{1,4}\b", answer):
        return "answer_type_mismatch"
    if kind in {"date", "birth date", "death date"} and not re.search(r"\d|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", answer, re.I):
        return "answer_type_mismatch"
    return None


def _same_entity(left: str, right: str, document: str) -> bool:
    if normalized_text(left) == normalized_text(right):
        return True
    # Mere co-occurrence of two names in one identity quote is not equivalence.
    return (any(normalized_text(right) == normalized_text(name) for name in _identity_names(left, document))
            or any(normalized_text(left) == normalized_text(name) for name in _identity_names(right, document)))


def _is_life_date(value: str) -> bool:
    """Recognize dates, not arbitrary parenthesized numeric intervals."""
    tokens = normalized_text(value).split()
    months = {"january", "february", "march", "april", "may", "june", "july", "august",
              "september", "october", "november", "december", "jan", "feb", "mar", "apr",
              "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "bc", "bce", "ad", "ce"}
    if not tokens or any(not token.isdigit() and token not in months for token in tokens):
        return False
    return any(token.isdigit() and len(token) == 4 for token in tokens) or (
        bool({"bc", "bce", "ad", "ce"} & set(tokens)) and any(token.isdigit() for token in tokens))


def _supported_life_range(relation: str, answer: str, entity: str, span: str,
                          document: str) -> bool:
    if relation not in {"birth_date", "death_date"}:
        return False
    for match in re.finditer(r"\(([^()\n]{3,90}?)\s*(?:–|—|\s-\s)\s*([^()\n]{3,90}?)\)", span):
        left, right = match.group(1).strip(), match.group(2).strip()
        if not _is_life_date(left) or not _is_life_date(right):
            continue
        prefix = re.split(r"[.;!?]\s*", span[:match.start()])[-1]
        if not supported_entity_surface(entity, prefix, document)[0]:
            continue
        if re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\b", prefix, re.I):
            continue
        value = left if relation == "birth_date" else right
        if _contains(answer, value):
            return True
    return False


def _attribute_reason(relation: str, answer: str, entity: str, span: str,
                      support: str, document: str) -> Optional[str]:
    if relation not in {"nationality", "birth_date", "birth_place", "death_date"}:
        return None
    surface, _ = supported_entity_surface(entity, span, document)
    anchored_support, _ = supported_entity_surface(entity, support, document)
    # Co-reference requires an explicit topic identity in this same document.
    pronoun = bool(_identity_names(entity, document)) and bool(anchored_support) and bool(
        re.search(r"(?:^|[.!?]\s+)(?:He|She|They|His|Her)\b", span))
    if not surface and not pronoun:
        return "relationship_subject_not_supported"
    if relation == "nationality":
        # An Italian film directed by X establishes the film's origin, not X's.
        escaped = re.escape(answer)
        unrelated = re.search(escaped + r"\s+(?:film|movie|novel|book|television\s+series|language|production)\b", span, re.I)
        if unrelated and not re.match(r"\s+(?:director|maker|producer|composer|writer|actor|actress)\b", span[unrelated.end():], re.I):
            return "answer_attribute_belongs_to_other_entity"
        if re.search(r"\b(?:born|birth)\b", span, re.I) and not re.search(
                r"\b(?:nationality|citizen|citizenship|national)\b|" + escaped +
                r"\s+(?:director|writer|author|actor|actress|composer|politician|scientist|singer|musician|filmmaker|screenwriter)\b", span, re.I):
            return "birthplace_does_not_establish_nationality"
        if not re.search(r"\b(?:nationality|citizen|citizenship|national|director|writer|author|actor|actress|composer|politician|person|scientist|singer|musician|filmmaker|screenwriter|is|was)\b", span, re.I):
            return "requested_relationship_not_in_quote"
        own_clause = False
        for clause in re.split(r"[.;!?]\s*", span):
            if _contains(answer, clause) and (supported_entity_surface(entity, clause, document)[0]
                                             or (pronoun and re.match(r"(?:He|She|They|His|Her)\b", clause.strip()))):
                own_clause = True
                break
        if not own_clause:
            return "answer_attribute_belongs_to_other_entity"
    else:
        marker = r"\b(?:born|birth)\b" if relation.startswith("birth") else r"\b(?:died|death)\b"
        matches = list(re.finditer(marker, span, re.I))
        if not matches:
            if _supported_life_range(relation, answer, entity, span, document):
                return None
            return "requested_relationship_not_in_quote"
        for match in matches:
            end = re.search(r"[);.!?]|\b(?:died|death|became|appointed|elected|served)\b", span[match.end():], re.I)
            suffix = span[match.end():match.end() + end.start()] if end else span[match.end():]
            before = span[:match.start()]
            head = re.split(r"[.;!?]\s*", before)[-1]
            if not supported_entity_surface(entity, head, document)[0] and not (
                    pronoun and re.match(r"(?:He|She|They|His|Her)\b", head.strip())):
                continue
            # Possessive relative statements must not move a parent's date to X.
            if re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\b[^.;!?]*$", before, re.I):
                continue
            if _contains(answer, suffix):
                return None
        return "answer_not_supported_for_requested_relationship"
    return None


def verify_typed_hypotheses(payload: dict, documents: Dict[int, str], question: str,
                            answer_type: str, validation: str = "legacy") -> Tuple[List[dict], List[dict]]:
    """Check quoted answers, same-document aliases, subject and attribute type."""
    if not isinstance(payload.get("hypotheses"), list):
        return [], [{"reason": "invalid_hypotheses_list"}]
    accepted, rejected, seen = [], [], set()
    if validation not in {"legacy", "strict_relation", "conservative_relation"}:
        raise ValueError(f"Unknown evidence binding validation: {validation}")
    spec = relation_spec(question) if validation != "legacy" else None
    if validation == "conservative_relation" and spec["relation"] in {"other", "comparison"}:
        return verify_hypotheses(payload, documents)
    relation = spec["relation"] if spec else requested_relation(question)
    requested_subject = None if spec else _question_subject(question, relation)
    for item in payload["hypotheses"]:
        base, failures = verify_hypotheses({"hypotheses": [item]}, documents)
        if not isinstance(item, dict):
            rejected.extend(failures)
            continue
        answer = str(item.get("answer", "")).strip()
        alias_kind = "literal"
        if not base and failures and failures[0].get("reason") == "answer_not_supported_in_quote":
            doc_id = failures[0]["doc_id"]
            surface, alias_kind = supported_entity_surface(answer, str(item.get("evidence", "")), documents[doc_id])
            if surface:
                base, failures = verify_hypotheses({"hypotheses": [dict(item, answer=surface)]}, documents)
        if not base:
            rejected.extend(failures)
            continue
        doc_id, span = base[0]["doc_id"], base[0]["evidence"]
        document = documents[doc_id]
        entity = str(item.get("answer_entity", "")).strip()
        claimed_relation = str(item.get("answer_relation", "")).strip()
        support = str(item.get("subject_evidence", "")).strip()
        reason = _typed_answer_reason(answer, answer_type)
        if not entity or len(entity) > 200:
            reason = reason or "missing_relationship_subject"
        elif len(support) < 8 or len(support) > 1500 or support not in document:
            reason = reason or "subject_evidence_not_exact_substring"
        elif not supported_entity_surface(entity, support, document)[0]:
            reason = reason or "relationship_subject_not_supported"
        elif not (relation_name_matches(claimed_relation, spec) if spec and relation not in {
                "birth_date", "birth_place", "death_date", "nationality"} else
                _relation_matches(claimed_relation, relation)):
            reason = reason or "requested_relationship_mismatch"
        elif requested_subject and not _same_entity(entity, requested_subject, document):
            reason = reason or "relationship_subject_mismatch"
        if not reason:
            if spec:
                guard = (conservative_relation_reason if validation == "conservative_relation"
                         else strict_relation_reason)
                reason = guard(spec, question, answer, answer_type, entity, span,
                               document, supported_entity_surface, _same_entity)
        if not reason:
            reason = _attribute_reason(relation, answer, entity, span, support, document)
        uncovered = (validation == "conservative_relation" and
                     rejection_disposition(reason) == "unrecognized_construction")
        retained_reason = reason if uncovered else None
        if uncovered:
            reason = None
        if reason:
            rejected.append({"answer": answer[:200], "doc_id": doc_id, "reason": reason})
            continue
        identity = (normalized_text(answer), doc_id, span)
        if identity in seen:
            continue
        seen.add(identity)
        checked = dict(base[0], answer=answer, answer_entity=entity,
                       answer_relation=claimed_relation, subject_evidence=support,
                       relation_supported=not uncovered,
                       binding_support={"answer_surface": alias_kind,
                                        "validation": (validation if spec else "source_grounded_typed"),
                                        "requested_relation": relation})
        if spec:
            checked["binding_support"]["mechanical_relation_checked"] = not uncovered
            checked["binding_support"]["entailment_guaranteed"] = False
        if validation == "conservative_relation":
            checked["binding_support"]["guard_disposition"] = (
                "literal_retained_unrecognized_construction" if uncovered else "mechanical_supported")
            if retained_reason:
                checked["binding_support"]["unrecognized_construction_reason"] = retained_reason
        accepted.append(checked)
    accepted.sort(key=lambda x: (-x["confidence"], x["doc_id"], normalized_text(x["answer"])))
    return accepted, rejected


class BindingImprovementMixin:
    def _verify(self, question: str, answer_type: str, docs: Dict[int, str]):
        if "binding" not in self.improvements:
            return super()._verify(question, answer_type, docs)
        validation = getattr(getattr(self, "cfg", None), "evidence_binding_validation", "legacy")
        if validation not in {"legacy", "strict_relation", "conservative_relation"}:
            raise ValueError(f"Unknown evidence binding validation: {validation}")
        spec = relation_spec(question) if validation != "legacy" else None
        relation = spec["relation"] if spec else requested_relation(question)
        key = question + "::" + ",".join(map(str, docs))
        if validation == "conservative_relation" and (
                relation in {"other", "comparison"} or
                (spec.get("subject") and re.search(r"\b(?:who|whose|that|which)\b", spec["subject"], re.I))):
            # Preserve the original request and its repair budget for relations
            # the local grammar cannot judge. Do not issue a second verifier.
            result = super()._verify(question, answer_type, docs)
            diagnostic = self._verification_diagnostics.get(key)
            if diagnostic is not None:
                diagnostic["binding_mode"] = "conservative_relation_literal_fallback"
                diagnostic["local_guard_reason"] = "unrecognized_construction"
            return result
        if spec and relation in {"other", "comparison"}:
            unsupported = ("comparison_requires_atomic_attribute_evidence" if relation == "comparison"
                           else "unrecognized_requested_relationship")
            self._verification_diagnostics[key] = {
                "attempts": 0, "outputs": [], "binding_mode": "strict_relation",
                "local_guard_reason": unsupported}
            return [], [{"reason": unsupported}], None
        messages = [
            {"role": "system", "content": (
                "Verify answers to a retrieval subquestion using ONLY the supplied passages. "
                "Return ONLY JSON {hypotheses:[{answer,evidence,doc_id,confidence,answer_entity,"
                "answer_relation,subject_evidence}]}. Retain up to 3 distinct supported answers. "
                "evidence and subject_evidence must each be exact contiguous quotes from the cited D-number. "
                "evidence must establish the requested relationship; subject_evidence must explicitly "
                "identify its subject. answer_entity is the entity whose relation is answered: for "
                "'Who directed Film X?' it is Film X; for 'When was Person X born?' it is Person X. "
                "answer_relation must name the requested relation, preferably the provided canonical label. "
                "Never transfer an attribute between nearby entities: 'Italian film directed by Sergio "
                "Nasca' proves the film's origin, not Nasca's nationality. A birth date must belong to "
                "the requested person, not their relative or another mentioned person. "
                "For a full-name answer, quote the preceding identity sentence too when practical. "
                "A title or first-sentence full name may license its literal surname or middle-name "
                "variant in the answer quote, but a different person with the same surname cannot. "
                "Do not guess aliases, expand ambiguous names, invent entities, or output hashes. "
                "Return an empty hypotheses list when the requested relationship is not supported.")},
            {"role": "user", "content": (
                f"Subquestion: {question}\nAnswer type: {answer_type}\nRequested relation: {relation}\n\n" +
                "\n\n".join(f"[D{k}]\n{text}" for k, text in docs.items()))},
        ]
        if spec:
            messages[0]["content"] += (
                " Strict relation mode: answer_entity must be the specific subject asked about, "
                "not another entity sharing a passage. A country containing a port does not "
                "inherit the port's location; a neighboring country in another country's sentence "
                "does not establish the queried adjacency. A date for one polity must never be "
                "bound to another polity. For the birth date or nationality of a work's creator, "
                "this same passage must explicitly link that work to that creator. Return no "
                "hypotheses for unresolved comparison questions or an unrecognized requested "
                "relation. Copy the clause that establishes subject, relation and answer together.")
            messages[1]["content"] += "\nLocal relation requirements: " + json.dumps(spec, ensure_ascii=False)
            if validation == "conservative_relation":
                messages[0]["content"] += (
                    " Conservative relation validation retains literal source-supported proposals "
                    "when local grammar cannot establish their relation, but never retains an "
                    "explicitly contradictory subject, attribute owner, or direction.")
        diagnostics = {"attempts": 0, "outputs": [], "binding_mode": (
            validation if spec else "source_grounded_typed")}
        self._verification_diagnostics[key] = diagnostics
        all_rejected = []
        reason = None
        for attempt in range(2):
            payload, reason = self._infer_object(messages)
            diagnostics["attempts"] += 1
            diagnostics["outputs"].append(payload)
            accepted, rejected = (([], [{"reason": reason}]) if reason else
                                  verify_typed_hypotheses(payload, docs, question, answer_type, validation))
            all_rejected.extend(rejected)
            if accepted or (not rejected and not reason):
                return accepted, all_rejected, reason
            if attempt == 0:
                messages += [
                    {"role": "assistant", "content": json.dumps(payload or {}, ensure_ascii=False)},
                    {"role": "user", "content": (
                        "These proposals failed local source checks: " + json.dumps(rejected, ensure_ascii=False) +
                        ". Recheck the original question and passages. Correct exact quotations, entity "
                        "identity and ownership of the requested property. Supply all seven fields. "
                        "Return {\"hypotheses\":[]} if unsupported. Never edit passage text or invent support.")},
                ]
        return [], all_rejected, reason
