"""Local, optional source witnesses for existing model-produced hypotheses.

The checks recognize explicit English clause frames and source-owned aliases.
They are not a general entailment model. Unknown original answers are retained;
only strictly supported formatting/identity repairs can become new answers.
"""
from __future__ import annotations

import math
import re

from .evidence_binding import _identity_names, _same_entity, _typed_answer_reason, supported_entity_surface
from .evidence_package import _title
from .evidence_relation_guard import relation_spec
from .evidence_retrieval import normalized_text, verify_hypotheses


_PUNCT = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\u00a0": " "}
_STOP = set("who what which where when how is was are were do does did have has had the a an of in on at to for from with by its it this that these those name called known number many much more less".split())
# Lexical alternations apply only after concrete entity ownership and the
# clause's argument direction have been established. No dataset routing occurs.
_IRREGULAR = {"written": "write", "wrote": "write", "spoken": "speak", "spoke": "speak", "made": "make", "won": "win", "sang": "sing", "sung": "sing", "built": "build", "born": "birth", "b": "birth", "died": "die", "death": "die", "people": "person", "men": "man", "women": "woman"}
_PREDICATES = {"write", "speak", "make", "win", "sing", "build", "play", "portray", "voice", "star", "record", "perform", "release", "design", "invent", "found", "establish", "dedicate", "name", "rename", "locate", "situate", "base", "headquarter", "reside", "live", "use", "command", "captain", "member", "belong", "collaborate", "serve", "work", "produce", "manufacture", "own", "contain", "include", "house", "create", "direct", "compose", "award", "receive", "study", "license", "criticize", "border", "marry", "spouse"}
_PERSON = {"person", "human", "player", "actor", "actress", "author", "writer", "director", "composer", "singer"}
_MONTHS = set("january february march april may june july august september october november december".split())


def source_witness_status(check, reason):
    return "supported" if check else "contradicted" if str(reason).startswith("contradicted_") else "unknown"


def _literal(value, text):
    value = normalized_text(value)
    return bool(value) and f" {value} " in f" {normalized_text(text)} "


def _normalized_source(text):
    """Normalize only spacing and punctuation, retaining each source offset."""
    chars, offsets = [], []
    for i, char in enumerate(str(text)):
        char = _PUNCT.get(char, char)
        if char.isspace():
            char = " "
        if char == " " and chars and chars[-1] == " ":
            continue
        chars.append(char); offsets.append(i)
    return "".join(chars), offsets


def literal_source_span(quote, document):
    """Find an exact or uniquely reversible formatting span, never a paraphrase."""
    quote, document = str(quote), str(document)
    if not quote or len(quote) < 8 or len(quote) > 1500:
        return None, "unknown_quote_size"
    position = document.find(quote)
    if position >= 0:
        return (position, position + len(quote)), "exact"
    if "..." in quote or "…" in quote:
        return None, "unknown_ellipsis_is_not_contiguous_quote"
    needle, _ = _normalized_source(quote)
    haystack, offsets = _normalized_source(document)
    found = [m.start() for m in re.finditer(re.escape(needle), haystack)]
    if len(found) == 1:
        start = found[0]
        return (offsets[start], offsets[start + len(needle) - 1] + 1), "reversible_whitespace_or_punctuation"
    # Token-adjacent punctuation changes remain reversible, but punctuation is
    # read from the source again during clause/direction checks. Never ignore
    # letters, digits, ellipses, or skipped words.
    def skeleton(value):
        out, mapping = [], []
        for i, c in enumerate(value):
            if c.isalnum() or c.isspace():
                c = " " if c.isspace() else c
                if c == " " and out and out[-1] == " ":
                    continue
                out.append(c); mapping.append(i)
        return "".join(out), mapping
    needle, _ = skeleton(quote)
    needle = needle.strip()
    haystack, mapping = skeleton(document)
    # Trimming can invalidate offsets when source starts with spaces; use only
    # word-started sources in this fallback. Exact whitespace mapping above
    # covers other cases without changing index arithmetic.
    if not needle or not document or document[:1].isspace() or len(needle) < 8:
        return None, "unknown_quote_not_in_source"
    found = [m.start() for m in re.finditer(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack)]
    if len(found) != 1:
        return None, "unknown_quote_not_unique" if found else "unknown_quote_not_in_source"
    start = found[0]
    return (mapping[start], mapping[start + len(needle) - 1] + 1), "reversible_punctuation"


def _sentences(text, base=0):
    """Return exact spans; keep initials, honorifics and b./e.g. abbreviations."""
    ends = []
    for match in re.finditer(r"[.!?;]\s+|\n+", text):
        if text[match.start():match.start()+1] == "." and re.search(
                r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|St|Sr|Jr|b)|\b[A-Z])$", text[:match.start()]):
            continue
        ends.append(match.end())
    starts, ends = [0] + ends, ends + [len(text)]
    result = []
    for start, end in zip(starts, ends):
        left = len(text[start:end]) - len(text[start:end].lstrip())
        right = len(text[start:end].rstrip())
        if right > left:
            result.append((base+start+left, base+start+right, text[start+left:start+right]))
    return result


def _lemma(word):
    w = normalized_text(word)
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w in _PREDICATES:
        return w
    for suffix in ["ing", "ed", "er", "or", "es", "s"]:
        if len(w) > len(suffix)+2 and w.endswith(suffix):
            root = w[:-len(suffix)]
            for candidate in [root, root+"e", root[:-1] if len(root)>1 and root[-1]==root[-2] else root]:
                if candidate in _PREDICATES:
                    return candidate
    return w


def _tokens(text):
    return [(m.group(), m.start(), m.end(), _lemma(m.group())) for m in re.finditer(r"\w+(?:[-’']\w+)*", text)]


def _anchors(question, node, bindings):
    deps = [str(bindings[d]) for d in node.get("depends_on", []) if d in bindings]
    spec = relation_spec(question)
    finite_subject = spec.get("subject") if spec["relation"] in {
        "birth_place", "birth_date", "death_date", "nationality", "director", "author",
        "composer", "established_date"
    } else None
    # Structural templates recover literal titles even when their spelling is
    # lowercase or contains punctuation; they do not invent a missing subject.
    patterns = [r"how\s+many\s+.+?\s+(?:does|did|do)\s+(.+?)\s+have[?.]*$",
                r"(?:where|who)\s+(?:is|was|are|were)\s+(.+?)(?:\s+(?:located|situated|based|headquartered|from))?[?.]*$",
                r"(?:what|which)\s+.+?\s+(?:of|for)\s+(.+?)[?.]*$"]
    explicit = [finite_subject] if finite_subject else []
    for pattern in patterns:
        if finite_subject and pattern.startswith("(?:where|who)"):
            continue
        match = re.fullmatch(pattern, question.strip(), re.I)
        if match and not re.search(r"\b(?:who|whose|that|which|and|or)\b", match.group(1), re.I):
            value = match.group(1).strip().rstrip("?.")
            if not (pattern.startswith("(?:where|who)") and re.match(r"who\b", question, re.I)
                    and re.search(r"\b(?:the|of|commander|director|author|composer|inventor|founder)\b", value, re.I)):
                explicit.append(value)
    quoted = re.findall(r'"([^"\n]{2,150})"|“([^”\n]{2,150})”', question)
    names = re.findall(r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’.-]*(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’.-]*)*", question)
    ignored = {"Who", "What", "Which", "Where", "When", "How", "Is", "Was", "Are", "Does", "Did", "The", "A", "In", "At", "On"}
    names = [n.rstrip("?.") for n in names if n.rstrip("?.") not in ignored and normalized_text(n) not in _MONTHS]
    # Exact structural subjects contain their shorter proper-name fragments.
    values = deps + explicit + [a or b for a,b in quoted] + names
    return list(dict.fromkeys(v for v in values if v and not any(
        v != longer and _literal(v, longer) for longer in deps+explicit)))


def _topic_identity(entity, document):
    return _same_entity(entity, _title(document), document) or bool(_identity_names(entity, document))


def _adjacent_identity(entity, document, start):
    if not _topic_identity(entity, document):
        return None
    previous = _sentences(document[:start])
    if not previous:
        return None
    a, b, clause = previous[-1]
    if b-a > 550 or re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\b", clause, re.I):
        return None
    surface = supported_entity_surface(entity, clause, document)[0]
    if not surface:
        return None
    occurrence = re.search(re.escape(surface), clause, re.I)
    if not occurrence or clause[:occurrence.start()].strip() or re.match(r"['’]s\b", clause[occurrence.end():]):
        return None
    rest = clause[:occurrence.start()] + "TOPIC" + clause[occurrence.end():]
    if re.search(r"\b[A-Z][\w'’-]+\s+[A-Z][\w'’-]+", rest):
        return None
    return a


def _surface(entity, clause, document, start):
    """Literal/owned alias or an adjacent topic pronoun, never fuzzy similarity."""
    surface, kind = supported_entity_surface(entity, clause, document)
    if surface:
        if kind == "document_anchored_surname":
            previous = _sentences(document[:start])
            if previous and not _literal(entity, previous[-1][2]):
                other = re.findall(r"\b[A-Z][\w'’-]+\s+[A-Z][\w'’-]+", previous[-1][2])
                if any(not _same_entity(entity, n, document) for n in other):
                    return None
        pattern = r"\s+".join(re.escape(part) for part in surface.split())
        literal = re.search(pattern, clause, re.I)
        return literal.group() if literal else surface
    pronoun = re.match(r"\s*(?:(?:In|During)\s+[^,;.!?]{1,35},\s*)?(He|She|It|They|His|Her|Its)\b", clause, re.I)
    if pronoun and _adjacent_identity(entity, document, start) is not None:
        return pronoun.group(1)
    return None


def _type_conflict(answer, kind, quote):
    if _typed_answer_reason(answer, kind):
        return "contradicted_type:answer_type_mismatch"
    if normalized_text(kind) in _PERSON:
        people = re.split(r"\s+(?:and|&)\s+|\s*;\s*", answer)
        name = r"[A-ZÀ-ÖØ-Þ][\w'’.-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'’.-]*)*"
        if len(people) > 1 and all(re.fullmatch(name, value) for value in people):
            return "contradicted_type:multiple_people_for_single_person"
    if normalized_text(kind) in {"boolean", "bool"} and normalized_text(answer) not in {"yes", "no", "true", "false"}:
        return "contradicted_type:non_boolean_answer"
    return None


def _constraints(question, clause, document, start, node, bindings, answer):
    years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", question))
    found = set(re.findall(r"\b(?:18|19|20)\d{2}\b", clause))
    if years and not years <= found:
        # Another event's year is not evidence that the requested event never
        # happened in the requested year. Treat a missing qualifier as unknown.
        return "unknown_missing_year_qualifier"
    for anchor in _anchors(question, node, bindings):
        if _same_entity(anchor, answer, document):
            continue
        if not _surface(anchor, clause, document, start):
            return "unknown_missing_named_constraint"
    for qualifier in ["first", "last", "largest", "smallest", "oldest", "youngest", "only", "former", "current", "national", "international"]:
        if re.search(r"\b"+qualifier+r"(?:ly)?\b", question, re.I) and not re.search(r"\b"+qualifier+r"(?:ly)?\b", clause, re.I):
            return "unknown_missing_explicit_qualifier"
    season = re.search(r"\bseason\s+(\d+)\b", question, re.I)
    if season and not re.search(r"\bseason\s+"+re.escape(season.group(1))+r"\b", clause, re.I):
        return "unknown_missing_season_qualifier"
    return None


def _owned_head(surface, before, entity, document):
    occurrence = re.search(r"(?<!\w)"+re.escape(surface)+r"(?!\w)", before, re.I)
    if not occurrence:
        return False, None
    prefix, between = before[:occurrence.start()].strip(), before[occurrence.end():]
    if re.match(r"['’]s\s+(?:father|mother|brother|sister|son|daughter|wife|husband)\b", between, re.I):
        return False, "contradicted_owner:relative_attribute"
    if re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\s+of\s*$", prefix, re.I):
        return False, "contradicted_owner:relative_attribute"
    if prefix and not re.fullmatch(r"(?:The\s+|A\s+|An\s+)?(?:In\s+[^,]{1,35},\s*)?", prefix, re.I):
        return False, None
    if re.search(r"\b(?:who|whose|which)\b|\band\s+[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)*\s+(?:was|is|had|has|won|built|made|created)\b", between):
        return False, "unknown_clause_owner_changed"
    return True, None


def _generic_clause(node, question, proof, clause, document, start, bindings):
    answer = proof["answer"]
    answer_surface = _surface(answer, clause, document, start)
    if not answer_surface:
        return None, "unknown_answer_surface"
    anchors = _anchors(question, node, bindings)
    entities = [a for a in anchors if not _same_entity(a, answer, document)]
    if not entities:
        return None, "unknown_no_concrete_subject"
    words = _tokens(clause)
    requested = {_lemma(w) for w in re.findall(r"\w+", question) if normalized_text(w) not in _STOP}
    shared = requested & _PREDICATES
    simple_agent = bool(re.match(r"\s*Who\s+(?!(?:is|was|are|were)\b)", question, re.I))
    requested_mode = "agent" if simple_agent else "object"
    direction_certain = simple_agent or bool(re.match(r"\s*(?:What|Which)\b.*\b(?:does|did|do)\b", question, re.I))
    if re.search(r"\b(?:commander|performer|inventor|founder|creator|director|author|composer)\b", question, re.I) and re.match(r"\s*Who\b", question, re.I):
        requested_mode = "agent"; direction_certain = True
    role_question = re.match(r"\s*Who\s+(?:is|was)\s+(?:the\s+)?(\w+)\s+of\b", question, re.I)
    if role_question and _lemma(role_question.group(1)) in _PREDICATES:
        requested_mode = "agent"; direction_certain = True
    query_tokens = _tokens(question)
    first_action = next((a for t,a,b,l in query_tokens if l in _PREDICATES), None)
    if (first_action is not None and re.match(r"\s*(?:What|Which)\b", question, re.I)
            and not re.search(r"\b(?:is|was|are|were|does|did|do)\b", question[:first_action], re.I)
            and not any(_literal(e, question[:first_action]) for e in entities)):
        requested_mode = "agent"; direction_certain = True
    for entity in entities:
        entity_surface = _surface(entity, clause, document, start)
        if not entity_surface:
            continue
        for token, pstart, pend, lemma in words:
            if lemma not in shared:
                continue
            before, after = clause[:pstart], clause[pend:]
            e_before = _literal(entity_surface, before); a_before = _literal(answer_surface, before)
            e_after = _literal(entity_surface, after); a_after = _literal(answer_surface, after)
            # Passive voice reverses the agent/object alignment explicitly.
            passive = bool(re.match(r"\s+by\b", after, re.I))
            if requested_mode == "agent":
                owner = answer_surface if a_before and e_after and not passive else entity_surface if e_before and a_after and passive else None
            else:
                owner = entity_surface if e_before and a_after and not passive else answer_surface if a_before and e_after and passive else None
            if owner:
                supported, conflict = _owned_head(owner, before, entity, document)
                if conflict:
                    return None, conflict
                if supported and not re.search(r"\b(?:not(?!\s+only)|never|neither|without)\b", before+after, re.I):
                    return "source_owned_predicate_frame:"+lemma, None
            # Only flag a direction contradiction when both arguments and the
            # same predicate are explicit in one simple owning clause.
            if direction_certain and requested_mode == "agent" and e_before and a_after and not passive:
                ok, _ = _owned_head(entity_surface, before, entity, document)
                if ok:
                    return None, "contradicted_direction:agent_object_reversed"
            if direction_certain and requested_mode == "object" and a_before and e_after and not passive:
                ok, _ = _owned_head(answer_surface, before, answer, document)
                if ok:
                    return None, "contradicted_direction:agent_object_reversed"
        # An identity question asks for an owned copular description, not an
        # arbitrary relation merely because its names share one article.
        if re.fullmatch(r"\s*Who\s+(?:is|was)\s+"+re.escape(entity)+r"[?. ]*", question, re.I):
            marker = re.search(r"\b(?:is|was)\b", clause, re.I)
            if marker and _literal(answer_surface, clause[marker.end():]):
                ok, conflict = _owned_head(entity_surface, clause[:marker.start()], entity, document)
                if conflict:
                    return None, conflict
                if ok:
                    return "source_owned_copular_identity", None
        # Exact count and its requested unit must be adjacent in an owned
        # statement. A number elsewhere in a biography/table is not a witness.
        count = re.match(r"\s*How\s+many\s+(.+?)\s+(?:does|did|do)\b", question, re.I)
        if count:
            units = {_lemma(w) for w in re.findall(r"\w+", count.group(1)) if normalized_text(w) not in _STOP}
            pos = re.search(re.escape(answer_surface), clause, re.I)
            epos = re.search(re.escape(entity_surface), clause, re.I)
            if pos and epos and epos.start()<pos.start() and units:
                following = [_lemma(w) for w in re.findall(r"\w+", clause[pos.end():pos.end()+45])]
                suffix = set(following[:len(units)])
                ok, conflict = _owned_head(entity_surface, clause[:pos.start()], entity, document)
                if conflict:
                    return None, conflict
                if ok and units <= suffix:
                    return "source_owned_count_and_unit", None
    return None, "unknown_no_owned_predicate_frame"


def source_witness_check(node, bound_question, proof, document, bindings):
    """Return (check, reason); check exists only for bounded supported facts.

    Reasons prefixed contradicted_ mark explicit conflicts; unknown_* denotes
    incomplete grammar/source coverage and must never be upgraded to support.
    """
    if not isinstance(proof, dict) or not isinstance(node, dict):
        return None, "unknown_invalid_proof"
    answer, quote = str(proof.get("answer", "")).strip(), str(proof.get("evidence", "")).strip()
    position = document.find(quote)
    if not answer or len(answer)>200 or len(quote)<8 or len(quote)>1500 or position<0:
        return None, "unknown_quote_not_literal_source"
    if re.search(r"\b(?:not|never|without|neither)\b", bound_question, re.I):
        return None, "unknown_negative_question_semantics"
    conflict = _type_conflict(answer, node.get("answer_type", ""), quote)
    if conflict:
        return None, conflict
    candidates = [(start,end,c) for start,end,c in _sentences(quote, position) if _surface(answer,c,document,start)]
    if not candidates:
        return None, "unknown_answer_not_grounded"
    # Existing finite relation checks remain available. This late import avoids
    # a module cycle when the DAG finalizer installs this function as callback.
    from .evidence_dag_package import _proof_relation
    old_reason = None
    reasons = []
    for start,end,clause in candidates:
        if re.search(r"\b(?:not(?!\s+only)|never|neither|without)\b", clause, re.I) and not re.search(r"\b(?:not|never|without)\b", bound_question, re.I):
            requested = {_lemma(word) for word in re.findall(r"\w+", bound_question)}
            negated = [match.group(1) for match in re.finditer(
                r"\b(?:not(?!\s+only)|never|neither|without)\s+"
                r"(?:(?:ever|been|being|be|to|have|having|\w+ly)\s+){0,2}(\w+)", clause, re.I)]
            same_predicate = any(_lemma(word) in requested & (_PREDICATES | {"birth", "die"})
                                 for word in negated)
            reasons.append("contradicted_negation:positive_question_negated_source" if same_predicate
                           else "unknown_negation_of_other_or_ambiguous_predicate")
            continue
        constraint = _constraints(bound_question, clause, document, start, node, bindings, answer)
        if constraint:
            reasons.append(constraint); continue
        old_check, old_reason = _proof_relation(node, bound_question, dict(proof, evidence=clause), document, bindings)
        if old_check:
            return "source_existing_relation:"+old_check, None
        check, reason = _generic_clause(node, bound_question, proof, clause, document, start, bindings)
        if check:
            return check, None
        reasons.append(reason)
    # A quoted opposite life event or explicit relative ownership is a conflict,
    # not merely unrecognized paraphrase. Other legacy reasons stay unknown.
    spec = relation_spec(bound_question)
    if old_reason in {"answer_attribute_belongs_to_other_entity", "relationship_direction_conflict", "biography_lead_identity_conflict"}:
        return None, "contradicted_owner:"+old_reason
    if spec["relation"] in {"birth_date","birth_place"} and re.search(r"\b(?:died|death)\b", quote, re.I) and not re.search(r"\b(?:born|birth)\b", quote, re.I):
        return None, "contradicted_direction:death_does_not_establish_birth"
    return None, next((r for r in reasons if str(r).startswith("contradicted_")), reasons[0] if reasons else "unknown_relationship")


def _recover_candidate(item, document, question, answer_type):
    """Repair one existing proposal, preserving a real continuous source span."""
    if not isinstance(item, dict):
        return None, "unknown_invalid_hypothesis"
    answer, raw_quote = str(item.get("answer", "")).strip(), str(item.get("evidence", "")).strip()
    span, mode = literal_source_span(raw_quote, document)
    if span is None:
        return None, mode
    start, end = span; quote = document[start:end]
    if not _literal(answer, quote):
        reference = _surface(answer, quote, document, start)
        if not reference or not _topic_identity(answer, document):
            return None, "unknown_no_anchored_answer_alias"
        # Restore the nearest literal full identity before the predicate, never
        # a long-distance arbitrary title plus an unrelated named subject.
        before = document[:start]
        full = list(re.finditer(r"(?<!\w)"+re.escape(answer)+r"(?!\w)", before, re.I))
        if not full:
            return None, "unknown_full_identity_not_literal"
        left = full[-1].start()
        sentence = next((a for a,b,c in _sentences(document[:start]) if a<=left<b), left)
        if end-sentence>1500:
            return None, "unknown_identity_span_too_long"
        start = sentence; quote = document[start:end];mode += "+contiguous_identity"
    recovered = dict(item, answer=answer, evidence=quote)
    raw_id = str(item.get("doc_id", ""))
    if not re.fullmatch(r"D?\d+", raw_id):
        return None, "unknown_document_reference"
    try:
        confidence = float(item.get("confidence", .5))
    except (TypeError, ValueError, OverflowError):
        return None, "unknown_invalid_confidence"
    if not math.isfinite(confidence):
        return None, "unknown_invalid_confidence"
    doc_id=int(raw_id.lstrip("D")); recovered["doc_id"]=doc_id
    basic, _ = verify_hypotheses({"hypotheses":[recovered]}, {doc_id:document})
    if not basic:
        return None, "unknown_repaired_quote_not_basic_grounded"
    recovered=dict(item,**basic[0])
    check, reason = source_witness_check({"answer_type":answer_type,"depends_on":[]},question,recovered,document,{})
    if not check:
        return None, reason
    recovered["source_witness"]={"status":"supported","check":check,"reason":None,
                                 "source_span":[start,end],"repair_mode":mode,
                                 "original_evidence":raw_quote,"direct_extra_requests":0}
    return recovered, mode


class SourceWitnessMixin:
    """Optional verifier wrapper using only its parent's actual model outputs."""

    def _verify(self, question, answer_type, docs):
        result=super()._verify(question,answer_type,docs)
        if "source_witness" not in self.improvements:
            return result
        accepted,rejected,reason=result
        key=question+"::"+",".join(map(str,docs))
        diagnostic=self._verification_diagnostics.get(key)
        if not isinstance(diagnostic,dict):
            diagnostic={"attempts":0,"outputs":[]};self._verification_diagnostics[key]=diagnostic
        stat={"direct_extra_requests":0,"may_enable_downstream_requests":True,
              "accepted_supported":0,"accepted_unknown":0,"contradicted_count":0,
              "repair_attempts":0,"repair_accepted":0,"repair_rejected":0,
              "repairs":[],"rejections":[]}
        diagnostic["source_witness"]=stat
        kept=[];extra_rejected=[]
        for item in accepted:
            doc_id=item.get("doc_id");source=docs.get(doc_id,"")
            check,local_reason=source_witness_check({"answer_type":answer_type,"depends_on":[]},question,item,source,{})
            status=source_witness_status(check,local_reason)
            if status=="contradicted":
                entry={"doc_id":doc_id,"answer":item.get("answer"),"reason":local_reason}
                stat["contradicted_count"]+=1;stat["rejections"].append(entry);extra_rejected.append(entry)
                continue
            quote=str(item.get("evidence",""));start=source.find(quote)
            metadata={"status":status,"check":check,"reason":local_reason,
                      "source_span":[start,start+len(quote)] if start>=0 else None,"direct_extra_requests":0}
            kept.append(dict(item,source_witness=metadata));stat["accepted_"+status]+=1
        # Preserve existing accepted hypotheses and their order. Repair only an
        # empty accepted set; do not replace a valid answer with a new alias beam.
        if kept:
            return kept,list(rejected)+extra_rejected,reason
        seen=set()
        for payload in diagnostic.get("outputs",[]):
            if not isinstance(payload,dict) or not isinstance(payload.get("hypotheses",[]),list):
                continue
            for item in payload.get("hypotheses",[]):
                if not isinstance(item,dict):continue
                raw_id=str(item.get("doc_id",""));match=re.fullmatch(r"D?(\d+)",raw_id)
                if not match or int(match.group(1)) not in docs:continue
                unique=(int(match.group(1)),str(item.get("answer","")),str(item.get("evidence","")))
                if unique in seen:continue
                seen.add(unique);stat["repair_attempts"]+=1
                repair,repair_reason=_recover_candidate(item,docs[int(match.group(1))],question,answer_type)
                if repair:
                    if not any(normalized_text(v['answer'])==normalized_text(repair['answer']) for v in kept):
                        kept.append(repair);stat["repair_accepted"]+=1;stat["repairs"].append({"doc_id":repair["doc_id"],"answer":repair["answer"],"source_span":repair["source_witness"]["source_span"],"repair_mode":repair_reason})
                else:
                    stat["repair_rejected"]+=1;stat["rejections"].append({"doc_id":int(match.group(1)),"answer":item.get("answer"),"reason":repair_reason})
                    if source_witness_status(None,repair_reason)=="contradicted":stat["contradicted_count"]+=1
        return kept[:3],list(rejected)+extra_rejected,None if kept else reason
