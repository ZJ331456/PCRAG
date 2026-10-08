"""Experimental local selection of connected, complementary condition evidence.

Only an existing Top10 passage can replace an unprotected Top5 passage. A
literal title edge must connect it to an already selected query-grounded root,
and its locally owned text must improve original-question condition coverage.
Lexical support is not entailment; no binding is changed and no requests occur.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .evidence_anchor_companion import _literal, _norm, _plan_reason, _protected, _specific, _title
from .evidence_binding import supported_entity_surface


PREFIX_K, FIXED_K, TOP_K = 5, 2, 10
MIN_ROOT_TERMS = 2
MIN_CONDITION_TERMS = 2
MIN_DISCRIMINATIVE_IDF = 1.35
MIN_ROOT_NOVELTY_WEIGHT = 1.20
MIN_VICTIM_GAIN = .75
MAX_SENTENCE_CHARS = 700
_STOP = frozenset("a an the and or of in on at to for from by with as is are was were be "
                  "been being who what which where when why how do does did have has had "
                  "it its their his her he she they this that these those one name kind "
                  "type called would could can individual person year years".split())


def _word(word):
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def _words(text):
    return [_word(t) for t in _norm(text).split() if t not in _STOP and (len(t) > 2 or t.isdigit())]


def _terms(text):
    return set(_words(text))


def _sentences(source):
    body = source.partition("\n")[2]
    # Boundaries determine eligible local evidence only, never proof rejection.
    return [s.strip() for s in re.split(r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z])|\n+", body)
            if 8 <= len(s.strip()) <= MAX_SENTENCE_CHARS]


def _owned(source, title):
    """Literal identity, an owning introductory lead, or an adjacent pronoun."""
    rows, sentences = [], _sentences(source)
    title_words = _words(title)
    lead_owned = False
    for i, sentence in enumerate(sentences):
        surface, reason = supported_entity_surface(title, sentence, source)
        if surface:
            # A sentence about a named relative does not own that relative's fact.
            position = _norm(sentence).find(_norm(surface))
            prefix = _norm(sentence)[:position]
            if re.search(r"\b(?:father|mother|son|daughter|wife|husband)\s+of\s*$", prefix):
                continue
            check = "document_identity:" + str(reason)
        elif i == 0 and len(title_words) >= 2 and sentence.lstrip('"“').casefold().startswith(
                _norm(title).split()[0]) and set(title_words) <= _terms(sentence):
            check = "introductory_title_terms"
        elif i == 1 and lead_owned and re.match(r"(?:He|She|It|They|His|Her|Its)\b", sentence):
            check = "adjacent_owned_lead_pronoun"
        else:
            continue
        if re.search(r"\b(?:not|never|without)\b", sentence, re.I):
            continue
        lead_owned = lead_owned or i == 0
        rows.append({"sentence": sentence, "identity_check": check, "terms": _terms(sentence)})
    return rows


def _phrase(query_words, sentence, title, idf):
    """An original adjacent content phrase with a locally discriminative word."""
    sentence_words, title_words = _words(sentence), _terms(title)
    for i in range(len(query_words) - 1):
        pair = query_words[i:i + 2]
        if not any(idf.get(w, 0) >= MIN_DISCRIMINATIVE_IDF for w in pair) or set(pair) <= title_words:
            continue
        if any(sentence_words[j:j + 2] == pair for j in range(len(sentence_words) - 1)):
            return " ".join(pair)
    return None


def rerank_condition_companions(query, order, state, document):
    """Return one bounded swap, or the original order with gate diagnostics."""
    original = [int(d) for d in order]
    diag = {"enabled": True, "policy": "literal_connected_condition_complement",
            "gold_labels_used": False, "extra_requests": 0, "entailment_guaranteed": False,
            "promotions": [], "roots": [], "options": [], "gate_counts": {}, "abstain_reason": None,
            "top2_preserved": True, "top10_set_preserved": True, "suffix_preserved": True,
            "document_set_preserved": True, "max_promotions": 1}
    gates = diag["gate_counts"]

    def count(key):
        gates[key] = gates.get(key, 0) + 1

    if len(original) != len(set(original)) or len(original) <= PREFIX_K:
        diag["abstain_reason"] = "duplicate_documents_or_short_ranking"
        return original, diag
    trace = state.get("evidence_trace", {})
    reason = _plan_reason(query, trace, state)
    if reason:
        diag["abstain_reason"] = reason
        return original, diag
    sources = {d: str(document(d)) for d in original[:TOP_K]}
    titles = {d: _title(s) for d, s in sources.items()}
    ambiguous = {_norm(t) for t in titles.values() if sum(_norm(t) == _norm(v) for v in titles.values()) > 1}
    query_words, query_terms = _words(query), _terms(query)
    document_terms = {d: _terms(s.partition("\n")[2]) for d, s in sources.items()}
    idf = {w: 1. + math.log((len(sources) + 1.) / (1. + sum(w in terms for terms in document_terms.values())))
           for w in query_terms}
    owned = {d: _owned(s, titles[d]) for d, s in sources.items()}
    protected = _protected(trace, original)
    protected.update(d for d in original[:PREFIX_K] if _specific(titles[d]) and _literal(titles[d], query))
    diag["protected_doc_ids"], diag["query_term_idf"] = sorted(protected), idf
    roots = []
    for root in original[:PREFIX_K]:
        count("roots_examined")
        if not _specific(titles[root]) or _norm(titles[root]) in ambiguous:
            count("root_unspecific_or_ambiguous_title")
            continue
        best = None
        for row in owned[root]:
            hits = row["terms"] & query_terms
            discriminative = {w for w in hits if idf[w] >= MIN_DISCRIMINATIVE_IDF}
            phrase = _phrase(query_words, row["sentence"], titles[root], idf)
            anchor = _literal(titles[root], query)
            if not (anchor or phrase or len(discriminative) >= MIN_ROOT_TERMS):
                continue
            score = sum(idf[w] for w in hits)
            if best is None or score > best[0]:
                best = (score, row, hits, phrase, anchor)
        if best is None:
            count("root_no_owned_query_condition")
            continue
        _, row, hits, phrase, anchor = best
        roots.append((root, hits))
        count("qualified_roots")
        diag["roots"].append({"doc_id": root, "title": titles[root], "source_sentence": row["sentence"],
                              "query_hits": sorted(hits), "query_phrase": phrase, "literal_query_anchor": anchor,
                              "identity_check": row["identity_check"]})
    options = []
    for candidate in original[PREFIX_K:TOP_K]:
        count("candidates_examined")
        if not _specific(titles[candidate]) or _norm(titles[candidate]) in ambiguous:
            count("candidate_unspecific_or_ambiguous_title")
            continue
        for root, root_hits in roots:
            count("pairs_examined")
            forward = next((s for s in _sentences(sources[root]) if _literal(titles[candidate], s)), None)
            reverse = next((s for s in _sentences(sources[candidate]) if _literal(titles[root], s)), None)
            if not (forward or reverse):
                count("no_literal_title_edge")
                continue
            count("literal_title_edges")
            if forward and re.search(r"\b(?:starring|cast|actors|actresses|directors|composers)\b", forward, re.I):
                count("creator_or_cast_enumeration")
                continue
            if not owned[candidate]:
                count("candidate_no_owned_sentence")
            condition_terms = query_terms - _terms(titles[root])
            for row in owned[candidate]:
                count("owned_candidate_sentences")
                hits = row["terms"] & condition_terms
                if len(hits) < MIN_CONDITION_TERMS:
                    count("insufficient_condition_terms")
                    continue
                novelty = hits - root_hits
                novelty_weight = sum(idf[w] for w in novelty)
                if novelty_weight < MIN_ROOT_NOVELTY_WEIGHT:
                    count("no_discriminative_root_complement")
                    continue
                victims = [d for d in original[FIXED_K:PREFIX_K] if d not in protected and d != root]
                if not victims:
                    count("no_unprotected_victim")
                    continue
                victim = min(victims, key=lambda d: (sum(idf[w] for w in condition_terms & document_terms[d]),
                                                      -original.index(d)))
                victim_hits = condition_terms & document_terms[victim]
                gain = sum(idf[w] for w in hits) - sum(idf[w] for w in victim_hits)
                if gain < MIN_VICTIM_GAIN or not (novelty - victim_hits):
                    count("no_weighted_victim_gain")
                    continue
                count("qualified_options")
                evidence = {"doc_id": candidate, "root_doc_id": root, "victim": victim,
                            "source_sentence": row["sentence"], "identity_check": row["identity_check"],
                            "forward_link_sentence": forward, "reverse_link_sentence": reverse,
                            "condition_hits": sorted(hits), "root_novel_terms": sorted(novelty),
                            "victim_condition_hits": sorted(victim_hits), "weighted_gain": gain,
                            "novelty_weight": novelty_weight}
                options.append((gain + .25 * novelty_weight, candidate, victim, evidence))
                diag["options"].append(evidence)
    final = list(original)
    if options:
        score, candidate, victim, evidence = max(options, key=lambda x: (x[0], -original.index(x[1]), -x[1]))
        a, b = final.index(candidate), final.index(victim)
        final[a], final[b] = final[b], final[a]
        diag["promotions"].append(dict(evidence, score=score, from_rank=a + 1, to_rank=b + 1))
    else:
        diag["abstain_reason"] = "no_qualified_condition_companion"
    diag.update(top2_preserved=final[:FIXED_K] == original[:FIXED_K],
                top10_set_preserved=set(final[:TOP_K]) == set(original[:TOP_K]),
                suffix_preserved=final[TOP_K:] == original[TOP_K:],
                document_set_preserved=set(final) == set(original),
                protected_proofs_preserved=protected <= set(final[:PREFIX_K]))
    if not all(diag[k] for k in ("top2_preserved", "top10_set_preserved", "suffix_preserved",
                                  "document_set_preserved", "protected_proofs_preserved")):
        diag["abstain_reason"], diag["promotions"] = "ranking_invariant_failure", []
        return original, diag
    return final, diag


class ConditionCompanionMixin:
    def finalize(self, query, ids, scores, ctx, state):
        result = super().finalize(query, ids, scores, ctx, state)
        if "condition_companion" not in self.improvements or getattr(self, "stage", 4) < 4:
            return result
        original_ids, original_scores, trace = result
        final, diagnostic = rerank_condition_companions(query, original_ids, dict(state, evidence_trace=trace), self._document)
        trace = dict(trace, improvement_condition_companion=diagnostic)
        if np.array_equal(np.asarray(final), original_ids):
            return original_ids, original_scores, trace
        old = {p["doc_id"]: p for p in trace.get("selected_prefix", [])}
        trace["selected_prefix"] = [dict(old.get(d, {"doc_id": d, "selection_source": "condition_companion"}),
                                          original_greedy_rank=list(original_ids).index(d) + 1)
                                    for d in final[:PREFIX_K]]
        return np.asarray(final, dtype=int), np.arange(len(final), 0, -1, dtype=float) / max(1, len(final)), trace
