"""Bounded anchor completion and locally supported companion selection.

This optional postprocessor uses the question, existing passages and saved
retrieval proofs. It issues no requests and never changes a binding. Its lexical
checks are selection evidence, not general entailment. Top2, the Top10 set and
the suffix after Top10 remain fixed; Top5 quality still requires evaluation.
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np

from .evidence_binding import supported_entity_surface
from .evidence_selection import _ancestors


TOP_K = 10
PREFIX_K = 5
FIXED_K = 2
MIN_CONFIDENCE = .85
MIN_CONDITION_HITS = 2
ANCHOR_THRESHOLD = .79
COMPANION_THRESHOLD = .80
MAX_SENTENCE_CHARS = 600
_STOP = frozenset("a an the and or of in on at to for from by with as is are was were be "
                  "been being who what which where when why how do does did have has had "
                  "it its their his her he she they this that these those one name named "
                  "kind type called would could can show known former individual person".split())
_GENERIC = _STOP | frozenset("film movie song album book television city country county "
                            "town state river university company religion leadership "
                            "history music actor actress composer director president "
                            "birthday birthplace nationality football basketball".split())
_COMPARATIVE = re.compile(r"\b(?:older|younger|earlier|later|longer|shorter|taller|"
                          r"larger|smaller|compare|compared|versus)\b", re.I)
_FAMILIES = (
    (frozenset({"theme", "themes"}), frozenset({"theme", "themes", "spiritual"})),
    (frozenset({"born", "birth", "birthplace"}), frozenset({"born", "birth"})),
    (frozenset({"died", "death"}), frozenset({"died", "death"})),
    (frozenset({"composer", "composed"}), frozenset({"composer", "composed"})),
    (frozenset({"director", "directed"}), frozenset({"director", "directed"})),
    (frozenset({"author", "wrote", "written", "screenplay"}),
     frozenset({"author", "wrote", "written", "screenplay"})),
    (frozenset({"libretto"}), frozenset({"libretto"})),
    (frozenset({"county", "city", "town", "located", "location"}),
     frozenset({"county", "city", "town", "located", "location"})),
    (frozenset({"president", "officer", "chief"}), frozenset({"president", "officer", "chief"})),
    (frozenset({"valuable", "award", "awards", "won", "winner"}),
     frozenset({"valuable", "award", "awards", "won", "winner"})),
    (frozenset({"released", "release", "premiered"}), frozenset({"released", "release", "premiered"})),
)


def _norm(value):
    # Accent folding is a spelling normalization, not a surname/name alias.
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    return " ".join(re.findall(r"\w+", "".join(c for c in text if not unicodedata.combining(c))))


def _literal(phrase, text):
    phrase = _norm(phrase)
    return bool(phrase) and f" {phrase} " in f" {_norm(text)} "


def _title(document):
    lines = str(document).splitlines()
    return re.sub(r"\s*\([^()]*\)\s*$", "", lines[0]).strip() if len(lines) > 1 else ""


def _specific(title):
    tokens = _norm(title).split()
    content = [t for t in tokens if t not in _GENERIC]
    return bool(content) and (len(content) >= 2 or len(content[0]) >= 7) and len(tokens) <= 12


def _terms(text):
    return set(_norm(text).split()) - _STOP


def _sentences(document):
    body = str(document).partition("\n")[2]
    # Unknown boundaries merely prevent local evidence from passing. They
    # never invalidate an existing proof or remove a candidate passage.
    return [s.strip() for s in re.split(r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z])|\n+", body)
            if 8 <= len(s.strip()) <= MAX_SENTENCE_CHARS]


def _plan_reason(query, trace, state):
    if _COMPARATIVE.search(query):
        return "comparative_question_out_of_scope"
    plan = state.get("_evidence_plan", trace.get("plan", []))
    ancestors, order, error = _ancestors(plan) if isinstance(plan, list) else ({}, [], "invalid_plan")
    if error:
        return error
    if any(len(node.get("depends_on", [])) > 1 for node in plan):
        return "multi_parent_join_out_of_scope"
    if sum(not n.get("depends_on") for n in plan) > 1:
        return "parallel_roots_out_of_scope"
    if any(len(ancestors[nid]) > 1 for nid in order):
        return "deep_plan_out_of_scope"
    return None


def _protected(trace, original):
    diagnostic = trace.get("improvement_dag_package", {})
    eligible = diagnostic.get("eligible_proofs", {}) if isinstance(diagnostic, dict) else {}
    if not isinstance(eligible, dict):
        return set()
    protected = set()
    for rows in eligible.values():
        if isinstance(rows, (list, tuple)):
            for proof in rows:
                if isinstance(proof, dict) and proof.get("doc_id") in original[:PREFIX_K]:
                    protected.add(proof["doc_id"])
    return protected


def _saved_support(trace, doc_id, source):
    """Supplementary literal citations never become a new accepted binding."""
    answers = []
    bindings = trace.get("bindings", {})
    for branch in trace.get("branch_scores", []):
        if branch.get("bindings") != bindings:
            continue
        for node_id, proof in branch.get("proofs", {}).items():
            try:
                confidence = float(proof.get("confidence", 0))
            except (TypeError, ValueError, OverflowError):
                continue
            quote, answer = str(proof.get("evidence", "")), str(proof.get("answer", ""))
            if (proof.get("doc_id") == doc_id and math.isfinite(confidence) and confidence >= MIN_CONFIDENCE
                    and 8 <= len(quote) <= 1500 and quote in source and _literal(answer, quote)
                    and _norm(answer) == _norm(bindings.get(node_id, ""))):
                answers.append(answer)
    return answers


def _conditions(query_terms, sentence, roots):
    hits = (query_terms - set().union(*(_terms(r) for r in roots))) & _terms(sentence)
    families = [sorted(q & query_terms) for q, passage in _FAMILIES
                if q & query_terms and passage & _terms(sentence)]
    return hits, families


def rerank_anchor_companions(query, order, state, document, mode="combined"):
    """Swap at most one existing Top10 passage per enabled mechanism into Top5."""
    original = [int(d) for d in order]
    diag = {"enabled": True, "mode": mode, "policy": "bounded_literal_anchor_companions",
            "gold_labels_used": False, "extra_requests": 0, "entailment_guaranteed": False,
            "promotions": [], "rejected_candidates": [], "abstain_reason": None,
            "top2_preserved": True, "top10_set_preserved": True, "suffix_preserved": True,
            "document_set_preserved": True, "max_promotions_per_mechanism": 1}
    if mode not in {"anchor", "companion", "combined"} or len(original) != len(set(original)):
        diag["abstain_reason"] = "invalid_mode_or_duplicate_documents"
        return original, diag
    trace = state.get("evidence_trace", {})
    reason = _plan_reason(query, trace, state)
    if reason or len(original) <= PREFIX_K:
        diag["abstain_reason"] = reason or "short_ranking"
        return original, diag
    sources = {d: str(document(d)) for d in original[:TOP_K]}
    titles = {d: _title(s) for d, s in sources.items()}
    duplicate_titles = {_norm(t) for t in titles.values()
                        if sum(_norm(t) == _norm(other) for other in titles.values()) > 1}
    query_terms = _terms(query)
    anchors = {d for d, title in titles.items() if _specific(title) and _literal(title, query)
               and _norm(title) not in duplicate_titles}
    protected = _protected(trace, original) | (anchors & set(original[:PREFIX_K]))
    diag["protected_doc_ids"] = sorted(protected)
    final = list(original)

    def promote(options, mechanism):
        victims = [d for d in final[FIXED_K:PREFIX_K] if d not in protected]
        if not options or not victims:
            return
        checked = []
        for score, candidate, evidence in options:
            terms = set(evidence.get("condition_terms", query_terms))
            victim = min(victims, key=lambda d: (len(terms & _terms(sources[d])), -final.index(d)))
            old_hits = len(terms & _terms(sources[victim]))
            if mechanism == "companion" and len(evidence["condition_hits"]) <= old_hits:
                continue
            checked.append((score, candidate, victim, dict(evidence, victim_condition_hits=old_hits)))
        if not checked:
            return
        score, candidate, victim, evidence = max(
            checked, key=lambda item: (item[0], -original.index(item[1]), -item[1]))
        if candidate in final[:PREFIX_K]:
            return
        index, target = final.index(candidate), final.index(victim)
        final[index], final[target] = victim, candidate
        protected.add(candidate)
        diag["promotions"].append(dict(evidence, mechanism=mechanism, doc_id=candidate, victim=victim,
                                       score=score, from_rank=index + 1, to_rank=target + 1))

    if mode in {"anchor", "combined"}:
        options = []
        for d in original[PREFIX_K:TOP_K]:
            if d in anchors:
                hits = (query_terms - _terms(titles[d])) & _terms(sources[d].partition("\n")[2])
                score = .75 + .04 * min(4, len(hits))
                if score >= ANCHOR_THRESHOLD:
                    options.append((score, d, {"anchor": titles[d], "condition_hits": sorted(hits),
                                              "source_sentence": sources[d].partition("\n")[2][:600]}))
        promote(options, "anchor")

    if mode in {"companion", "combined"}:
        options = []
        roots = [d for d in final[:PREFIX_K] if d in anchors]
        for d in final[PREFIX_K:TOP_K]:
            if not _specific(titles[d]) or _norm(titles[d]) in duplicate_titles:
                continue
            for root in roots:
                root_sentences, candidate_sentences = _sentences(sources[root]), _sentences(sources[d])
                direct = next((s for s in root_sentences if _literal(titles[d], s)), None)
                reverse = next((s for s in candidate_sentences if _literal(titles[root], s)), None)
                literal_binding = next((a for a in _saved_support(trace, root, sources[root])
                                        if _literal(titles[d], a)), None)
                if not (direct or reverse or literal_binding):
                    continue
                link = direct or reverse
                if direct and re.search(r"\b(?:starring|cast|actors|actresses|directors|composers)\b", direct, re.I):
                    continue  # Enumerated creator/cast names need richer ownership evidence.
                for sentence in candidate_sentences:
                    surface, identity_check = supported_entity_surface(titles[d], sentence, sources[d])
                    if not surface:
                        continue
                    hits, families = _conditions(query_terms, sentence, [titles[root], titles[d]])
                    if len(hits) < MIN_CONDITION_HITS or not families:
                        continue
                    if re.search(r"\b(?:not|never|without)\b", sentence, re.I):
                        continue
                    score = .70 + .04 * min(4, len(hits)) + .05 * bool(direct and reverse) + .03 * bool(literal_binding)
                    if score >= COMPANION_THRESHOLD:
                        options.append((score, d, {"root_doc_id": root, "root_title": titles[root],
                                                  "candidate_title": titles[d], "link_sentence": link,
                                                  "literal_binding": literal_binding, "source_sentence": sentence,
                                                  "subject_surface": surface, "identity_check": identity_check,
                                                  "condition_terms": sorted(query_terms - _terms(titles[root]) - _terms(titles[d])),
                                                  "condition_hits": sorted(hits), "condition_families": families}))
        promote(options, "companion")

    diag.update(top2_preserved=final[:FIXED_K] == original[:FIXED_K],
                top10_set_preserved=set(final[:TOP_K]) == set(original[:TOP_K]),
                suffix_preserved=final[TOP_K:] == original[TOP_K:],
                document_set_preserved=set(final) == set(original),
                protected_proofs_preserved=_protected(trace, original) <= set(final[:PREFIX_K]))
    if not all(diag[k] for k in ("top2_preserved", "top10_set_preserved", "suffix_preserved",
                                  "document_set_preserved", "protected_proofs_preserved")):
        diag["abstain_reason"], diag["promotions"] = "ranking_invariant_failure", []
        return original, diag
    return final, diag


class AnchorCompanionMixin:
    """Opt-in finalizer, preserving parent arrays on a no-op."""

    def finalize(self, query, ids, scores, ctx, state):
        result = super().finalize(query, ids, scores, ctx, state)
        anchor, companion = "anchor_completion" in self.improvements, "bridge_companion" in self.improvements
        if not (anchor or companion) or getattr(self, "stage", 4) < 4:
            return result
        original_ids, original_scores, trace = result
        mode = "combined" if anchor and companion else "anchor" if anchor else "companion"
        replay_state = dict(state, evidence_trace=trace)
        final, diagnostic = rerank_anchor_companions(query, original_ids, replay_state, self._document, mode)
        trace = dict(trace, improvement_anchor_companion=diagnostic)
        if np.array_equal(np.asarray(final), original_ids):
            return original_ids, original_scores, trace
        previous = {p["doc_id"]: p for p in trace.get("selected_prefix", [])}
        trace["selected_prefix"] = [dict(previous.get(d, {"doc_id": d, "selection_source": "anchor_companion"}),
                                          original_greedy_rank=list(original_ids).index(d) + 1)
                                    for d in final[:PREFIX_K]]
        output_scores = np.arange(len(final), 0, -1, dtype=float) / max(1, len(final))
        return np.asarray(final, dtype=int), output_scores, trace
