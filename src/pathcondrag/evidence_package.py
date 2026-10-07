"""Bounded, source-grounded condition packages within the existing Top10.

The selector recognizes a few explicit English relations. It does not infer
arbitrary entailment, fetch passages, or alter the retrieval candidate pool.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .evidence_binding import _same_entity, _typed_answer_reason
from .evidence_relation_guard import _owned_clause, relation_spec, strict_relation_reason
from .evidence_retrieval import bind_question, branch_matches, normalized_text
from .evidence_terminal import _literal, _relation_check, _same_subject, _subject_surface
from .evidence_selection import _ancestors


def _title(document):
    lines = str(document).splitlines()
    return re.sub(r"\s*\([^()]+\)\s*$", "", lines[0]).strip() if len(lines) > 1 else ""


def _identity(entity, document):
    """An exact title or an explicitly supported full name, never surname alone."""
    title = _title(document)
    if not title:
        return False
    if _same_entity(entity, title, document):
        return True
    # Geographic titles sometimes append a region after the actual place name.
    if "," in title and normalized_text(title.split(",", 1)[0]) == normalized_text(entity):
        lead = " ".join(document.splitlines()[1:])[:450]
        return bool(re.match(re.escape(entity) + r"\s+(?:is|was)\b", lead, re.I))
    return False


def _clauses(document):
    body = " ".join(str(document).splitlines()[1:])
    return [c.strip() for c in re.split(r"(?<=[.!?;])\s+", body) if c.strip()]


def _proper_anchors(question):
    names = re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*", question)
    ignore = {"Who", "What", "Which", "Where", "When", "How", "Is", "Was", "The", "A"}
    return [n.rstrip("?.") for n in names if n.rstrip("?.") not in ignore]


def _year_conflict(question, clauses):
    years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", question))
    # Only compare the actual relation-bearing sentences, not unrelated birth
    # dates or other works anywhere in the passage.
    mentioned = set(re.findall(r"\b(?:18|19|20)\d{2}\b", " ".join(clauses)))
    return bool(years and mentioned and years.isdisjoint(mentioned))


def _founder_witness(question, answer, quote, document):
    if not re.search(r"\b(?:co[- ]?)?founded\b|\bfounder\b", question, re.I):
        return None
    for anchor in _proper_anchors(question):
        if _literal(anchor, answer):
            continue
        for marker in re.finditer(r"\b(?:co[- ]?)?founded\s+by\b", quote, re.I):
            if (_owned_clause(anchor, quote[:marker.start()], document, _subject_surface, _same_subject)
                    and _literal(answer, quote[marker.end():])
                    and not re.search(r"\b(?:not|never)\b", quote, re.I)):
                return {"relation": "founder", "anchor": anchor, "quote": quote}
        # Active founder construction: the answer owns the predicate.
        for marker in re.finditer(r"\b(?:co[- ]?)?founded\b", quote, re.I):
            if (_owned_clause(answer, quote[:marker.start()], document, _subject_surface, _same_subject)
                    and _literal(anchor, quote[marker.end():])
                    and not re.search(r"\b(?:not|never)\b", quote, re.I)):
                return {"relation": "founder", "anchor": anchor, "quote": quote}
    return None


def _grounded_proofs(nodes, branch, bindings, candidates, rank, document, diag):
    proofs = {}
    for nid, proof in (branch.get("proofs", {}).items() if isinstance(branch.get("proofs"), dict) else []):
        reason = None
        if nid not in nodes or not isinstance(proof, dict):
            continue
        try:
            doc_id, confidence = int(proof.get("doc_id")), float(proof.get("confidence", 0))
        except (ValueError, TypeError, OverflowError):
            diag["rejected_proofs"].append({"node": nid, "reason": "invalid_proof_fields"})
            continue
        answer, quote = str(proof.get("answer", "")), str(proof.get("evidence", ""))
        requirements = proof.get("requirements", {})
        if doc_id not in candidates or doc_id not in rank:
            reason = "proof_outside_existing_pool"
        elif nid not in bindings or normalized_text(answer) != normalized_text(bindings[nid]):
            reason = "proof_conflicts_with_winning_binding"
        elif not isinstance(requirements, dict) or not branch_matches(requirements, bindings):
            reason = "proof_conflicts_with_winning_branch"
        elif not math.isfinite(confidence) or confidence < .85:
            reason = "invalid_or_low_confidence"
        elif len(quote) < 8 or quote not in document(doc_id) or not _literal(answer, quote):
            reason = "proof_not_literal_grounded"
        elif _typed_answer_reason(answer, nodes[nid].get("answer_type", "")):
            reason = "answer_type_conflict"
        elif re.search(r"\b(?:not|never|without)\b", quote, re.I):
            reason = "negated_quote_not_supported"
        if reason:
            diag["rejected_proofs"].append({"node": nid, "doc_id": doc_id, "reason": reason})
        else:
            proofs[nid] = dict(proof, doc_id=doc_id)
    return proofs


def _local_citation(node, proof, bindings, document):
    """Validate the promoted fact; predecessor citations stay literal anchors."""
    question = bind_question(node.get("question", ""), bindings)
    if question is None:
        return None, "unbound_node_question"
    check, reason = _relation_check(node, question, proof, document, bindings)
    if check:
        return {"check": check, "quote": proof["evidence"]}, None
    deps = node.get("depends_on", [])
    if len(deps) != 1 or deps[0] not in bindings:
        return None, reason
    entity, answer, quote = bindings[deps[0]], proof["answer"], proof["evidence"]
    if not _identity(entity, document):
        return None, "candidate_identity_mismatch"
    # An explicitly owned geographic/government membership fact.
    if re.search(r"\b(?:belong|government|region|located|situated|within)\b", question, re.I):
        for clause in _clauses(document):
            marker = re.search(r"\b(?:is|was)\b.*?\b(?:within|part of|in|located in|situated in)\b", clause, re.I)
            if (marker and _owned_clause(entity, clause[:marker.start()], document, _subject_surface, _same_subject)
                    and _literal(answer, clause[marker.start():])
                    and not re.search(r"\b(?:not|never)\b", clause, re.I)):
                return {"check": "owned_geographic_membership", "quote": clause}, None
    # A film based on a bound work may refer to itself as It in the actual
    # director sentence. Its same-document lead must establish the adaptation.
    creator = re.search(r"\b(?:directed|director)\b", question, re.I)
    if creator and re.search(r"\bfilm\b.*\bbased\b", question, re.I):
        lead = _clauses(document)[0] if _clauses(document) else ""
        if (_literal(entity, lead) and re.search(r"\bfilm\b.*\bbased\s+on\b", lead, re.I)
                and not _year_conflict(question, [lead])):
            direct_question = f"Who directed {entity}?"
            reason = strict_relation_reason(relation_spec(direct_question), direct_question, answer,
                                            node.get("answer_type", ""), entity, quote, document,
                                            _subject_surface, _same_subject)
            if reason is None:
                return {"check": "owned_adaptation_creator", "quote": quote, "identity_quote": lead}, None
    return None, reason or "unknown_local_relation"


def _predecessor_supported(query, node, proof, bindings, document):
    """A literal predecessor still needs an owned relation, not confidence."""
    question = bind_question(node.get("question", ""), bindings)
    if question is None:
        return False
    if _relation_check(node, question, proof, document, bindings)[0]:
        return True
    if _founder_witness(question, proof["answer"], proof["evidence"], document):
        return True
    quote = proof["evidence"]
    if re.search(r"\b(?:grew|grow|growing|grown)\s+up\b", question, re.I):
        for entity in _proper_anchors(question):
            marker = re.search(r"\b(?:grew|grow|growing|grown)\s+up\b", quote, re.I)
            if (marker and _owned_clause(entity, quote[:marker.start()], document, _subject_surface, _same_subject)
                    and _literal(proof["answer"], quote[marker.end():])):
                return True
    # Same-document song identity licenses a quoted 'part of ... rock opera'
    # construction. The song/title must be named in the actual input question.
    title, clauses = _title(document), _clauses(document)
    if (title and _literal(title, query) and clauses and _literal(title, clauses[0])
            and re.search(r"\bis\s+a\s+song\b", clauses[0], re.I)
            and re.search(r"\brock\s+opera\b", question, re.I)
            and re.match(r"\s*Part\s+of\b", quote, re.I)
            and re.search(r"\brock\s+opera\b", quote, re.I)
            and _literal(proof["answer"], quote)):
        return True
    return False


def _identity_package(query, node, proof, target, document, selected_documents):
    """Bind a founder's identity page to both explicit question conditions.

    This deliberately narrow construction needs an existing founder citation,
    an owned founder statement on the identity page, and a selected actor page
    explicitly connecting that same person to the requested actor's films.
    """
    answer = str(proof["answer"])
    if not _identity(answer, target):
        return None, "candidate_identity_mismatch"
    if normalized_text(node.get("answer_type", "")) not in {"person", "people", "name", "entity"}:
        return None, "identity_answer_type_not_supported"
    lead = _clauses(target)[0] if _clauses(target) else ""
    if (re.search(r"\bis\b.*\b(?:film|movie|song|book|novel|city|suburb|country|company)\b", lead, re.I)
            and not re.search(r"\b(?:filmmaker|actor|actress|writer|director|composer|producer|author|person)\b", lead, re.I)):
        return None, "identity_document_type_conflict"
    # Reuse established creator guards: both the selected source and the
    # identity article must explicitly attribute the same work to this person.
    # The identity title alone is never a reason for a promotion.
    specification = relation_spec(node.get("question", ""))
    if specification["relation"] in {"director", "author", "composer"} and specification.get("subject"):
        work = specification["subject"]
        source_reason = strict_relation_reason(specification, node["question"], answer,
                                               node.get("answer_type", ""), work, proof["evidence"], document,
                                               _subject_surface, _same_subject)
        if source_reason:
            return None, "creator_source_not_owned"
        for clause in _clauses(target):
            reason = strict_relation_reason(specification, node["question"], answer,
                                            node.get("answer_type", ""), work, clause, target,
                                            _subject_surface, _same_subject)
            if (reason is None and not _year_conflict(query, [clause])
                    and not re.search(r"\b(?:not|never)\b", clause, re.I)):
                return {"check": "creator_identity_and_owned_same_work", "quote": clause,
                        "source_quote": proof["evidence"], "work": work}, None
        return None, "identity_same_work_creator_not_supported"
    if not re.search(r"\bproduc(?:ed|er)\b.*\b(?:movies|films)\b.*\b(?:starring|starred)\b", query, re.I):
        return None, "unknown_identity_package_conditions"
    witness = _founder_witness(query, answer, proof["evidence"], document)
    if not witness:
        return None, "founder_citation_not_owned"
    actor = re.search(r"\b(?:starring|starred)\s+(.+?)[?.]*$", query, re.I)
    actor = actor.group(1).strip() if actor else ""
    if not actor or not re.fullmatch(r"[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){1,4}", actor):
        return None, "unresolved_actor_condition"
    founder_clauses = []
    for clause in _clauses(target):
        if (not _literal(witness["anchor"], clause) or
                not re.search(r"\b(?:co[- ]?)?founded\b", clause, re.I) or
                re.search(r"\b(?:not|never)\s+(?:\w+\s+){0,2}(?:co[- ]?)?founded\b", clause, re.I)):
            continue
        # A topic pronoun must own the predicate, never another named founder.
        topic_pronoun = re.search(r"\b(?:he|she)\s+(?:co[- ]?)?founded\b", clause, re.I)
        other_person = re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:is|was|had|has)\b", clause)
        if topic_pronoun and not other_person:
            founder_clauses.append(clause)
        elif _founder_witness(query, answer, clause, target):
            founder_clauses.append(clause)
    if not founder_clauses or _year_conflict(query, founder_clauses):
        return None, "identity_founder_or_date_condition_not_supported"
    if not re.search(r"\b(?:co[- ]?)?produc(?:ed|er)|\bfilmmaker\b", target, re.I):
        return None, "identity_production_type_not_supported"
    for doc_id, actor_doc in selected_documents:
        if not _identity(actor, actor_doc):
            continue
        for clause in _clauses(actor_doc):
            # Require the actor to own the acting relation and the full bound
            # name to modify films. Mere names co-occurring is insufficient.
            if not _owned_clause(actor, clause, actor_doc, _subject_surface, _same_subject):
                continue
            match = re.search(re.escape(answer) + r"(?:['’]s)?\s+(?:movies|films)\b", clause, re.I)
            if (match and re.search(r"\b(?:roles|starred|starring|acted)\b", clause, re.I)
                    and not re.search(r"\b(?:not|never)\b", clause, re.I)):
                return {"check": "founder_identity_and_owned_actor_films", "quote": founder_clauses[0],
                        "actor_quote": clause, "actor_doc_id": doc_id, "founder_quote": witness["quote"]}, None
    return None, "selected_actor_film_condition_not_supported"


def rerank_evidence_packages(query, order, state, document, max_promotions=2):
    """Return a bounded reorder and diagnostics; never inspect labels or types."""
    original = [int(d) for d in order]
    order = list(original)
    diag = {"enabled": True, "policy": "literal_winning_condition_packages", "extra_requests": 0,
            "gold_labels_used": False, "entailment_guaranteed": False, "promotions": [],
            "rejected_proofs": [], "rejected_packages": [], "deferred": [], "max_promotions": 2,
            "top2_preserved": True, "top10_set_preserved": True, "top200_set_preserved": True,
            "document_set_preserved": True, "top5_preserved": True}
    trace = state.get("evidence_trace", {})
    plan = state.get("_evidence_plan", trace.get("plan", []))
    bindings = state.get("_evidence_winning_bindings", trace.get("bindings", {}))
    ancestors, node_order, error = _ancestors(plan) if isinstance(plan, list) else ({}, [], "invalid_plan")
    routing = trace.get("planning_outputs", {}).get("routing", {})
    if routing.get("expand") and routing.get("kind") in {"bridge_comparison", "parallel_comparison", "parallel_attribute"}:
        error = "successful_parallel_route_not_replaced"
    if len(set(original)) != len(original) or not isinstance(bindings, dict):
        error = error or "invalid_order_or_bindings"
    if not node_order or len(node_order) > 2:
        error = error or "outside_one_or_two_node_scope"
    if len(node_order) == 2 and len(ancestors.get(node_order[-1], set())) != 1:
        error = error or "independent_nodes_not_replaced"
    diag["scope_reason"] = error
    if error:
        return original, diag
    branches = state.get("_evidence_beams") or trace.get("branch_scores", [])
    branch = next((b for b in branches if isinstance(b, dict) and isinstance(b.get("bindings"), dict)
                   and set(b["bindings"]) == set(bindings) and branch_matches(b["bindings"], bindings)), None)
    if branch is None:
        diag["scope_reason"] = "missing_exact_winning_branch"
        return original, diag
    nodes = {n["id"]: n for n in plan}
    candidates, rank = state.get("evidence_candidates", {}), {d: i + 1 for i, d in enumerate(original)}
    proofs = _grounded_proofs(nodes, branch, bindings, candidates, rank, document, diag)
    protected = {p["doc_id"] for p in proofs.values()}
    proposals = []
    for nid in node_order:
        if nid not in proofs:
            continue
        proof, required = proofs[nid], ancestors[nid]
        bundle = [proofs[n]["doc_id"] for n in node_order if n in required and n in proofs]
        if (any(n not in proofs for n in required) or not set(bundle) <= set(original[:5])
                or any(not _predecessor_supported(query, nodes[n], proofs[n], bindings,
                                                 document(proofs[n]["doc_id"])) for n in required)):
            diag["deferred"].append({"node": nid, "reason": "missing_selected_literal_predecessor"})
            continue
        doc_id = proof["doc_id"]
        if 6 <= rank[doc_id] <= 10:
            check, reason = _local_citation(nodes[nid], proof, bindings, document(doc_id))
            if check:
                proposals.append(dict(check, node=nid, doc_id=doc_id, bundle_doc_ids=bundle + [doc_id]))
            else:
                diag["rejected_packages"].append({"node": nid, "doc_id": doc_id, "reason": reason})
        # An already selected root citation can corroborate a missing identity
        # article even when there is no explicit DAG leaf for that condition.
        if not required and proof["doc_id"] in original[:5]:
            for candidate_id in original[5:10]:
                if candidate_id not in candidates:
                    continue
                check, reason = _identity_package(query, nodes[nid], proof, document(candidate_id),
                                                  document(proof["doc_id"]),
                                                  [(d, document(d)) for d in original[:5]])
                if check:
                    other_doc = [check["actor_doc_id"]] if "actor_doc_id" in check else []
                    proposals.append(dict(check, node=nid, doc_id=candidate_id,
                                          bundle_doc_ids=[proof["doc_id"]] + other_doc + [candidate_id]))
                elif _identity(proof["answer"], document(candidate_id)):
                    diag["rejected_packages"].append({"node": nid, "doc_id": candidate_id, "reason": reason})
    seen = set()
    for proposal in sorted(proposals, key=lambda p: (rank[p["doc_id"]], p["node"])):
        doc_id = proposal["doc_id"]
        if doc_id in seen or doc_id in order[:5]:
            continue
        if len(diag["promotions"]) >= min(2, max(0, int(max_promotions))):
            diag["deferred"].append({"doc_id": doc_id, "reason": "promotion_budget"})
            continue
        protected.update(proposal["bundle_doc_ids"])
        victim = next((d for d in reversed(order[2:5]) if d not in protected
                       and not candidates.get(d, {}).get("verified")), None)
        if victim is None:
            diag["deferred"].append({"doc_id": doc_id, "reason": "no_unprotected_prefix_slot"})
            continue
        target, source = order.index(victim), order.index(doc_id)
        order[target], order[source] = doc_id, victim
        seen.add(doc_id)
        diag["promotions"].append(dict(proposal, from_rank=rank[doc_id], to_rank=target + 1, victim=victim))
    diag.update(top2_preserved=order[:2] == original[:2], top5_preserved=order[:5] == original[:5],
                top10_set_preserved=set(order[:10]) == set(original[:10]),
                top200_set_preserved=set(order[:200]) == set(original[:200]),
                document_set_preserved=set(order) == set(original), literal_winning_proofs=len(proofs))
    return order, diag


class EvidencePackageMixin:
    """Opt-in local finalizer; no flag means exact delegation to the parent."""

    def finalize(self, query, ids, scores, ctx, state):
        result = super().finalize(query, ids, scores, ctx, state)
        if "package" not in self.improvements:
            return result
        original_ids, original_scores, original_trace = result
        final, diagnostic = rerank_evidence_packages(query, original_ids, state, self._document)
        trace = dict(original_trace, improvement_package=diagnostic)
        if np.array_equal(np.asarray(final), original_ids):
            return original_ids, original_scores, trace
        previous = {p["doc_id"]: p for p in trace.get("selected_prefix", [])}
        trace["selected_prefix"] = [dict(previous.get(d, {"doc_id": d, "selection_source": "condition_package"}),
                                         original_greedy_rank=list(original_ids).index(d) + 1) for d in final[:5]]
        return np.asarray(final, dtype=int), np.arange(len(final), 0, -1, dtype=float) / max(1, len(final)), trace
