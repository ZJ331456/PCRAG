"""Finite, source-grounded selection of ancestor-closed DAG evidence packages.

Only existing winning answers and exact source quotations are considered. The
local grammar checks are bounded syntactic support, not general entailment.
No model requests, document fetches, or benchmark annotations are used.
"""
from __future__ import annotations

from itertools import product
import math
import re

import numpy as np

from .evidence_binding import _same_entity, _typed_answer_reason
from .evidence_package import _founder_witness, _local_citation, _title
from .evidence_relation_guard import relation_spec, strict_relation_reason
from .evidence_retrieval import bind_question, branch_matches, normalized_text
from .evidence_selection import _ancestors
from .evidence_terminal import _literal, _relation_check, _same_subject, _subject_surface


def _topic_pronoun_safe(entity, quote, document):
    """A topic pronoun needs an adjacent owning identity, not distant co-occurrence."""
    if not _same_entity(entity, _title(document), document):
        return False
    offset = document.find(quote)
    before = document[:offset].strip()
    clauses = [c.strip() for c in re.split(r"(?<=[.!?])\s+|\n+", before) if c.strip()]
    if not clauses:
        return False
    previous = clauses[-1]
    # A title immediately followed by its quoted lead is a same-document owner.
    if normalized_text(previous) == normalized_text(_title(document)):
        return True
    if len(previous) > 550:
        return False
    surface = _subject_surface(entity, previous, document)[0]
    if not surface:
        return False
    start = re.search(re.escape(surface), previous, re.I)
    return bool(start and not previous[:start.start()].strip()
                and not re.match(r"['’]s\b", previous[start.end():])
                and not re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\b",
                                  previous[start.end():], re.I))


def _intro_birth_date(entity, answer, quote, document):
    """Explicit biography lead: title and named lead identify the same person.

    Reordered full names are permitted only when the document title matches the
    bound person and the lead contains exactly the same name tokens. This is not
    a general token-permutation alias rule.
    """
    if not _same_entity(entity, _title(document), document):
        return False
    body = " ".join(document.splitlines()[1:]).lstrip()
    if not body.startswith(quote):
        return False
    lead = re.match(r"([A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*){1,5})\s*\((?:born|b\.)\s+([^)]{3,90})\)", quote)
    if not lead:
        return False
    name, date = lead.groups()
    exact_identity = _same_entity(entity, name, document)
    same_tokens = (sorted(normalized_text(entity).split()) == sorted(normalized_text(name).split())
                   and 2 <= len(normalized_text(entity).split()) <= 5)
    return bool((exact_identity or same_tokens) and _literal(answer, date) and re.search(r"\d", answer))


def _proof_relation(node, question, proof, document, bindings):
    """Recheck the actual predicate; confidence and same-page names are insufficient."""
    quote, answer = proof["evidence"], proof["answer"]
    spec = relation_spec(question)
    entity = spec.get("subject")
    if not entity and len(node.get("depends_on", [])) == 1:
        entity = bindings.get(node["depends_on"][0])
    if entity:
        clauses = re.split(r"(?<=[.!?])\s+|\n+", quote)
        for clause in clauses:
            if (re.match(r"\s*(?:He|She|It|They|His|Her|Its)\b", clause, re.I)
                    and not _topic_pronoun_safe(entity, clause.strip(), document)):
                return None, "ambiguous_document_topic_pronoun"
        if spec["relation"] in {"birth_date", "birth_place", "death_date", "nationality"}:
            body = " ".join(document.splitlines()[1:]).lstrip()
            lead = re.match(r"([A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*){1,5})(?=\s*\(|\s+(?:is|was)\b)", quote)
            if lead and body.startswith(quote) and _same_entity(entity, _title(document), document):
                name = lead.group(1)
                if (not _same_entity(entity, name, document)
                        and sorted(normalized_text(entity).split()) != sorted(normalized_text(name).split())):
                    return None, "biography_lead_identity_conflict"
    if spec["relation"] == "birth_date" and entity and _intro_birth_date(entity, answer, quote, document):
        return "owned_intro_birth_date", None
    check, reason = _relation_check(node, question, proof, document, bindings)
    # The terminal helper's broad unknown-relation cue is unsuitable for joint
    # promotion: require explicit bounded ownership checks instead.
    if check and check != "matched_literal_relation_cue":
        return check, None
    if (entity and spec["relation"] in {"director", "author", "composer", "location", "established_date"}
            and strict_relation_reason(spec, question, answer, node.get("answer_type", ""),
                                       entity, quote, document, _subject_surface, _same_subject) is None):
        return "owned_known_relation", None
    local, local_reason = _local_citation(node, proof, bindings, document)
    if local and local["check"] != "matched_literal_relation_cue":
        return local["check"], None
    if _founder_witness(question, answer, quote, document):
        return "owned_founder_relation", None
    return None, reason or local_reason or "unknown_relation_not_promoted"


def _collect_proofs(nodes, bindings, winning, candidates, rank, document, diagnostic, proof_check=None):
    choices = {nid: [] for nid in nodes}
    for nid, proof in winning.get("proofs", {}).items():
        if nid in choices and isinstance(proof, dict):
            choices[nid].append(proof)
    for doc_id, candidate in candidates.items():
        if not isinstance(candidate, dict):
            continue
        for proof in candidate.get("verified", []):
            if not isinstance(proof, dict):
                continue
            goal = str(proof.get("goal", ""))
            if goal.startswith("dag:") and goal[4:] in choices:
                choices[goal[4:]].append(dict(proof, doc_id=doc_id))
    valid = {}
    for nid, items in choices.items():
        checked, seen = [], set()
        for raw in items:
            reason, check = None, None
            try:
                doc_id, confidence = int(raw.get("doc_id")), float(raw.get("confidence", 0))
            except (ValueError, TypeError, OverflowError):
                diagnostic["rejected_proofs"].append({"node": nid, "reason": "invalid_proof_fields"})
                continue
            answer, quote = str(raw.get("answer", "")), str(raw.get("evidence", ""))
            if (doc_id, answer, quote) in seen:
                continue
            seen.add((doc_id, answer, quote))
            requirements = raw.get("requirements", {})
            source = document(doc_id) if doc_id in rank and rank[doc_id] <= 200 else ""
            if doc_id not in candidates or not source:
                reason = "proof_outside_existing_top200_pool"
            elif nid not in bindings or not _same_entity(answer, bindings[nid], source):
                reason = "proof_conflicts_with_winning_binding"
            elif not isinstance(requirements, dict) or not branch_matches(requirements, bindings):
                reason = "proof_conflicts_with_winning_branch"
            elif not math.isfinite(confidence) or confidence < .85:
                reason = "low_or_invalid_confidence"
            elif len(quote) < 8 or len(quote) > 1500 or quote not in source:
                reason = "proof_quote_not_literal_grounded"
            elif not _subject_surface(answer, quote, source)[0] and not _literal(answer, quote):
                reason = "answer_not_source_grounded"
            elif _typed_answer_reason(answer, nodes[nid].get("answer_type", "")):
                reason = "answer_type_conflict"
            elif re.search(r"\b(?:not|never|without)\b", quote, re.I):
                reason = "negated_quote_not_supported"
            question = bind_question(nodes[nid].get("question", ""), bindings)
            if not reason and question is None:
                reason = "unbound_node_question"
            if not reason:
                proof = dict(raw, answer=answer, evidence=quote)
                check, reason = (proof_check or _proof_relation)(nodes[nid], question, proof, source, bindings)
                if check:
                    # Multi-parent joins need each input literally anchored in
                    # the actual quote; a topic title cannot silently join roots.
                    deps = nodes[nid].get("depends_on", [])
                    if len(deps) > 1 and not all(
                            _subject_surface(bindings.get(dep, ""), quote, source)[0] for dep in deps):
                        reason, check = "multi_parent_join_inputs_not_grounded", None
            if reason:
                diagnostic["rejected_proofs"].append({"node": nid, "doc_id": doc_id, "reason": reason})
            elif check:
                checked.append({"doc_id": doc_id, "check": check, "confidence": confidence})
        # Fixed bounded branching: three source alternatives per node, at most
        # 3**6 assignments for any of the 64 closed node subsets.
        by_doc = {p["doc_id"]: p for p in checked}
        if by_doc:
            valid[nid] = sorted(by_doc.values(), key=lambda p: (rank[p["doc_id"]], p["doc_id"]))[:3]
    return valid


def _closed_coverage(selected, choices, ancestors, node_order):
    covered = set()
    for nid in node_order:
        if nid in choices and ancestors[nid] <= covered and any(p["doc_id"] in selected for p in choices[nid]):
            covered.add(nid)
    return covered


def _package_sets(choices, ancestors, node_order):
    """Enumerate source alternatives for every supported ancestor-closed subset."""
    packages, seen = [], set()
    for mask in range(1, 1 << len(node_order)):
        subset = {nid for i, nid in enumerate(node_order) if mask & (1 << i)}
        if any(nid not in choices or not ancestors[nid] <= subset for nid in subset):
            continue
        ordered = [nid for nid in node_order if nid in subset]
        for assignment in product(*(choices[nid] for nid in ordered)):
            docs = frozenset(p["doc_id"] for p in assignment)
            # Different node subsets may share the same documents. Coverage is
            # recalculated from all their supported facts, so one doc set suffices.
            if docs not in seen:
                seen.add(docs)
                packages.append(docs)
    return packages


def _optimize_prefix(order, original, size, fixed_count, choices, ancestors, node_order, leaves, packages, diagnostic):
    if len(order) <= fixed_count:
        return order
    size = min(size, len(order))
    rank = {d: i + 1 for i, d in enumerate(original)}
    fixed = list(order[:fixed_count])
    supported_docs = {p["doc_id"] for rows in choices.values() for p in rows}
    protected = set(order[:size]) & supported_docs
    must_keep = set(fixed) | protected
    old_selected = set(order[:size])
    old_coverage = _closed_coverage(old_selected, choices, ancestors, node_order)
    baseline = (len(old_coverage & leaves), len(old_coverage))
    best_key, best_set, best_coverage = baseline + (0, 0), old_selected, old_coverage
    examined, blocked = 0, 0
    for package in packages:
        required = must_keep | set(package)
        if len(required) > size:
            blocked += 1
            continue
        selected = set(required)
        for d in order:
            if len(selected) >= size:
                break
            selected.add(d)
        coverage = _closed_coverage(selected, choices, ancestors, node_order)
        additions = selected - old_selected
        key = (len(coverage & leaves), len(coverage), -len(additions), -sum(rank[d] for d in additions))
        examined += 1
        if key > best_key:
            best_key, best_set, best_coverage = key, selected, coverage
    summary = {"top_k": size, "packages_examined": examined, "capacity_deferred": blocked,
               "coverage_before": sorted(old_coverage), "coverage_after": sorted(best_coverage),
               "complete_terminals_before": sorted(old_coverage & leaves),
               "complete_terminals_after": sorted(best_coverage & leaves),
               "protected_doc_ids": sorted(protected), "fixed_doc_ids": fixed}
    diagnostic["prefix_optimization"].append(summary)
    if best_key[:2] <= baseline:
        return order
    tail_selected = sorted(best_set - set(fixed), key=lambda d: (rank[d], d))
    prefix = fixed + tail_selected
    final = prefix + [d for d in order if d not in best_set]
    for d in prefix:
        if d not in old_selected:
            diagnostic["promotions"].append({"doc_id": d, "top_k": size,
                                             "from_rank": order.index(d) + 1,
                                             "to_rank": final.index(d) + 1,
                                             "closed_nodes": sorted(best_coverage),
                                             "bundle_doc_ids": sorted(best_set & supported_docs)})
    return final


def rerank_dag_packages(order, state, document, proof_check=None):
    """Optimize Top5 then Top10 with fixed front documents and source closures."""
    original = [int(d) for d in order]
    diagnostic = {"enabled": True, "policy": "finite_ancestor_closed_package_optimization",
                  "gold_labels_used": False, "extra_requests": 0, "entailment_guaranteed": False,
                  "eligible_proofs": {}, "packages": [], "promotions": [], "deferred": [],
                  "rejected_proofs": [], "prefix_optimization": [], "plan_error": None,
                  "top2_preserved": True, "top200_set_preserved": True,
                  "document_set_preserved": True, "unique_documents": True, "continuous_prefix": True}
    trace = state.get("evidence_trace", {})
    plan = state.get("_evidence_plan", trace.get("plan", []))
    bindings = state.get("_evidence_winning_bindings", trace.get("bindings", {}))
    if not isinstance(plan, list) or not isinstance(bindings, dict) or len(set(original)) != len(original):
        diagnostic["plan_error"] = "invalid_plan_binding_or_ranking"
        return original, diagnostic
    ancestors, node_order, error = _ancestors(plan)
    depth = {}
    for nid in node_order:
        deps = next(n.get("depends_on", []) for n in plan if n["id"] == nid)
        depth[nid] = 1 + max((depth[d] for d in deps), default=0)
    if error or not node_order or len(node_order) > 6 or max(depth.values(), default=0) > 4:
        diagnostic["plan_error"] = error or "empty_or_out_of_budget_plan"
        return original, diagnostic
    branches = state.get("_evidence_beams") or trace.get("branch_scores", [])
    winning = next((b for b in branches if isinstance(b, dict) and isinstance(b.get("bindings"), dict)
                    and set(b["bindings"]) == set(bindings) and branch_matches(b["bindings"], bindings)), None)
    if not winning or not isinstance(winning.get("proofs"), dict):
        diagnostic["plan_error"] = "missing_exact_winning_branch"
        return original, diagnostic
    nodes = {n["id"]: n for n in plan}
    choices = _collect_proofs(nodes, bindings, winning, state.get("evidence_candidates", {}),
                             {d: i + 1 for i, d in enumerate(original)}, document, diagnostic, proof_check)
    diagnostic["eligible_proofs"] = choices
    packages = _package_sets(choices, ancestors, node_order)
    diagnostic["eligible_package_count"] = len(packages)
    # Compact summaries rather than thousands of redundant alternative sets.
    diagnostic["packages"] = [sorted(p) for p in packages[:64]]
    children = {dep for n in plan for dep in n.get("depends_on", [])}
    leaves = set(node_order) - children
    for nid in node_order:
        missing = ({nid} | ancestors[nid]) - set(choices)
        if missing:
            diagnostic["deferred"].append({"node": nid, "reason": "missing_supported_ancestor_closure",
                                           "missing_nodes": sorted(missing)})
    final = _optimize_prefix(original, original, 5, 2, choices, ancestors, node_order, leaves, packages, diagnostic)
    final = _optimize_prefix(final, original, 10, 5, choices, ancestors, node_order, leaves, packages, diagnostic)
    diagnostic.update(top2_preserved=final[:2] == original[:2],
                      top200_set_preserved=set(final[:200]) == set(original[:200]),
                      document_set_preserved=set(final) == set(original),
                      unique_documents=len(set(final)) == len(final),
                      top5_set_preserved=set(final[:5]) == set(original[:5]),
                      top10_set_preserved=set(final[:10]) == set(original[:10]))
    if not all(diagnostic[k] for k in ("top2_preserved", "top200_set_preserved", "document_set_preserved", "unique_documents")):
        diagnostic["deferred"].append({"reason": "ranking_invariant_failure"})
        diagnostic["promotions"] = []
        return original, diagnostic
    return final, diagnostic


class DAGPackageMixin:
    """Optional stage4 finalizer; disabled flags delegate exact parent objects."""

    def finalize(self, query, ids, scores, ctx, state):
        result = super().finalize(query, ids, scores, ctx, state)
        if "dag_package" not in self.improvements or getattr(self, "stage", 4) < 4:
            return result
        original_ids, original_scores, original_trace = result
        proof_check = None
        if "source_witness" in self.improvements:
            from .evidence_source_witness import source_witness_check
            proof_check = source_witness_check
        final, diagnostic = rerank_dag_packages(original_ids, state, self._document, proof_check)
        trace = dict(original_trace, improvement_dag_package=diagnostic)
        if np.array_equal(np.asarray(final), original_ids):
            return original_ids, original_scores, trace
        old = {p["doc_id"]: p for p in trace.get("selected_prefix", [])}
        trace["selected_prefix"] = [dict(old.get(d, {"doc_id": d, "selection_source": "dag_package"}),
                                          original_greedy_rank=list(original_ids).index(d) + 1)
                                    for d in final[:5]]
        output_scores = np.arange(len(final), 0, -1, dtype=float) / max(1, len(final))
        return np.asarray(final, dtype=int), output_scores, trace
