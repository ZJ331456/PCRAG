"""Bounded preservation of already accepted ancestor citation documents.

This postprocessor uses only the winning plan branch and literal source checks.
It does not upgrade a citation into semantic relation support or issue requests.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .evidence_retrieval import branch_matches, normalized_text
from .evidence_selection import _ancestors


def _same_bindings(left, right):
    return (isinstance(left, dict) and isinstance(right, dict) and
            set(left) == set(right) and branch_matches(left, right))


class AncestorSupportMixin:
    """Keep the original greedy prefix, then protect at most two tail citations.

    The optional bounded swap replaces at most one unverified near duplicate in
    the first five. Its displaced document is retained near the front of the
    tail. Neither policy removes a document from the retrieval result.
    """

    def finalize(self, query, ids, scores, ctx, state):
        result = super().finalize(query, ids, scores, ctx, state)
        if "support" not in self.improvements:
            return result
        original_ids, original_scores, original_trace = result
        order = [int(doc_id) for doc_id in original_ids]
        trace = dict(original_trace)
        mode = getattr(self.cfg, "evidence_support_mode", "tail_only")
        diagnostic = {"enabled": True, "mode": mode, "policy": "winning_literal_ancestor_citations",
                      "max_tail_promotions": 2, "max_top5_swaps": int(mode == "bounded_swap"),
                      "semantic_relation_upgrade": False, "gold_labels_used": False,
                      "promotions": [], "rejected_proofs": [], "deferred_chains": []}
        trace["improvement_support"] = diagnostic
        candidates = state.get("evidence_candidates", {})
        bindings = state.get("_evidence_winning_bindings", trace.get("bindings", {}))
        plan = state.get("_evidence_plan", trace.get("plan", []))
        ancestors, node_order, error = (_ancestors(plan or []) if isinstance(plan, (list, tuple)) else
                                        ({}, [], "invalid_plan_type"))
        prefix, tail_promotions, swapped = order[:5], [], None
        if mode not in {"tail_only", "bounded_swap"} or len(set(order)) != len(order):
            error = error or "invalid_mode_or_duplicate_ranking"
        diagnostic["plan_error"] = error
        branches = state.get("_evidence_beams") or trace.get("branch_scores", [])
        branch = next((b for b in branches if isinstance(b, dict) and
                       _same_bindings(b.get("bindings"), bindings)), None)
        valid = {}
        if not error and branch:
            proofs = branch.get("proofs", {})
            for nid, proof in (proofs.items() if isinstance(proofs, dict) else []):
                reason = None
                if not isinstance(proof, dict):
                    continue
                try:
                    doc_id = int(proof.get("doc_id"))
                    confidence = float(proof.get("confidence", 0.))
                except (ValueError, TypeError, OverflowError):
                    diagnostic["rejected_proofs"].append({"node": nid, "reason": "invalid_proof_fields"})
                    continue
                answer, quote = str(proof.get("answer", "")), str(proof.get("evidence", ""))
                if nid not in ancestors or doc_id not in candidates or doc_id not in order:
                    reason = "proof_node_or_document_missing"
                elif nid not in bindings or normalized_text(answer) != normalized_text(bindings[nid]):
                    reason = "proof_conflicts_with_winning_binding"
                elif not isinstance(proof.get("requirements", {}), dict) or not branch_matches(proof.get("requirements", {}), bindings):
                    reason = "proof_conflicts_with_winning_branch"
                elif not math.isfinite(confidence) or confidence < .85:
                    reason = "literal_proof_low_or_invalid_confidence"
                elif len(quote) < 8 or quote not in self._document(doc_id):
                    reason = "proof_quote_not_grounded"
                elif not normalized_text(answer) or (" " + normalized_text(answer) + " ") not in (" " + normalized_text(quote) + " "):
                    reason = "literal_answer_not_in_quote"
                if reason:
                    diagnostic["rejected_proofs"].append({"node": nid, "doc_id": doc_id, "reason": reason})
                else:
                    valid[nid] = doc_id

        def duplicate_victim(current_prefix):
            goals = {d: self._goal_scores(candidates.get(d, {"sources": [], "verified": []}), bindings)
                     for d in current_prefix}
            for victim in reversed(current_prefix):
                if candidates.get(victim, {}).get("verified"):
                    continue
                text = self._document(victim)
                lines = text.splitlines()
                if len(lines) < 2 or not normalized_text(lines[0]):
                    continue
                title, tokens = normalized_text(lines[0]), set(normalized_text(text).split())
                other_goals = {g for d in current_prefix if d != victim for g, value in goals[d].items() if value > 0}
                if any(value > 0 and g not in other_goals for g, value in goals[victim].items()):
                    continue
                for partner in current_prefix:
                    if partner == victim:
                        continue
                    other = self._document(partner)
                    other_lines = other.splitlines()
                    if len(other_lines) < 2 or normalized_text(other_lines[0]) != title:
                        continue
                    # Distinct dates/counts or negation must not be collapsed as
                    # duplicate evidence simply because most words overlap.
                    if set(re.findall(r"\d+", text)) != set(re.findall(r"\d+", other)):
                        continue
                    negations = {"not", "never", "no", "without"}
                    other_tokens = set(normalized_text(other).split())
                    if tokens & negations != other_tokens & negations:
                        continue
                    similarity = len(tokens & other_tokens) / max(1, len(tokens | other_tokens))
                    if similarity >= .85:
                        return victim, partner, similarity
            return None

        def assemble(current_prefix, promoted, swap):
            front = current_prefix + promoted + ([swap[0]] if swap else [])
            seen = set(front)
            return front + [d for d in order if d not in seen]

        # A child already selected by exp4 is the anchor. Do not promote proofs
        # from losing branches or build a new ranking around unselected leaves.
        children = [nid for nid in node_order if nid in valid and ancestors[nid] and valid[nid] in order[:5]]
        for child in children:
            missing = [nid for nid in node_order if nid in ancestors[child] and nid not in valid]
            if missing:
                diagnostic["deferred_chains"].append({"child": child, "reason": "missing_valid_ancestor", "nodes": missing})
                continue
            required = list(dict.fromkeys(valid[nid] for nid in node_order if nid in ancestors[child]))
            current = assemble(prefix, tail_promotions, swapped)
            if set(required) <= set(current[:5] if mode == "bounded_swap" else current[:10]):
                continue
            proposed_prefix, proposed_tail, proposed_swap = list(prefix), list(tail_promotions), swapped
            to_add = [d for d in required if d not in proposed_prefix and d not in proposed_tail]
            if mode == "bounded_swap" and proposed_swap is None and to_add:
                duplicate = duplicate_victim(proposed_prefix)
                if duplicate:
                    victim, partner, similarity = duplicate
                    promoted = to_add.pop(0)
                    proposed_prefix[proposed_prefix.index(victim)] = promoted
                    proposed_swap = (victim, promoted, partner, similarity)
            proposed_tail.extend(to_add)
            if len(proposed_tail) > 2:
                diagnostic["deferred_chains"].append({"child": child, "reason": "ancestor_bundle_exceeds_budget", "doc_ids": required})
                continue
            proposed = assemble(proposed_prefix, proposed_tail, proposed_swap)
            if not set(required) <= set(proposed[:10]):
                diagnostic["deferred_chains"].append({"child": child, "reason": "ancestor_bundle_not_in_top10", "doc_ids": required})
                continue
            previous = set(prefix + tail_promotions)
            for doc_id in proposed_prefix + proposed_tail:
                if doc_id not in previous:
                    is_swap = proposed_swap and doc_id == proposed_swap[1]
                    detail = {"child": child, "doc_id": doc_id, "from_rank": order.index(doc_id) + 1,
                              "reason": "missing_winning_ancestor_citation", "scope": "top5_swap" if is_swap else "top10_tail"}
                    if is_swap:
                        detail.update(victim=proposed_swap[0], duplicate_partner=proposed_swap[2], jaccard=proposed_swap[3])
                    diagnostic["promotions"].append(detail)
            prefix, tail_promotions, swapped = proposed_prefix, proposed_tail, proposed_swap
        final = assemble(prefix, tail_promotions, swapped)
        for detail in diagnostic["promotions"]:
            detail["to_rank"] = final.index(detail["doc_id"]) + 1
        diagnostic.update(top5_preserved=final[:5] == order[:5], r1_preserved=final[:1] == order[:1],
                          document_set_preserved=set(final) == set(order),
                          winning_literal_proofs=len(valid), displaced_document_retained=swapped[0] in final if swapped else True)
        if swapped:
            old = {d["doc_id"]: d for d in trace.get("selected_prefix", [])}
            trace["selected_prefix"] = [dict(old.get(d, {"doc_id": d, "verified": candidates.get(d, {}).get("verified", []),
                                                        "selection_source": "ancestor_support"}),
                                              original_greedy_rank=order.index(d) + 1) for d in final[:5]]
        output_scores = np.arange(len(final), 0, -1, dtype=float) / max(1, len(final))
        return np.asarray(final, dtype=int), output_scores, trace
