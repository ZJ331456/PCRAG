"""Continuous retrieval prefixes with optional reliable proof closure.

This module uses retrieval scores, generated plans and passage-grounded bindings.
It never reads benchmark support labels or calls an embedding/LLM service.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evidence_retrieval import branch_matches, normalized_text


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def _ancestors(plan):
    nodes = {}
    for node in plan:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            return {}, [], "invalid_plan_node"
        nid, deps = node["id"], node.get("depends_on", [])
        if nid in nodes or not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            return {}, [], "invalid_plan_dependencies"
        nodes[nid] = dict(node, depends_on=deps)
    if any(dep not in nodes or dep == nid for nid, n in nodes.items() for dep in n["depends_on"]):
        return {}, [], "unknown_or_self_dependency"
    ancestors, ordered = {}, []
    while len(ordered) < len(nodes):
        ready = [nid for nid, n in nodes.items() if nid not in ancestors
                 and all(dep in ancestors for dep in n["depends_on"])]
        if not ready:
            return {}, [], "cyclic_dependencies"
        for nid in ready:
            ancestors[nid] = set(nodes[nid]["depends_on"])
            for dep in nodes[nid]["depends_on"]:
                ancestors[nid].update(ancestors[dep])
            ordered.append(nid)
    return ancestors, ordered, None


class SelectionImprovementMixin:
    """Add selection/closure without changing the unselected exp4 control."""

    def _reliable_proof_bundles(self, state, candidates, bindings):
        """Return document closures only for fully supported winning paths.

        An explicit relation check is preferred. Legacy literal proofs need high
        confidence, a literal answer and named subject/dependency mentions. This
        conservative fallback is citation-based; it is not relabelled as a
        semantic relation check. Unreliable/missing ancestors disable promotion
        of the affected proof while its passage remains in the base ranking.
        """
        plan = state.get("_evidence_plan", state.get("evidence_trace", {}).get("plan", []))
        ancestors, node_order, error = _ancestors(plan)
        diagnostics = {"scope": "reliable_winning_proof_documents", "plan_error": error,
                       "reliable_proofs": {}, "rejected_proofs": [], "soft_fallback_docs": []}
        if error or not node_order:
            return {}, set(), diagnostics
        nodes = {node["id"]: node for node in plan}
        proofs = {}
        branches = list(state.get("_evidence_beams", []))
        branches.extend(state.get("evidence_trace", {}).get("branch_scores", []))
        for branch in branches:
            if branch.get("bindings") and branch_matches(branch["bindings"], bindings):
                proofs.update(branch.get("proofs", {}))
                break
        for doc_id, candidate in candidates.items():
            for proof in candidate.get("verified", []):
                goal = str(proof.get("goal", ""))
                if goal.startswith("dag:") and goal[4:] not in proofs:
                    proofs[goal[4:]] = dict(proof, doc_id=doc_id)
        reliable, proof_docs, fallback = {}, {}, set()
        for nid, proof in proofs.items():
            if not isinstance(proof, dict):
                continue
            try:
                doc_id = int(proof.get("doc_id"))
            except (TypeError, ValueError, OverflowError):
                continue
            reason, mode = None, None
            quote, answer = str(proof.get("evidence", "")), str(proof.get("answer", ""))
            if nid not in nodes or doc_id not in candidates:
                reason = "proof_node_or_document_missing"
            elif nid not in bindings or normalized_text(answer) != normalized_text(bindings[nid]):
                reason = "proof_conflicts_with_winning_binding"
            elif not branch_matches(proof.get("requirements", {}), bindings):
                reason = "proof_conflicts_with_branch"
            elif len(quote) < 8 or quote not in self._document(doc_id):
                reason = "proof_quote_not_grounded"
            elif proof.get("relation_supported") is True:
                if _finite(proof.get("confidence", 0.0)) < 0.6:
                    reason = "relation_proof_low_confidence"
                else:
                    mode = "relation_checked"
            elif _finite(proof.get("confidence", 0.0)) < 0.85:
                reason = "literal_proof_low_confidence"
            elif (" " + normalized_text(answer) + " ") not in (" " + normalized_text(quote) + " "):
                reason = "literal_answer_not_in_quote"
            else:
                deps = nodes[nid]["depends_on"]
                if deps:
                    if not all(dep in bindings and normalized_text(bindings[dep]) in normalized_text(quote)
                               for dep in deps):
                        reason = "dependency_entity_not_in_quote"
                else:
                    names = re.findall(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", nodes[nid].get("question", ""))
                    names = [n for n in names if n not in {"Who", "What", "Which", "Where", "When", "How", "Is", "The", "A"}]
                    if not names or not any(normalized_text(n) in normalized_text(quote) for n in names):
                        reason = "named_subject_not_in_quote"
                if reason is None:
                    mode = "conservative_literal"
            proof_docs[nid] = doc_id
            if reason:
                fallback.add(doc_id)
                diagnostics["rejected_proofs"].append({"node": nid, "doc_id": doc_id, "reason": reason})
            else:
                reliable[nid] = doc_id
                diagnostics["reliable_proofs"][nid] = {"doc_id": doc_id, "check": mode}
        # Protect every node proved by one passage, including the uncommon case
        # where that same passage provides more than one plan node's answer.
        protected = {}
        for nid, doc_id in reliable.items():
            protected.setdefault(doc_id, set()).add(nid)
        bundles = {}
        for doc_id in protected:
            required, pending, missing = {doc_id}, [doc_id], False
            while pending and not missing:
                current = pending.pop()
                for nid in protected.get(current, ()):
                    for ancestor in ancestors[nid]:
                        if ancestor not in reliable:
                            missing = True
                            break
                        parent_doc = reliable[ancestor]
                        if parent_doc not in required:
                            required.add(parent_doc)
                            pending.append(parent_doc)
            if missing:
                fallback.add(doc_id)
                diagnostics["rejected_proofs"].append({"doc_id": doc_id, "reason": "missing_reliable_ancestor"})
            else:
                bundles[doc_id] = required
        diagnostics["soft_fallback_docs"] = sorted(fallback)
        diagnostics["ancestor_document_bundles"] = {str(d): sorted(required) for d, required in bundles.items()}
        return bundles, fallback, diagnostics

    def finalize(self, query, ids, scores, ctx, state):
        selection = "selection" in self.improvements
        closure = "closure" in self.improvements
        if not selection and not closure:
            return super().finalize(query, ids, scores, ctx, state)
        trace = state.get("evidence_trace", self._trace(self.stage))
        candidates = state.get("evidence_candidates", {})
        if self.stage < 2 or not candidates:
            return ids, scores, trace
        ids, scores = np.asarray(ids, dtype=int), np.asarray(scores, dtype=float)
        finite = np.where(np.isfinite(scores), scores, 0.0)
        span = float(finite.max() - finite.min()) if len(finite) else 0.0
        norm = (finite - finite.min()) / span if span > 0 else np.ones(len(finite))
        base = dict(zip(ids.tolist(), norm.tolist()))
        base_order = list(dict.fromkeys(int(d) for d in ids))
        base_rank = {doc_id: rank for rank, doc_id in enumerate(base_order)}
        bindings = state.get("_evidence_winning_bindings", {})
        goals = {d: self._goal_scores(c, bindings) for d, c in candidates.items()}
        goal_total = max(1, len({g for values in goals.values() for g in values}))
        base_w = _finite(getattr(self.cfg, "evidence_base_weight", .4), .4)
        relation_w = _finite(getattr(self.cfg, "evidence_relation_weight", .25), .25)
        coverage_w = _finite(getattr(self.cfg, "evidence_coverage_weight", .45), .45)
        redundancy_w = _finite(getattr(self.cfg, "evidence_redundancy_weight", .15), .15)
        merged = {d: base_w * v for d, v in base.items()}
        for d, candidate in candidates.items():
            merged[d] = base_w * base.get(d, candidate.get("base_score", 0.0)) + (1. - base_w) * max(goals[d].values(), default=0.)
        old_tail = sorted(merged, key=lambda d: (-merged[d], d))
        # Candidate-only passages stay retrievable, but cannot displace every
        # passage in the established base channel merely through one goal score.
        stable_order = base_order + [d for d in old_tail if d not in base_rank]
        budget = (max(1, min(20, int(getattr(self.cfg, "evidence_selection_top_k", 10))))
                  if selection else self.budget)
        budget = min(budget, len(stable_order))
        document_tokens = {d: set(normalized_text(self._document(d)).split()) for d in candidates}
        bundles, fallback, closure_diagnostics = ({}, set(), {})
        if closure:
            bundles, fallback, closure_diagnostics = self._reliable_proof_bundles(state, candidates, bindings)
        covered, selected, details = {}, [], []
        promoted, baseline_fills, bundle_count = 0, 0, 0
        jaccard_cache = {}

        def marginal(doc_id, prefix):
            local_covered = dict(covered)
            for previous in prefix[len(selected):]:
                for goal, value in goals.get(previous, {}).items():
                    local_covered[goal] = max(local_covered.get(goal, 0.), value)
            values = goals.get(doc_id, {})
            gain = sum(max(0., value - local_covered.get(goal, 0.)) for goal, value in values.items()) / goal_total
            meaningful = any(value - local_covered.get(goal, 0.) >= max(1e-9, .15 * value)
                             for goal, value in values.items() if value > 0.)
            tokens = document_tokens.get(doc_id)
            redundancy = 0.
            if tokens is not None:
                for previous in prefix:
                    other = document_tokens.get(previous)
                    if other is None:
                        continue
                    key = tuple(sorted((doc_id, previous)))
                    if key not in jaccard_cache:
                        jaccard_cache[key] = len(tokens & other) / max(1, len(tokens | other))
                    redundancy = max(redundancy, jaccard_cache[key])
            base_value = base.get(doc_id, candidates.get(doc_id, {}).get("base_score", 0.))
            value = base_w * base_value + relation_w * max(values.values(), default=0.) + coverage_w * gain - redundancy_w * redundancy
            return value, gain, redundancy, meaningful

        def extension(doc_id):
            required = bundles.get(doc_id, {doc_id}) - set(selected)
            if len(selected) + len(required) > budget:
                return None
            result = list(selected)
            while required:
                eligible = [d for d in required if not (bundles.get(d, {d}) - set(result) - {d})]
                if not eligible:
                    return None
                chosen = max(eligible, key=lambda d: (marginal(d, result)[0], -base_rank.get(d, len(base_order)), -d))
                result.append(chosen)
                required.remove(chosen)
            return result

        while len(selected) < budget:
            chosen_set = set(selected)
            next_base = next((d for d in stable_order if d not in chosen_set), None)
            if next_base is None:
                break
            base_extension = extension(next_base) if next_base in bundles else selected + [next_base]
            if base_extension is None:
                base_extension = selected + [next_base]
            # A base fallback is allowed even when its binding cannot support a
            # complete path. It is not counted as a reliable proof promotion.
            base_delta = sum(marginal(d, base_extension[:index])[0]
                             for index, d in enumerate(base_extension) if index >= len(selected))
            base_option = (base_delta / max(1, len(base_extension) - len(selected)), 0,
                           -base_rank.get(next_base, len(base_order)), -next_base, base_extension, False)
            # Closure is an independent ablation: without selection, retain the
            # original forced candidate-prefix greedy objective. Base protection
            # and meaningful-gain gating belong only to the selection flag.
            best = base_option if selection else (float("-inf"), 0, 0, 0, None, False)
            for doc_id in candidates:
                if doc_id in chosen_set or doc_id in fallback:
                    continue
                value, gain, redundancy, meaningful = marginal(doc_id, selected)
                if selection and not meaningful:
                    continue
                proposed = extension(doc_id)
                if proposed is None:
                    continue
                delta = sum(marginal(d, proposed[:index])[0]
                            for index, d in enumerate(proposed) if index >= len(selected))
                score = delta / max(1, len(proposed) - len(selected))
                if selection:
                    contender = (score, 1 if doc_id in bundles else 0,
                                 -base_rank.get(doc_id, len(base_order)), -doc_id, proposed, True)
                else:
                    contender = (score, merged[doc_id], -doc_id, 0, proposed, True)
                if contender[:4] > best[:4]:
                    best = contender
            if best[4] is None:
                best = base_option
            chosen_extension, is_promotion = best[4], best[5]
            added = chosen_extension[len(selected):]
            if len(added) > 1:
                bundle_count += 1
            for doc_id in added:
                value, gain, redundancy, meaningful = marginal(doc_id, selected)
                mode = "proof_bundle" if is_promotion and len(added) > 1 else "coverage" if is_promotion else "base_fallback"
                details.append({"doc_id": doc_id, "utility": float(value), "coverage_gain": float(gain),
                                "redundancy": float(redundancy), "goals": goals.get(doc_id, {}),
                                "verified": candidates.get(doc_id, {}).get("verified", []), "selection_source": mode})
                selected.append(doc_id)
                for goal, value in goals.get(doc_id, {}).items():
                    covered[goal] = max(covered.get(goal, 0.), value)
                promoted += int(is_promotion)
                baseline_fills += int(not is_promotion)
        tail_source = stable_order if selection else old_tail
        selected_set = set(selected)
        final = selected + [d for d in tail_source if d not in selected_set]
        # Strict monotone rank scores preserve the actual order for downstream
        # exporters. The trace retains the original utility and coverage values.
        output_scores = np.arange(len(final), 0, -1, dtype=float) / max(1, len(final))
        trace["selected_prefix"] = details
        trace["covered_goals"] = covered
        trace["improvement_selection"] = {"enabled": selection, "top_k": budget,
                                           "continuous_prefix": True, "base_tail_preserved": selection,
                                           "coverage_promotions": promoted, "base_fills": baseline_fills,
                                           "meaningful_gain_fraction": .15, "gold_labels_used": False}
        if closure:
            def proof_closed_at(k):
                prefix = set(final[:k])
                return all(required <= prefix for d, required in bundles.items() if d in prefix)
            closure_diagnostics.update(enabled=True, selected_bundle_count=bundle_count,
                                       strong_scope="promoted_reliable_proof_documents",
                                       fallback_policy="preserve_base_passage_without_forced_proof",
                                       reliable_proof_closed_at_5=proof_closed_at(5),
                                       reliable_proof_closed_at_10=proof_closed_at(10))
            trace["improvement_closure"] = closure_diagnostics
        return np.asarray(final, dtype=int), output_scores, trace
