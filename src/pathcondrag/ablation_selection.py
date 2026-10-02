"""Binding checks and bounded evidence selection for the exp4 ablations.

The selectors consume already generated candidates and one fixed binding branch.
They never inspect gold labels, generate bridge hypotheses, or call a model. The
joint selector searches proof-closed seeds plus greedy completions and bounded
single replacements; it is not a globally exact subset or binding optimizer.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .evidence_retrieval import _HASH, _REFERENCE, normalized_text, verify_hypotheses


def _doc_id(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    match = re.fullmatch(r"D?(\d+)", str(value))
    return int(match.group(1)) if match else None


def verify_string_hypotheses(payload: dict, documents: Mapping[int, str]):
    """Use the literal control's answer checks without requiring a citation.

    This deliberately accepts answers not present in the referenced passage.
    It is the ungrounded binding control, not a claim of answer correctness.
    """
    accepted, rejected, seen = [], [], set()
    items = payload.get("hypotheses", []) if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return [], [{"reason": "invalid_hypotheses_list"}]
    for item in items:
        if not isinstance(item, dict):
            rejected.append({"reason": "invalid_hypothesis"})
            continue
        answer = str(item.get("answer") or "").strip()
        doc_id = _doc_id(item.get("doc_id"))
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError, OverflowError):
            confidence = float("nan")
        reason = None
        if doc_id not in documents:
            reason = "unknown_document_reference"
        elif not answer or len(answer) > 200 or _HASH.search(answer) or _REFERENCE.search(answer):
            reason = "invalid_answer_or_hash"
        elif normalized_text(answer) in {"unknown", "none", "not known", "not stated", "no answer"}:
            reason = "missing_answer"
        elif not math.isfinite(confidence):
            reason = "invalid_confidence"
        elif confidence < 0.25:
            reason = "low_claimed_confidence"
        if reason:
            rejected.append({"answer": answer[:200], "doc_id": doc_id, "reason": reason})
            continue
        evidence = str(item.get("evidence") or "").strip()
        key = (normalized_text(answer), doc_id, evidence)
        if key in seen:
            continue
        seen.add(key)
        accepted.append({"answer": answer, "doc_id": doc_id, "evidence": evidence,
                         "confidence": min(1.0, max(0.0, confidence))})
    accepted.sort(key=lambda h: (-h["confidence"], h["doc_id"], normalized_text(h["answer"])))
    return accepted, rejected


def verify_relation_verdicts(payload: dict, hypotheses: List[dict],
                             documents: Mapping[int, str]):
    """Require literal grounding and an explicit verdict for each same claim.

    Payload schema: ``{verdicts: [{doc_id, answer, evidence,
    label: 'entailed', relation_supported: true}]}``. A verdict cannot swap the
    document, answer, or original quote. Missing, contradictory, and duplicated
    conflicting verdicts are rejected. Entailment remains an LLM judgment; this
    parser validates the output contract rather than proving the relationship.
    """
    literal, rejected = verify_hypotheses({"hypotheses": hypotheses}, dict(documents))
    verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
    if not isinstance(verdicts, list):
        return [], rejected + [{"reason": "invalid_relation_verdicts_list"}]
    by_claim = defaultdict(list)
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            rejected.append({"reason": "invalid_relation_verdict"})
            continue
        key = (_doc_id(verdict.get("doc_id")),
               normalized_text(str(verdict.get("answer") or "")),
               str(verdict.get("evidence") or "").strip())
        by_claim[key].append(verdict)
    accepted, used = [], set()
    for hypothesis in literal:
        key = (hypothesis["doc_id"], normalized_text(hypothesis["answer"]), hypothesis["evidence"])
        claims = by_claim.get(key, [])
        used.add(key)
        if not claims:
            reason = "missing_matching_relation_verdict"
        elif not all(str(v.get("label", "")).strip().casefold() == "entailed"
                     and v.get("relation_supported") is True for v in claims):
            reason = "relation_not_entailed"
        else:
            accepted.append(dict(hypothesis, relation_supported=True,
                                 relation_label="entailed"))
            continue
        rejected.append({"answer": hypothesis["answer"], "doc_id": hypothesis["doc_id"],
                         "reason": reason})
    for key in by_claim.keys() - used:
        rejected.append({"answer": key[1], "doc_id": key[0],
                         "reason": "relation_verdict_changed_claim_or_quote"})
    return accepted, rejected


def _finite(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def _plan_ancestors(plan: List[dict]):
    nodes = {}
    for node in plan:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            return None, None, "invalid_plan_node"
        nid = node["id"]
        deps = node.get("depends_on", [])
        if not nid or nid in nodes or not isinstance(deps, list):
            return None, None, "invalid_or_duplicate_plan_node"
        if any(not isinstance(dep, str) for dep in deps):
            return None, None, "invalid_plan_dependency"
        nodes[nid] = list(dict.fromkeys(deps))
    if len(nodes) > 4:
        return None, None, "plan_exceeds_four_nodes"
    if any(dep not in nodes or dep == nid for nid, deps in nodes.items() for dep in deps):
        return None, None, "unknown_or_self_dependency"
    ancestors, ordered = {}, []
    while len(ordered) < len(nodes):
        ready = [nid for nid, deps in nodes.items() if nid not in ancestors
                 and all(dep in ancestors for dep in deps)]
        if not ready:
            return None, None, "cyclic_dependencies"
        for nid in ready:
            ancestors[nid] = set(nodes[nid])
            for dep in nodes[nid]:
                ancestors[nid].update(ancestors[dep])
            ordered.append(nid)
    return ancestors, ordered, None


class _SelectionProblem:
    def __init__(self, candidates, plan, proofs, goal_scores, document_tokens,
                 bindings, budget, base_scores, weights, utility_callback, goal_total):
        self.candidates = candidates
        self.documents = sorted(candidates)
        self.budget = budget
        self.bindings = dict(bindings or {})
        self.base = {d: _finite((base_scores or {}).get(d, candidates[d].get("base_score", 0.0)))
                     for d in self.documents}
        self.goals = {d: {str(g): _finite(v) for g, v in goal_scores.get(d, {}).items()}
                      for d in self.documents}
        self.goal_total = goal_total or max(1, len({g for goals in self.goals.values() for g in goals}))
        self.tokens = {d: set(document_tokens.get(d, set())) for d in self.documents}
        self.weights = weights
        self.utility_callback = utility_callback
        self.ancestors, self.node_order, self.plan_error = _plan_ancestors(plan)
        self.proofs, self.invalid_proofs = {}, []
        self.protected = defaultdict(set)
        self.blocked = set()
        self.jaccard_cache = {}
        self.evaluations = 0
        if self.plan_error:
            return
        for nid, hypothesis in (proofs or {}).items():
            reason = None
            doc_id = _doc_id(hypothesis.get("doc_id")) if isinstance(hypothesis, dict) else None
            if nid not in self.ancestors:
                reason = "proof_node_not_in_plan"
            elif doc_id not in candidates:
                reason = "proof_document_not_in_candidates"
            elif nid in self.bindings and normalized_text(hypothesis.get("answer", "")) != normalized_text(self.bindings[nid]):
                reason = "proof_answer_conflicts_with_binding"
            elif not math.isfinite(_finite(hypothesis.get("quality", 0.0), float("nan"))):
                reason = "invalid_proof_quality"
            elif any(k not in self.bindings or normalized_text(v) != normalized_text(self.bindings[k])
                     for k, v in hypothesis.get("requirements", {}).items()):
                reason = "proof_conflicts_with_branch"
            if reason:
                self.invalid_proofs.append({"node": nid, "doc_id": doc_id, "reason": reason})
                if doc_id in candidates:
                    self.blocked.add(doc_id)
                continue
            self.proofs[nid] = dict(hypothesis, doc_id=doc_id)
        for doc_id, candidate in candidates.items():
            for proof in candidate.get("verified", []):
                goal = str(proof.get("goal", ""))
                if not goal.startswith("dag:"):
                    continue
                nid = goal[4:]
                winning = self.proofs.get(nid)
                answer = proof.get("answer")
                if winning is None or winning["doc_id"] != doc_id or (
                    answer is not None and normalized_text(answer) != normalized_text(winning.get("answer", ""))
                ):
                    self.blocked.add(doc_id)
                    self.invalid_proofs.append({"node": nid, "doc_id": doc_id,
                                                "reason": "candidate_proof_incompatible_with_winning_branch"})
                else:
                    self.protected[doc_id].add(nid)
        # A winning proof is protected even if the caller did not duplicate it in
        # candidate['verified']; otherwise an ancestor constraint could be bypassed.
        for nid, hypothesis in self.proofs.items():
            self.protected[hypothesis["doc_id"]].add(nid)
        for doc_id in list(self.protected):
            if self.bundle(doc_id, set()) is None:
                self.blocked.add(doc_id)

    def jaccard(self, first, second):
        key = tuple(sorted((first, second)))
        if key not in self.jaccard_cache:
            a, b = self.tokens[first], self.tokens[second]
            self.jaccard_cache[key] = len(a & b) / max(1, len(a | b))
        return self.jaccard_cache[key]

    def utility(self, doc_id, selected):
        self.evaluations += 1
        covered = {}
        for previous in selected:
            for goal, score in self.goals[previous].items():
                covered[goal] = max(covered.get(goal, 0.0), score)
        goals = self.goals[doc_id]
        gain = sum(max(0.0, score - covered.get(goal, 0.0))
                   for goal, score in goals.items()) / self.goal_total
        redundancy = max((self.jaccard(doc_id, previous) for previous in selected), default=0.0)
        relation = max(goals.values(), default=0.0)
        base_w, relation_w, coverage_w, redundancy_w = self.weights
        value = (base_w * self.base[doc_id] + relation_w * relation
                 + coverage_w * gain - redundancy_w * redundancy)
        if self.utility_callback is not None:
            value = _finite(self.utility_callback(doc_id, tuple(selected)), float("-inf"))
        return value, gain, redundancy

    def tie_score(self, doc_id):
        return self.weights[0] * self.base[doc_id] + (1.0 - self.weights[0]) * max(self.goals[doc_id].values(), default=0.0)

    def objective(self, selected):
        return sum(self.utility(doc_id, selected[:index])[0] for index, doc_id in enumerate(selected))

    def bundle(self, doc_id, selected):
        """Return the full ancestor-document closure needed by this document."""
        if doc_id in self.blocked:
            return None
        docs, pending = {doc_id}, [doc_id]
        while pending:
            current = pending.pop()
            if current in self.blocked:
                return None
            for nid in self.protected.get(current, set()):
                for ancestor in self.ancestors[nid]:
                    hypothesis = self.proofs.get(ancestor)
                    if hypothesis is None:
                        return None
                    ancestor_doc = hypothesis["doc_id"]
                    if ancestor_doc not in docs:
                        docs.add(ancestor_doc)
                        pending.append(ancestor_doc)
        return docs - set(selected)

    def closed(self, selected):
        selected = set(selected)
        return all(self.bundle(doc_id, selected) == set() for doc_id in selected)

    def order(self, subset, initial=()):
        """Order a closed subset by marginal score without preceding ancestors."""
        remaining, selected = set(subset) - set(initial), list(initial)
        while remaining:
            eligible = []
            for doc_id in remaining:
                needed = self.bundle(doc_id, set(selected))
                if needed is not None and needed <= {doc_id}:
                    utility, gain, redundancy = self.utility(doc_id, selected)
                    eligible.append((utility, self.tie_score(doc_id), -doc_id, doc_id))
            if not eligible:
                return None
            doc_id = max(eligible)[3]
            selected.append(doc_id)
            remaining.remove(doc_id)
        return selected

    def greedy(self, initial=(), closure=True, banned=frozenset()):
        selected = self.order(initial) if closure else list(initial)
        if selected is None or len(selected) > self.budget:
            return None
        while len(selected) < self.budget:
            ranked = []
            for doc_id in self.documents:
                if doc_id in selected or doc_id in banned:
                    continue
                bundle = self.bundle(doc_id, selected) if closure else {doc_id}
                if not bundle or len(selected) + len(bundle) > self.budget or bundle & set(banned):
                    continue
                extension = self.order(bundle, selected) if closure else selected + [doc_id]
                if extension is None:
                    continue
                delta = sum(self.utility(extension[index], extension[:index])[0]
                            for index in range(len(selected), len(extension)))
                ranked.append((delta / len(bundle), delta, self.tie_score(doc_id), -doc_id, extension))
            if not ranked:
                break
            selected = max(ranked, key=lambda item: item[:4])[4]
        return selected


def select_evidence_prefix(
    candidates: Mapping[int, dict], plan: List[dict], proofs: Mapping[str, dict],
    goal_scores: Mapping[int, Mapping[str, float]], document_tokens: Mapping[int, Iterable[str]],
    *, bindings: Optional[dict] = None, budget: int = 5, policy: str = "closure",
    base_scores: Optional[Mapping[int, float]] = None, base_weight: float = 0.4,
    relation_weight: float = 0.25, coverage_weight: float = 0.45,
    redundancy_weight: float = 0.15,
    utility_callback: Optional[Callable[[int, Tuple[int, ...]], float]] = None,
    replacement_pool_size: int = 24, goal_total: Optional[int] = None,
):
    """Return a prefix and auditable selection diagnostics.

    ``coverage`` uses ordinary marginal greedy selection. ``closure`` greedily
    selects complete ancestor proof-document bundles. ``joint`` enumerates at
    most 16 proof seeds for a four-node plan, greedily completes each seed, then
    tries one-document replacements drawn from a fixed top-24 shortlist. It
    retains the closure solution as a candidate, so its *selection objective*
    cannot be lower; recall is not guaranteed to improve. Bindings are fixed.

    The common objective is the sum of the existing prefix marginal utilities:
    base relevance + relation relevance + newly covered goals - maximum token
    Jaccard redundancy against preceding documents. Scores are independent of
    benchmark answers or support labels. ``utility_callback`` can override the
    marginal utility consistently for all three policies.
    ``goal_total`` can fix the coverage denominator across generated branches;
    otherwise the denominator is the number of goals in the supplied scores.
    """
    if policy == "ancestor":
        policy = "closure"
    if policy not in {"coverage", "closure", "joint"}:
        raise ValueError("Unknown evidence selection policy")
    if not isinstance(budget, int) or isinstance(budget, bool) or not 0 <= budget <= 5:
        raise ValueError("Evidence selection budget must be an integer from 0 to 5")
    if any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in candidates):
        raise ValueError("Candidate document IDs must be nonnegative integers")
    if goal_total is not None and (
        not isinstance(goal_total, int) or isinstance(goal_total, bool) or goal_total < 1
    ):
        raise ValueError("goal_total must be a positive integer when provided")
    weights = tuple(_finite(w) for w in (base_weight, relation_weight, coverage_weight, redundancy_weight))
    problem = _SelectionProblem(candidates, plan, proofs, goal_scores, document_tokens,
                                bindings, budget, base_scores, weights, utility_callback, goal_total)
    diagnostics = {"policy": policy, "budget": budget, "candidate_count": len(candidates),
                   "binding_optimization": False, "bindings_fixed": True,
                   "closure_scope": "verified_dag_proof_documents", "goal_total": problem.goal_total,
                   "objective": "sum_prefix_marginal_utility",
                   "weights": {"base": weights[0], "relation": weights[1],
                               "coverage": weights[2], "redundancy": weights[3]},
                   "global_optimum_guaranteed": False}
    if problem.plan_error and policy != "coverage":
        diagnostics.update(status="invalid_plan", error=problem.plan_error, selected_prefix=[])
        return [], diagnostics
    if policy == "coverage":
        selected = problem.greedy(closure=False)
        diagnostics.update(search_method="ordinary_greedy", proof_seed_count=0)
    else:
        baseline = problem.greedy() or []
        baseline_score = problem.objective(baseline)
        selected = baseline
        diagnostics.update(closure_baseline_objective=baseline_score,
                           search_method="greedy_ancestor_bundles", proof_seed_count=1)
        if policy == "joint":
            nodes = [nid for nid in problem.node_order if nid in problem.proofs]
            seeds = {tuple()}
            for mask in range(1 << len(nodes)):
                documents = set()
                valid = True
                for index, nid in enumerate(nodes):
                    if not mask & (1 << index):
                        continue
                    bundle = problem.bundle(problem.proofs[nid]["doc_id"], set())
                    if bundle is None:
                        valid = False
                        break
                    documents.update(bundle)
                if valid and len(documents) <= budget:
                    seeds.add(tuple(sorted(documents)))
            results = [baseline]
            for seed in sorted(seeds):
                result = problem.greedy(seed)
                if result is not None:
                    results.append(result)
            selected = max(results, key=lambda docs: (problem.objective(docs), tuple(-d for d in docs)))
            # Bounded local improvement. All candidate docs remain available for
            # greedy completion; only replacement starts are shortlisted.
            shortlist = sorted((d for d in problem.documents if d not in problem.blocked),
                               key=lambda d: (-problem.utility(d, [])[0], -problem.tie_score(d), d))
            shortlist = shortlist[:max(0, int(replacement_pool_size))]
            replacement_trials = 0
            snapshot = list(selected)
            best_score = problem.objective(selected)
            for removed in snapshot:
                retained = set(snapshot) - {removed}
                for added in shortlist:
                    if added in snapshot:
                        continue
                    replacement_trials += 1
                    subset = retained | {added}
                    if not problem.closed(subset):
                        continue
                    result = problem.greedy(subset, banned=frozenset({removed}))
                    if result is None:
                        continue
                    score = problem.objective(result)
                    if score > best_score + 1e-12 or (
                        abs(score - best_score) <= 1e-12 and tuple(result) < tuple(selected)
                    ):
                        selected, best_score = result, score
            diagnostics.update(search_method="bounded_proof_seed_search_with_single_replacement",
                               proof_seed_count=len(seeds), proof_mask_count=1 << len(nodes),
                               replacement_pool_size=len(shortlist), replacement_trials=replacement_trials,
                               exact_scope="all proof masks only; completions and ordering are greedy")
    selected = selected or []
    details, covered = [], {}
    for index, doc_id in enumerate(selected):
        utility, gain, redundancy = problem.utility(doc_id, selected[:index])
        details.append({"doc_id": doc_id, "utility": utility, "coverage_gain": gain,
                        "redundancy": redundancy, "goals": problem.goals[doc_id],
                        "verified": list(candidates[doc_id].get("verified", [])),
                        "protected_nodes": sorted(problem.protected.get(doc_id, set()))})
        for goal, score in problem.goals[doc_id].items():
            covered[goal] = max(covered.get(goal, 0.0), score)
    diagnostics.update(status="ok", objective_value=problem.objective(selected),
                       selected_prefix=details, covered_goals=covered,
                       utility_evaluations=problem.evaluations,
                       rejected_proofs=problem.invalid_proofs,
                       blocked_proof_document_ids=sorted(problem.blocked),
                       proof_closed=problem.closed(selected) if not problem.plan_error else False)
    return selected, diagnostics
