"""Optional plan capacity and failure-triggered dependency search improvements.

The planner uses only the question and a depth hint. Neither component reads
benchmark answers, support annotations, question types, or gold document counts.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from .evidence_retrieval import normalized_text, plan_layers, validate_plan


def validate_retrieval_plan(payload: dict, node_budget: int,
                            depth_budget: int) -> Tuple[List[dict], Optional[str]]:
    """Keep capacity and dependency depth separate; reject derived comparisons."""
    nodes, reason = validate_plan(payload, node_budget)
    if reason:
        return nodes, reason
    if len(plan_layers(nodes)) > depth_budget:
        return [], "dependency_depth_exceeded"
    for node in nodes:
        # Comparing two retrieved answers is computation, not passage retrieval.
        answer_type = normalized_text(node["answer_type"])
        if len(node["depends_on"]) >= 2 and answer_type in {
            "comparison", "boolean comparison", "final answer", "aggregation"
        }:
            return [], "derived_comparison_is_not_retrieval"
    return nodes, None


class PlannerImprovementMixin:
    """Allow parallel atomic branches without inflating the depth hint."""

    def _plan(self, query: str, hops: int):
        if "planning" not in self.improvements:
            return super()._plan(query, hops)
        node_budget = max(1, int(getattr(self.cfg, "evidence_plan_node_budget", 6)))
        depth_budget = max(1, int(getattr(self.cfg, "evidence_plan_depth_budget", 4)))
        depth_hint = max(1, int(hops))
        messages = [
            {"role": "system", "content": (
                "Plan evidence retrieval using ONLY the input question. Return ONLY JSON "
                "{nodes:[{id,question,depends_on,answer_type}]}. Each node asks for one "
                "atomic relation or attribute that a single passage can establish. "
                "Keep every unknown intermediate entity as a literal ${ID.answer} "
                "placeholder; list exactly those IDs in depends_on. Never guess an answer. "
                "The depth hint is the expected length of one dependency chain, NOT "
                "the total number of nodes. Independent comparison branches may each "
                "have their own complete chain. Retrieve their attributes separately. "
                "Do not add a final comparison, arithmetic, counting, or aggregation "
                "node: these operations combine retrieved facts, not passages. "
                "For 'which director of Film A and Film B was born earlier?', use four "
                "nodes: s1='Who directed Film A?', s2='When was ${s1.answer} born?', "
                "s3='Who directed Film B?', s4='When was ${s3.answer} born?'. "
                "Do not compress either branch into 'When was the director of Film A born?'. "
                "For 'Where was the writer of Work X born?', retrieve the writer first "
                "and then retrieve that person's birthplace. Preserve all nested "
                "location, time, and relation qualifiers. Use no hashes and no cycles. "
                "Use the fewest atomic nodes that preserve the question's evidence needs."
            )},
            {"role": "user", "content": (
                f"Question: {query}\nExpected dependency depth hint: {depth_hint}\n"
                f"Maximum total retrieval nodes: {node_budget}\n"
                f"Maximum dependency depth: {depth_budget}\n"
                'Syntax: {"nodes":[{"id":"s1","question":"Who wrote Work X?",'
                '"depends_on":[],"answer_type":"person"},{"id":"s2",'
                '"question":"Where was ${s1.answer} born?","depends_on":["s1"],'
                '"answer_type":"place"}]}'
            )},
        ]
        diagnostics = {"attempts": 0, "outputs": [], "validation_errors": [],
                       "depth_hint": depth_hint, "node_budget": node_budget,
                       "depth_budget": depth_budget, "capacity_policy": "atomic_parallel"}
        self._plan_diagnostics[query] = diagnostics
        reason = None
        for attempt in range(2):
            payload, reason = self._infer_object(messages)
            diagnostics["attempts"] += 1
            diagnostics["outputs"].append(payload)
            if not reason:
                nodes, reason = validate_retrieval_plan(payload, node_budget, depth_budget)
                if not reason:
                    diagnostics["actual_depth"] = len(plan_layers(nodes))
                    return nodes, None
            diagnostics["validation_errors"].append(reason)
            if attempt == 0:
                messages.extend([
                    {"role": "assistant", "content": json.dumps(payload or {}, ensure_ascii=False)},
                    {"role": "user", "content": (
                        f"Validation failed: {reason}. Correct the JSON with at most "
                        f"{node_budget} atomic retrieval nodes and depth at most {depth_budget}. "
                        "Every dependency must appear as ${ID.answer}. Preserve separate "
                        "parallel branches; omit derived comparison operations. Do not "
                        "guess unknown intermediate entities. Return ONLY corrected JSON."
                    )},
                ])
        return [], reason


class AdaptiveSearchMixin:
    """Spend extra verification only on missing evidence or close alternatives.

    Place this mixin before a binding verifier in the method resolution order,
    so both the first and fallback batches use the same binding rules.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "adaptive" in self.improvements:
            self.beam_width = 2

    def _verification_documents(self, ids: List[int]) -> Dict[int, str]:
        count = 6 if "adaptive" in self.improvements else 3
        return {int(doc_id): self._document(int(doc_id))[:4000] for doc_id in ids[:count]}

    def _verify(self, question: str, answer_type: str, docs: Dict[int, str]):
        if "adaptive" not in self.improvements or len(docs) <= 3:
            return super()._verify(question, answer_type, docs)
        entries = list(docs.items())
        first, extra = dict(entries[:3]), dict(entries[3:6])
        accepted, rejected, reason = super()._verify(question, answer_type, first)
        first_key = question + "::" + ",".join(map(str, first))
        first_diagnostic = dict(self._verification_diagnostics.get(first_key, {"attempts": 1}))
        diagnostics = {"attempts": first_diagnostic["attempts"],
                       "outputs": list(first_diagnostic.get("outputs", [])),
                       "adaptive_extra_batches": 0,
                       "batch_diagnostics": [first_diagnostic],
                       "verified_document_batches": [list(first)]}
        # Protocol failures and HTTP exceptions must not silently become a retry
        # against different evidence. HTTP exceptions propagate from super().
        if not accepted and reason is None and extra:
            next_accepted, next_rejected, reason = super()._verify(question, answer_type, extra)
            accepted = next_accepted
            rejected = rejected + next_rejected
            extra_key = question + "::" + ",".join(map(str, extra))
            extra_diagnostic = self._verification_diagnostics.get(extra_key, {"attempts": 1})
            diagnostics["attempts"] += extra_diagnostic["attempts"]
            diagnostics["outputs"].extend(extra_diagnostic.get("outputs", []))
            diagnostics["batch_diagnostics"].append(dict(extra_diagnostic))
            diagnostics["adaptive_extra_batches"] = 1
            diagnostics["verified_document_batches"].append(list(extra))
        key = question + "::" + ",".join(map(str, docs))
        self._verification_diagnostics[key] = diagnostics
        return accepted, rejected, reason

    def _prune_beams(self, beams: List[dict], total: int) -> List[dict]:
        ordered = super()._prune_beams(beams, total)
        if "adaptive" not in self.improvements or len(ordered) < 2:
            return ordered
        margin = max(0.0, float(getattr(self.cfg, "evidence_ambiguity_margin", 0.06)))
        gap = self._beam_score(ordered[0], total) - self._beam_score(ordered[1], total)
        # Every retained alternative is already supported by an accepted quote.
        # A clear winner keeps the single-path behavior and cost of exp4.
        return ordered[:2] if gap <= margin else ordered[:1]
