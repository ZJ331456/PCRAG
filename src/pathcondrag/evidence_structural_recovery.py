"""Opt-in recovery of an invalid or incompletely bound retrieval structure.

The existing planner and dependency search run first. A question gets at most
one repair planning request; its old DAG and candidate pool remain the fallback.
Only a fully cited repaired DAG can replace that fallback. All decisions use
the input question and online planning/search diagnostics, never gold labels.
"""
from __future__ import annotations

import copy
import json
import re

from .evidence_planning import (
    canonicalize_plan_dependencies, prune_derived_comparison_nodes,
    relation_faithfulness_error, terminal_attribute_error, validate_retrieval_plan,
)
from .evidence_retrieval import normalized_text, validate_plan, verify_hypotheses


_REFERENCE = re.compile(r"\$\{([A-Za-z][\w-]*)\.answer\}")
_DESCRIPTION = re.compile(
    r"\b(?:the|a|an)\s+(?:[\w-]+\s+){0,2}"
    r"(?:writer|author|director|composer|performer|singer|actor|actress|"
    r"maker|manufacturer|creator|founder|owner|operator|spouse|parent|"
    r"father|mother|birthplace|hometown|city|country|region|company|"
    r"university|team|league|album|record\s+label)\s+"
    r"(?:of|for|behind|where|whose|that|which|who)\b", re.I)
_POSSESSIVE_DESCRIPTION = re.compile(
    r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*['’]s\s+"
    r"(?:writer|author|director|composer|performer|maker|manufacturer|"
    r"creator|owner|spouse|parent)\b")
_QUESTION_WORDS = {
    "Who", "Whose", "What", "Which", "When", "Where", "How", "On", "In",
    "At", "Is", "Are", "Was", "Were", "Did", "Does", "Do", "The", "A", "An",
}
_NAMED = re.compile(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*")
_QUALIFIERS = re.compile(
    r"\b(?:main|lead|original|historic|national|international|female|male)\b", re.I)
_RELATION_FAMILIES = (
    r"\b(?:born|birth|birthplace|birthday)\b",
    r"\b(?:die|dies|died|death|dead|deceased)\b",
    r"\b(?:direct|directed|directing|director)\b",
    r"\b(?:write|wrote|written|writer|author|authored)\b",
    r"\b(?:compose|composed|composer)\b",
    r"\b(?:found|founded|founder|establish|established)\b",
    r"\b(?:design|designed|designer|architect)\b",
    r"\b(?:marry|married|spouse|wife|husband)\b",
    r"\b(?:perform|performed|performer|sing|sang|singer)\b",
    r"\b(?:publish|published|publisher)\b",
    r"\b(?:release|released)\b",
    r"\b(?:operate|operated|operating|operator)\b",
    r"\b(?:part\s+of|belong|belongs|belonged)\b",
)


def _literal(value, text):
    value = normalized_text(value)
    return bool(value) and " " + value + " " in " " + normalized_text(text) + " "


def _named_spans(text):
    spans = []
    for match in _NAMED.finditer(_REFERENCE.sub("", text)):
        words = match.group(0).rstrip(".,?").split()
        while words and words[0] in _QUESTION_WORDS:
            words.pop(0)
        if words:
            spans.append(" ".join(words))
    return spans


def structural_recovery_trigger(state):
    """Return a text/diagnostic failure signal, or a reason to keep the DAG."""
    plan = state.get("_evidence_plan", [])
    bindings = state.get("_evidence_winning_bindings", {})
    if plan and all(node["id"] in bindings for node in plan):
        return None, "existing_complete_dag"
    trace = state.get("evidence_trace", {})
    if not plan:
        errors = trace.get("planning_outputs", {}).get("validation_errors", [])
        failure = next((str(item) for item in reversed(errors) if item), None)
        return {"kind": "invalid_plan" if failure else "empty_plan",
                "reason": failure or "no_accepted_retrieval_plan"}, None
    used = {dep for node in plan for dep in node.get("depends_on", [])}
    unresolved = [node["id"] for node in plan if node["id"] in used and node["id"] not in bindings]
    if unresolved:
        return {"kind": "unbound_intermediate", "node_ids": unresolved,
                "reason": "a_required_intermediate_has_no_verified_binding"}, None
    compound = [node["id"] for node in plan if not node.get("depends_on")
                and node["id"] not in bindings
                and (_DESCRIPTION.search(node["question"])
                     or _POSSESSIVE_DESCRIPTION.search(node["question"]))]
    if compound:
        return {"kind": "compound_unknown_root", "node_ids": compound,
                "reason": "unknown_entity_description_is_not_a_bound_variable"}, None
    return None, "no_diagnostic_structure_failure"


def _plan_signature(nodes):
    """Ignore ID renaming so an unchanged repair does not rerun verification."""
    positions = {node["id"]: index for index, node in enumerate(nodes)}
    return tuple((normalized_text(_REFERENCE.sub(
        lambda match: f"VARIABLE{positions.get(match.group(1), -1)}", node["question"])),
        tuple(positions.get(dep, -1) for dep in node["depends_on"]),
        normalized_text(node.get("answer_type", "unknown"))) for node in nodes)


def validate_structural_repair(payload, query):
    """Check syntax, bounded capacity and explicit input constraints.

    Lexical constraints conservatively reject dropped names/years/relations;
    they are not a claim that every remaining plan semantically entails the
    original question. The unchanged verifier still checks all actual bindings.
    """
    canonical, changes, reason = canonicalize_plan_dependencies(payload)
    if reason:
        return [], reason, {"canonicalization_changes": changes, "pruned_nodes": []}
    nodes, reason = validate_plan(canonical, 7)
    if reason:
        return [], reason, {"canonicalization_changes": changes, "pruned_nodes": []}
    if any(not re.fullmatch(r"r[1-7]", node["id"]) for node in nodes):
        return [], "repair_requires_letter_prefixed_ids", {"canonicalization_changes": changes, "pruned_nodes": []}
    nodes, removed = prune_derived_comparison_nodes(nodes)
    detail = {"canonicalization_changes": changes, "pruned_nodes": removed}
    nodes, reason = validate_retrieval_plan({"nodes": nodes}, 6, 4)
    if reason:
        return [], reason, detail
    reason = relation_faithfulness_error(query, nodes) or terminal_attribute_error(query, nodes)
    if reason:
        return [], reason, detail
    questions = "\n".join(node["question"] for node in nodes)
    for value in re.findall(r"\b\d+(?:[./-]\d+)*\b", query):
        if not _literal(value, questions):
            return [], "repair_dropped_numeric_qualifier", detail
    for value in re.findall(r'["“]([^"”]+)["”]', query) + _named_spans(query):
        if not _literal(value, questions):
            return [], "repair_dropped_named_identity", detail
    for value in _QUALIFIERS.findall(query):
        if not _literal(value, questions):
            return [], "repair_dropped_role_or_scope_qualifier", detail
    for pattern in _RELATION_FAMILIES:
        if re.search(pattern, query, re.I) and not re.search(pattern, questions, re.I):
            return [], "repair_dropped_explicit_relation", detail
    for value in _named_spans(questions):
        if not _literal(value, query):
            return [], "repair_introduced_named_identity", detail
    return nodes, None, detail


def _complete_cited_repair(state, plan, document):
    bindings = state.get("_evidence_winning_bindings", {})
    if not all(node["id"] in bindings for node in plan):
        return False
    proofs = {proof.get("goal"): proof for candidate in state.get("evidence_candidates", {}).values()
              for proof in candidate.get("verified", [])}
    for node in plan:
        proof = proofs.get(f"dag:{node['id']}", {})
        if normalized_text(proof.get("answer", "")) != normalized_text(bindings[node["id"]]):
            return False
        doc_id = proof.get("doc_id")
        if not isinstance(doc_id, int) or not verify_hypotheses(
                {"hypotheses": [proof]}, {doc_id: document(doc_id)[:4000]})[0]:
            return False
    return True


class StructuralRecoveryMixin:
    """Place first in the evidence MRO; require ``structural_recovery``."""

    def _verify(self, question, answer_type, docs):
        if "structural_recovery" not in self.improvements:
            return super()._verify(question, answer_type, docs)
        key = (question, answer_type, tuple(docs.items()))
        cache = getattr(self, "_structural_verification_cache", {})
        diagnostic_key = question + "::" + ",".join(map(str, docs))
        if getattr(self, "_structural_recovery_reusing", False) and key in cache:
            result, diagnostic = copy.deepcopy(cache[key])
            diagnostic.update(cached_attempts=diagnostic.get("attempts", 0), attempts=0,
                              structural_recovery_cache_hit=True)
            self._verification_diagnostics[diagnostic_key] = diagnostic
            return result
        result = super()._verify(question, answer_type, docs)
        if hasattr(self, "_structural_verification_cache"):
            cache[key] = copy.deepcopy((result, self._verification_diagnostics.get(diagnostic_key, {})))
        return result

    def _structural_repair_plan(self, state, trigger):
        ctx = state.get("base", (None, None, {}))[2] or {}
        depth_hint = max(1, int(ctx.get("hops", 1)))
        messages = [
            {"role": "system", "content": (
                "Repair a diagnosed evidence-retrieval structure using ONLY the input question. "
                "Return ONLY JSON {nodes:[{id,question,depends_on,answer_type}]}. Use letter-prefixed "
                "IDs r1,r2,..., never numeric IDs. Each node retrieves one relation or attribute "
                "from a passage. Resolve the innermost unknown entity first; every dependent "
                "question must contain each dependency as the literal ${r1.answer}, ${r2.answer}, "
                "etc., and list exactly those IDs in depends_on. A pronoun or 'the director' is "
                "not a variable reference. Never guess an unknown person's/work's name or any "
                "answer. The original rejected/partial plan explains the failure and is not "
                "evidence about the world. Static subquestions are optional structural hints, "
                "not facts; retain one only if it preserves the actual input question. Keep all "
                "named works/entities, years, nationality, role, location, temporal qualifiers, "
                "relation directions and the final requested attribute. Do not replace a "
                "difficult relation with a familiar birth, nationality, or creator question. "
                "Use the fewest necessary atomic nodes, at most 6 nodes and dependency depth 4. "
                "The depth hint is a capacity reference, never a requirement to invent hops. "
                "Independent comparison branches may be parallel. Do not add a final comparison, "
                "arithmetic or aggregation over already retrieved values. If the structural "
                "failure cannot be repaired faithfully, return {\"nodes\":[]}." )},
            {"role": "user", "content": (
                f"Question: {state['query']}\nAuthorized dependency depth hint: {depth_hint}\n"
                "Specific structure failure: " + json.dumps(trigger, ensure_ascii=False) + "\n"
                "Original accepted/partial plan: " + json.dumps(state.get("_evidence_plan", []), ensure_ascii=False) + "\n"
                "Original planning diagnostics: " + json.dumps({
                    "validation_errors": state["evidence_trace"].get("planning_outputs", {}).get("validation_errors", []),
                    "rejected_outputs": state["evidence_trace"].get("planning_outputs", {}).get("outputs", [])[-2:],
                }, ensure_ascii=False) + "\n"
                "Static subquestions: " + json.dumps([
                    question for question in state.get("static_sub_questions", []) if isinstance(question, str)
                ][:4], ensure_ascii=False))},
        ]
        payload, reason = self._infer_object(messages)
        nodes, detail = [], {}
        if not reason:
            nodes, reason, detail = validate_structural_repair(payload, state["query"])
        return nodes, reason, payload, detail

    def _dependency_search(self, states, executor):
        enabled = "structural_recovery" in self.improvements
        if enabled:
            self._structural_verification_cache = {}
            self._structural_recovery_reusing = False
        super()._dependency_search(states, executor)
        if not enabled:
            return
        pending = []
        for state in states:
            trace = state["evidence_trace"]
            trigger, reason = structural_recovery_trigger(state)
            diagnostic = {"trigger": trigger, "extra_plan_requests": 0, "extra_searches": 0,
                          "extra_verification_requests": 0, "applied": False,
                          "result": reason, "uses_gold_annotations": False,
                          "original_fallback_preserved": True, "search_budget": self.max_searches}
            trace["improvement_structural_recovery"] = diagnostic
            if not trigger:
                continue
            if trace["search_count"] >= self.max_searches:
                diagnostic["result"] = "existing_search_budget_exhausted"
                continue
            pending.append((state, trigger))
        results = self._collect_jobs(executor, [(self._structural_repair_plan, (state, trigger))
                                              for state, trigger in pending])
        repairs = []
        for (state, _trigger), (plan, reason, payload, detail) in zip(pending, results):
            trace = state["evidence_trace"]
            diagnostic = trace["improvement_structural_recovery"]
            diagnostic.update(extra_plan_requests=1, validation_error=reason, repair_output=payload,
                              validation_detail=detail, original_plan=copy.deepcopy(state.get("_evidence_plan", [])))
            trace["llm_plan_calls"] += 1
            if reason:
                diagnostic["result"] = "invalid_repair_preserved_original_dag"
                continue
            if _plan_signature(plan) == _plan_signature(state.get("_evidence_plan", [])):
                diagnostic["result"] = "unchanged_plan_skipped"
                continue
            clone = copy.deepcopy(state)
            clone["_evidence_plan"] = plan
            clone["evidence_trace"]["plan"] = plan
            for doc_id, candidate in list(clone["evidence_candidates"].items()):
                candidate["sources"] = [source for source in candidate.get("sources", [])
                                        if not str(source.get("source", "")).startswith("dag:")]
                candidate["verified"] = [proof for proof in candidate.get("verified", [])
                                         if not str(proof.get("goal", "")).startswith("dag:")]
                if not candidate.get("base_score", 0) and not candidate["sources"] and not candidate["verified"]:
                    del clone["evidence_candidates"][doc_id]
            clone["evidence_trace"]["routes"] = [route for route in clone["evidence_trace"].get("routes", [])
                                                 if not str(route.get("source", "")).startswith("dag:")]
            for key in ("_evidence_winning_bindings", "_evidence_layers", "_evidence_beams"):
                clone.pop(key, None)
            repairs.append((state, clone, plan, trace["search_count"], trace["llm_verification_calls"]))
        self._structural_recovery_reusing = True
        try:
            # Existing thread pool batches LLM requests; embedding stays on the caller thread.
            super()._dependency_search([entry[1] for entry in repairs], executor)
        finally:
            self._structural_recovery_reusing = False
        for state, clone, plan, searches_before, verification_before in repairs:
            trace, clone_trace = state["evidence_trace"], clone["evidence_trace"]
            diagnostic = trace["improvement_structural_recovery"]
            diagnostic.update(extra_searches=clone_trace["search_count"] - searches_before,
                              extra_verification_requests=clone_trace["llm_verification_calls"] - verification_before,
                              repair_bindings=dict(clone.get("_evidence_winning_bindings", {})),
                              reused_verification_count=sum(bool(output.get("structural_recovery_cache_hit"))
                                  for output in clone_trace.get("verification_outputs", [])
                                  [len(trace.get("verification_outputs", [])):]),
                              search_budget_preserved=clone_trace["search_count"] <= self.max_searches)
            if not _complete_cited_repair(clone, plan, self._document):
                # Preserve ranking inputs, but account for actual spend and retain reusable searches.
                trace["search_count"] = clone_trace["search_count"]
                trace["llm_verification_calls"] = clone_trace["llm_verification_calls"]
                state.setdefault("_evidence_search_cache", {}).update(clone.get("_evidence_search_cache", {}))
                diagnostic["result"] = "incomplete_repair_preserved_original_dag"
                diagnostic["repair_verification_outputs"] = clone_trace.get("verification_outputs", [])[
                    len(trace.get("verification_outputs", [])):]
                continue
            diagnostic.update(applied=True, result="complete_cited_repair_applied")
            clone_trace["improvement_structural_recovery"] = diagnostic
            clone_trace.setdefault("planning_outputs", {})["structural_repair"] = {
                "attempts": 1, "outputs": [diagnostic["repair_output"]], "validation_errors": []}
            state.update(clone)
        self._structural_verification_cache = {}
