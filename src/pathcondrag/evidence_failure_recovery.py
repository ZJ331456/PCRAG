"""One bounded, failure-specific repair with program-rendered answer slots.

The parent pipeline is the measured fallback. A correct atomic DAG whose root
only failed quote validation is not replanned. Repairs use the question and
online diagnostics, never benchmark labels, and replace ranking inputs only
after their complete task structure has source citations.
"""
from __future__ import annotations

import copy
import json
import re

from .evidence_planning import (
    canonicalize_plan_dependencies, prune_derived_comparison_nodes,
    relation_faithfulness_error, terminal_attribute_error, validate_retrieval_plan,
)
from .evidence_retrieval import bind_question, normalized_text, validate_plan
from .evidence_structural_recovery import (
    _DESCRIPTION, _POSSESSIVE_DESCRIPTION, _complete_cited_repair, _plan_signature,
)
from .question_structure import route_question_structure


_SLOT = re.compile(r"\{\{([a-z][a-z_]*)\}\}")
_SLOT_ROLES = {"subject", "person", "work", "city", "state", "country", "organization", "party",
               "team", "location", "parent", "child", "performer", "director", "writer", "company",
               "region", "institution", "entity", "first", "second", "left", "right", "value",
               "date", "year", "label", "railroad", "film", "song", "actor", "member", "town",
               "county", "station", "governor", "university", "birthplace", "headquarters",
               "manufacturer", "star", "body", "source", "target", "ancestor"}
_REFERENCE = re.compile(r"\$\{([A-Za-z][\w-]*)\.answer\}")
_NAMES = re.compile(r"\b[A-ZÀ-ÖØ-Þ][\w'’.-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'’.-]*)*")
_INTERROGATIVES = {"Who", "Whose", "What", "Which", "When", "Where", "How", "On", "In", "At",
                  "Is", "Are", "Was", "Were", "Did", "Does", "Do", "The", "A", "An"}
_CITATION_FAILURES = {"evidence_not_exact_substring", "answer_not_supported_in_quote",
                      "answer_not_source_grounded", "original_quote_not_exact"}
_PERSON = {"person", "people", "actor", "actress", "author", "writer", "director", "performer", "singer"}
_TYPE_FAMILIES = {
    "town": "city", "city": "city", "state": "state", "country": "country",
    "political party": "party", "party": "party", "person": "person", "actor": "person",
    "performer": "person", "singer": "person", "writer": "person", "director": "person",
    "date": "time", "year": "time", "time": "time",
}
_RELATIONS = (
    r"\b(?:born|birth|birthplace|birthday)\b", r"\b(?:die|died|death|deceased)\b",
    r"\b(?:direct|directed|director)\b", r"\b(?:write|wrote|written|writer|author)\b",
    r"\b(?:compose|composed|composer)\b", r"\b(?:found|founded|founder|establish|established)\b",
    r"\b(?:design|designed|designer|architect)\b", r"\b(?:marry|married|spouse|wife|husband)\b",
    r"\b(?:publish|published|publisher)\b", r"\b(?:release|released)\b",
    r"\b(?:operate|operated|operating|operator)\b",
    r"\b(?:part\s+of|belong|belongs|belonged|include|includes|included)\b",
    r"\b(?:produce|produced|producer)\b", r"\b(?:manufacture|manufactured|manufacturer)\b",
    r"\b(?:headquarters|headquartered|based)\b",
)
_QUALIFIERS = re.compile(r"\b(?:main|lead|original|historic|national|international|female|male)\b", re.I)
_ENTITY_DESCRIPTION = re.compile(
    r"\b(?:the|a|an)\s+(?:[\w-]+\s+){0,2}"
    r"(?:actors?|actress|stars?|writer|author|director|performer|singer|manufacturer|"
    r"company|city|country|region|political party|record label)\s+"
    r"(?:of|for|in|from|behind|whose|that|which|who|releasing)\b", re.I)
_SNAPSHOT_FIELDS = ("plan", "bindings", "branch_scores", "routes", "search_count", "llm_plan_calls",
                    "llm_verification_calls", "planning_outputs", "verification_outputs")


def _literal(value, text):
    value = normalized_text(value)
    return bool(value) and " " + value + " " in " " + normalized_text(text) + " "


def _identity_spans(text):
    # Possession is grammar, not part of a person's/work's canonical identity.
    text = re.sub(r"['’]s\b", "", _REFERENCE.sub("", str(text)))
    names = []
    for match in _NAMES.finditer(text):
        words = match.group(0).rstrip(".,?").split()
        while words and words[0] in _INTERROGATIVES:
            words.pop(0)
        if words:
            names.append(" ".join(words))
    return names


def render_repair_templates(payload):
    """Render declared semantic slots; never append an unrelated dependency.

    ``inputs`` maps slot names to predecessor IDs. An optional ``depends_on``
    must agree exactly with those IDs. Bare pronouns, unused inputs, undeclared
    slots, guessed literal answer references and cyclic IDs remain invalid.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
        return None, "invalid_template_nodes"
    if not 1 <= len(payload["nodes"]) <= 7:
        return None, "invalid_template_node_count"
    nodes = []
    for item in payload["nodes"]:
        if not isinstance(item, dict):
            return None, "invalid_template_node"
        nid = item.get("id", "")
        template, inputs = item.get("question_template"), item.get("inputs")
        if not isinstance(nid, str) or not re.fullmatch(r"r[1-7]", nid):
            return None, "invalid_repair_node_id"
        if not isinstance(template, str) or not template.strip() or not isinstance(inputs, dict):
            return None, "invalid_template_or_inputs"
        if _REFERENCE.search(template) or "${" in template:
            return None, "model_written_answer_reference_not_allowed"
        slots = set(_SLOT.findall(template))
        stripped = _SLOT.sub("", template)
        if "{{" in stripped or "}}" in stripped:
            return None, "malformed_semantic_slot"
        if slots != set(inputs):
            return None, "slot_input_mismatch"
        if any(slot not in _SLOT_ROLES and slot.rsplit("_", 1)[-1] not in _SLOT_ROLES for slot in slots):
            return None, "ambiguous_semantic_slot_role"
        if re.search(r"\b(?:regarding|concerning|with respect to)\s+\{\{", template, re.I):
            return None, "unrelated_dependency_decoration"
        if re.search(r"\?\s*\{\{", template):
            return None, "unrelated_dependency_decoration"
        if any(not re.fullmatch(r"[a-z][a-z_]*", key) or not isinstance(value, str)
               or not re.fullmatch(r"r[1-7]", value) for key, value in inputs.items()):
            return None, "invalid_slot_dependency"
        deps = list(dict.fromkeys(inputs[slot] for slot in _SLOT.findall(template)))
        if "depends_on" in item and (not isinstance(item["depends_on"], list)
                                       or any(not isinstance(dep, str) for dep in item["depends_on"])
                                       or set(item["depends_on"]) != set(deps)):
            return None, "declared_dependency_without_semantic_slot"
        question = _SLOT.sub(lambda match: "${" + inputs[match.group(1)] + ".answer}", template)
        nodes.append({"id": nid, "question": question, "depends_on": deps,
                      "answer_type": str(item.get("answer_type", "unknown"))[:80]})
    canonical, _changes, reason = canonicalize_plan_dependencies({"nodes": nodes})
    return (None, reason) if reason else (canonical, None)


def _node_type(node):
    question = node.get("question", "")
    # A nominal explicit requested class is stronger than broad ``place``.
    match = re.search(r"^\s*(?:what|which)\s+(?:is\s+the\s+)?(political party|town|city|state|country)\b", question, re.I)
    if match:
        return _TYPE_FAMILIES[match.group(1).casefold()]
    return _TYPE_FAMILIES.get(normalized_text(node.get("answer_type", "")))


def _reference_type_conflicts(nodes):
    by_id = {node["id"]: node for node in nodes}
    conflicts = []
    for node in nodes:
        for match in re.finditer(r"\b(town|city|state|country|political party|party|person|actor|performer|singer|writer|director)\s+"
                                 r"\$\{([A-Za-z][\w-]*)\.answer\}", node["question"], re.I):
            expected = _TYPE_FAMILIES[match.group(1).casefold()]
            actual = _node_type(by_id.get(match.group(2), {}))
            if actual and actual != expected:
                conflicts.append({"node": node["id"], "input": match.group(2),
                                  "expected_type": expected, "producer_type": actual})
    return conflicts


def _unused_constraint_branches(query, nodes):
    if route_question_structure(query)["expand"]:
        return []
    used = {dep for node in nodes for dep in node["depends_on"]}
    sinks = [node for node in nodes if node["id"] not in used]
    if len(sinks) <= 1:
        return []
    # Only flag a clearly different terminal task. Unknown plural/conjunctive
    # requests may legitimately retrieve independent terminal evidence.
    if re.match(r"^\s*(?:when|in what year|what year|on what date)\b", query, re.I):
        matching = [node for node in sinks if _node_type(node) == "time"
                    or re.match(r"\s*(?:when|what year|in what year)\b", node["question"], re.I)]
    elif re.match(r"^\s*where\b", query, re.I):
        matching = [node for node in sinks if re.match(r"\s*where\b", node["question"], re.I)]
    else:
        return []
    if len(matching) == 1:
        return [node["id"] for node in sinks if node is not matching[0]]
    return []


def _compound_unknown(node):
    question = node["question"]
    descriptive = bool(_DESCRIPTION.search(question) or _POSSESSIVE_DESCRIPTION.search(question)
                       or _ENTITY_DESCRIPTION.search(question))
    # "Who is the writer of X?" already resolves that relation; "Where was
    # the writer of X born?" leaves a person unresolved inside a place lookup.
    attribute = bool(re.match(r"\s*(?:where|when|how|(?:what|which)\s+(?:year|date|place|country|city|nationality|role|symbol|attribute))\b", question, re.I)
                     or re.search(r"\bknown\s+for\b|\bparent\s+company\s+of\b", question, re.I))
    nested = len(re.findall(r"\b(?:of|in|from)\b", question, re.I)) >= 2
    return descriptive and (attribute or nested)


def _person_list(answer, evidence):
    parts = [part.strip() for part in re.split(r"\s*(?:,|;|\band\b|&)\s*", answer) if part.strip()]
    if not 2 <= len(parts) <= 8:
        return []
    name = r"[A-ZÀ-ÖØ-Þ][\w'’.-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'’.-]*)+"
    if not all(re.fullmatch(name, part) and _literal(part, evidence) for part in parts):
        return []
    return sorted(set(parts), key=normalized_text)


def _proof_conflicts(node, proof):
    """Recognize explicit class/direction conflicts, not arbitrary entailment."""
    answer, quote = str(proof.get("answer", "")), str(proof.get("evidence", ""))
    question = node["question"]
    if re.search(r"\bpolitical party\b", question, re.I) and re.search(r"\b(?:senate|congress|house of representatives|parliament)\b", answer, re.I):
        return "political_body_bound_as_party"
    if re.search(r"\bchild.in.law\b", question, re.I) and re.search(r"\b(?:their|his|her)\s+(?:daughter|son)\s+(?:was|is)\b", quote, re.I) and not re.search(r"\bin.law\b", quote, re.I):
        return "child_bound_as_child_in_law"
    # A different explicitly named object immediately follows the requested
    # predicate, while the accepted answer occurs elsewhere in the quote.
    if re.search(r"\breleased by\b", question, re.I):
        match = re.search(r"\breleased by\s+(?:the\s+)?([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*)", quote)
        if match and not _literal(answer, match.group(1)) and _literal(answer, quote):
            return "released_by_object_conflict"
    return None


def classify_recovery_failure(state):
    """Classify recoverable structure; citation-only atomic failures keep it."""
    plan, trace = state.get("_evidence_plan", []), state["evidence_trace"]
    bindings = state.get("_evidence_winning_bindings", {})
    if not plan:
        errors = trace.get("planning_outputs", {}).get("validation_errors", [])
        return {"kind": "invalid_dependency_structure", "errors": errors[-2:]}, None
    type_conflicts = _reference_type_conflicts(plan)
    if type_conflicts:
        return {"kind": "variable_type_conflict", "conflicts": type_conflicts}, None
    unused = _unused_constraint_branches(state["query"], plan)
    if unused:
        return {"kind": "unjoined_constraint_branch", "node_ids": unused}, None
    proofs = {proof.get("goal", "").removeprefix("dag:"): proof
              for candidate in state.get("evidence_candidates", {}).values()
              for proof in candidate.get("verified", []) if str(proof.get("goal", "")).startswith("dag:")}
    for node in plan:
        proof = proofs.get(node["id"], {})
        if _person_list(str(bindings.get(node["id"], "")), str(proof.get("evidence", ""))):
            return {"kind": "multi_person_binding", "node_ids": [node["id"]]}, None
        conflict = _proof_conflicts(node, proof)
        if conflict:
            return {"kind": "explicit_relation_conflict", "node_ids": [node["id"]], "reason": conflict}, None
    # SourceWitness may distinguish an explicit contradiction from unknown
    # support. Only contradiction enables this extra structural attempt.
    for output in trace.get("verification_outputs", []):
        if output.get("node") in bindings:
            continue
        witness = output.get("source_witness", {})
        if witness.get("status") == "contradicted" or witness.get("contradicted_count", 0):
            return {"kind": "source_relation_contradiction", "node_ids": [output.get("node")],
                    "reason": witness.get("reason", "explicit_source_conflict")}, None
    for rejected in trace.get("rejected_hypotheses", []):
        if rejected.get("node") not in bindings and str(rejected.get("reason", "")).startswith("contradicted_"):
            return {"kind": "source_relation_contradiction", "node_ids": [rejected.get("node")],
                    "reason": rejected["reason"]}, None
    compound = [node["id"] for node in plan if _compound_unknown(node)]
    if compound:
        return {"kind": "compound_unknown_relation", "node_ids": compound}, None
    if all(node["id"] in bindings for node in plan):
        return None, "complete_structure_without_explicit_conflict"
    rejects = trace.get("rejected_hypotheses", [])
    if rejects and all(item.get("reason") in _CITATION_FAILURES for item in rejects):
        return None, "citation_only_failure_keep_structure"
    return None, "no_explicit_repairable_structure_failure"


def validate_failure_repair(payload, query):
    rendered, reason = render_repair_templates(payload)
    if reason:
        return [], reason, {}
    nodes, reason = validate_plan(rendered, 7)
    if reason:
        return [], reason, {}
    nodes, removed = prune_derived_comparison_nodes(nodes)
    nodes, reason = validate_retrieval_plan({"nodes": nodes}, 6, 4)
    detail = {"pruned_nodes": removed, "program_rendered_references": True}
    if reason:
        return [], reason, detail
    reason = relation_faithfulness_error(query, nodes) or terminal_attribute_error(query, nodes)
    if reason:
        return [], reason, detail
    questions = "\n".join(node["question"] for node in nodes)
    for value in re.findall(r"\b\d+(?:st|nd|rd|th)?(?:[./-]\d+)*\b", query, re.I):
        if not _literal(value, questions):
            return [], "repair_dropped_numeric_qualifier", detail
    for value in re.findall(r'["“]([^"”]+)["”]', query) + _identity_spans(query):
        if not _literal(value, questions):
            return [], "repair_dropped_named_identity", detail
    for value in _QUALIFIERS.findall(query):
        if not _literal(value, questions):
            return [], "repair_dropped_role_qualifier", detail
    for pattern in _RELATIONS:
        if re.search(pattern, query, re.I) and not re.search(pattern, questions, re.I):
            return [], "repair_dropped_relation", detail
    for direction in re.findall(r"\b(?:north|south|east|west)\s+of\b", query, re.I):
        if not _literal(direction, questions):
            return [], "repair_dropped_direction", detail
    for value in _identity_spans(questions):
        if not _literal(value, re.sub(r"['’]s\b", "", query)):
            return [], "repair_introduced_named_identity", detail
    if _reference_type_conflicts(nodes):
        return [], "repair_variable_type_conflict", detail
    if _unused_constraint_branches(query, nodes):
        return [], "repair_unjoined_constraint_branch", detail
    if any(_compound_unknown(node) for node in nodes):
        return [], "repair_still_has_unbound_unknown_description", detail
    return nodes, None, detail


class FailureRecoveryMixin:
    """Opt-in failure classification and one schema-constrained repair."""

    def _verify(self, question, answer_type, docs):
        if "failure_recovery" not in self.improvements:
            return super()._verify(question, answer_type, docs)
        cache_key = (question, answer_type, tuple(docs.items()))
        diagnostic_key = question + "::" + ",".join(map(str, docs))
        cache = getattr(self, "_failure_verification_cache", {})
        repair = getattr(self, "_failure_repair_active", False)
        if repair and cache_key in cache:
            result, diagnostic = copy.deepcopy(cache[cache_key])
            diagnostic.update(cached_attempts=diagnostic.get("attempts", 0), attempts=0,
                              failure_recovery_cache_hit=True)
            self._verification_diagnostics[diagnostic_key] = diagnostic
        else:
            result = super()._verify(question, answer_type, docs)
            if hasattr(self, "_failure_verification_cache"):
                cache[cache_key] = copy.deepcopy((result, self._verification_diagnostics.get(diagnostic_key, {})))
        if not repair or normalized_text(answer_type) not in _PERSON:
            return result
        accepted, rejected, reason = result
        normalized, splits = [], []
        for proof in accepted:
            people = _person_list(proof["answer"], proof.get("evidence", ""))
            if people:
                splits.append({"original_answer": proof["answer"], "answers": people[:3]})
                normalized.extend(dict(proof, answer=person) for person in people[:3])
            else:
                normalized.append(proof)
        if splits:
            self._verification_diagnostics[diagnostic_key]["failure_recovery_person_splits"] = splits
        return normalized, rejected, reason

    def _failure_repair_plan(self, state, failure):
        ctx = state.get("base", (None, None, {}))[2] or {}
        messages = [
            {"role": "system", "content": (
                "Repair a diagnosed retrieval structure using ONLY the input question. Return ONLY JSON. "
                "Each node has id (r1,r2,...), question_template, inputs (object mapping named slots to "
                "predecessor IDs), and answer_type. Root inputs is {}. The program renders {{subject}} "
                "to the referenced predecessor answer. Every inputs key MUST occur literally as a "
                "{{key}} slot in the grammatical question_template; unused inputs and bare pronouns "
                "are invalid. Do not write ${ID.answer} yourself. Example syntax: {\"nodes\":["
                "{\"id\":\"r1\",\"question_template\":\"Who wrote Work X?\",\"inputs\":{},\"answer_type\":\"person\"},"
                "{\"id\":\"r2\",\"question_template\":\"Where was {{person}} born?\","
                "\"inputs\":{\"person\":\"r1\"},\"answer_type\":\"place\"}]}. "
                "These syntax examples never change the input's actual requested relation. "
                "Resolve each unknown innermost entity before asking its outer attribute. Each node "
                "must ask one atomic passage-supported relation. Never guess a person/work/place. "
                "Preserve names, works, dates, nationality, role, location, relative directions, "
                "negation and final requested attribute. Use an actual person as a person slot, not "
                "their occupation or a list of names. A city is not a state, a political party is "
                "not a legislature, child is not child-in-law, released-by is not signed-with. "
                "Every qualifying evidence branch must connect to the requested terminal fact. "
                "Parallel comparisons may retrieve independent attribute leaves, but do not add "
                "comparison/arithmetic nodes over existing values. Never attach a dependency merely "
                "as an unrelated 'regarding X' phrase. Use the fewest necessary nodes, maximum 6 and "
                "depth 4. The authorized depth hint is a capacity cue, not a required node count. "
                "If no faithful repair is possible, return {\"nodes\":[]}." )},
            {"role": "user", "content": (
                f"Question: {state['query']}\nAuthorized depth hint: {max(1, int(ctx.get('hops', 1)))}\n"
                "Diagnosed failure: " + json.dumps(failure, ensure_ascii=False) + "\n"
                "Original plan (diagnostic only, not facts): " + json.dumps(state.get("_evidence_plan", []), ensure_ascii=False) + "\n"
                "Optional static subquestions: " + json.dumps([
                    question for question in state.get("static_sub_questions", []) if isinstance(question, str)
                ][:4], ensure_ascii=False))},
        ]
        payload, reason = self._infer_object(messages)
        nodes, detail = [], {}
        if not reason:
            nodes, reason, detail = validate_failure_repair(payload, state["query"])
        return nodes, reason, payload, detail

    def _failure_task_support(self, state, plan):
        """Require actual predicate witnesses in addition to exact citations."""
        from .evidence_source_witness import source_witness_check
        bindings = state.get("_evidence_winning_bindings", {})
        proofs = {proof.get("goal", "").removeprefix("dag:"): proof
                  for candidate in state.get("evidence_candidates", {}).values()
                  for proof in candidate.get("verified", []) if str(proof.get("goal", "")).startswith("dag:")}
        failures = []
        for node in plan:
            proof = proofs.get(node["id"])
            question = bind_question(node["question"], bindings)
            if not proof or question is None:
                failures.append({"node": node["id"], "reason": "missing_bound_source_proof"})
                continue
            check, reason = source_witness_check(node, question, proof, self._document(proof["doc_id"]), bindings)
            if not check:
                failures.append({"node": node["id"], "reason": reason or "unknown_source_relation"})
        return failures

    def _dependency_search(self, states, executor):
        enabled = "failure_recovery" in self.improvements
        if enabled:
            self._failure_verification_cache, self._failure_repair_active = {}, False
        super()._dependency_search(states, executor)
        if not enabled:
            return
        pending = []
        for state in states:
            trace = state["evidence_trace"]
            failure, reason = classify_recovery_failure(state)
            diagnostic = {"failure": failure, "result": reason, "applied": False, "uses_gold_annotations": False,
                          "extra_plan_requests": 0, "extra_searches": 0, "extra_verification_requests": 0,
                          "original_parent_fields": copy.deepcopy({key: trace.get(key) for key in _SNAPSHOT_FIELDS})}
            trace["improvement_failure_recovery"] = diagnostic
            if failure and trace["search_count"] < self.max_searches:
                pending.append((state, failure))
            elif failure:
                diagnostic["result"] = "existing_search_budget_exhausted"
        results = self._collect_jobs(executor, [(self._failure_repair_plan, (state, failure)) for state, failure in pending])
        repairs = []
        for (state, _failure), (plan, reason, payload, detail) in zip(pending, results):
            trace = state["evidence_trace"]
            diagnostic = trace["improvement_failure_recovery"]
            diagnostic.update(extra_plan_requests=1, validation_error=reason, repair_output=payload,
                              validation_detail=detail, repair_plan=plan)
            trace["llm_plan_calls"] += 1
            if reason:
                diagnostic["result"] = "invalid_repair_preserved_original"
                continue
            if (_plan_signature(plan) == _plan_signature(state.get("_evidence_plan", []))
                    and _failure.get("kind") != "multi_person_binding"):
                diagnostic["result"] = "unchanged_structure_skipped"
                continue
            clone = copy.deepcopy(state)
            clone["_evidence_plan"], clone["evidence_trace"]["plan"] = plan, plan
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
        self._failure_repair_active = True
        try:
            super()._dependency_search([entry[1] for entry in repairs], executor)
        finally:
            self._failure_repair_active = False
        for state, clone, plan, searches_before, verifies_before in repairs:
            trace, clone_trace = state["evidence_trace"], clone["evidence_trace"]
            diagnostic = trace["improvement_failure_recovery"]
            additional_outputs = clone_trace.get("verification_outputs", [])[len(trace.get("verification_outputs", [])):]
            diagnostic.update(extra_searches=clone_trace["search_count"] - searches_before,
                              extra_verification_requests=clone_trace["llm_verification_calls"] - verifies_before,
                              repair_bindings=copy.deepcopy(clone.get("_evidence_winning_bindings", {})),
                              repair_verification_outputs=copy.deepcopy(additional_outputs),
                              reused_verification_count=sum(bool(output.get("failure_recovery_cache_hit")) for output in additional_outputs),
                              search_budget_preserved=clone_trace["search_count"] <= self.max_searches)
            # Old contradiction diagnostics explain the repair trigger; they
            # must not condemn a newly supported DAG after IDs change.
            new_ids = {node["id"] for node in plan}
            validation_state = dict(clone, evidence_trace=dict(
                clone_trace, verification_outputs=additional_outputs,
                rejected_hypotheses=[item for item in clone_trace.get("rejected_hypotheses", [])
                                     if item.get("node") in new_ids]))
            failure_after, _keep_reason = classify_recovery_failure(validation_state)
            complete = _complete_cited_repair(clone, plan, self._document)
            support_failures = self._failure_task_support(clone, plan) if complete else []
            diagnostic["task_support_failures"] = support_failures
            if not complete or failure_after or support_failures:
                trace["search_count"] = clone_trace["search_count"]
                trace["llm_verification_calls"] = clone_trace["llm_verification_calls"]
                state.setdefault("_evidence_search_cache", {}).update(clone.get("_evidence_search_cache", {}))
                diagnostic.update(result="unsupported_or_incomplete_repair_preserved_original",
                                  remaining_failure=failure_after,
                                  supported_subgraph_node_ids=list(clone.get("_evidence_winning_bindings", {})))
                continue
            diagnostic.update(applied=True, result="complete_task_repair_applied")
            clone_trace["improvement_failure_recovery"] = diagnostic
            clone_trace.setdefault("planning_outputs", {})["failure_repair"] = {
                "attempts": 1, "outputs": [diagnostic["repair_output"]], "validation_errors": []}
            state.update(clone)
        self._failure_verification_cache = {}
