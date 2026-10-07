"""Optional plan capacity and failure-triggered dependency search improvements.

The planner uses only the question and a depth hint. Neither component reads
benchmark answers, support annotations, question types, or gold document counts.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from .evidence_retrieval import normalized_text, plan_layers, validate_plan
from .question_structure import route_dependency_depth, route_question_structure


_PLAN_REFERENCE = re.compile(r"\$\{([A-Za-z][\w-]*)\.answer\}")
_ANSWER_REFERENCE_LIKE = re.compile(r"\b[A-Za-z][\w-]*\.answer\b")
_DEATH_ATTRIBUTE = re.compile(r"\b(?:die|dies|died|death|dead|deceased|passed\s+away)\b", re.I)
_BIRTH_ATTRIBUTE = re.compile(r"\b(?:born|birth|birthday|birthdate)\b", re.I)
_END_ATTRIBUTE = re.compile(r"\b(?:end|ends|ended|ceased|terminated)\b", re.I)
_TIME_QUESTION = re.compile(r"\b(?:when|what\s+(?:date|year|time)|which\s+(?:date|year|time))\b", re.I)
_REQUESTED_TIME = re.compile(
    r"^\s*(?:when|(?:in|on)\s+what\s+(?:date|year|time)|(?:what|which)\s+(?:date|year|time))\b", re.I)
_REQUESTED_PLACE = re.compile(
    r"^\s*(?:where|(?:in|at)\s+(?:what|which)\s+(?:city|country|place|region)|"
    r"(?:what|which)\s+(?:city|country|place|region))\b", re.I)
_PURE_EQUALITY = re.compile(
    r"(?:are|is|were|was) VALUE(?: (?:and|or) VALUE)+ (?:the )?"
    r"(?:same|identical|different)(?: (?P<attribute>person|people|individual|individuals|entity|entities|"
    r"answer|answers|value|values|name|names|date|dates|country|countries|nationality|nationalities))?", re.I)
_PURE_ORDERING = re.compile(
    r"(?:which (?:date|year|number|value) is (?:earlier|later|higher|lower|larger|smaller|greater|less) "
    r"VALUE(?: (?:and|or) VALUE)+|"
    r"(?:is|was) VALUE (?:earlier|later|higher|lower|larger|smaller|greater|less) than VALUE)", re.I)
_COMPARABLE_TYPES = {"date", "year", "time", "number", "numeric", "integer", "quantity", "value"}
_IDENTITY_ATTRIBUTE_TYPES = {
    "person": {"person", "people", "individual"}, "people": {"person", "people", "individual"},
    "individual": {"person", "people", "individual"}, "individuals": {"person", "people", "individual"},
    "date": {"date", "year", "time"}, "dates": {"date", "year", "time"},
    "country": {"country"}, "countries": {"country"},
    "nationality": {"nationality"}, "nationalities": {"nationality"},
}


def canonicalize_plan_dependencies(payload: dict) -> Tuple[dict, List[dict], Optional[str]]:
    """Derive dependency metadata from existing literal references only.

    The questions and node identifiers are never rewritten. Missing/redundant
    metadata can be corrected when the question already names its dependencies.
    A declared dependency without a placeholder has no such deterministic repair.
    Full ordering, cycle, capacity and depth checks remain with the validator.
    """
    changes = []
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
        return payload, changes, "invalid_node_count"
    nodes = payload["nodes"]
    ids = set()
    for item in nodes:
        if not isinstance(item, dict):
            return payload, changes, "invalid_node"
        nid = str(item.get("id", "")).strip()
        if not re.fullmatch(r"[A-Za-z][\w-]*", nid) or nid in ids:
            return payload, changes, "invalid_or_duplicate_node_id"
        ids.add(nid)
    rewritten = []
    for item in nodes:
        question = str(item.get("question", "")).strip()
        matches = list(_PLAN_REFERENCE.finditer(question))
        # Reject malformed, incomplete or merely similar reference text. It
        # must never become a root question by losing a syntactically bad ref.
        if (any(start.start() not in {match.start() for match in matches}
                for start in re.finditer(r"\$\{", question))
                or any(not any(match.start() <= token.start() < token.end() <= match.end()
                               for match in matches)
                       for token in _ANSWER_REFERENCE_LIKE.finditer(question))):
            return payload, changes, "invalid_dependency_placeholder"
        refs = list(dict.fromkeys(match.group(1) for match in matches))
        raw_deps = item.get("depends_on", [])
        if not isinstance(raw_deps, list) or any(not isinstance(dep, str) for dep in raw_deps):
            return payload, changes, "invalid_question_or_dependencies"
        deps = [dep.strip() for dep in raw_deps]
        if any(not re.fullmatch(r"[A-Za-z][\w-]*", dep) for dep in deps):
            return payload, changes, "invalid_dependency_id"
        nid = str(item["id"]).strip()
        if any(dep not in ids or dep == nid for dep in refs + deps):
            return payload, changes, "unknown_or_self_dependency"
        if deps and not refs:
            return payload, changes, "declared_dependency_without_placeholder"
        if raw_deps != refs:
            changes.append({"node_id": nid, "original_depends_on": list(raw_deps),
                            "canonical_depends_on": refs,
                            "added": [dep for dep in refs if dep not in deps],
                            "removed": [dep for dep in dict.fromkeys(deps) if dep not in refs]})
        rewritten.append(dict(item, depends_on=refs))
    return dict(payload, nodes=rewritten), changes, None


def relation_faithfulness_error(query: str, nodes: List[dict]) -> Optional[str]:
    """Reject explicit birth/death/end attribute swaps without inferring answers.

    This is a narrow lexical safeguard, not a claim of semantic entailment. An
    intermediate birth node is allowed if a requested death node is also present.
    """
    questions = "\n".join(node["question"] for node in nodes)
    requested_death = bool(_DEATH_ATTRIBUTE.search(query))
    requested_birth = bool(_BIRTH_ATTRIBUTE.search(query))
    planned_death = bool(_DEATH_ATTRIBUTE.search(questions))
    planned_birth = bool(_BIRTH_ATTRIBUTE.search(questions))
    if requested_death and planned_birth and not planned_death:
        return "relation_attribute_mismatch_death_to_birth"
    if requested_birth and planned_death and not planned_birth:
        return "relation_attribute_mismatch_birth_to_death"
    if (not requested_birth and _TIME_QUESTION.search(query) and _END_ATTRIBUTE.search(query)
            and planned_birth and not _END_ATTRIBUTE.search(questions)):
        return "relation_attribute_mismatch_end_to_birth"
    return None


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


def prune_derived_comparison_nodes(nodes: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Remove only terminal computations over at least two explicit answers.

    This deliberately accepts a small grammar. A conjunction, boolean type or
    multi-parent node alone never establishes that no external fact is needed.
    All predecessor evidence nodes and their identifiers remain unchanged.
    """
    by_id = {node["id"]: node for node in nodes}
    used = {dep for node in nodes for dep in node["depends_on"]}
    removed, keep = [], []
    for node in nodes:
        deps = node["depends_on"]
        reason = None
        if node["id"] not in used and len(deps) >= 2:
            expression = _PLAN_REFERENCE.sub("VALUE", node["question"])
            expression = " ".join(re.findall(r"\w+", expression))
            equality = _PURE_EQUALITY.fullmatch(expression)
            if equality:
                attribute = (equality.group("attribute") or "").casefold()
                required_types = _IDENTITY_ATTRIBUTE_TYPES.get(attribute)
                # Two person names do not already establish their countries
                # or nationalities; those still require separate lookups.
                if (required_types is None or
                        all(normalized_text(by_id[dep]["answer_type"]) in required_types for dep in deps)):
                    reason = "literal_identity_comparison_of_bound_values"
            elif (_PURE_ORDERING.fullmatch(expression)
                  and all(normalized_text(by_id[dep]["answer_type"]) in _COMPARABLE_TYPES
                          for dep in deps)):
                reason = "ordering_of_bound_numeric_or_temporal_values"
        if reason:
            removed.append(dict(node, reason=reason))
        else:
            keep.append(node)
    return keep, removed


def terminal_attribute_error(query: str, nodes: List[dict]) -> Optional[str]:
    """Guard explicit date/place swaps for the optional atomic-chain planner.

    Only the main interrogative and terminal retrieval nodes are examined. A
    temporal qualifier inside a person question must not be treated as its
    requested answer type. Unknown types remain acceptable.
    """
    used = {dep for node in nodes for dep in node["depends_on"]}
    sinks = [node for node in nodes if node["id"] not in used]
    if _REQUESTED_TIME.search(query):
        for node in sinks:
            if (_REQUESTED_PLACE.search(node["question"])
                    or normalized_text(node["answer_type"]) in {"place", "location", "city", "country", "region"}):
                return "terminal_attribute_mismatch_time_to_place"
    if _REQUESTED_PLACE.search(query):
        for node in sinks:
            if (_REQUESTED_TIME.search(node["question"])
                    or normalized_text(node["answer_type"]) in {"date", "year", "time"}):
                return "terminal_attribute_mismatch_place_to_time"
    return None


class PlannerImprovementMixin:
    """Allow parallel atomic branches without inflating the depth hint."""

    def _plan(self, query: str, hops: int):
        if "planning" not in self.improvements:
            return self._original_plan(query, hops)
        routing_mode = getattr(self.cfg, "evidence_plan_routing", "all")
        if routing_mode not in {"all", "question_structure", "dependency_depth"}:
            raise ValueError("Unknown evidence_plan_routing")
        routing = (route_dependency_depth(query, hops) if routing_mode == "dependency_depth"
                   else route_question_structure(query) if routing_mode == "question_structure" else None)
        if routing is not None and not routing["expand"]:
            nodes, reason = self._original_plan(query, hops)
            self._plan_diagnostics.setdefault(query, {})["routing"] = routing
            return nodes, reason
        node_budget = max(1, int(getattr(self.cfg, "evidence_plan_node_budget", 6)))
        depth_budget = max(1, int(getattr(self.cfg, "evidence_plan_depth_budget", 4)))
        depth_hint = max(1, int(hops))
        validation_mode = ("canonical_refs" if routing is not None
                           else getattr(self.cfg, "evidence_plan_validation", "strict"))
        if validation_mode not in {"strict", "canonical_refs"}:
            raise ValueError("Unknown evidence_plan_validation")
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
        if routing is not None:
            diagnostics["routing"] = routing
        chain_prompt = routing is not None and routing["kind"] in {"dependency_chain", "nested_bridge"}
        if chain_prompt:
            messages[0]["content"] = (
                "Plan evidence retrieval using ONLY the input question. Return ONLY JSON "
                "{nodes:[{id,question,depends_on,answer_type}]}. Each node retrieves one "
                "relation or attribute that one passage can establish. Resolve an unknown "
                "entity in the innermost description first; keep its answer as the literal "
                "${ID.answer} in the next node. List exactly those referenced IDs in "
                "depends_on. Never guess an intermediate entity or an answer. "
                "Trace each nested relation in its original direction. Do not compress "
                "'the label of the performer of a work' into one lookup: first identify "
                "the performer, then that performer's label, then the requested fact. "
                "Likewise a region containing a city is different from the city itself. "
                "Retain identifying qualifiers such as genre, nationality, role, edition, "
                "location, dates and relation direction in the node identifying that entity. "
                "Do not move a relation about an outer entity onto the inner entity. "
                "Preserve the input's final requested attribute exactly: when asks for "
                "a time/date/year, where asks for a place, who asks for an entity. "
                "Birth, death, release, founding, membership and tenure end are distinct "
                "relations; never replace one by a more familiar relation. "
                "The dependency depth hint is an upper-capacity cue, NOT a required "
                "number of nodes. Use only the relations explicitly needed by the question; "
                "do not invent extra facts to match the hint. Parallel branches keep their "
                "own literal intermediate variables. Omit a final comparison, arithmetic "
                "or aggregation of already retrieved answers. Use no hashes or cycles."
            )
            # A neutral structural syntax avoids suggesting birthplace for a
            # question whose requested final attribute is a year or death date.
            messages[1]["content"] = (
                f"Question: {query}\nExpected dependency depth hint: {depth_hint}\n"
                f"Maximum total retrieval nodes: {node_budget}\n"
                f"Maximum dependency depth: {depth_budget}\n"
                "Syntax: each node has id, question, depends_on, answer_type; root "
                "nodes have depends_on=[], and dependent questions contain every "
                "listed dependency as an existing node's literal ${ID.answer}. "
                "Choose the fewest atomic nodes that retain all necessary relations."
            )
            diagnostics["capacity_policy"] = "question_relations_with_dependency_depth_hint"
        if validation_mode == "canonical_refs":
            if chain_prompt:
                messages[0]["content"] += (
                    " Relation faithfulness: preserve the input's exact relation, "
                    "requested attribute, and identifying qualifiers. Birth, death, "
                    "founding, release and tenure ending are distinct facts. Every "
                    "unknown intermediate entity must appear as an existing node's "
                    "literal ${ID.answer}; do not leave a compound unknown entity "
                    "description unbound or substitute a guessed name."
                )
            else:
                messages[0]["content"] += (
                    " Relation faithfulness: preserve the exact requested relation and attribute. "
                    "A death date/place is not a birth date/place; a tenure ending is not a "
                    "birth event. Examples illustrate structure, never override the question. "
                    "For 'which director of Film A and Film B died earlier?', use "
                    "s1='Who directed Film A?', s2='When did ${s1.answer} die?', "
                    "s3='Who directed Film B?', s4='When did ${s3.answer} die?'. "
                    "Use birth questions only when the input asks about birth. Every unknown "
                    "intermediate entity must appear as an existing node's literal "
                    "${ID.answer}; do not replace it with an unbound definite description."
                )
            diagnostics.update(validation_mode=validation_mode, canonicalization_changes=[],
                               relation_faithfulness="explicit_birth_death_end_swap_check")
        self._plan_diagnostics[query] = diagnostics
        if "plan_prune" in self.improvements:
            diagnostics["comparison_pruning"] = []
        reason = None
        for attempt in range(2):
            payload, reason = self._infer_object(messages)
            diagnostics["attempts"] += 1
            diagnostics["outputs"].append(payload)
            if not reason:
                validated_payload = payload
                if validation_mode == "canonical_refs":
                    validated_payload, changes, reason = canonicalize_plan_dependencies(payload)
                    diagnostics["canonicalization_changes"].append(
                        {"attempt": attempt + 1, "changes": changes, "error": reason})
                if not reason and "plan_prune" in self.improvements:
                    # At most one superfluous computation may exceed the
                    # retrieval-node capacity; real retrieval nodes still must
                    # pass the original capacity/depth check after pruning.
                    structural, reason = validate_plan(validated_payload, node_budget + 1)
                    if not reason:
                        structural, removed = prune_derived_comparison_nodes(structural)
                        diagnostics["comparison_pruning"].append(
                            {"attempt": attempt + 1, "removed": removed})
                        validated_payload = dict(validated_payload, nodes=structural)
                if not reason:
                    nodes, reason = validate_retrieval_plan(validated_payload, node_budget, depth_budget)
                if not reason and validation_mode == "canonical_refs":
                    reason = relation_faithfulness_error(query, nodes)
                if not reason and chain_prompt:
                    reason = terminal_attribute_error(query, nodes)
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
                if validation_mode == "canonical_refs":
                    messages[-1]["content"] += (
                        " Keep the input question's requested relation and attribute exactly: "
                        "death, birth and tenure end are different facts. If a node depends "
                        "on an earlier answer, write its existing ${ID.answer} explicitly "
                        "in that node's question. Do not remove a needed dependency or "
                        "guess the intermediate entity to bypass validation."
                    )
        return [], reason

    def _original_plan(self, query: str, hops: int):
        nodes, reason = super()._plan(query, hops)
        if not reason and "plan_prune" in self.improvements:
            nodes, removed = prune_derived_comparison_nodes(nodes)
            self._plan_diagnostics.setdefault(query, {})["comparison_pruning"] = [
                {"attempt": 1, "removed": removed}]
        return nodes, reason


class AdaptiveSearchMixin:
    """Spend extra verification only on missing evidence or close alternatives.

    Place this mixin before a binding verifier in the method resolution order,
    so both the first and fallback batches use the same binding rules.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_mode = getattr(self.cfg, "evidence_adaptive_mode", "both")
        if self.adaptive_mode not in {"both", "verify_only", "beam_only"}:
            raise ValueError("Unknown evidence_adaptive_mode")
        self.expand_verification = "adaptive" in self.improvements and self.adaptive_mode in {"both", "verify_only"}
        self.retain_alternatives = "adaptive" in self.improvements and self.adaptive_mode in {"both", "beam_only"}
        if self.retain_alternatives:
            self.beam_width = 2

    def _verification_documents(self, ids: List[int]) -> Dict[int, str]:
        count = 6 if self.expand_verification else 3
        return {int(doc_id): self._document(int(doc_id))[:4000] for doc_id in ids[:count]}

    def _verify(self, question: str, answer_type: str, docs: Dict[int, str]):
        if not self.expand_verification or len(docs) <= 3:
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
        if not self.retain_alternatives or len(ordered) < 2:
            return ordered
        margin = max(0.0, float(getattr(self.cfg, "evidence_ambiguity_margin", 0.06)))
        gap = self._beam_score(ordered[0], total) - self._beam_score(ordered[1], total)
        # Every retained alternative is already supported by an accepted quote.
        # A clear winner keeps the single-path behavior and cost of exp4.
        return ordered[:2] if gap <= margin else ordered[:1]
