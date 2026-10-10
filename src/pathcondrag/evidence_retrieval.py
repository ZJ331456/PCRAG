"""Evidence-preserving retrieval and bounded dependency search.

Only LLM planning/answer verification runs in worker threads. Embedding and graph
operations stay on the caller thread. This module never reads benchmark answers,
support labels, or annotated question decompositions.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_REFERENCE = re.compile(r"\$\{([A-Za-z][\w-]*)\.answer\}")
_HASH = re.compile(r"\b(?:entity-)?[0-9a-f]{32}\b", re.I)


def normalized_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", str(value).casefold()))


def parse_object(raw: str) -> Tuple[Optional[dict], Optional[str]]:
    raw = str(raw or "").strip()
    variants = [raw]
    left, right = raw.find("{"), raw.rfind("}")
    if 0 <= left < right:
        variants.append(raw[left:right + 1])
    for variant in variants:
        try:
            obj = json.loads(variant)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj, None
    return None, "invalid_json_object"


def validate_plan(payload: dict, max_nodes: int) -> Tuple[List[dict], Optional[str]]:
    """Validate references and return a stable topological ordering."""
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes or len(nodes) > max_nodes:
        return [], "invalid_node_count"
    clean: List[dict] = []
    seen = set()
    for item in nodes:
        if not isinstance(item, dict):
            return [], "invalid_node"
        nid, question = str(item.get("id", "")).strip(), str(item.get("question", "")).strip()
        if not re.fullmatch(r"[A-Za-z][\w-]*", nid) or nid in seen:
            return [], "invalid_or_duplicate_node_id"
        deps = item.get("depends_on", [])
        if not question or _HASH.search(question) or not isinstance(deps, list):
            return [], "invalid_question_or_dependencies"
        deps = list(dict.fromkeys(str(x) for x in deps))
        refs = set(_REFERENCE.findall(question))
        if refs != set(deps):
            return [], "dependency_reference_mismatch"
        clean.append({"id": nid, "question": question, "depends_on": deps,
                      "answer_type": str(item.get("answer_type", "unknown"))[:80]})
        seen.add(nid)
    if any(dep not in seen or dep == n["id"] for n in clean for dep in n["depends_on"]):
        return [], "unknown_or_self_dependency"
    ordered, done = [], set()
    while len(ordered) < len(clean):
        ready = [n for n in clean if n["id"] not in done and set(n["depends_on"]) <= done]
        if not ready:
            return [], "cyclic_dependencies"
        for node in ready:
            ordered.append(node)
            done.add(node["id"])
    return ordered, None


def plan_layers(nodes: List[dict]) -> List[List[dict]]:
    layers, finished = [], set()
    while len(finished) < len(nodes):
        ready = [n for n in nodes if n["id"] not in finished and set(n["depends_on"]) <= finished]
        if not ready:
            raise ValueError("Plan is cyclic or has missing dependencies")
        layers.append(ready)
        finished.update(n["id"] for n in ready)
    return layers


def bind_question(question: str, bindings: dict) -> Optional[str]:
    refs = _REFERENCE.findall(question)
    if any(ref not in bindings for ref in refs):
        return None
    if any(_HASH.search(str(bindings[ref])) for ref in refs):
        return None
    return _REFERENCE.sub(lambda m: str(bindings[m.group(1)]), question)


def verify_hypotheses(payload: dict, documents: Dict[int, str]) -> Tuple[List[dict], List[dict]]:
    """Require an exact quoted span and an answer supported inside that span."""
    accepted, rejected, seen = [], [], set()
    items = payload.get("hypotheses", [])
    if not isinstance(items, list):
        return [], [{"reason": "invalid_hypotheses_list"}]
    for item in items:
        if not isinstance(item, dict):
            rejected.append({"reason": "invalid_hypothesis"})
            continue
        answer, span = str(item.get("answer", "")).strip(), str(item.get("evidence", "")).strip()
        raw_id = str(item.get("doc_id", ""))
        match = re.fullmatch(r"D?(\d+)", raw_id)
        doc_id = int(match.group(1)) if match else -1
        reason = None
        if doc_id not in documents:
            reason = "unknown_document_reference"
        elif not answer or len(answer) > 200 or _HASH.search(answer) or _REFERENCE.search(answer):
            reason = "invalid_answer_or_hash"
        elif len(span) < 8 or len(span) > 1500 or span not in documents[doc_id]:
            reason = "evidence_not_exact_substring"
        elif normalized_text(answer) not in normalized_text(span):
            reason = "answer_not_supported_in_quote"
        elif normalized_text(answer) in {"unknown", "none", "not known", "not stated", "no answer"}:
            reason = "missing_answer"
        try:
            confidence = float(item.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.0
        if not np.isfinite(confidence):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        if confidence < 0.25 and reason is None:
            reason = "low_claimed_confidence"
        if reason:
            rejected.append({"answer": answer[:200], "doc_id": doc_id, "reason": reason})
            continue
        unique = (normalized_text(answer), doc_id, span)
        if unique in seen:
            continue
        seen.add(unique)
        accepted.append({"answer": answer, "doc_id": doc_id, "evidence": span,
                         "confidence": confidence})
    accepted.sort(key=lambda x: (-x["confidence"], x["doc_id"], normalized_text(x["answer"])))
    return accepted, rejected


def branch_matches(requirements: dict, bindings: dict) -> bool:
    return all(k in bindings and normalized_text(v) == normalized_text(bindings[k])
               for k, v in requirements.items())


def local_rank_scores(scores: np.ndarray, constant: float) -> List[float]:
    """Local ranks always restart at one; later subquestions get equal priors."""
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0:
        return []
    finite = np.where(np.isfinite(values), values, 0.0)
    low, high = float(finite.min()), float(finite.max())
    semantic = (finite - low) / (high - low) if high > low else np.ones(len(finite))
    constant = max(1.0, float(constant))
    return [float(0.75 * constant / (constant + rank) + 0.25 * semantic[rank - 1])
            for rank in range(1, len(finite) + 1)]


class EvidenceRetrieval:
    """Cumulative stages: candidate fidelity, coverage, dependency binding, beam."""

    def __init__(self, rag):
        self.rag, self.cfg = rag, rag.pcrag_config
        self.stage = int(getattr(self.cfg, "improvement_stage", 0))
        self.top_k = int(getattr(self.cfg, "evidence_candidate_top_k", 20))
        self.pool_size = int(getattr(self.cfg, "evidence_pool_size", 80))
        self.budget = int(getattr(self.cfg, "evidence_budget", 5))
        self.beam_width = int(getattr(self.cfg, "evidence_beam_width", 3)) if self.stage >= 5 else 1
        self.max_searches = int(getattr(self.cfg, "evidence_max_searches", 20))
        self.rank_constant = float(getattr(self.cfg, "evidence_local_rank_constant", 10.0))
        self._documents: Dict[int, str] = {}
        self._plan_diagnostics: Dict[str, dict] = {}
        self._verification_diagnostics: Dict[str, dict] = {}

    def _document(self, doc_id: int) -> str:
        if doc_id not in self._documents:
            key = self.rag.passage_node_keys[doc_id]
            row = self.rag.chunk_embedding_store.get_row(key)
            self._documents[doc_id] = str(row.get("content", "")) if isinstance(row, dict) else ""
        return self._documents[doc_id]

    @staticmethod
    def _trace(stage: int) -> dict:
        return {"stage": stage, "candidate_count": 0, "search_count": 0,
                "llm_plan_calls": 0, "llm_verification_calls": 0,
                "semantic_failures": [], "rejected_hypotheses": [],
                "plan": [], "bindings": {}, "branch_scores": [],
                "routes": [], "selected_prefix": []}

    def _infer_object(self, messages: List[dict]) -> Tuple[Optional[dict], Optional[str]]:
        # HTTP failures deliberately propagate through future.result().
        ret = self.rag.llm_model.infer(messages=messages, temperature=0.0)
        response = ret[0] if isinstance(ret, tuple) else ret
        metadata = ret[1] if isinstance(ret, tuple) and len(ret) > 1 else {}
        if isinstance(metadata, dict) and metadata.get("finish_reason") == "length":
            return None, "truncated_response"
        text = self.rag._extract_llm_text(response)
        return parse_object(text)

    @staticmethod
    def _collect_jobs(executor, jobs: List[Tuple[Any, tuple]]) -> list:
        if executor is None:
            return [func(*args) for func, args in jobs]
        futures = [executor.submit(func, *args) for func, args in jobs]
        return [future.result() for future in futures]

    def _plan(self, query: str, hops: int) -> Tuple[List[dict], Optional[str]]:
        max_nodes = max(1, min(4, int(hops)))
        messages = [
            {"role": "system", "content": (
                "Plan evidence retrieval for a question. Output ONLY JSON with nodes. "
                "Each node is {id, question, depends_on, answer_type}. "
                "Each question must be answerable from one passage. Unknown answers must "
                "stay as placeholders ${s1.answer}, never guess an intermediate person, "
                "place or answer. List every referenced node in depends_on. Use no other "
                "references. Independent comparison branches should be independent nodes. "
                "Preserve every nested location/relation qualifier in the question. "
                "Resolve the innermost unknown entity before asking the final relationship. "
                "For example, 'When was the person that A was compared with hired by B?' "
                "starts with 'Who was A compared with?', followed by 'When was ${s1.answer} hired by B?'. "
                "Do not carry 'hired by B' into the first subquestion. "
                "For a region north of the region containing X, first locate X, then ask "
                "for the region north of ${s1.answer}; do not replace this with north of X. "
                "Every depends_on ID must appear literally as ${ID.answer} in that node's question. "
                "No cycles. A node's answer_type describes what its passage must establish." )},
            {"role": "user", "content": (
                f"Question: {query}\nMaximum nodes: {max_nodes}\n"
                'Example syntax: {"nodes":[{"id":"s1","question":"Who wrote Work X?",'
                '"depends_on":[],"answer_type":"person"},{"id":"s2",'
                '"question":"Where was ${s1.answer} born?","depends_on":["s1"],'
                '"answer_type":"place"}]}\nUse only the input question to construct the plan.' )},
        ]
        diagnostics = {"attempts": 0, "outputs": [], "validation_errors": []}
        self._plan_diagnostics[query] = diagnostics
        for attempt in range(2):
            payload, reason = self._infer_object(messages)
            diagnostics["attempts"] += 1
            diagnostics["outputs"].append(payload)
            nodes, reason = ([], reason) if reason else validate_plan(payload, max_nodes)
            if not reason:
                return nodes, None
            diagnostics["validation_errors"].append(reason)
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": json.dumps(payload or {}, ensure_ascii=False)},
                    {"role": "user", "content": (
                        f"The plan failed validation: {reason}. Correct the JSON plan, using at most {max_nodes} nodes. "
                        "Every declared dependency must occur as a literal ${ID.answer} placeholder. "
                        "Keep the original question's nested qualifiers, keep unknown entities unbound, "
                        "and return only the corrected JSON. Do not add guessed answers.")},
                ]
        return [], reason

    def _verify(self, question: str, answer_type: str, docs: Dict[int, str]):
        messages = [
            {"role": "system", "content": (
                "Verify possible answers to one retrieval subquestion using ONLY the "
                "provided passages. Output ONLY JSON {hypotheses:[{answer,evidence,"
                "doc_id,confidence}]}. evidence must be copied exactly from that passage "
                "and contain the literal answer. doc_id must match its D-number. "
                "Include the sentence that directly answers the subquestion. "
                "If a full name appears before the relevant sentence, include that preceding "
                "sentence in the same contiguous quote, or use the answer's literal surface in the quote. "
                "Check that the quoted sentence establishes the requested relationship; "
                "a mentioned name alone is insufficient. Return an empty hypotheses list "
                "when unsupported. Retain up to 3 genuinely different supported answers. "
                "Do not resolve unknown placeholders, invent entities, or output hashes." )},
            {"role": "user", "content": (
                f"Subquestion: {question}\nAnswer type: {answer_type}\n\n" +
                "\n\n".join(f"[D{k}]\n{text}" for k, text in docs.items()))},
        ]
        key = question + "::" + ",".join(map(str, docs))
        diagnostics = {"attempts": 0, "outputs": []}
        self._verification_diagnostics[key] = diagnostics
        all_rejected = []
        for attempt in range(2):
            payload, reason = self._infer_object(messages)
            diagnostics["attempts"] += 1
            diagnostics["outputs"].append(payload)
            if reason:
                accepted, rejected = [], [{"reason": reason}]
            else:
                accepted, rejected = verify_hypotheses(payload, docs)
            all_rejected.extend(rejected)
            if accepted or (not rejected and not reason):
                return accepted, all_rejected, reason
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": json.dumps(payload or {}, ensure_ascii=False)},
                    {"role": "user", "content": (
                        "The proposed citations failed validation: " + json.dumps(rejected, ensure_ascii=False) +
                        ". Recheck the original subquestion and the provided passages. "
                        "Return a corrected hypothesis only if the exact original quote contains the literal answer "
                        "and establishes the requested relationship. Otherwise return {\"hypotheses\":[]}. "
                        "Do not change passage text or infer unsupported relationships.")},
                ]
        return [], all_rejected, reason

    def _candidate(self, state: dict, doc_id: int) -> dict:
        candidates = state["evidence_candidates"]
        if doc_id not in candidates:
            candidates[doc_id] = {"doc_id": doc_id, "base_score": 0.0,
                                  "sources": [], "verified": []}
        return candidates[doc_id]

    def _add_route(self, state: dict, source: str, goal: str, query: str,
                   requirements: Optional[dict] = None) -> Optional[List[int]]:
        trace = state["evidence_trace"]
        requirements = dict(requirements or {})
        cache = state["_evidence_search_cache"]
        if query in cache:
            ids, raw = cache[query]
        else:
            if trace["search_count"] >= self.max_searches:
                trace["semantic_failures"].append({"node": goal, "reason": "search_budget_exhausted"})
                return None
            ids, raw = self.rag.dense_passage_retrieval(query)
            ids = [int(i) for i in np.asarray(ids)[:self.top_k]]
            raw = np.asarray(raw, dtype=float)[:len(ids)]
            cache[query] = (ids, raw)
            trace["search_count"] += 1
        local_scores = local_rank_scores(raw, self.rank_constant)
        for rank, (doc_id, score, original) in enumerate(zip(ids, local_scores, raw), 1):
            self._candidate(state, doc_id)["sources"].append({
                "source": source, "goal": goal, "rank": rank, "raw_score": float(original),
                "score": score, "question": query, "requirements": requirements})
        trace["routes"].append({"source": source, "goal": goal, "query": query,
                                "doc_ids": list(ids), "requirements": requirements})
        return ids

    @staticmethod
    def _goal_for_question(question: str, static_questions: List[str], fallback: str) -> str:
        tokens = set(normalized_text(question).split())
        best = (0.0, fallback)
        for index, static in enumerate(static_questions):
            other = set(normalized_text(static).split())
            sim = len(tokens & other) / max(1, len(tokens | other))
            if sim > best[0]:
                best = (sim, f"static:s{index + 1}")
        return best[1] if best[0] >= 0.25 else fallback

    def _prepare_state(self, state: dict):
        state["evidence_candidates"] = {}
        state["evidence_trace"] = self._trace(self.stage)
        state["_evidence_search_cache"] = {}
        if "base" in state:
            ids, scores, ctx = state["base"]
            top_ids = np.asarray(ids)[:self.pool_size]
            top_scores = np.asarray(scores, dtype=float)[:len(top_ids)]
            values = local_rank_scores(top_scores, self.rank_constant)
            for doc_id, score in zip(top_ids, values):
                self._candidate(state, int(doc_id))["base_score"] = score
        static = [q for q in state.get("static_sub_questions", []) if isinstance(q, str) and q.strip()]
        for index, question in enumerate(static[:4]):
            self._add_route(state, f"static:{index + 1}", f"static:s{index + 1}", question)
        pcqd = state.get("pcqd_sub_questions") or []
        for index, item in enumerate(pcqd[:4]):
            question = item.get("question", "") if isinstance(item, dict) else str(item)
            if not question.strip():
                continue
            goal = self._goal_for_question(question, static, f"pcqd:p{index + 1}")
            self._add_route(state, f"pcqd:{index + 1}", goal, question)

    @staticmethod
    def _beam_score(beam: dict, total: int) -> float:
        qualities = beam["qualities"]
        if not qualities:
            return 0.0
        coverage = len(beam["bindings"]) / max(1, total)
        return float(0.45 * min(qualities) + 0.35 * np.mean(qualities) + 0.20 * coverage)

    def _prune_beams(self, beams: List[dict], total: int) -> List[dict]:
        unique = {}
        for beam in beams:
            key = tuple(sorted((k, normalized_text(v)) for k, v in beam["bindings"].items()))
            if key not in unique or self._beam_score(beam, total) > self._beam_score(unique[key], total):
                unique[key] = beam
        ordered = sorted(unique.values(), key=lambda b: (
            -self._beam_score(b, total), tuple(sorted(b["bindings"].items()))))
        return ordered[:self.beam_width]

    def _verification_documents(self, ids: List[int]) -> Dict[int, str]:
        """The original verifier receives the first three bounded passages."""
        return {doc_id: self._document(doc_id)[:4000] for doc_id in ids[:3]}

    def _dependency_search(self, states: List[dict], executor):
        active = [s for s in states if s.get("_evidence_plan")]
        for state in active:
            state["_evidence_layers"] = plan_layers(state["_evidence_plan"])
            state["_evidence_beams"] = [{"bindings": {}, "proofs": {}, "qualities": []}]
        depth = max((len(s["_evidence_layers"]) for s in active), default=0)
        for level in range(depth):
            tasks, task_lookup = [], {}
            # All embedding calls below execute on this caller thread.
            for state in active:
                if level >= len(state["_evidence_layers"]):
                    continue
                for node in state["_evidence_layers"][level]:
                    for beam_index, beam in enumerate(state["_evidence_beams"]):
                        question = bind_question(node["question"], beam["bindings"])
                        if question is None:
                            state["evidence_trace"]["semantic_failures"].append({
                                "node": node["id"], "reason": "unbound_dependency",
                                "requirements": list(node["depends_on"])})
                            continue
                        requirements = dict(beam["bindings"])
                        ids = self._add_route(state, f"dag:{node['id']}", f"dag:{node['id']}", question, requirements)
                        if not ids:
                            continue
                        # The verifier sees exactly these spans, not unbounded full documents.
                        docs = self._verification_documents(ids)
                        task = (state, node, beam_index, list(docs), question, docs)
                        task_lookup[(id(state), node["id"], beam_index)] = len(tasks)
                        tasks.append(task)
            results = self._collect_jobs(executor, [
                (self._verify, (question, node["answer_type"], docs))
                for state, node, beam_index, ids, question, docs in tasks])
            for task, (hypotheses, rejected, reason) in zip(tasks, results):
                state, node, beam_index, ids, question, docs = task
                trace = state["evidence_trace"]
                diagnostic_key = question + "::" + ",".join(map(str, docs))
                verification_diagnostic = self._verification_diagnostics.get(diagnostic_key, {"attempts": 1})
                trace["llm_verification_calls"] += verification_diagnostic["attempts"]
                trace.setdefault("verification_outputs", []).append(dict(verification_diagnostic, node=node["id"]))
                trace["rejected_hypotheses"].extend(dict(item, node=node["id"]) for item in rejected)
                if reason:
                    trace["semantic_failures"].append({"node": node["id"], "reason": reason})
                for hyp in hypotheses:
                    local_rank = ids.index(hyp["doc_id"]) + 1
                    hyp["quality"] = float(0.6 * self.rank_constant / (self.rank_constant + local_rank)
                                           + 0.4 * hyp["confidence"])
                hypotheses.sort(key=lambda h: (-h["quality"], h["doc_id"], normalized_text(h["answer"])))
                # Keep at most one proof per answer hypothesis to avoid a fake beam of duplicates.
                unique, seen = [], set()
                for hyp in hypotheses:
                    answer_key = normalized_text(hyp["answer"])
                    if answer_key not in seen:
                        unique.append(hyp)
                        seen.add(answer_key)
                results[task_lookup[(id(state), node["id"], beam_index)]] = unique[:self.beam_width]
                if not unique:
                    trace["semantic_failures"].append({"node": node["id"], "reason": "no_verified_answer"})
            # Cartesian expansion for independent nodes, preserving every ancestor binding.
            for state in active:
                if level >= len(state["_evidence_layers"]):
                    continue
                expanded = []
                for beam_index, ancestor in enumerate(state["_evidence_beams"]):
                    descendants = [ancestor]
                    for node in state["_evidence_layers"][level]:
                        index = task_lookup.get((id(state), node["id"], beam_index))
                        choices = results[index] if index is not None else []
                        if not choices:
                            continue
                        new = []
                        for descendant in descendants:
                            for hyp in choices:
                                child = {"bindings": dict(descendant["bindings"]),
                                         "proofs": dict(descendant["proofs"]),
                                         "qualities": list(descendant["qualities"])}
                                child["bindings"][node["id"]] = hyp["answer"]
                                child["proofs"][node["id"]] = dict(hyp)
                                child["qualities"].append(hyp["quality"])
                                new.append(child)
                        descendants = self._prune_beams(new, len(state["_evidence_plan"]))
                    expanded.extend(descendants)
                state["_evidence_beams"] = self._prune_beams(expanded, len(state["_evidence_plan"]))
        for state in active:
            beams = state["_evidence_beams"]
            total = len(state["_evidence_plan"])
            best = beams[0] if beams else {"bindings": {}, "proofs": {}, "qualities": []}
            trace = state["evidence_trace"]
            trace["bindings"] = dict(best["bindings"])
            trace["branch_scores"] = [
                {"bindings": dict(b["bindings"]), "score": self._beam_score(b, total),
                 "proofs": dict(b["proofs"])} for b in beams]
            state["_evidence_winning_bindings"] = best["bindings"]
            for node_id, hyp in best["proofs"].items():
                candidate = self._candidate(state, hyp["doc_id"])
                candidate["verified"].append(dict(hyp, goal=f"dag:{node_id}"))

    @staticmethod
    def _goal_scores(candidate: dict, bindings: dict) -> Dict[str, float]:
        goals: Dict[str, float] = {}
        for source in candidate["sources"]:
            if not branch_matches(source["requirements"], bindings):
                continue
            goal = source["goal"]
            goals[goal] = max(goals.get(goal, 0.0), source["score"])
        for proof in candidate["verified"]:
            goals[proof["goal"]] = max(goals.get(proof["goal"], 0.0), proof["quality"])
        return goals

    def _trim_pool(self, state: dict):
        candidates = state["evidence_candidates"]
        bindings = state.get("_evidence_winning_bindings", {})
        ranked = sorted(candidates, key=lambda d: (
            -bool(candidates[d]["verified"]),
            -(0.4 * candidates[d]["base_score"] +
              0.6 * max(self._goal_scores(candidates[d], bindings).values(), default=0.0)), d))
        # Reserve one good candidate per subgoal before filling the shared pool.
        goals = defaultdict(list)
        for doc_id in ranked:
            for goal, score in self._goal_scores(candidates[doc_id], bindings).items():
                goals[goal].append((score, doc_id))
        reserved = [d for d in ranked if candidates[d]["verified"]]
        reserved.extend(min(values, key=lambda x: (-x[0], x[1]))[1] for values in goals.values())
        keep = list(dict.fromkeys(reserved + ranked))[:self.pool_size]
        state["evidence_candidates"] = {d: candidates[d] for d in keep}
        state["evidence_trace"]["candidate_count"] = len(keep)

    def process_window(self, states: List[dict], executor=None):
        if self.stage < 2:
            return
        for state in states:
            self._prepare_state(state)
        if self.stage >= 4:
            requests = []
            for state in states:
                ctx = state.get("base", (None, None, {}))[2]
                requests.append((self._plan, (state["query"], int(ctx.get("hops", 1)))))
            for state, (plan, reason) in zip(states, self._collect_jobs(executor, requests)):
                trace = state["evidence_trace"]
                planning_diagnostic = self._plan_diagnostics.get(state["query"], {"attempts": 1})
                trace["llm_plan_calls"] += planning_diagnostic["attempts"]
                trace["planning_outputs"] = planning_diagnostic
                trace["plan"] = plan
                state["_evidence_plan"] = plan
                if reason:
                    trace["semantic_failures"].append({"node": "plan", "reason": reason})
            self._dependency_search(states, executor)
        for state in states:
            self._trim_pool(state)
            state.pop("_evidence_search_cache", None)

    def finalize(self, query: str, ids: np.ndarray, scores: np.ndarray,
                 ctx: dict, state: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        mode = getattr(self.cfg, "evidence_scoring_mode", "legacy")
        # Default configurations execute the original scoring implementation.
        # Explicit experiment modes additionally fingerprint their shared inputs.
        if not hasattr(self.cfg, "evidence_scoring_mode"):
            return self._finalize_legacy(query, ids, scores, ctx, state)
        from .evidence_dependency_scoring import dependency_scored_prefix, scoring_input_sha256
        input_hash = scoring_input_sha256(query, ids, scores, state)
        legacy_ids, legacy_scores, trace = self._finalize_legacy(query, ids, scores, ctx, state)
        trace["evidence_scoring_input_sha256"] = input_hash
        trace["finalizer_input_hash"] = input_hash
        trace["evidence_scoring_mode"] = mode
        if mode != "dependency" or self.stage < 3 or not state.get("evidence_candidates"):
            return legacy_ids, legacy_scores, trace
        output_ids, output_scores, details, diagnostic = dependency_scored_prefix(
            query, legacy_ids, legacy_scores, state, self.cfg, self._document,
            signal_ids=ids, signal_scores=scores)
        diagnostic["input_sha256"] = input_hash
        diagnostic["legacy_top10"] = np.asarray(legacy_ids)[:10].tolist()
        diagnostic["scored_top10"] = np.asarray(output_ids)[:10].tolist()
        trace["dependency_scoring"] = diagnostic
        if details:
            trace["selected_prefix"] = details
            trace["covered_goals"] = {goal: 1.0 for goal in diagnostic.get("covered_goals", [])}
        return output_ids, output_scores, trace

    def _finalize_legacy(self, query: str, ids: np.ndarray, scores: np.ndarray,
                         ctx: dict, state: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        trace = state.get("evidence_trace", self._trace(self.stage))
        candidates = state.get("evidence_candidates", {})
        if self.stage < 2 or not candidates:
            return ids, scores, trace
        ids, scores = np.asarray(ids, dtype=int), np.asarray(scores, dtype=float)
        finite = np.where(np.isfinite(scores), scores, 0.0)
        span = float(finite.max() - finite.min()) if len(finite) else 0.0
        norm = (finite - finite.min()) / span if span > 0 else np.ones(len(finite))
        base_by_id = dict(zip(ids.tolist(), norm.tolist()))
        base_weight = float(getattr(self.cfg, "evidence_base_weight", 0.4))
        merged = {doc_id: base_weight * value for doc_id, value in base_by_id.items()}
        bindings = state.get("_evidence_winning_bindings", {})
        goal_by_id = {}
        for doc_id, candidate in candidates.items():
            goals = self._goal_scores(candidate, bindings)
            goal_by_id[doc_id] = goals
            relation = max(goals.values(), default=0.0)
            merged[doc_id] = base_weight * base_by_id.get(doc_id, candidate["base_score"]) + (1.0 - base_weight) * relation
        ordered = sorted(merged, key=lambda d: (-merged[d], d))
        if self.stage < 3:
            trace["selected_prefix"] = [{"doc_id": d, "utility": merged[d],
                                          "goals": goal_by_id.get(d, {})} for d in ordered[:self.budget]]
            return np.asarray(ordered), np.asarray([merged[d] for d in ordered]), trace

        covered, selected, selected_tokens, details = {}, [], [], []
        remaining = set(candidates)
        goal_ids = set(goal for goals in goal_by_id.values() for goal in goals)
        goal_total = max(1, len(goal_ids))
        coverage_weight = float(getattr(self.cfg, "evidence_coverage_weight", 0.45))
        relation_weight = float(getattr(self.cfg, "evidence_relation_weight", 0.25))
        redundancy_weight = float(getattr(self.cfg, "evidence_redundancy_weight", 0.15))
        document_tokens = {d: set(normalized_text(self._document(d)).split()) for d in remaining}
        for position in range(min(self.budget, len(remaining))):
            ranked = []
            for doc_id in remaining:
                goals = goal_by_id[doc_id]
                gain = sum(max(0.0, score - covered.get(goal, 0.0)) for goal, score in goals.items()) / goal_total
                tokens = document_tokens[doc_id]
                redundancy = max((len(tokens & previous) / max(1, len(tokens | previous))
                                  for previous in selected_tokens), default=0.0)
                relation = max(goals.values(), default=0.0)
                utility = (base_weight * base_by_id.get(doc_id, candidates[doc_id]["base_score"])
                           + relation_weight * relation + coverage_weight * gain
                           - redundancy_weight * redundancy)
                ranked.append((utility, merged[doc_id], -doc_id, gain, redundancy, doc_id))
            utility, _score, _tie, gain, redundancy, chosen = max(ranked)
            selected.append(chosen)
            selected_tokens.append(document_tokens[chosen])
            remaining.remove(chosen)
            for goal, score in goal_by_id[chosen].items():
                covered[goal] = max(covered.get(goal, 0.0), score)
            details.append({"doc_id": chosen, "utility": float(utility),
                            "coverage_gain": float(gain), "redundancy": float(redundancy),
                            "goals": goal_by_id[chosen], "verified": candidates[chosen]["verified"]})
        # Keep greedy marginal-utility order as the actual retrieval prefix.
        # Tail scores encode the tail's existing order below the selected prefix.
        tail = [d for d in ordered if d not in set(selected)]
        final_ids = selected + tail
        top_score = max(merged.values(), default=1.0) + 1.0
        prefix_scores = [top_score + (len(selected) - position) / max(1, len(selected))
                         for position in range(len(selected))]
        final_scores = prefix_scores + [merged[d] for d in tail]
        trace["selected_prefix"] = details
        trace["covered_goals"] = covered
        return np.asarray(final_ids), np.asarray(final_scores, dtype=float), trace
