"""Controlled experiments for the stage-four evidence retrieval mechanism.

Only question text, frozen candidate hashes/scores and a reference call count are
accepted as experimental inputs. Benchmark answers and decomposition text never
enter this runtime. Embedding remains on the caller thread; worker jobs carry an
explicit state through thread-local storage to keep per-question budgets isolated.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, List

import numpy as np

from .evidence_retrieval import (
    EvidenceRetrieval, bind_question, branch_matches, local_rank_scores, normalized_text, parse_object, plan_layers,
)
from .prompts.linking import get_query_instruction


class EvidenceAblationRetrieval(EvidenceRetrieval):
    MODES = {"normal", "budget_dag", "budget_qd", "budget_iterative", "fixed_pool",
             "validation", "selection"}

    def __init__(self, rag):
        super().__init__(rag)
        self.mode = str(getattr(self.cfg, "evidence_ablation_mode", "normal"))
        self.binding_mode = str(getattr(self.cfg, "evidence_binding_mode", "literal"))
        self.selection_mode = str(getattr(self.cfg, "evidence_selection_mode", "coverage"))
        if self.mode not in self.MODES:
            raise ValueError(f"Unknown evidence ablation mode: {self.mode}")
        if self.binding_mode not in {"string", "literal", "relation"}:
            raise ValueError(f"Unknown evidence binding mode: {self.binding_mode}")
        if self.selection_mode not in {"coverage", "ancestor", "joint"}:
            raise ValueError(f"Unknown evidence selection mode: {self.selection_mode}")
        self._local = threading.local()
        self._single_flight_registry_lock = threading.Lock()
        self._single_flight_prompt_locks = {}
        self.query_inputs: Dict[str, dict] = {}
        self._frozen_pools: Dict[str, tuple] = {}
        # BaseRAG constructs passage_node_keys when retrieval is prepared, after
        # the evidence runtime itself is instantiated.
        self._chunk_to_id = None
        if self.mode == "selection":
            # All selection controls share the exact same width-three search.
            # Joint selection may choose a different generated binding branch.
            self.beam_width = max(1, int(getattr(self.cfg, "evidence_beam_width", 3)))
        input_file = str(getattr(self.cfg, "evidence_ablation_inputs_file", "") or "")
        if input_file:
            self.set_query_inputs(json.loads(Path(input_file).read_text(encoding="utf-8")))
        elif self.mode.startswith("budget_") or self.mode == "fixed_pool":
            raise ValueError(f"{self.mode} requires evidence_ablation_inputs_file")

    def set_query_inputs(self, payload):
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Ablation inputs must have schema_version=1")
        if set(payload) - {"schema_version", "records"}:
            raise ValueError("Unexpected ablation input fields")
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("Ablation records must be a list")
        inputs = {}
        for record in records:
            allowed = {"question", "sample_id", "query_index", "pool", "evidence_call_budget"}
            if not isinstance(record, dict) or set(record) - allowed:
                raise ValueError("Ablation records must contain only question/candidates/call counts")
            question = record.get("question")
            budget = record.get("evidence_call_budget")
            if not isinstance(question, str) or not question.strip() or question in inputs:
                raise ValueError("Missing or duplicate question in ablation inputs")
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
                raise ValueError("Reference evidence call budget must be a nonnegative integer")
            pool = record.get("pool")
            if not isinstance(pool, list):
                raise ValueError("Frozen pool must be a list")
            hashes = []
            for item in pool:
                if not isinstance(item, dict) or set(item) != {"doc_hash", "score"}:
                    raise ValueError("Pool entries must contain only doc_hash and score")
                digest = item["doc_hash"]
                if not isinstance(digest, str) or not digest.startswith("chunk-") or len(digest) != 38:
                    raise ValueError("Invalid frozen chunk hash")
                try:
                    int(digest[6:], 16)
                    score = float(item["score"])
                except (TypeError, ValueError):
                    raise ValueError("Invalid frozen hash or score") from None
                if not np.isfinite(score):
                    raise ValueError("Frozen scores must be finite")
                hashes.append(digest)
            if len(hashes) != len(set(hashes)):
                raise ValueError("Frozen pool contains duplicate documents")
            if self.mode == "fixed_pool" and len(pool) != 200:
                raise ValueError("Fixed candidate experiments require exactly 200 unique documents")
            inputs[question] = record
        self.query_inputs = inputs
        self._frozen_pools.clear()
        if self.mode.startswith("budget_"):
            # All budget controls use the same input-file maximum. A productive
            # flat round may return four queries; the independent search cap must
            # not prevent later allowed LLM rounds from executing their searches.
            self.max_searches = max(self.max_searches, 8 + 4 * max(
                (record["evidence_call_budget"] for record in inputs.values()), default=0))

    def _record(self, state):
        question = state["query"]
        if question not in self.query_inputs:
            raise ValueError(f"No ablation input for question: {question[:160]}")
        return self.query_inputs[question]

    def _frozen_pool(self, state):
        question = state["query"]
        if question not in self._frozen_pools:
            if self._chunk_to_id is None:
                self._chunk_to_id = {str(key): i for i, key in enumerate(self.rag.passage_node_keys)}
            pool = self._record(state)["pool"]
            missing = [x["doc_hash"] for x in pool if x["doc_hash"] not in self._chunk_to_id]
            if missing:
                raise ValueError(f"Frozen pool contains chunks absent from the shared index: {missing[:3]}")
            ids = np.asarray([self._chunk_to_id[x["doc_hash"]] for x in pool], dtype=int)
            scores = np.asarray([float(x["score"]) for x in pool], dtype=float)
            # Check content as well as the index key, so a mismatched index cannot
            # quietly turn an experimental pool into a different document set.
            for doc_id, item in zip(ids, pool):
                actual = "chunk-" + hashlib.md5(self._document(int(doc_id)).encode("utf-8")).hexdigest()
                if actual != item["doc_hash"]:
                    raise ValueError("Frozen candidate hash does not match indexed document content")
            self._frozen_pools[question] = (ids, scores)
        return self._frozen_pools[question]

    def _infer_object(self, messages):
        state = getattr(self._local, "state", None)
        if state is None:
            raise RuntimeError("Evidence LLM call has no explicit owning query state")
        with state["_ablation_call_lock"]:
            limit = state["_ablation_call_limit"]
            if limit is not None and state["_ablation_call_count"] >= limit:
                return None, "llm_budget_exhausted"
            state["_ablation_call_count"] += 1
        self._local.job_calls += 1
        # Preserve the parent call contract and expose logical token cost even
        # when a shared experiment cache supplies a previous model response.
        # The existing SQLite cache releases its DB lock while HTTP is running.
        # Two equal worker prompts could otherwise generate different responses
        # and overwrite one key, preventing exact selection-control replay.
        # Generation defaults and the LLM are fixed for this runtime instance.
        # This registry serializes only equal requests, never unrelated prompts.
        request_key = hashlib.sha256(json.dumps(
            {"messages": messages, "temperature": 0.0}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        with self._single_flight_registry_lock:
            prompt_lock = self._single_flight_prompt_locks.setdefault(request_key, threading.Lock())
        with prompt_lock:
            # CacheOpenAI persists a miss before infer returns; the next equal
            # logical call therefore observes exactly the first response.
            ret = self.rag.llm_model.infer(messages=messages, temperature=0.0)
        response = ret[0] if isinstance(ret, tuple) else ret
        metadata = ret[1] if isinstance(ret, tuple) and len(ret) > 1 else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        cache_hit = bool(ret[2]) if isinstance(ret, tuple) and len(ret) > 2 else False
        def token_count(key):
            try:
                return max(0, int(metadata.get(key, 0) or 0))
            except (ValueError, TypeError, OverflowError):
                return 0
        with state["_ablation_call_lock"]:
            usage = state["_ablation_usage"]
            usage["logical_prompt_tokens"] += token_count("prompt_tokens")
            usage["logical_completion_tokens"] += token_count("completion_tokens")
            usage["cache_hits"] += int(cache_hit)
            usage["logical_network_calls"] += int(not cache_hit)
        if metadata.get("finish_reason") == "length":
            return None, "truncated_response"
        return parse_object(self.rag._extract_llm_text(response))

    def _owned_job(self, state, function, *args):
        previous = getattr(self._local, "state", None)
        previous_calls = getattr(self._local, "job_calls", 0)
        self._local.state = state
        self._local.job_calls = 0
        try:
            return function(*args)
        finally:
            self._local.last_job_calls = self._local.job_calls
            self._local.state = previous
            self._local.job_calls = previous_calls

    def _plan_owned(self, state, hops):
        if state["_ablation_call_limit"] == 0:
            result = ([], "llm_budget_exhausted")
            calls = 0
            self._plan_diagnostics[state["query"]] = {"attempts": 0, "outputs": [], "validation_errors": []}
        else:
            result = self._owned_job(state, self._plan, state["query"], hops)
            calls = self._local.last_job_calls
        diagnostic = self._plan_diagnostics[state["query"]]
        diagnostic["attempts"] = calls
        return result

    def _verify_owned(self, state, question, answer_type, docs):
        result = self._owned_job(state, self._verify, question, answer_type, docs)
        diagnostic = dict(getattr(self._local, "verification_diagnostic", {}))
        diagnostic["attempts"] = self._local.last_job_calls
        # Carry the diagnostic with the result. Identical subquestions in two
        # simultaneously running original questions cannot overwrite ownership.
        return (*result, diagnostic)

    def _verify(self, question, answer_type, docs):
        if self.binding_mode in {"literal", "string"}:
            return self._literal_verify(question, answer_type, docs)
        from .ablation_selection import verify_relation_verdicts
        # Relation verification begins with the exact same literal extraction.
        # An independent judgement must then entail the requested relation/type
        # and every qualifier; token presence by itself does not pass this gate.
        hypotheses, rejected, reason = self._literal_verify(question, answer_type, docs)
        if not hypotheses:
            return hypotheses, rejected, reason
        diagnostic = self._local.verification_diagnostic
        messages = [
            {"role": "system", "content": (
                "Independently judge each proposed answer against its exact quoted evidence. "
                "Output only JSON {verdicts:[{doc_id,answer,evidence,label,relation_supported}]}. "
                "Copy doc_id, answer and evidence from the proposal unchanged. label is entailed, "
                "contradicted or insufficient. relation_supported must be true only if the quoted "
                "text establishes the requested relationship, answer type, identity, time and every "
                "nested qualifier. Merely mentioning a name or an associated event is insufficient. "
                "Do not use world knowledge or assume missing relations. Judge every proposal.")},
            {"role": "user", "content": json.dumps({"subquestion": question, "answer_type": answer_type,
             "proposals": hypotheses, "documents": {f"D{k}": v for k, v in docs.items()}}, ensure_ascii=False)},
        ]
        payload, judgement_reason = self._infer_object(messages)
        diagnostic["attempts"] += 1
        diagnostic.setdefault("relation_outputs", []).append(payload)
        if judgement_reason:
            return [], rejected + [{"reason": judgement_reason, "gate": "relation"}], judgement_reason
        accepted, relation_rejected = verify_relation_verdicts(payload, hypotheses, docs)
        return accepted, rejected + [dict(x, gate="relation") for x in relation_rejected], reason

    def _literal_verify(self, question, answer_type, docs):
        # Keep the original stage-four literal prompt and bounded repair behavior.
        # A local diagnostic object makes concurrently shared question text safe.
        from .evidence_retrieval import verify_hypotheses
        if self.binding_mode == "string":
            from .ablation_selection import verify_string_hypotheses
            validate = verify_string_hypotheses
        else:
            validate = verify_hypotheses
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
                "Do not resolve unknown placeholders, invent entities, or output hashes.")},
            {"role": "user", "content": f"Subquestion: {question}\nAnswer type: {answer_type}\n\n" +
             "\n\n".join(f"[D{k}]\n{text}" for k, text in docs.items())},
        ]
        diagnostic = {"attempts": 0, "outputs": []}
        self._local.verification_diagnostic = diagnostic
        self._verification_diagnostics[question + "::" + ",".join(map(str, docs))] = diagnostic
        all_rejected = []
        for attempt in range(2):
            payload, reason = self._infer_object(messages)
            diagnostic["attempts"] += 1
            diagnostic["outputs"].append(payload)
            accepted, rejected = ([], [{"reason": reason}]) if reason else validate(payload, docs)
            all_rejected.extend(rejected)
            if accepted or (not rejected and not reason):
                return accepted, all_rejected, reason
            if reason == "llm_budget_exhausted":
                break
            if attempt == 0:
                messages = messages + [{"role": "assistant", "content": json.dumps(payload or {}, ensure_ascii=False)},
                                       {"role": "user", "content": (
                                 "The proposed citations failed validation: " + json.dumps(rejected, ensure_ascii=False) +
                                 ". Recheck the original subquestion and the provided passages. "
                                 "Return a corrected hypothesis only if the exact original quote contains the literal answer "
                                 "and establishes the requested relationship. Otherwise return {\"hypotheses\":[]}. "
                                 "Do not change passage text or infer unsupported relationships.")}]
        return [], all_rejected, reason

    def _add_route(self, state, source, goal, query, requirements=None):
        if self.mode != "fixed_pool":
            return super()._add_route(state, source, goal, query, requirements)
        trace, cache = state["evidence_trace"], state["_evidence_search_cache"]
        requirements = dict(requirements or {})
        if query in cache:
            ids, raw = cache[query]
        else:
            if trace["search_count"] >= self.max_searches:
                trace["semantic_failures"].append({"node": goal, "reason": "search_budget_exhausted"})
                return None
            pool_ids, _ = self._frozen_pool(state)
            embedding = self.rag.query_to_embedding.get("passage", {}).get(query)
            if embedding is None:
                embedding = self.rag.embedding_model.batch_encode(
                    query, instruction=get_query_instruction("query_to_passage"), norm=True)
            vector = np.asarray(embedding).reshape(-1)
            values = np.dot(self.rag.passage_embeddings[pool_ids], vector)
            if not np.all(np.isfinite(values)):
                raise ValueError("Nonfinite frozen-pool similarity")
            order = np.lexsort((pool_ids, -values))[:self.top_k]
            ids = pool_ids[order].tolist()
            raw = values[order]
            cache[query] = (ids, raw)
            trace["search_count"] += 1
        for rank, (doc_id, score, original) in enumerate(zip(ids, local_rank_scores(raw, self.rank_constant), raw), 1):
            self._candidate(state, int(doc_id))["sources"].append({
                "source": source, "goal": goal, "rank": rank, "raw_score": float(original),
                "score": score, "question": query, "requirements": requirements})
        trace["routes"].append({"source": source, "goal": goal, "query": query,
                                "doc_ids": list(ids), "requirements": requirements,
                                "restricted_to_frozen_pool": True})
        return ids

    def _prepare_state(self, state):
        state["_ablation_call_lock"] = threading.Lock()
        state["_ablation_call_count"] = 0
        state["_ablation_usage"] = {"logical_prompt_tokens": 0, "logical_completion_tokens": 0,
                                    "cache_hits": 0, "logical_network_calls": 0}
        state["_ablation_call_limit"] = (self._record(state)["evidence_call_budget"]
                                          if self.mode.startswith("budget_") else None)
        if self.mode == "fixed_pool":
            ids, scores = self._frozen_pool(state)
            ctx = state.get("base", (None, None, {}))[2]
            state["base"] = (ids, scores, ctx)
        super()._prepare_state(state)
        trace = state["evidence_trace"]
        trace.update({"ablation_mode": self.mode, "binding_mode": self.binding_mode,
                      "selection_mode": self.selection_mode, "budget_scope": "evidence_module",
                      "reference_llm_call_budget": state["_ablation_call_limit"],
                      "actual_evidence_llm_calls": 0})
        trace["ablation"] = {"mode": self.mode, "binding_mode": self.binding_mode,
                             "selection_mode": self.selection_mode, "budget_scope": "evidence_module",
                             "call_budget": state["_ablation_call_limit"], "llm_calls": 0,
                             "budget_exhausted": False}
        trace["ablation"]["search_limit"] = self.max_searches
        if self.mode == "selection":
            trace["ablation"].update({"selection_shared_branches": True,
                                       "shared_beam_width": self.beam_width,
                                       "joint_search_scope": "generated_bounded_binding_branches"})
        if self.mode == "fixed_pool":
            ids, scores = self._frozen_pool(state)
            priors = local_rank_scores(scores, self.rank_constant)
            for doc_id, prior in zip(ids, priors):
                self._candidate(state, int(doc_id))["base_score"] = prior
            trace["frozen_pool_count"] = len(ids)
            trace["frozen_pool_hashes"] = [x["doc_hash"] for x in self._record(state)["pool"]]

    def _flat_request(self, state, iteration):
        iterative = self.mode == "budget_iterative"
        previous = [x["query"] for x in state["evidence_trace"]["routes"] if x["source"].startswith("budget:")]
        context = ""
        if iterative:
            ids = list(dict.fromkeys(state.get("_flat_context_ids", [])))[:3]
            context = "\n\nRetrieved passages:\n" + "\n\n".join(
                f"[D{d}]\n{self._document(d)[:4000]}" for d in ids)
        messages = [
            {"role": "system", "content": (
                "Generate productive next retrieval queries for the original question. Output only JSON "
                "{queries:[string]}, with one to four concrete retrieval questions. " +
                ("Use the retrieved passages to identify missing information and refine the next query. "
                 "Intermediate facts are provisional; there is no verification gate. " if iterative else
                 "Decompose the original question into flat standalone retrieval subquestions. "
                 "Do not invent bridge answers or create dependency placeholders. " ) +
                "Avoid repeating the preceding queries. Do not output answers, a dependency graph or hashes.")},
            {"role": "user", "content": f"Original question: {state['query']}\nRound: {iteration + 1}\n" +
             "Previous queries: " + json.dumps(previous, ensure_ascii=False) + context},
        ]
        return self._owned_job(state, self._infer_object, messages)

    def _flat_search(self, states, executor):
        for state in states:
            # The iterative control starts from the same baseline top passages.
            state["_flat_context_ids"] = np.asarray(state.get("base", ([], [], {}))[0])[:3].tolist()
        iteration = 0
        while True:
            active = [s for s in states if s["_ablation_call_count"] < s["_ablation_call_limit"]]
            if not active:
                break
            results = self._collect_jobs(executor, [(self._flat_request, (s, iteration)) for s in active])
            for state, (payload, reason) in zip(active, results):
                trace = state["evidence_trace"]
                trace["llm_plan_calls"] += 1
                trace.setdefault("flat_query_outputs", []).append(payload)
                if reason:
                    trace["semantic_failures"].append({"node": f"round{iteration + 1}", "reason": reason})
                    continue
                queries = payload.get("queries")
                if not isinstance(queries, list) or not queries or len(queries) > 4:
                    trace["semantic_failures"].append({"node": f"round{iteration + 1}", "reason": "invalid_flat_queries"})
                    continue
                clean = []
                for query in queries:
                    if not isinstance(query, str) or not query.strip() or "${" in query:
                        continue
                    if query not in clean:
                        clean.append(query.strip())
                retrieved = []
                for index, query in enumerate(clean):
                    ids = self._add_route(state, f"budget:{iteration + 1}:{index + 1}",
                                          f"flat:s{index + 1}", query)
                    retrieved.extend((ids or [])[:3])
                if retrieved:
                    state["_flat_context_ids"] = list(dict.fromkeys(retrieved))[:3]
            iteration += 1

    def _dependency_search(self, states, executor):
        """Parent DAG execution, with explicit state ownership for every LLM job."""
        active = [s for s in states if s.get("_evidence_plan")]
        for state in active:
            state["_evidence_layers"] = plan_layers(state["_evidence_plan"])
            state["_evidence_beams"] = [{"bindings": {}, "proofs": {}, "qualities": []}]
        depth = max((len(s["_evidence_layers"]) for s in active), default=0)
        for level in range(depth):
            tasks, lookup = [], {}
            for state in active:
                if level >= len(state["_evidence_layers"]):
                    continue
                for node in state["_evidence_layers"][level]:
                    for beam_index, beam in enumerate(state["_evidence_beams"]):
                        question = bind_question(node["question"], beam["bindings"])
                        if question is None:
                            state["evidence_trace"]["semantic_failures"].append({
                                "node": node["id"], "reason": "unbound_dependency", "requirements": node["depends_on"]})
                            continue
                        ids = self._add_route(state, f"dag:{node['id']}", f"dag:{node['id']}", question, beam["bindings"])
                        if not ids:
                            continue
                        docs = {doc_id: self._document(doc_id)[:4000] for doc_id in ids[:3]}
                        lookup[(id(state), node["id"], beam_index)] = len(tasks)
                        tasks.append((state, node, beam_index, ids[:3], question, docs))
            results = self._collect_jobs(executor, [
                (self._verify_owned, (state, question, node["answer_type"], docs))
                for state, node, beam_index, ids, question, docs in tasks])
            choices_by_task = []
            for task, (hypotheses, rejected, reason, diagnostic) in zip(tasks, results):
                state, node, beam_index, ids, question, docs = task
                trace = state["evidence_trace"]
                trace["llm_verification_calls"] += diagnostic["attempts"]
                trace.setdefault("verification_outputs", []).append(dict(diagnostic, node=node["id"]))
                trace["rejected_hypotheses"].extend(dict(x, node=node["id"]) for x in rejected)
                if reason:
                    trace["semantic_failures"].append({"node": node["id"], "reason": reason})
                for hyp in hypotheses:
                    rank = ids.index(hyp["doc_id"]) + 1
                    hyp["quality"] = float(.6 * self.rank_constant / (self.rank_constant + rank) + .4 * hyp["confidence"])
                hypotheses.sort(key=lambda h: (-h["quality"], h["doc_id"], normalized_text(h["answer"])))
                unique, seen = [], set()
                for hyp in hypotheses:
                    key = normalized_text(hyp["answer"])
                    if key not in seen:
                        unique.append(hyp)
                        seen.add(key)
                choices_by_task.append(unique[:self.beam_width])
                if not unique:
                    trace["semantic_failures"].append({"node": node["id"], "reason": "no_verified_answer"})
            for state in active:
                if level >= len(state["_evidence_layers"]):
                    continue
                expanded = []
                for beam_index, ancestor in enumerate(state["_evidence_beams"]):
                    descendants = [ancestor]
                    for node in state["_evidence_layers"][level]:
                        task_index = lookup.get((id(state), node["id"], beam_index))
                        choices = choices_by_task[task_index] if task_index is not None else []
                        if not choices:
                            continue
                        new = []
                        for descendant in descendants:
                            for hyp in choices:
                                child = {"bindings": dict(descendant["bindings"]), "proofs": dict(descendant["proofs"]),
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
            best = beams[0] if beams else {"bindings": {}, "proofs": {}, "qualities": []}
            trace = state["evidence_trace"]
            trace["bindings"] = dict(best["bindings"])
            trace["branch_scores"] = [{"bindings": dict(b["bindings"]),
                                      "score": self._beam_score(b, len(state["_evidence_plan"])),
                                      "proofs": dict(b["proofs"])} for b in beams]
            state["_evidence_winning_bindings"] = best["bindings"]
            proof_beams = beams if self.mode == "selection" else [best]
            for proof_beam in proof_beams:
                for node_id, hyp in proof_beam["proofs"].items():
                    proof = dict(hyp, goal=f"dag:{node_id}")
                    if self.mode == "selection":
                        proof["requirements"] = dict(proof_beam["bindings"])
                    self._candidate(state, hyp["doc_id"])["verified"].append(proof)

    @staticmethod
    def _goal_scores(candidate, bindings):
        # In the selection family, proofs from different generated binding
        # branches coexist in one common candidate pool. Only compatible proof
        # and route scores contribute to any particular branch's objective.
        compatible = dict(candidate)
        compatible["verified"] = [p for p in candidate["verified"]
                                   if branch_matches(p.get("requirements", {}), bindings)]
        return EvidenceRetrieval._goal_scores(compatible, bindings)

    def _trim_pool(self, state):
        if self.mode != "fixed_pool":
            return super()._trim_pool(state)
        ids, _ = self._frozen_pool(state)
        candidates = state["evidence_candidates"]
        frozen = set(ids.tolist())
        if set(candidates) != frozen:
            raise RuntimeError("Evidence candidate set escaped the frozen pool")
        state["evidence_trace"]["candidate_count"] = len(ids)

    def process_window(self, states: List[dict], executor=None):
        if self.stage < 2:
            return
        for state in states:
            self._prepare_state(state)
        if self.mode in {"budget_qd", "budget_iterative"}:
            self._flat_search(states, executor)
        elif self.stage >= 4:
            requests = [(self._plan_owned, (state, int(state.get("base", (None, None, {}))[2].get("hops", 1))))
                        for state in states]
            for state, (plan, reason) in zip(states, self._collect_jobs(executor, requests)):
                trace = state["evidence_trace"]
                diagnostic = self._plan_diagnostics[state["query"]]
                trace["llm_plan_calls"] += diagnostic["attempts"]
                trace["planning_outputs"] = diagnostic
                trace["plan"] = plan
                state["_evidence_plan"] = plan
                if reason:
                    trace["semantic_failures"].append({"node": "plan", "reason": reason})
            self._dependency_search(states, executor)
        for state in states:
            self._trim_pool(state)
            trace = state["evidence_trace"]
            trace["actual_evidence_llm_calls"] = state["_ablation_call_count"]
            trace["ablation"]["llm_calls"] = state["_ablation_call_count"]
            trace["ablation"].update(state["_ablation_usage"])
            trace["ablation"]["budget_exhausted"] = (
                state["_ablation_call_limit"] is not None and
                state["_ablation_call_count"] >= state["_ablation_call_limit"])
            trace["unused_llm_call_budget"] = (None if state["_ablation_call_limit"] is None else
                                               state["_ablation_call_limit"] - state["_ablation_call_count"])
            if trace["actual_evidence_llm_calls"] != trace["llm_plan_calls"] + trace["llm_verification_calls"]:
                raise RuntimeError("Evidence call accounting mismatch")
            state.pop("_evidence_search_cache", None)

    def finalize(self, query, ids, scores, ctx, state):
        if self.mode == "fixed_pool":
            ids, scores = self._frozen_pool(state)
        if self.selection_mode == "coverage" and self.mode != "selection":
            output = super().finalize(query, ids, scores, ctx, state)
        else:
            output = self._finalize_selection(query, ids, scores, ctx, state)
        if self.mode == "fixed_pool":
            frozen, _ = self._frozen_pool(state)
            actual = np.asarray(output[0], dtype=int)
            if len(actual) != len(frozen) or set(actual.tolist()) != set(frozen.tolist()):
                raise RuntimeError("Final retrieval results changed the frozen Top200 document set")
            output[2]["fixed_pool_set_preserved"] = True
        return output

    def _finalize_selection(self, query, ids, scores, ctx, state):
        from .ablation_selection import select_evidence_prefix
        candidates = state.get("evidence_candidates", {})
        if not candidates:
            return super().finalize(query, ids, scores, ctx, state)
        ids, scores = np.asarray(ids, dtype=int), np.asarray(scores, dtype=float)
        finite = np.where(np.isfinite(scores), scores, 0.0)
        span = float(finite.max() - finite.min()) if len(finite) else 0.
        norm = (finite - finite.min()) / span if span > 0 else np.ones(len(finite))
        base = dict(zip(ids.tolist(), norm.tolist()))
        bindings = state.get("_evidence_winning_bindings", {})
        base_weight = float(getattr(self.cfg, "evidence_base_weight", .4))
        beams = state.get("_evidence_beams", [])
        if not beams:
            beams = [{"bindings": bindings, "proofs": {}, "qualities": []}]
        evaluated = beams if self.selection_mode == "joint" else beams[:1]
        goal_union = {source["goal"] for c in candidates.values() for source in c["sources"]}
        goal_union.update(p["goal"] for c in candidates.values() for p in c["verified"])
        tokens = {d: set(normalized_text(self._document(d)).split()) for d in candidates}
        branch_results = []
        for index, beam in enumerate(evaluated):
            branch_bindings = beam["bindings"]
            views = {d: dict(c, verified=[p for p in c["verified"]
                                         if branch_matches(p.get("requirements", {}), branch_bindings)])
                     for d, c in candidates.items()}
            goals = {d: self._goal_scores(c, branch_bindings) for d, c in views.items()}
            selected, diagnostic = select_evidence_prefix(
                views, state.get("_evidence_plan", []), beam["proofs"], goals, tokens,
                bindings=branch_bindings, budget=self.budget,
                policy={"ancestor": "closure", "coverage": "coverage", "joint": "joint"}[self.selection_mode],
                base_scores=base, base_weight=base_weight,
                relation_weight=float(getattr(self.cfg, "evidence_relation_weight", .25)),
                coverage_weight=float(getattr(self.cfg, "evidence_coverage_weight", .45)),
                redundancy_weight=float(getattr(self.cfg, "evidence_redundancy_weight", .15)),
                goal_total=max(1, len(goal_union)))
            branch_results.append((float(diagnostic.get("objective_value", 0.)), index, selected, diagnostic, goals, views))
        _, branch_index, selected, diagnostic, goals, views = max(
            branch_results, key=lambda x: (x[0], -x[1]))
        chosen_beam = evaluated[branch_index]
        if self.selection_mode == "joint":
            bindings = dict(chosen_beam["bindings"])
            state["_evidence_winning_bindings"] = bindings
            state["evidence_trace"]["bindings"] = bindings
        diagnostic["evaluated_binding_branches"] = [
            {"bindings": dict(evaluated[index]["bindings"]), "objective_value": objective,
             "selected_prefix": prefix} for objective, index, prefix, diag, gs, vs in branch_results]
        diagnostic["chosen_binding_branch"] = branch_index
        diagnostic["binding_optimization"] = self.selection_mode == "joint"
        diagnostic["bindings_fixed"] = self.selection_mode != "joint"
        diagnostic["binding_search_scope"] = "generated_bounded_binding_branches"
        diagnostic["global_binding_optimum_guaranteed"] = False
        diagnostic["closure_valid"] = bool(diagnostic.get("proof_closed", False))
        merged = {d: base_weight * value for d, value in base.items()}
        for d, candidate in candidates.items():
            merged[d] = base_weight * base.get(d, candidate["base_score"]) + (1. - base_weight) * max(goals[d].values(), default=0.)
        ordered = sorted(merged, key=lambda d: (-merged[d], d))
        tail = [d for d in ordered if d not in set(selected)]
        top = max(merged.values(), default=1.) + 1.
        final_scores = [top + (len(selected) - i) / max(1, len(selected)) for i in range(len(selected))] + [merged[d] for d in tail]
        trace = state["evidence_trace"]
        trace["selection_diagnostics"] = diagnostic
        trace["selected_prefix"] = [{"doc_id": d, "goals": goals.get(d, {}),
                                      "verified": views[d]["verified"]} for d in selected]
        return np.asarray(selected + tail), np.asarray(final_scores), trace


# Public factory name used by PathCondRAG's runtime switch.
EvidenceAblation = EvidenceAblationRetrieval
