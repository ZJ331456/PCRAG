"""Apply the evaluated support guard and semantic veto after parent retrieval.

Each window freezes the real parent ranking and trace before proposing one
swap. Proposal and independent review use the existing trial prompts unchanged.
Only independent HTTP calls enter the worker pool; source reads and ranking
updates stay on the caller thread. No gold documents or answers are accepted.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from .evidence_semantic_swap_gate import build_verifier_prompt, validate_verification
from .evidence_support_guard import support_guard
from .evidence_typed_joint_selector import apply_joint_selection, build_joint_prompt


def _state(trace):
    # Match the frozen small-sample protocol, rather than using a different
    # internal beam representation at deployment time.
    return {"evidence_trace": trace, "_evidence_plan": trace.get("plan", []),
            "_evidence_winning_bindings": trace.get("bindings", {}),
            "_evidence_beams": trace.get("branch_scores", [])}


def _request(runtime, messages):
    result = runtime.rag.llm_model.infer(messages, response_format={"type": "json_object"})
    response = result[0] if isinstance(result, tuple) else result
    metadata = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
    return {"response": runtime.rag._extract_llm_text(response),
            "metadata": metadata if isinstance(metadata, dict) else {},
            "cache_hit": bool(result[2]) if isinstance(result, tuple) and len(result) > 2 else None}


def _preflight(runtime, messages):
    if not hasattr(runtime, "_support_semantic_tokenizer"):
        from transformers import AutoTokenizer
        runtime._support_semantic_tokenizer = AutoTokenizer.from_pretrained(
            runtime.cfg.evidence_review_tokenizer, trust_remote_code=True, local_files_only=True)
    count = len(runtime._support_semantic_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    limit = int(runtime.cfg.evidence_review_context_length)
    required = count + int(runtime.cfg.max_new_tokens)
    return {"prompt_tokens": count, "output_token_budget": int(runtime.cfg.max_new_tokens),
            "context_limit": limit, "allowed": required <= limit,
            "messages_sha256": hashlib.sha256(json.dumps(
                messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}


def _fetch(runtime, executor, contexts, stage):
    futures = []
    for ctx in contexts:
        messages = ctx.get(stage + "_messages")
        if messages is None:
            continue
        check = _preflight(runtime, messages)
        ctx[stage + "_preflight"] = check
        if check["allowed"]:
            # At most one stage's window is outstanding, bounded by workers.
            futures.append((ctx, executor.submit(_request, runtime, messages)))
    for ctx, future in futures:
        # Terminal HTTP/cache exceptions propagate; they are never reported as
        # a successful semantic abstention.
        ctx[stage + "_response"] = future.result()


def postprocess_support_semantic_veto(runtime, inputs):
    """Update QuerySolutions from real parent IDs before retrieval evaluation."""
    workers = max(1, int(runtime.cfg.llm_prefetch_workers))
    counts = Counter()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pcrag-support-review") as executor:
        with tqdm(total=len(inputs), desc="Evidence support/semantic veto") as progress:
            for start in range(0, len(inputs), workers):
                contexts = []
                for solution, ids in inputs[start:start + workers]:
                    old = [int(value) for value in ids]
                    if len(old) != len(solution.docs) or len(old) != len(set(old)):
                        raise ValueError("Parent IDs and exported documents must be unique and aligned")
                    # Preserve physical source identity; equal headings are
                    # allowed, equal physical documents are not collapsed.
                    if any(runtime._document(doc_id) != text for doc_id, text in zip(old, solution.docs)):
                        raise ValueError("Parent document text does not match its physical ID")
                    trace = deepcopy(solution.retrieval_trace["evidence"])
                    state = _state(trace)
                    messages = build_joint_prompt(solution.question, old, state, runtime._document)
                    contexts.append({"solution": solution, "old": old, "trace": trace, "state": state,
                                     "proposer_messages": messages})
                _fetch(runtime, executor, contexts, "proposer")
                for ctx in contexts:
                    query, old, state = ctx["solution"].question, ctx["old"], ctx["state"]
                    response = ctx.get("proposer_response")
                    proposed, proposal = apply_joint_selection(query, old, state, runtime._document,
                        response["response"] if response else '{"swap":null}',
                        response["metadata"].get("finish_reason", "unknown") if response else "stop")
                    if ctx.get("proposer_preflight", {}).get("allowed") is False:
                        proposal = dict(proposal, abstain_reason="proposer_prompt_exceeds_context_limit")
                    allowed, guard = support_guard(query, old, state, runtime._document, proposal)
                    ctx.update(proposed=proposed, proposal=proposal, guard=guard, guard_allowed=allowed)
                    # Review every accepted base proposal, including locally
                    # vetoed ones, to match the evaluated two-control protocol.
                    if proposed != old:
                        messages = build_verifier_prompt(query, old, state, runtime._document, proposal)
                        if messages is None:
                            raise ValueError("Source-validated proposal has no independent review prompt")
                        ctx["reviewer_messages"] = messages
                _fetch(runtime, executor, contexts, "reviewer")
                for ctx in contexts:
                    solution, old, state = ctx["solution"], ctx["old"], ctx["state"]
                    review = ctx.get("reviewer_response")
                    approved, semantic = validate_verification(solution.question, old, state,
                        runtime._document, ctx["proposal"], review["response"] if review else '{}',
                        review["metadata"].get("finish_reason", "unknown") if review else "stop")
                    if ctx.get("reviewer_preflight", {}).get("allowed") is False:
                        semantic = dict(semantic, rejection_reason="reviewer_prompt_exceeds_context_limit")
                    accepted = ctx["proposed"] != old and ctx["guard_allowed"] and approved
                    new = ctx["proposed"] if accepted else old
                    invariants = {"top2_preserved": new[:2] == old[:2],
                        "top10_set_preserved": set(new[:10]) == set(old[:10]),
                        "top200_set_preserved": set(new[:200]) == set(old[:200]),
                        "suffix_preserved": new[10:] == old[10:],
                        "unique_documents": len(new) == len(set(new)),
                        "protected_support_preserved": set(ctx["guard"]["protected_doc_ids"]) <= set(new[:5])}
                    if not all(invariants.values()):
                        raise ValueError("Support/semantic review violated a ranking or source invariant")
                    stages = {stage: {"preflight": ctx.get(stage + "_preflight"),
                                      "output": ctx.get(stage + "_response")}
                              for stage in ("proposer", "reviewer")}
                    calls = sum(ctx.get(stage + "_response") is not None for stage in stages)
                    diagnostic = {"enabled": True, "gold_labels_used": False,
                        "policy": "existing_support_guard_and_independent_semantic_veto",
                        "entailment_guaranteed": False, "accepted": bool(accepted),
                        "original_parent_doc_ids": old, "final_doc_ids": new, "extra_requests": calls,
                        "base_proposal": ctx["proposal"], "support_guard": ctx["guard"],
                        "independent_verification": semantic, "llm_stages": stages,
                        "promotions": ctx["proposal"].get("promotions", []) if accepted else [], **invariants}
                    trace = dict(ctx["trace"], improvement_support_semantic_veto=diagnostic)
                    if accepted:
                        selected = {item["doc_id"]: item for item in trace.get("selected_prefix", [])}
                        trace["selected_prefix"] = [dict(selected.get(doc_id, {
                            "doc_id": doc_id, "selection_source": "support_semantic_veto"}),
                            parent_rank=old.index(doc_id) + 1) for doc_id in new[:5]]
                        by_id = dict(zip(old, solution.docs))
                        solution.docs = [by_id[doc_id] for doc_id in new]
                        solution.doc_scores = np.arange(len(new), 0, -1, dtype=float) / max(1, len(new))
                    solution.retrieval_trace = dict(solution.retrieval_trace, evidence=trace)
                    counts["questions"] += 1
                    counts["logical_extra_requests"] += calls
                    counts["base_proposals"] += ctx["proposed"] != old
                    counts["applied"] += accepted
                    counts["support_vetoes"] += ctx["proposed"] != old and not ctx["guard_allowed"]
                    counts["semantic_vetoes"] += ctx["proposed"] != old and not approved
                    counts["context_abstentions"] += any(
                        stage["preflight"] and not stage["preflight"]["allowed"] for stage in stages.values())
                    progress.update(1)
    return dict(counts)
