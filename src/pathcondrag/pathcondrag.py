from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from numbers import Integral
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from .BaseRAG import BaseRAG
from .evaluation.retrieval_eval import RetrievalRecall
from .utils.misc_utils import QuerySolution, compute_mdhash_id, min_max_normalize

from .config import PCRAGConfig, PathCondRAGConfig
from .path_optimizer import (
    PathCandidate,
    aggregate_path_set_to_passage_scores,
    greedy_path_set_selection,
    normalize_candidate_scores,
)

logger = logging.getLogger(__name__)


class PCRAG(BaseRAG):
    """Path-Centric RAG on the graph retrieval runtime."""

    def __init__(
        self,
        global_config: Optional[PCRAGConfig] = None,
        save_dir: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_embedding_endpoint: Optional[str] = None,
    ):
        if global_config is None:
            global_config = PCRAGConfig()

        super().__init__(
            global_config=global_config,
            save_dir=save_dir,
            llm_model_name=llm_model_name,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_model_name,
            embedding_base_url=embedding_base_url,
            azure_endpoint=azure_endpoint,
            azure_embedding_endpoint=azure_embedding_endpoint,
        )

        self.pcrag_config: PCRAGConfig = self.global_config
        self.evidence_runtime = None
        if self.pcrag_config.improvement_stage >= 2:
            from .evidence_retrieval import EvidenceRetrieval
            self.evidence_runtime = EvidenceRetrieval(self)
        self._query_hop_overrides: Optional[List[int]] = None
        self._query_hop_override_source: Optional[str] = None

        # Index-side artifacts
        self.entity_idf: Dict[str, float] = {}
        self.chunk_to_entities: Dict[str, Set[str]] = {}
        self.bridge_neighbor_cache: Dict[str, List[str]] = {}
        self.entity_surface_norm: Dict[str, str] = {}
        self.entity_token_to_keys: Dict[str, List[str]] = {}
        self.entity_key_to_local_idx: Dict[str, int] = {}
        self.passage_key_to_local_idx: Dict[str, int] = {}

        self._pcrag_artifacts_path = os.path.join(self.working_dir, "pcrag_index_artifacts.json")

        self.reset_retrieval_diagnostics()

        logger.info(
            "PathCondRAG initialized: enhanced_hop=%s qcappr=%s eba=%s path_set=%s "
            "iterative=%s entity_idf=%s bridge_cache=%s hop_force_max=%s",
            self.pcrag_config.use_enhanced_hop_estimation,
            self.pcrag_config.use_qcappr,
            self.pcrag_config.use_eba,
            self.pcrag_config.use_path_set_optimization,
            self.pcrag_config.use_iterative_retrieval,
            self.pcrag_config.use_entity_idf_index,
            self.pcrag_config.use_bridge_cache_index,
            self.pcrag_config.hop_force_max,
        )

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------
    def set_query_hop_overrides(self, hops: List[int], source: str) -> None:
        """Use one externally supplied hop count for each query in the next retrieve.

        The list is aligned by position, never by question text.  Explicit
        labels bypass ``hop_force_max``, which only caps the heuristic estimate.
        ``retrieve`` validates the list length before starting any work.
        """
        if not isinstance(hops, list) or any(
            isinstance(hop, bool) or not isinstance(hop, Integral) or not 1 <= hop <= 4
            for hop in hops
        ):
            raise ValueError("hops must be a list of integers from 1 through 4")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("hop override source must be a nonempty string")
        self._query_hop_overrides = [int(hop) for hop in hops]
        self._query_hop_override_source = source.strip()

    def clear_query_hop_overrides(self) -> None:
        """Return subsequent retrievals to heuristic hop estimation."""
        self._query_hop_overrides = None
        self._query_hop_override_source = None

    def _qd_subquestion_limit(self, hops: int) -> int:
        """Preserve the old three-question prompt except for labeled four-hop items."""
        configured = self.pcrag_config.qd_max_sub_questions
        if self._query_hop_overrides is None:
            return configured
        return min(configured, max(3, hops))

    def reset_retrieval_diagnostics(self):
        super().reset_retrieval_diagnostics()
        cfg = getattr(self, "pcrag_config", None) or getattr(self, "global_config", None)

        def _flag(name: str) -> int:
            if cfg is None:
                return 0
            return int(bool(getattr(cfg, name, False)))

        self.retrieval_diagnostics.update(
            {
                "hop_counter": {"1": 0, "2": 0, "3": 0, "4": 0},
                "hop_override_source": getattr(self, "_query_hop_override_source", None),
                "avg_bridge_entities": 0.0,
                "avg_path_candidates": 0.0,
                "avg_selected_paths": 0.0,
                "avg_iterative_round2_seeds": 0.0,
                "iterative_used_count": 0,
                "qd_used_count": 0,
                "avg_qd_sub_questions": 0.0,
                "avg_qd_seq_rewrites": 0.0,
                "qd_query_ppr_used_count": 0,
                "qd_query_ppr_failed_count": 0,
                "pcqd_used_count": 0,
                "avg_pcqd_sub_questions": 0.0,
                "avg_path_hints": 0.0,
                "pcqd_entity_replacement_rate": 0.0,
                "pcqd_fallback_rate": 0.0,
                "pcqd_conflict_rate": 0.0,
                "_pcqd_entity_replacement_num": 0.0,
                "_pcqd_entity_replacement_den": 0.0,
                "_pcqd_fallback_count": 0,
                "_pcqd_conflict_sum": 0.0,
                "fallback_query_ppr_count": 0,
                "fallback_strategy_counter": {},
                "module_usage": {
                    "enhanced_hop_estimation": _flag("use_enhanced_hop_estimation"),
                    "qcappr": _flag("use_qcappr"),
                    "eba": _flag("use_eba"),
                    "path_set_opt": _flag("use_path_set_optimization"),
                    "iterative_retrieval": _flag("use_iterative_retrieval"),
                    "query_decomposition": _flag("use_query_decomposition"),
                    "path_conditioned_qd": _flag("use_path_conditioned_qd"),
                    "entity_idf_index": _flag("use_entity_idf_index"),
                    "bridge_cache_index": _flag("use_bridge_cache_index"),
                                                                                                                                            "mpce": _flag("use_mpce"),
                                                                                                    "no_facts_weak_entity_seed": _flag("no_facts_enable_weak_entity_seed"),
                    "qd_sequential_dependency": _flag("qd_enable_sequential_dependency"),
                    "pcqd_adaptive_fusion": _flag("pcqd_adaptive_fusion"),
                },
                "per_query": [],
            }
        )

    def get_retrieval_diagnostics(self) -> Dict[str, Any]:
        diag = json.loads(json.dumps(self.retrieval_diagnostics))
        total = int(diag.get("total_queries", 0))
        safe_total = max(1, total)
        fallback = int(diag.get("fallback_to_dpr_count", 0))
        query_ppr_fallback = int(diag.get("fallback_query_ppr_count", 0))
        no_facts = int(diag.get("no_facts_count", 0))
        diag["fallback_rate"] = float(fallback / safe_total)
        diag["no_facts_rate"] = float(no_facts / safe_total)
        diag["fallback_to_dpr_rate"] = float(fallback / safe_total)
        diag["fallback_query_ppr_rate"] = float(query_ppr_fallback / safe_total)
        # hide internal accumulators
        diag.pop("_pcqd_entity_replacement_num", None)
        diag.pop("_pcqd_entity_replacement_den", None)
        diag.pop("_pcqd_fallback_count", None)
        diag.pop("_pcqd_conflict_sum", None)
        return diag

    # ---------------------------------------------------------------------
    # Index-side innovations
    # ---------------------------------------------------------------------
    def index(self, docs: List[str]):
        super().index(docs)

        if not (
            self.pcrag_config.use_entity_idf_index
            or self.pcrag_config.use_bridge_cache_index
            or self.pcrag_config.no_facts_enable_weak_entity_seed
        ):
            return

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self._build_index_side_artifacts()

    def _build_index_side_artifacts(self):
        if self.ent_node_to_chunk_ids is None:
            logger.warning("ent_node_to_chunk_ids is empty; skipping pcrag index artifacts.")
            return

        self.entity_key_to_local_idx = {k: i for i, k in enumerate(self.entity_node_keys)}
        self.passage_key_to_local_idx = {k: i for i, k in enumerate(self.passage_node_keys)}

        # Build chunk -> entities reverse map (index innovation #1)
        chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)
        for ent_key, chunk_ids in self.ent_node_to_chunk_ids.items():
            for cid in chunk_ids:
                chunk_to_entities[cid].add(ent_key)
        self.chunk_to_entities = chunk_to_entities

        # Query-time weak matching index (used by no-facts root-cause recovery).
        self.entity_surface_norm = {}
        token_to_keys: Dict[str, Set[str]] = defaultdict(set)
        for ent_key in self.entity_node_keys:
            surf = self._normalize_text(self._entity_label(ent_key))
            self.entity_surface_norm[ent_key] = surf
            if not surf:
                continue
            for tok in set(surf.split()):
                if len(tok) < 2:
                    continue
                token_to_keys[tok].add(ent_key)
        self.entity_token_to_keys = {k: sorted(v) for k, v in token_to_keys.items()}

        # Build entity idf prior (index innovation #2)
        if self.pcrag_config.use_entity_idf_index:
            num_chunks = max(1, len(self.passage_node_keys))
            entity_idf: Dict[str, float] = {}
            for ent_key, chunk_ids in self.ent_node_to_chunk_ids.items():
                df = max(1, len(chunk_ids))
                entity_idf[ent_key] = float(math.log((num_chunks + 1) / (df + 1)) + 1.0)
            self.entity_idf = entity_idf

        # Precompute bridge neighbor cache (optional, can be expensive)
        self.bridge_neighbor_cache = {}
        if self.pcrag_config.use_bridge_cache_index:
            limit = min(self.pcrag_config.bridge_cache_entity_limit, len(self.entity_node_keys))
            for ent_key in self.entity_node_keys[:limit]:
                self.bridge_neighbor_cache[ent_key] = self._rank_bridge_neighbors(ent_key)


        payload = {
            "num_entities": len(self.entity_node_keys),
            "num_passages": len(self.passage_node_keys),
            "chunk_to_entities_size": len(self.chunk_to_entities),
            "entity_surface_index_size": len(self.entity_surface_norm),
            "entity_token_index_size": len(self.entity_token_to_keys),
            "entity_idf_size": len(self.entity_idf),
            "bridge_cache_size": len(self.bridge_neighbor_cache),
            "config": {
                "use_entity_idf_index": self.pcrag_config.use_entity_idf_index,
                "use_bridge_cache_index": self.pcrag_config.use_bridge_cache_index,
            },
        }
        with open(self._pcrag_artifacts_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("PCRAG index artifacts built: chunk_to_entities=%d, entity_idf=%d, bridge_cache=%d", len(self.chunk_to_entities), len(self.entity_idf), len(self.bridge_neighbor_cache))

    def _extract_passage_title(content: str) -> str:
        if not content:
            return ""
        for line in str(content).splitlines():
            title = line.strip()
            if title:
                return title
        return ""

    def _entity_label(self, ent_key: str) -> str:
        try:
            row = self.entity_embedding_store.get_row(ent_key)
            if isinstance(row, dict):
                content = str(row.get("content", "")).strip()
                if content:
                    return content
        except Exception:
            pass
        return self._entity_surface(ent_key)

    def _apply_mpce(
        self,
        sorted_doc_ids: np.ndarray,
        sorted_doc_scores: np.ndarray,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Multi-Path Consensus Evidence (MPCE) re-scoring.

        Boost passages supported by multiple selected paths using direct path
        membership and shared high-IDF entity consensus.
        """
        if not self.pcrag_config.use_mpce:
            return sorted_doc_ids, sorted_doc_scores, {"mpce_applied": False}

        ctx = ctx or {}
        hops = int(ctx.get("hops", 2))
        if self.pcrag_config.mpce_only_on_two_hop and hops < 2:
            return sorted_doc_ids, sorted_doc_scores, {"mpce_applied": False}

        selected_paths: List[Dict[str, Any]] = ctx.get("selected_path_candidates", [])
        if len(selected_paths) < int(self.pcrag_config.mpce_consensus_min_paths):
            return sorted_doc_ids, sorted_doc_scores, {"mpce_applied": False}

        candidate_top_k = min(len(sorted_doc_ids), int(self.pcrag_config.mpce_candidate_top_k))
        if candidate_top_k <= 1:
            return sorted_doc_ids, sorted_doc_scores, {"mpce_applied": False}

        path_passage_keys: Set[str] = set()
        path_entity_sets: List[Set[str]] = []
        for path in selected_paths:
            pkey = str(path.get("passage_key", ""))
            if pkey:
                path_passage_keys.add(pkey)
            ents = path.get("covered_entities", [])
            if ents:
                path_entity_sets.append({str(e) for e in ents if str(e)})

        # Build consensus entity map: entity → weighted score (path_count × IDF).
        # v2 improvement: keep ALL entities that appear in >= min_paths selected
        # paths (not just top-3), then use a soft weighted-overlap score rather
        # than a hard top-k cutoff.  This captures more second-hop signals.
        consensus_entities: Dict[str, float] = {}
        if self.pcrag_config.mpce_use_entity_consensus and path_entity_sets:
            entity_path_count: Dict[str, int] = {}
            for ent_set in path_entity_sets:
                for ent in ent_set:
                    entity_path_count[ent] = entity_path_count.get(ent, 0) + 1

            min_paths = int(self.pcrag_config.mpce_consensus_min_paths)
            for ent, count in entity_path_count.items():
                if count < min_paths:
                    continue
                idf = float(self.entity_idf.get(ent, 1.0))
                if idf < 2.0:
                    continue
                consensus_entities[ent] = float(count) * idf

            # v2: soft top-k — keep up to entity_consensus_top_k×5 candidates
            # so that the boost is distributed over more second-hop entities.
            # A hard cutoff of 3 was too narrow; bridge entities may rank lower
            # when many paths share the same seed entity.
            top_k_ent = int(self.pcrag_config.mpce_entity_consensus_top_k) * 5
            if len(consensus_entities) > top_k_ent:
                consensus_entities = dict(
                    sorted(consensus_entities.items(), key=lambda x: x[1], reverse=True)[:top_k_ent]
                )

        gamma = float(self.pcrag_config.mpce_gamma)
        cap = float(self.pcrag_config.mpce_boost_cap)
        ent_w = float(self.pcrag_config.mpce_entity_consensus_weight)

        candidate_ids = [int(x) for x in sorted_doc_ids[:candidate_top_k].tolist()]
        base_scores = np.asarray(
            min_max_normalize(sorted_doc_scores[:candidate_top_k]), dtype=np.float32
        )

        boosts: Dict[int, float] = {}
        consensus_count = 0
        max_possible = sum(consensus_entities.values()) if consensus_entities else 0.0

        for rank, doc_id in enumerate(candidate_ids):
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]

            path_support = 1.0 if pkey in path_passage_keys else 0.0

            # v2: weighted overlap — sum matched consensus scores normalised by
            # the total consensus weight (not hard top-3 exact match).
            entity_support = 0.0
            if consensus_entities:
                pkey_ents = self.chunk_to_entities.get(pkey, set())
                if pkey_ents:
                    matched_score = sum(
                        consensus_entities[e] for e in pkey_ents if e in consensus_entities
                    )
                    if matched_score > 0:
                        entity_support = min(1.0, float(matched_score / max(1.0, max_possible)))

            if path_support > 0.0 or entity_support > 0.0:
                consensus_score = (1.0 - ent_w) * path_support + ent_w * entity_support
                base_score = float(base_scores[rank])
                boost = min(cap, gamma * consensus_score * (0.5 + 0.5 * base_score))
                if boost > 0:
                    boosts[doc_id] = boost
                    consensus_count += 1

        if not boosts:
            return sorted_doc_ids, sorted_doc_scores, {
                "mpce_applied": False,
                "mpce_consensus_passages": 0,
                "mpce_consensus_entities": int(len(consensus_entities)),
            }

        full_scores = np.zeros(len(self.passage_node_keys), dtype=np.float32)
        full_scores[sorted_doc_ids] = np.asarray(min_max_normalize(sorted_doc_scores), dtype=np.float32)

        for doc_id, boost in boosts.items():
            full_scores[doc_id] = float(full_scores[doc_id]) * (1.0 + float(boost))

        reranked_ids = np.argsort(full_scores)[::-1]
        reranked_scores = full_scores[reranked_ids]
        return reranked_ids, reranked_scores, {
            "mpce_applied": True,
            "mpce_consensus_passages": int(consensus_count),
            "mpce_consensus_entities": int(len(consensus_entities)),
        }
    def _iter_retrieval_states(self, queries: List[str]):
        """Bound outstanding LLM jobs to one small, ordered query window."""
        workers = self.pcrag_config.llm_prefetch_workers
        evidence_runtime = getattr(self, "evidence_runtime", None)
        if (workers <= 1 or not self.pcrag_config.use_query_decomposition) and evidence_runtime is None:
            for q_idx, query in enumerate(queries):
                yield {"query_idx": q_idx, "query": query}
            return

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pcrag-qd") as executor:
            for start in range(0, len(queries), workers):
                states: List[Dict[str, Any]] = []
                # Embed/read scores on the caller thread. Only independent LLM
                # fact-filter calls enter the bounded worker pool.
                for q_idx in range(start, min(start + workers, len(queries))):
                    query = queries[q_idx]
                    query_fact_scores = self.get_fact_scores(query)
                    state: Dict[str, Any] = {
                        "query_idx": q_idx,
                        "query": query,
                        "fact_scores": query_fact_scores,
                        "rerank_future": executor.submit(self.rerank_facts, query, query_fact_scores),
                    }
                    states.append(state)
                for state in states:
                    q_idx, query = state["query_idx"], state["query"]
                    query_fact_scores = state.pop("fact_scores")
                    top_k_fact_indices, top_k_facts, rerank_log = state.pop("rerank_future").result()
                    state["rerank"] = (query_fact_scores, top_k_fact_indices, top_k_facts, rerank_log)
                    if top_k_facts:
                        hop_override = (
                            self._query_hop_overrides[q_idx]
                            if self._query_hop_overrides is not None else None
                        )
                        base = self._path_graph_search(
                            query=query,
                            query_fact_scores=query_fact_scores,
                            top_k_facts=top_k_facts,
                            top_k_fact_indices=top_k_fact_indices,
                            hop_override=hop_override,
                            defer_qd=True,
                        )
                        state["base"] = base
                        _base_ids, _base_scores, ctx = base
                        if ctx.get("_qd_deferred") and int(ctx["hops"]) >= self.pcrag_config.qd_min_hops:
                            state["qd_plan"] = True
                    elif evidence_runtime is not None:
                        ids, scores = self.dense_passage_retrieval(query)
                        hops = (self._query_hop_overrides[q_idx] if self._query_hop_overrides is not None
                                else self._estimate_query_hops(query, top_k_facts=[]))
                        state["base"] = (ids, scores, {"hops": hops, "seed_entities": [],
                                                     "seed_distribution": {}, "bridges_by_seed": {}})
                        if self.pcrag_config.use_query_decomposition and hops >= self.pcrag_config.qd_min_hops:
                            state["qd_plan"] = True

                # Phase 1: parallelize independent static decomposition calls.
                for state in states:
                    if "qd_plan" in state:
                        state["static_future"] = executor.submit(
                            self._decompose_query,
                            state["query"],
                            self._qd_subquestion_limit(int(state["base"][2]["hops"])),
                        )
                for state in states:
                    if "static_future" in state:
                        state["static_sub_questions"] = state.pop("static_future").result()

                # Phase 2: build static tracks and path hints serially.  The
                # original path only asks PCQD after a usable static track.
                for state in states:
                    if "qd_plan" not in state:
                        continue
                    static_sub_questions = state["static_sub_questions"]
                    if not static_sub_questions:
                        continue
                    base_ids, base_scores, ctx = state["base"]
                    static_track = self._build_qd_track_ranking(
                        query=state["query"],
                        hops=int(ctx["hops"]),
                        sub_questions=static_sub_questions,
                        base_sorted_doc_ids=base_ids,
                        original_seed_entities=set(ctx["seed_entities"]),
                    )
                    state["static_track"] = static_track
                    static_ids, static_scores, static_stats = static_track
                    static_used = bool(static_stats.get("used", False) and static_ids is not None and static_scores is not None)
                    if static_used and self.pcrag_config.use_path_conditioned_qd:
                        state["path_hints"] = self._build_pcqd_path_hints(
                            base_sorted_doc_ids=base_ids,
                            base_sorted_doc_scores=base_scores,
                            seed_distribution=ctx["seed_distribution"],
                            bridges_by_seed=ctx["bridges_by_seed"],
                        )
                        state["hint_diagnostics"] = dict(getattr(self, "_last_pcqd_hint_stats", {}))

                # Phase 3: parallelize PCQD calls only where the old serial
                # path would make one, then finalize every query in input order.
                for state in states:
                    if "path_hints" in state and state["path_hints"][0]:
                        state["pcqd_future"] = executor.submit(
                            self._decompose_query_with_hints,
                            state["query"],
                            state["path_hints"][0],
                            self._qd_subquestion_limit(int(state["base"][2]["hops"])),
                        )
                for state in states:
                    if "pcqd_future" in state:
                        state["pcqd_sub_questions"] = state.pop("pcqd_future").result()
                if evidence_runtime is not None:
                    evidence_runtime.process_window(states, executor)
                for state in states:
                    yield state

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: Optional[int] = None,
        gold_docs: Optional[List[List[str]]] = None,
    ):
        if self._query_hop_overrides is not None and len(self._query_hop_overrides) != len(queries):
            raise ValueError(
                "query hop override count must match the number of queries: "
                f"{len(self._query_hop_overrides)} != {len(queries)}"
            )
        retrieve_start_time = time.time()
        self.reset_retrieval_diagnostics()

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        if not self.chunk_to_entities:
            # lazy recovery for runs using existing index not built by current process
            self._build_index_side_artifacts()

        self.entity_key_to_local_idx = {k: i for i, k in enumerate(self.entity_node_keys)}
        self.passage_key_to_local_idx = {k: i for i, k in enumerate(self.passage_node_keys)}

        self.get_query_embeddings(queries)

        retrieval_results: List[QuerySolution] = []

        for state in tqdm(self._iter_retrieval_states(queries), total=len(queries), desc="PCRAG Retrieving"):
            q_idx, query = state["query_idx"], state["query"]
            if "rerank" in state:
                query_fact_scores, top_k_fact_indices, top_k_facts, rerank_log = state["rerank"]
            else:
                query_fact_scores = self.get_fact_scores(query)
                top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            no_facts_reason = (
                rerank_log.get("no_facts_reason", "none")
                if isinstance(rerank_log, dict)
                else "none"
            )

            if len(top_k_facts) == 0:
                fallback_strategy = self.pcrag_config.empty_rerank_fallback
                if isinstance(rerank_log, dict):
                    rerank_log["fallback_strategy"] = fallback_strategy
                dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
                sorted_doc_ids, sorted_doc_scores = dpr_sorted_doc_ids, dpr_sorted_doc_scores
                used_weak_seed = False
                if self.pcrag_config.no_facts_enable_weak_entity_seed:
                    sorted_doc_ids, sorted_doc_scores, used_weak_seed = self._weak_entity_seed_ppr_fallback(
                        query=query,
                        dpr_sorted_doc_ids=dpr_sorted_doc_ids,
                        dpr_sorted_doc_scores=dpr_sorted_doc_scores,
                    )
                used_query_ppr = False
                if not used_weak_seed:
                    if fallback_strategy == "query_ppr":
                        sorted_doc_ids, sorted_doc_scores, used_query_ppr = self._query_embedding_ppr_fallback(
                            query=query,
                            dpr_sorted_doc_ids=dpr_sorted_doc_ids,
                            dpr_sorted_doc_scores=dpr_sorted_doc_scores,
                        )
                    else:
                        sorted_doc_ids, sorted_doc_scores = dpr_sorted_doc_ids, dpr_sorted_doc_scores
                fallback_to_dpr = not (used_query_ppr or used_weak_seed)
                self._record_retrieval_diagnostic(
                    query_idx=q_idx,
                    query=query,
                    fallback_to_dpr=fallback_to_dpr,
                    rerank_log=rerank_log,
                )
                if used_weak_seed:
                    self.retrieval_diagnostics["fallback_query_ppr_count"] += 1
                    self._increment_counter(self.retrieval_diagnostics["fallback_strategy_counter"], "weak_seed_query_ppr")
                elif used_query_ppr:
                    self.retrieval_diagnostics["fallback_query_ppr_count"] += 1
                    self._increment_counter(self.retrieval_diagnostics["fallback_strategy_counter"], "query_ppr")
                elif fallback_strategy == "query_ppr":
                    self._increment_counter(
                        self.retrieval_diagnostics["fallback_strategy_counter"],
                        "query_ppr_failed_to_dpr",
                    )
                elif fallback_strategy in {"dpr", "unfiltered"}:
                    self._increment_counter(self.retrieval_diagnostics["fallback_strategy_counter"], fallback_strategy)
                else:
                    self._increment_counter(self.retrieval_diagnostics["fallback_strategy_counter"], "unknown")
                noop_stats_a = {}
                noop_stats_b = {}
                trace = {"improvement_stage": getattr(self.pcrag_config, "improvement_stage", 0),
                         "fact_filter": rerank_log, "dense_fallback": fallback_to_dpr,
                         "assigned_hops": self._query_hop_overrides[q_idx] if self._query_hop_overrides is not None else None}
                evidence_runtime = getattr(self, "evidence_runtime", None)
                if evidence_runtime is not None:
                    sorted_doc_ids, sorted_doc_scores, evidence_trace = evidence_runtime.finalize(
                        query, sorted_doc_ids, sorted_doc_scores, state.get("base", (None, None, {}))[2], state)
                    trace["evidence"] = evidence_trace
                retrieval_results.append(
                    QuerySolution(
                        question=query,
                        docs=[self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]],
                        doc_scores=sorted_doc_scores[:num_to_retrieve],
                        retrieval_trace=trace,
                    )
                )
                self.retrieval_diagnostics["per_query"].append(
                    {
                        "query_idx": q_idx,
                        "no_facts": True,
                        "no_facts_reason": no_facts_reason,
                        "fallback_strategy": (
                            "weak_seed_query_ppr"
                            if used_weak_seed
                            else ("query_ppr" if used_query_ppr else fallback_strategy)
                        ),
                        "hops": None,
                        "hop_source": self._query_hop_override_source if self._query_hop_overrides is not None else "estimated",
                        "assigned_hops": self._query_hop_overrides[q_idx] if self._query_hop_overrides is not None else None,
                        "bridge_entities": 0,
                        "path_candidates": 0,
                        "selected_paths": 0,
                        "iterative_round2_seeds": 0,
                        "pcqd_used": False,
                        "pcqd_sub_questions": 0,
                        "qd_sub_query_ppr_used": 0,
                        "qd_sub_query_ppr_failed": 0,
                        "qd_seq_rewrites": 0,
                        "pcqd_path_hints": 0,
                        "pcqd_entity_replacement_rate": 0.0,
                        "pcqd_fallback": False,
                        "pcqd_conflict_rate": 0.0,
                    }
                )
                continue

            if "base" in state:
                sorted_doc_ids, sorted_doc_scores, ctx = state["base"]
                if "qd_plan" in state:
                    # Propagate terminal request errors: silently dropping QD or
                    # PCQD changes retrieval results without alerting the user.
                    static_sub_questions = state["static_sub_questions"]
                    pcqd_sub_questions = state.get("pcqd_sub_questions")
                    path_hints, conflict_rate = state.get("path_hints", ([], 0.0))
                    sorted_doc_ids, sorted_doc_scores, qd_stats = self._query_decomposition_retrieval(
                        query=query,
                        hops=int(ctx["hops"]),
                        base_sorted_doc_ids=sorted_doc_ids,
                        base_sorted_doc_scores=sorted_doc_scores,
                        seed_entities=ctx["seed_entities"],
                        seed_distribution=ctx["seed_distribution"],
                        bridges_by_seed=ctx["bridges_by_seed"],
                        prefetched_qd=(static_sub_questions, pcqd_sub_questions, path_hints, conflict_rate),
                        prefetched_static_track=state.get("static_track"),
                    )
                    ctx.update(qd_stats)
                ctx.pop("_qd_deferred", None)
            else:
                sorted_doc_ids, sorted_doc_scores, ctx = self._path_graph_search(
                    query=query,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    hop_override=(
                        self._query_hop_overrides[q_idx]
                        if self._query_hop_overrides is not None else None
                    ),
                )
            self._record_retrieval_diagnostic(
                query_idx=q_idx,
                query=query,
                fallback_to_dpr=False,
                rerank_log=rerank_log,
            )


            if self.pcrag_config.use_path_set_optimization:
                sorted_doc_ids, sorted_doc_scores, path_stats = self._apply_path_set_optimization(
                    query=query,
                    base_sorted_doc_ids=sorted_doc_ids,
                    base_sorted_doc_scores=sorted_doc_scores,
                    ctx=ctx,
                )
            else:
                path_stats = {
                    "path_candidates": 0,
                    "selected_paths": 0,
                    "selected_path_candidates": [],
                }
            ctx.update(path_stats)

            sorted_doc_ids, sorted_doc_scores, mpce_stats = self._apply_mpce(
                sorted_doc_ids=sorted_doc_ids,
                sorted_doc_scores=sorted_doc_scores,
                ctx=ctx,
            )
            ctx.update(mpce_stats)
            trace = {"improvement_stage": getattr(self.pcrag_config, "improvement_stage", 0),
                     "fact_filter": rerank_log, "dense_fallback": False,
                     "hops": int(ctx["hops"]), "seed_entities": ctx["seed_entities"],
                     "fact_seed_distribution": ctx.get("fact_seed_distribution", {}),
                     "static_sub_questions": state.get("static_sub_questions", []),
                     "pcqd_sub_questions": state.get("pcqd_sub_questions", []),
                     "pcqd_validation": getattr(self, "_pcqd_diagnostics_by_query", {}).get(query, {}),
                     "path_hints": state.get("path_hints", ([], 0.0))[0],
                     "hint_diagnostics": state.get("hint_diagnostics", {}),
                     "selected_path_candidates": path_stats.get("selected_path_candidates", [])}
            evidence_runtime = getattr(self, "evidence_runtime", None)
            if evidence_runtime is not None:
                sorted_doc_ids, sorted_doc_scores, evidence_trace = evidence_runtime.finalize(
                    query, sorted_doc_ids, sorted_doc_scores, ctx, state)
                trace["evidence"] = evidence_trace
            top_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in sorted_doc_ids[:num_to_retrieve]
            ]
            top_scores = sorted_doc_scores[:num_to_retrieve]
            retrieval_results.append(QuerySolution(question=query, docs=top_docs, doc_scores=top_scores,
                                                    retrieval_trace=trace))

            hops = int(ctx.get("hops", 2))
            self.retrieval_diagnostics["hop_counter"][str(hops)] += 1
            self.retrieval_diagnostics["avg_bridge_entities"] += float(ctx.get("bridge_count", 0))
            self.retrieval_diagnostics["avg_path_candidates"] += float(path_stats.get("path_candidates", 0))
            self.retrieval_diagnostics["avg_selected_paths"] += float(path_stats.get("selected_paths", 0))
            round2_seed_count = int(ctx.get("iterative_round2_seed_count", 0))
            self.retrieval_diagnostics["avg_iterative_round2_seeds"] += float(round2_seed_count)
            if ctx.get("iterative_used", False):
                self.retrieval_diagnostics["iterative_used_count"] += 1
            if ctx.get("qd_used", False):
                self.retrieval_diagnostics["qd_used_count"] += 1
            self.retrieval_diagnostics["avg_qd_sub_questions"] += float(ctx.get("qd_sub_questions", 0))
            self.retrieval_diagnostics["avg_qd_seq_rewrites"] += float(ctx.get("qd_seq_rewrites", 0))
            self.retrieval_diagnostics["qd_query_ppr_used_count"] += int(ctx.get("qd_sub_query_ppr_used", 0))
            self.retrieval_diagnostics["qd_query_ppr_failed_count"] += int(ctx.get("qd_sub_query_ppr_failed", 0))
            if ctx.get("pcqd_used", False):
                self.retrieval_diagnostics["pcqd_used_count"] += 1
            self.retrieval_diagnostics["avg_pcqd_sub_questions"] += float(ctx.get("pcqd_sub_questions", 0))
            self.retrieval_diagnostics["avg_path_hints"] += float(ctx.get("pcqd_path_hints", 0))
            self.retrieval_diagnostics["_pcqd_entity_replacement_num"] += float(
                ctx.get("pcqd_entity_replacement_num", 0.0)
            )
            self.retrieval_diagnostics["_pcqd_entity_replacement_den"] += float(
                ctx.get("pcqd_entity_replacement_den", 0.0)
            )
            if ctx.get("pcqd_fallback", False):
                self.retrieval_diagnostics["_pcqd_fallback_count"] += 1
            self.retrieval_diagnostics["_pcqd_conflict_sum"] += float(ctx.get("pcqd_conflict_rate", 0.0))

            self.retrieval_diagnostics["per_query"].append(
                {
                    "query_idx": q_idx,
                    "no_facts": False,
                    "no_facts_reason": no_facts_reason,
                    "hops": hops,
                    "hop_source": self._query_hop_override_source if self._query_hop_overrides is not None else "estimated",
                    "bridge_entities": int(ctx.get("bridge_count", 0)),
                    "path_candidates": int(path_stats.get("path_candidates", 0)),
                    "selected_paths": int(path_stats.get("selected_paths", 0)),
                    "iterative_round2_seeds": round2_seed_count,
                    "iterative_used": bool(ctx.get("iterative_used", False)),
                    "qd_used": bool(ctx.get("qd_used", False)),
                    "qd_sub_questions": int(ctx.get("qd_sub_questions", 0)),
                    "qd_sub_query_ppr_used": int(ctx.get("qd_sub_query_ppr_used", 0)),
                    "qd_sub_query_ppr_failed": int(ctx.get("qd_sub_query_ppr_failed", 0)),
                    "qd_seq_rewrites": int(ctx.get("qd_seq_rewrites", 0)),
                    "pcqd_used": bool(ctx.get("pcqd_used", False)),
                    "pcqd_sub_questions": int(ctx.get("pcqd_sub_questions", 0)),
                    "pcqd_path_hints": int(ctx.get("pcqd_path_hints", 0)),
                    "pcqd_entity_replacement_rate": float(ctx.get("pcqd_entity_replacement_rate", 0.0)),
                    "pcqd_fallback": bool(ctx.get("pcqd_fallback", False)),
                    "pcqd_conflict_rate": float(ctx.get("pcqd_conflict_rate", 0.0)),
                }
            )

        total = max(1, self.retrieval_diagnostics["total_queries"])
        self.retrieval_diagnostics["avg_bridge_entities"] = float(self.retrieval_diagnostics["avg_bridge_entities"] / total)
        self.retrieval_diagnostics["avg_path_candidates"] = float(self.retrieval_diagnostics["avg_path_candidates"] / total)
        self.retrieval_diagnostics["avg_selected_paths"] = float(self.retrieval_diagnostics["avg_selected_paths"] / total)
        self.retrieval_diagnostics["avg_iterative_round2_seeds"] = float(
            self.retrieval_diagnostics["avg_iterative_round2_seeds"] / total
        )
        self.retrieval_diagnostics["avg_qd_sub_questions"] = float(
            self.retrieval_diagnostics["avg_qd_sub_questions"] / total
        )
        self.retrieval_diagnostics["avg_qd_seq_rewrites"] = float(
            self.retrieval_diagnostics["avg_qd_seq_rewrites"] / total
        )
        self.retrieval_diagnostics["avg_pcqd_sub_questions"] = float(
            self.retrieval_diagnostics["avg_pcqd_sub_questions"] / total
        )
        self.retrieval_diagnostics["avg_path_hints"] = float(
            self.retrieval_diagnostics["avg_path_hints"] / total
        )
        repl_den = max(1e-9, float(self.retrieval_diagnostics.get("_pcqd_entity_replacement_den", 0.0)))
        self.retrieval_diagnostics["pcqd_entity_replacement_rate"] = float(
            float(self.retrieval_diagnostics.get("_pcqd_entity_replacement_num", 0.0)) / repl_den
        )
        self.retrieval_diagnostics["pcqd_fallback_rate"] = float(
            float(self.retrieval_diagnostics.get("_pcqd_fallback_count", 0.0)) / total
        )
        self.retrieval_diagnostics["pcqd_conflict_rate"] = float(
            float(self.retrieval_diagnostics.get("_pcqd_conflict_sum", 0.0)) / total
        )

        self.all_retrieval_time += time.time() - retrieve_start_time
        logger.info("[PCRAG] Total Retrieval Time %.2fs", self.all_retrieval_time)

        if gold_docs is not None:
            evaluator = RetrievalRecall(global_config=self.global_config)
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall, _ = evaluator.calculate_metric_scores(
                gold_docs=gold_docs,
                retrieved_docs=[x.docs for x in retrieval_results],
                k_list=k_list,
            )
            logger.info("[PCRAG] Retrieval evaluation: %s", overall)
            return retrieval_results, overall

        return retrieval_results

    # ---------------------------------------------------------------------
    # Core modules: QCAPPR + EBA
    # ---------------------------------------------------------------------
    def _estimate_query_hops(self, query: str, top_k_facts: Optional[List[Tuple]] = None) -> int:
        """
        Enhanced hop estimation using multiple signals:
        1. Keyword clues (original)
        2. Query entity count
        3. DPR entity distribution entropy
        """
        # Signal 1: Keyword clues
        q = f" {query.lower()} "
        clues = [
            " and ", " after ", " before ", " then ", " which ", " whose ",
            " where ", " when ", " first ", " second ", " former ", " latter ",
            " that ", " who ",
        ]
        keyword_score = sum(1 for c in clues if c in q)

        # Legacy estimator (for ablation): keyword-only
        if not self.pcrag_config.use_enhanced_hop_estimation:
            if keyword_score <= 1:
                hops = 1
            elif keyword_score <= 4:
                hops = 2
            else:
                hops = 3
            return min(self.pcrag_config.hop_force_max, hops)
        
        # Signal 2: Query entity count (more entities -> likely multi-hop)
        if top_k_facts is None:
            query_fact_scores = self.get_fact_scores(query)
            _, top_k_facts, _ = self.rerank_facts(query, query_fact_scores)
        query_entities = self._extract_seed_entities_from_facts(top_k_facts)
        entity_count = len(query_entities)
        
        # Signal 3: DPR entity distribution entropy
        # If top DPR docs have highly dispersed entities, likely multi-hop
        dpr_doc_ids, _ = self.dense_passage_retrieval(query)
        top_dpr_entities: Set[str] = set()
        for doc_id in dpr_doc_ids[:10].tolist():
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            top_dpr_entities |= self.chunk_to_entities.get(pkey, set())

        diversity_score = len(top_dpr_entities) / max(1, entity_count) if entity_count > 0 else 0

        # Signal 4: DPR top-1 seed-entity coverage (new single-hop signal)
        # For single-hop queries the top-1 DPR doc is usually self-sufficient:
        # it contains most seed entities because the answer lives in a single passage.
        # For 2-hop queries the first-hop doc only covers the "seed side" of the chain,
        # so the fraction of seed entities found in top-1 is lower on average.
        dpr_top1_coverage = 0.0
        if self.pcrag_config.hop_use_dpr_coverage_signal and query_entities and len(dpr_doc_ids) > 0:
            top1_id = int(dpr_doc_ids[0])
            if 0 <= top1_id < len(self.passage_node_keys):
                top1_pkey = self.passage_node_keys[top1_id]
                top1_ents = self.chunk_to_entities.get(top1_pkey, set())
                covered = len(query_entities & top1_ents)
                dpr_top1_coverage = float(covered) / float(len(query_entities))

        # Multi-hop detection (unchanged across both logic paths)
        high_hop_signals = 0
        if keyword_score >= self.pcrag_config.hop_keyword_multi_min:
            high_hop_signals += 1
        if entity_count >= self.pcrag_config.hop_entity_multi_min:
            high_hop_signals += 1
        if diversity_score >= self.pcrag_config.hop_diversity_multi_min:
            high_hop_signals += 1

        if self.pcrag_config.use_hop_scoring_detection:
            # ── New scoring-based single-hop detection (opt-in) ───────────────
            # keyword_score carries double weight (75 % precision on HotpotQA).
            # DPR-coverage is an independent strong predictor: for single-hop
            # queries the answer lives in the top-1 DPR doc, so most seed
            # entities appear there.
            single_hop_score = 0
            if keyword_score <= self.pcrag_config.hop_single_keyword_max:
                single_hop_score += 2
            if entity_count <= self.pcrag_config.hop_single_entity_max:
                single_hop_score += 1
            if diversity_score < self.pcrag_config.hop_single_diversity_max:
                single_hop_score += 1
            if dpr_top1_coverage >= self.pcrag_config.hop_single_dpr_coverage_threshold:
                single_hop_score += 2

            if single_hop_score >= self.pcrag_config.hop_single_min_score:
                hops = 1
            elif high_hop_signals >= self.pcrag_config.hop_multi_min_signals:
                hops = 3
            else:
                hops = 2
        else:
            # ── Legacy AND-based single-hop detection (default, backward-compat) ─
            if keyword_score <= 1 and entity_count <= 2 and diversity_score < 3:
                hops = 1
            elif high_hop_signals >= self.pcrag_config.hop_multi_min_signals:
                hops = 3
            else:
                hops = 2

        return min(self.pcrag_config.hop_force_max, hops)

    def _qcappr_params(self, hops: int) -> Tuple[float, float]:
        if not self.pcrag_config.use_qcappr:
            return self.global_config.damping, 1.0
        if hops <= 1:
            return self.pcrag_config.qcappr_single_hop_damping, self.pcrag_config.qcappr_single_hop_temperature
        if hops == 2:
            return self.pcrag_config.qcappr_two_hop_damping, self.pcrag_config.qcappr_two_hop_temperature
        return self.pcrag_config.qcappr_multi_hop_damping, self.pcrag_config.qcappr_multi_hop_temperature

    @staticmethod
    def _softmax(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
        if not scores:
            return {}
        t = max(1e-4, float(temperature))
        keys = list(scores.keys())
        arr = np.array([scores[k] for k in keys], dtype=np.float32) / t
        arr -= np.max(arr)
        exp = np.exp(arr)
        s = float(np.sum(exp)) + 1e-12
        return {k: float(v / s) for k, v in zip(keys, exp)}

    def _extract_seed_entities_from_facts(self, top_k_facts: List[Tuple]) -> Set[str]:
        seeds: Set[str] = set()
        for fact in top_k_facts:
            if len(fact) < 3:
                continue
            subj = compute_mdhash_id(content=str(fact[0]).lower(), prefix="entity-")
            obj = compute_mdhash_id(content=str(fact[2]).lower(), prefix="entity-")
            if subj in self.node_name_to_vertex_idx:
                seeds.add(subj)
            if obj in self.node_name_to_vertex_idx:
                seeds.add(obj)
        return seeds

    def _build_seed_distribution(self, query: str, seed_entities: Set[str], hops: int) -> Dict[str, float]:
        if not seed_entities:
            return {}

        query_emb = self.query_to_embedding["triple"].get(query)
        if query_emb is None:
            return {}

        raw_scores: Dict[str, float] = {}
        for ent in sorted(seed_entities):
            local_idx = self.entity_key_to_local_idx.get(ent)
            if local_idx is None or local_idx >= len(self.entity_embeddings):
                continue

            ent_emb = self.entity_embeddings[local_idx]
            sim = float(np.dot(ent_emb, query_emb))

            # index-side IDF prior innovation
            if self.pcrag_config.use_entity_idf_index:
                sim *= float(self.entity_idf.get(ent, 1.0))

            # query-conditioned hub penalty
            if self.pcrag_config.use_qcappr:
                v_idx = self.node_name_to_vertex_idx.get(ent)
                if v_idx is not None:
                    deg = float(self.graph.degree(v_idx))
                    sim *= 1.0 / ((1.0 + deg) ** self.pcrag_config.qcappr_hub_penalty_gamma)

            raw_scores[ent] = sim

        _, temp = self._qcappr_params(hops)
        return self._softmax(raw_scores, temp)

    def _rank_bridge_neighbors(self, entity_key: str) -> List[str]:
        v_idx = self.node_name_to_vertex_idx.get(entity_key)
        if v_idx is None:
            return []

        neighbors = self.graph.neighbors(v_idx)
        cands: List[Tuple[str, float]] = []
        for nb_idx in neighbors:
            nb_name = self.graph.vs[nb_idx]["name"]
            if not isinstance(nb_name, str) or not nb_name.startswith("entity-"):
                continue
            deg = float(self.graph.degree(nb_idx))
            idf = float(self.entity_idf.get(nb_name, 1.0))
            score = idf / (1.0 + deg)
            cands.append((nb_name, score))

        cands.sort(key=lambda x: (-x[1], x[0]))
        return [x[0] for x in cands[: self.pcrag_config.bridge_cache_top_k_per_entity]]

    def _detect_bridge_entities(self, query: str, seed_entities: Set[str]) -> Dict[str, List[str]]:
        """
        Enhanced bridge detection with semantic filtering.
        Only keep bridge entities that are:
        1. In first-hop DPR passages (structural signal)
        2. Semantically relevant to query (semantic signal)
        """
        if not self.pcrag_config.use_eba or not seed_entities:
            return {}

        dpr_doc_ids, _ = self.dense_passage_retrieval(query)
        first_hop_doc_ids = dpr_doc_ids[: self.pcrag_config.eba_first_hop_k].tolist()
        first_hop_passage_keys = [self.passage_node_keys[i] for i in first_hop_doc_ids if i < len(self.passage_node_keys)]

        first_hop_entities: Set[str] = set()
        for pkey in first_hop_passage_keys:
            first_hop_entities |= self.chunk_to_entities.get(pkey, set())

        # Get query embedding for semantic filtering
        query_emb = self.query_to_embedding["triple"].get(query)
        
        bridges: Dict[str, List[str]] = {}
        for seed in sorted(seed_entities):
            if seed in self.bridge_neighbor_cache:
                ranked_nb = self.bridge_neighbor_cache[seed]
            else:
                ranked_nb = self._rank_bridge_neighbors(seed)
                if self.pcrag_config.use_bridge_cache_index:
                    self.bridge_neighbor_cache[seed] = ranked_nb

            filtered = []
            for nb in ranked_nb:
                if nb not in first_hop_entities:
                    continue
                if query_emb is not None:
                    nb_idx = self.entity_key_to_local_idx.get(nb)
                    if nb_idx is not None and nb_idx < len(self.entity_embeddings):
                        nb_emb = self.entity_embeddings[nb_idx]
                        sim = float(np.dot(nb_emb, query_emb))
                        if sim < self.pcrag_config.eba_bridge_semantic_threshold:
                            continue
                filtered.append(nb)
            keep = filtered[: self.pcrag_config.eba_bridge_top_k]

            if keep:
                bridges[seed] = keep

        return bridges

    def _query_embedding_ppr_fallback(
        self,
        query: str,
        dpr_sorted_doc_ids: Optional[np.ndarray] = None,
        dpr_sorted_doc_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Fallback retrieval when fact reranking is empty.
        Uses query embedding -> entity seed distribution -> PPR, with light DPR passage anchoring.
        """
        if dpr_sorted_doc_ids is None or dpr_sorted_doc_scores is None:
            dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        query_emb = self.query_to_embedding["triple"].get(query)
        if query_emb is None or len(self.entity_embeddings) == 0:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        ent_scores = np.dot(self.entity_embeddings, query_emb)
        if ent_scores.size == 0:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        top_k = min(self.pcrag_config.query_ppr_seed_top_k, ent_scores.shape[0])
        if top_k <= 0:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        top_local_indices = np.argpartition(ent_scores, -top_k)[-top_k:]
        top_local_indices = top_local_indices[np.argsort(ent_scores[top_local_indices])[::-1]]

        raw_seed_scores: Dict[str, float] = {}
        for local_idx in top_local_indices.tolist():
            if local_idx >= len(self.entity_node_keys):
                continue
            ent_key = self.entity_node_keys[local_idx]
            score = float(ent_scores[local_idx])
            if self.pcrag_config.use_entity_idf_index:
                score *= float(self.entity_idf.get(ent_key, 1.0))
            raw_seed_scores[ent_key] = score

        seed_distribution = self._softmax(raw_seed_scores, temperature=0.7)
        if not seed_distribution:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        num_nodes = len(self.graph.vs["name"])
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)

        for ent_key, prob in seed_distribution.items():
            v_idx = self.node_name_to_vertex_idx.get(ent_key)
            if v_idx is not None:
                phrase_weights[v_idx] += float(prob)

        dpr_norm = min_max_normalize(dpr_sorted_doc_scores)
        for i, doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            pidx = self.node_name_to_vertex_idx.get(pkey)
            if pidx is None:
                continue
            passage_weights[pidx] = (
                float(dpr_norm[i])
                * self.global_config.passage_node_weight
                * self.pcrag_config.query_ppr_passage_weight
            )

        node_weights = phrase_weights + passage_weights
        if float(np.sum(node_weights)) <= 0:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        sorted_doc_ids, sorted_doc_scores = self.run_ppr(
            node_weights,
            damping=self.pcrag_config.query_ppr_damping,
        )
        return sorted_doc_ids, sorted_doc_scores, True

    def _weak_entity_seed_ppr_fallback(
        self,
        query: str,
        dpr_sorted_doc_ids: np.ndarray,
        dpr_sorted_doc_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        No-facts root-cause recovery:
        weak query->entity lexical matching -> soft seed distribution -> PPR.
        """
        if not self.pcrag_config.no_facts_enable_weak_entity_seed:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False
        if not self.entity_surface_norm or not self.entity_token_to_keys:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        q_norm = self._normalize_text(query)
        q_toks = [t for t in q_norm.split() if t]
        if not q_toks:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        # Candidate entities via token inverted index.
        candidate_keys: Set[str] = set()
        for t in set(q_toks):
            keys = self.entity_token_to_keys.get(t)
            if keys:
                candidate_keys.update(keys)
        if not candidate_keys:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        max_n = max(1, int(self.pcrag_config.no_facts_weak_ngram_max_n))
        ngram_set: Set[str] = set()
        n_tok = len(q_toks)
        for n in range(1, min(max_n, n_tok) + 1):
            for i in range(0, n_tok - n + 1):
                ngram_set.add(" ".join(q_toks[i : i + n]))

        scored: Dict[str, float] = {}
        min_score = float(self.pcrag_config.no_facts_weak_seed_min_score)
        for ent_key in sorted(candidate_keys):
            surf = self.entity_surface_norm.get(ent_key, "")
            if not surf:
                continue
            surf_toks = set(surf.split())
            if not surf_toks:
                continue

            overlap = len(surf_toks.intersection(q_toks))
            jacc = overlap / max(1.0, float(len(surf_toks.union(set(q_toks)))))
            exact = 1.0 if surf in q_norm else 0.0
            ngram = 1.0 if surf in ngram_set else 0.0
            score = 1.5 * exact + 1.0 * ngram + 1.5 * jacc
            if self.pcrag_config.use_entity_idf_index:
                score *= float(self.entity_idf.get(ent_key, 1.0))
            if score >= min_score:
                scored[ent_key] = float(score)

        if not scored:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        top_k = max(1, int(self.pcrag_config.no_facts_weak_seed_top_k))
        scored_items = sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        seed_distribution = self._softmax(dict(scored_items), temperature=0.7)
        if not seed_distribution:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        num_nodes = len(self.graph.vs["name"])
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)
        for ent_key, prob in seed_distribution.items():
            v_idx = self.node_name_to_vertex_idx.get(ent_key)
            if v_idx is not None:
                phrase_weights[v_idx] += float(prob)

        dpr_norm = min_max_normalize(dpr_sorted_doc_scores)
        for i, doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            pidx = self.node_name_to_vertex_idx.get(pkey)
            if pidx is None:
                continue
            passage_weights[pidx] = (
                float(dpr_norm[i])
                * self.global_config.passage_node_weight
                * self.pcrag_config.query_ppr_passage_weight
            )

        node_weights = phrase_weights + passage_weights
        if float(np.sum(node_weights)) <= 0:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, False

        sorted_doc_ids, sorted_doc_scores = self.run_ppr(
            node_weights,
            damping=self.pcrag_config.query_ppr_damping,
        )
        return sorted_doc_ids, sorted_doc_scores, True

    def _extract_anchor_entity_from_docs(self, doc_ids: np.ndarray, top_n: int, original_seed_entities: Set[str]) -> str:
        """
        Infer a lightweight intermediate entity from previous sub-question retrieval.
        """
        max_docs = max(1, int(top_n))
        candidates: Dict[str, float] = {}
        for doc_id in doc_ids[:max_docs].tolist():
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            for ent_key in sorted(self.chunk_to_entities.get(pkey, set())):
                if ent_key in original_seed_entities:
                    continue
                candidates[ent_key] = max(candidates.get(ent_key, 0.0), float(self.entity_idf.get(ent_key, 1.0)))
        if not candidates:
            return ""
        best_ent = min(candidates.items(), key=lambda x: (-x[1], x[0]))[0]
        return self._grounded_entity_surface(best_ent)

    @staticmethod
    def _rewrite_sub_question_with_anchor(sub_q: str, anchor_surface: str) -> str:
        if not sub_q or not anchor_surface:
            return sub_q
        # Lightweight pronoun/anaphora replacement without additional LLM calls.
        patterns = [
            r"\bit\b",
            r"\bthis\b",
            r"\bthat\b",
            r"\bthese\b",
            r"\bthose\b",
            r"\bthe same\b",
            r"\bsame one\b",
            r"\bsame entity\b",
        ]
        rewritten = sub_q
        for p in patterns:
            rewritten = re.sub(p, anchor_surface, rewritten, flags=re.IGNORECASE)
        return rewritten

    def _collect_entities_from_top_docs(self, doc_ids: List[int]) -> Set[str]:
        entities: Set[str] = set()
        for doc_id in doc_ids:
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            entities |= self.chunk_to_entities.get(pkey, set())
        return entities

    @staticmethod
    def _merge_rankings(
        num_passages: int,
        base_sorted_ids: np.ndarray,
        base_sorted_scores: np.ndarray,
        extra_sorted_ids: np.ndarray,
        extra_sorted_scores: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha = min(1.0, max(0.0, float(alpha)))
        base_full = np.zeros(num_passages, dtype=np.float32)
        extra_full = np.zeros(num_passages, dtype=np.float32)

        base_norm = np.asarray(min_max_normalize(base_sorted_scores), dtype=np.float32)
        extra_norm = np.asarray(min_max_normalize(extra_sorted_scores), dtype=np.float32)
        base_full[base_sorted_ids] = base_norm
        extra_full[extra_sorted_ids] = extra_norm

        merged = (1.0 - alpha) * base_full + alpha * extra_full
        merged_sorted_ids = np.argsort(merged)[::-1]
        merged_sorted_scores = merged[merged_sorted_ids]
        return merged_sorted_ids, merged_sorted_scores

    def _select_iterative_bridge_seeds(
        self,
        query: str,
        candidate_entities: Set[str],
        original_seed_entities: Set[str],
    ) -> List[Tuple[str, float]]:
        """
        Select focused bridge entity seeds for round-2 iterative retrieval.

        Three modes controlled by ``iterative_seed_mode``:
        - ``idf_novel``: rank by IDF only, keep entities NOT in original query seeds.
          Rationale: bridge entities are *novel* (not mentioned in the query) and
          *specific* (high IDF = rare in corpus).  Similarity to the query is
          intentionally ignored because the bridge entity is the *answer* to the
          first hop, which the query does not yet know.
        - ``idf_only``: rank by IDF only (no novelty filter).
        - ``sim_idf``: original behaviour – rank by sim * IDF.
        """
        mode = getattr(self.pcrag_config, "iterative_seed_mode", "idf_novel")
        novel_only = mode == "idf_novel"

        query_emb = self.query_to_embedding["triple"].get(query) if mode == "sim_idf" else None

        scored: List[Tuple[str, float]] = []
        for ent_key in sorted(candidate_entities):
            if novel_only and ent_key in original_seed_entities:
                continue
            local_idx = self.entity_key_to_local_idx.get(ent_key)
            if local_idx is None or local_idx >= len(self.entity_embeddings):
                continue

            idf = float(self.entity_idf.get(ent_key, 1.0)) if self.pcrag_config.use_entity_idf_index else 1.0

            if mode == "sim_idf" and query_emb is not None:
                sim = float(np.dot(self.entity_embeddings[local_idx], query_emb))
                score = sim * idf
            else:
                score = idf

            scored.append((ent_key, score))

        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[: self.pcrag_config.iterative_round2_seed_top_k]

    def _iterative_round2_search(
        self,
        query: str,
        round1_sorted_doc_ids: np.ndarray,
        round1_sorted_doc_scores: np.ndarray,
        hops: int,
        original_seed_entities: Set[str],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Focused round-2 retrieval seeded by high-IDF novel bridge entities.

        Key change vs. original: instead of taking ALL entities from top-5 docs
        (avg ~40 seeds), we take only the top-k highest-IDF entities from the
        top-1 doc that are NOT already in the query's seed set.  This gives a
        tight, specific signal pointing at the second hop document.
        """
        top_doc_ids = round1_sorted_doc_ids[: self.pcrag_config.iterative_round1_top_docs].tolist()
        candidate_entities = self._collect_entities_from_top_docs(top_doc_ids)
        scored_seeds = self._select_iterative_bridge_seeds(
            query=query,
            candidate_entities=candidate_entities,
            original_seed_entities=original_seed_entities,
        )

        if len(scored_seeds) < self.pcrag_config.iterative_min_seed_entities:
            return round1_sorted_doc_ids, round1_sorted_doc_scores, {
                "iterative_used": False,
                "iterative_round2_seed_count": int(len(scored_seeds)),
            }

        seed_distribution = self._softmax(dict(scored_seeds), temperature=0.7)

        num_nodes = len(self.graph.vs["name"])
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)
        for ent_key, prob in seed_distribution.items():
            v_idx = self.node_name_to_vertex_idx.get(ent_key)
            if v_idx is not None:
                phrase_weights[v_idx] += float(prob)

        round1_norm = min_max_normalize(round1_sorted_doc_scores)
        for i, doc_id in enumerate(round1_sorted_doc_ids.tolist()):
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            pidx = self.node_name_to_vertex_idx.get(pkey)
            if pidx is None:
                continue
            passage_weights[pidx] = float(round1_norm[i]) * self.global_config.passage_node_weight

        node_weights = phrase_weights + passage_weights
        if float(np.sum(node_weights)) <= 0:
            return round1_sorted_doc_ids, round1_sorted_doc_scores, {
                "iterative_used": False,
                "iterative_round2_seed_count": int(len(seed_distribution)),
            }

        if self.pcrag_config.iterative_damping > 0:
            damping = self.pcrag_config.iterative_damping
        else:
            damping, _ = self._qcappr_params(hops)
        round2_sorted_ids, round2_sorted_scores = self.run_ppr(node_weights, damping=damping)

        merged_ids, merged_scores = self._merge_rankings(
            num_passages=len(self.passage_node_keys),
            base_sorted_ids=round1_sorted_doc_ids,
            base_sorted_scores=round1_sorted_doc_scores,
            extra_sorted_ids=round2_sorted_ids,
            extra_sorted_scores=round2_sorted_scores,
            alpha=self.pcrag_config.iterative_merge_alpha,
        )

        return merged_ids, merged_scores, {
            "iterative_used": True,
            "iterative_round2_seed_count": int(len(seed_distribution)),
        }

    def _path_graph_search(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        top_k_facts: List[Tuple],
        top_k_fact_indices: List[int],
        hop_override: Optional[int] = None,
        defer_qd: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        num_nodes = len(self.graph.vs["name"])
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)

        if hop_override is not None and (
            isinstance(hop_override, bool)
            or not isinstance(hop_override, Integral)
            or not 1 <= hop_override <= 4
        ):
            raise ValueError("hop_override must be an integer from 1 through 4")
        hops = int(hop_override) if hop_override is not None else self._estimate_query_hops(query, top_k_facts=top_k_facts)
        seed_entities = self._extract_seed_entities_from_facts(top_k_facts)
        seed_distribution = self._build_seed_distribution(query, seed_entities, hops)
        fact_seed_distribution = {}
        if getattr(self.pcrag_config, "improvement_stage", 0) >= 2:
            # Retain relation relevance that the original reset discarded.
            # Correct for endpoint frequency, normalize, then blend with the
            # existing semantic entity prior without changing total reset mass.
            for fact_idx, fact in zip(top_k_fact_indices, top_k_facts):
                relevance = max(0.0, float(query_fact_scores[int(fact_idx)]))
                for entity in self._extract_seed_entities_from_facts([fact]):
                    if entity in seed_distribution:
                        degree = max(1, len(self.ent_node_to_chunk_ids.get(entity, [])))
                        fact_seed_distribution[entity] = fact_seed_distribution.get(entity, 0.0) + relevance / math.sqrt(degree)
            total = sum(fact_seed_distribution.values())
            if total > 0:
                fact_seed_distribution = {k: v / total for k, v in fact_seed_distribution.items()}
                seed_distribution = {k: 0.5 * v + 0.5 * fact_seed_distribution.get(k, 0.0)
                                     for k, v in seed_distribution.items()}

        # QCAPPR reset initialization
        for ent, prob in seed_distribution.items():
            v_idx = self.node_name_to_vertex_idx.get(ent)
            if v_idx is not None:
                phrase_weights[v_idx] += prob

        # EBA bridge augmentation
        bridges_by_seed = self._detect_bridge_entities(query, seed_entities)
        bridge_count = 0
        for seed, bridge_entities in sorted(bridges_by_seed.items()):
            seed_w = seed_distribution.get(seed, 0.0)
            for b in bridge_entities:
                b_idx = self.node_name_to_vertex_idx.get(b)
                if b_idx is None:
                    continue
                else:
                    bridge_weight = self.pcrag_config.eba_bridge_weight * max(seed_w, 1e-5)
                phrase_weights[b_idx] += bridge_weight
                bridge_count += 1


        # Keep DPR passage anchoring
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        dpr_norm = min_max_normalize(dpr_sorted_doc_scores)
        for i, doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            pidx = self.node_name_to_vertex_idx.get(pkey)
            if pidx is None:
                continue
            base_w = float(dpr_norm[i]) * self.global_config.passage_node_weight
            passage_weights[pidx] = base_w



        node_weights = phrase_weights + passage_weights
        if float(np.sum(node_weights)) <= 0:
            # safety fallback
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores, {
                "hops": hops,
                "seed_entities": list(seed_entities),
                "seed_distribution": seed_distribution,
                "bridges_by_seed": bridges_by_seed,
                "bridge_count": bridge_count,
                "iterative_used": False,
                "iterative_round2_seed_count": 0,
            }

        damping, _ = self._qcappr_params(hops)
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=damping)
        ctx = {
            "hops": hops,
            "seed_entities": list(seed_entities),
            "seed_distribution": seed_distribution,
            "fact_seed_distribution": fact_seed_distribution,
            "bridges_by_seed": bridges_by_seed,
            "bridge_count": bridge_count,
            "iterative_used": False,
            "iterative_round2_seed_count": 0,
            "qd_used": False,
                        "qd_sub_questions": 0,
                        "qd_sub_query_ppr_used": 0,
                        "qd_sub_query_ppr_failed": 0,
                        "qd_seq_rewrites": 0,
            "pcqd_used": False,
            "pcqd_sub_questions": 0,
            "pcqd_path_hints": 0,
            "pcqd_entity_replacement_num": 0.0,
            "pcqd_entity_replacement_den": 0.0,
            "pcqd_entity_replacement_rate": 0.0,
            "pcqd_fallback": False,
            "pcqd_conflict_rate": 0.0,
        }

        current_ids, current_scores = ppr_sorted_doc_ids, ppr_sorted_doc_scores

        if self.pcrag_config.use_iterative_retrieval:
            current_ids, current_scores, iter_stats = self._iterative_round2_search(
                query=query,
                round1_sorted_doc_ids=current_ids,
                round1_sorted_doc_scores=current_scores,
                hops=hops,
                original_seed_entities=seed_entities,
            )
            ctx.update(iter_stats)

        if self.pcrag_config.use_query_decomposition:
            if defer_qd:
                ctx["_qd_deferred"] = True
            else:
                current_ids, current_scores, qd_stats = self._query_decomposition_retrieval(
                    query=query,
                    hops=hops,
                    base_sorted_doc_ids=current_ids,
                    base_sorted_doc_scores=current_scores,
                    seed_entities=seed_entities,
                    seed_distribution=seed_distribution,
                    bridges_by_seed=bridges_by_seed,
                )
                ctx.update(qd_stats)

        return current_ids, current_scores, ctx

    # ---------------------------------------------------------------------
    # Query Decomposition
    # ---------------------------------------------------------------------
    _QD_SYSTEM_PROMPT = (
        "You are a multi-hop question decomposer. "
        "Given a complex question, break it into sequential sub-questions "
        "that can each be answered by a single passage. "
        "Output ONLY a JSON object with key \"sub_questions\" containing a list of strings. "
        "Keep sub-questions concise and self-contained. "
        "Do NOT include the original question in the list."
    )
    _PCQD_SYSTEM_PROMPT = (
        "You are decomposing a multi-hop question with retrieved reasoning hints. "
        "Produce grounded sub-questions using evidence-backed entities only. "
        "Do not hallucinate entities. "
        "Return JSON with key \"sub_questions\"; each item should be an object with "
        "fields: question (string), grounded_entities (list), supporting_hint_ids (list), confidence (0-1)."
    )

    @staticmethod
    def _extract_llm_text(response_obj: Any) -> str:
        """Best-effort extraction of plain text from heterogeneous LLM return formats."""
        if response_obj is None:
            return ""
        if isinstance(response_obj, str):
            return response_obj
        if isinstance(response_obj, dict):
            for key in ("content", "text", "message"):
                val = response_obj.get(key)
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    nested = PCRAG._extract_llm_text(val)
                    if nested:
                        return nested
            return str(response_obj)
        if isinstance(response_obj, list):
            if not response_obj:
                return ""
            # OpenAI-style list of messages / choices.
            first = response_obj[0]
            txt = PCRAG._extract_llm_text(first)
            if txt:
                return txt
            return "".join(PCRAG._extract_llm_text(x) for x in response_obj if x is not None)
        return str(response_obj)

    @staticmethod
    def _parse_sub_questions_from_text(raw_text: str, max_sub_questions: int) -> List[str]:
        """
        Parse sub-questions from JSON-ish LLM output.
        Accepts strict JSON, fenced JSON, and lightly noisy wrappers.
        """
        text = (raw_text or "").strip()
        if not text:
            return []

        candidates: List[str] = [text]
        # Try extracting the largest JSON object span as a backup.
        left = text.find("{")
        right = text.rfind("}")
        if 0 <= left < right:
            candidates.append(text[left:right + 1])

        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except Exception:
                continue
            sub_qs = parsed.get("sub_questions", []) if isinstance(parsed, dict) else []
            if not isinstance(sub_qs, list):
                continue
            cleaned = [str(q).strip() for q in sub_qs if str(q).strip()]
            if cleaned:
                return cleaned[:max_sub_questions]

        # Final fallback: treat lines as potential sub-questions.
        line_like: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # strip bullet/enumeration prefix
            line = line.lstrip("-*0123456789. )(").strip()
            if line:
                line_like.append(line)
        return line_like[:max_sub_questions]

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _entity_surface(ent_key: str) -> str:
        if not isinstance(ent_key, str):
            return ""
        return ent_key[7:] if ent_key.startswith("entity-") else ent_key

    def _grounded_entity_surface(self, ent_key: str) -> str:
        if getattr(self.pcrag_config, "improvement_stage", 0) >= 1:
            label = self._entity_label(ent_key)
            if re.fullmatch(r"(?:entity-)?[0-9a-fA-F]{32}", label or ""):
                return ""
            return label
        return self._entity_surface(ent_key)

    def _entity_supported_in_evidence(self, ent_key: str, evidence_text: str) -> bool:
        ent = self._normalize_text(self._grounded_entity_surface(ent_key))
        evidence = self._normalize_text(evidence_text)
        if not ent or not evidence:
            return False
        return ent in evidence

    def _decompose_query(self, query: str, max_sub_questions: int) -> List[str]:
        """
        Use the LLM to decompose a multi-hop query into ordered sub-questions.
        Results are cached in ``_qd_cache`` to avoid redundant LLM calls.
        """
        if not hasattr(self, "_qd_cache"):
            self._qd_cache: Dict[Tuple[str, int], List[str]] = {}

        cache_key = (query, max_sub_questions)
        if self.pcrag_config.qd_cache_decompositions and cache_key in self._qd_cache:
            return self._qd_cache[cache_key]

        messages = [
            {"role": "system", "content": self._QD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Decompose into at most {max_sub_questions} sub-questions:\n{query}"
                ),
            },
        ]

        try:
            # Do NOT pass response_format=json_object: vLLM guided decoding
            # (xgrammar) can crash the whole engine on this stack. Prompt + parser
            # already expect JSON text.
            infer_ret = self.llm_model.infer(
                messages=messages,
                temperature=self.pcrag_config.qd_llm_temperature,
            )

            # Compatible with backends returning:
            # (message, metadata), (message, metadata, cache_hit), or bare message
            response_obj = infer_ret
            if isinstance(infer_ret, tuple):
                if len(infer_ret) >= 1:
                    response_obj = infer_ret[0]
                else:
                    response_obj = ""

            raw = self._extract_llm_text(response_obj)
            sub_qs = self._parse_sub_questions_from_text(raw, max_sub_questions=max_sub_questions)
        except Exception as e:
            logger.exception("[QD] decompose_query failed for %r: %s", query[:60], e)
            raise

        if self.pcrag_config.qd_cache_decompositions:
            self._qd_cache[cache_key] = sub_qs
        return sub_qs

    def _decompose_query_with_hints(
        self,
        query: str,
        path_hints: List[Dict[str, Any]],
        max_sub_questions: int,
    ) -> List[Dict[str, Any]]:
        if not path_hints:
            return []

        if not hasattr(self, "_pcqd_cache"):
            self._pcqd_cache: Dict[str, List[Dict[str, Any]]] = {}
        cache_key = f"{query}::{json.dumps(path_hints, ensure_ascii=False, sort_keys=True)}::{max_sub_questions}"
        if self.pcrag_config.qd_cache_decompositions and cache_key in self._pcqd_cache:
            return self._pcqd_cache[cache_key]

        hint_lines: List[str] = []
        for i, h in enumerate(path_hints):
            src = str(h.get("source_entity", ""))
            bridge = str(h.get("candidate_answer_or_bridge", ""))
            evidence = str(h.get("evidence", ""))
            conf = float(h.get("confidence", 0.0))
            hint_lines.append(
                f"[{i}] {src} -> {bridge}; confidence={conf:.3f}; evidence={evidence}"
            )

        grounding_instruction = (
            "Replace ambiguous mentions using supported entities from hints."
            if not self.pcrag_config.pcqd_disable_entity_grounding
            else "Do not force explicit entity replacement; only rewrite with clearer wording."
        )

        user_prompt = (
            f"Original question:\n{query}\n\n"
            f"Retrieved reasoning hints:\n" + "\n".join(hint_lines) + "\n\n"
            f"Task: generate at most {max_sub_questions} grounded sub-questions.\n"
            f"Rules:\n"
            f"1. Use entities from hints only when evidence supports them.\n"
            f"2. {grounding_instruction}\n"
            f"3. Do not invent unsupported entities.\n"
            f"4. Return JSON with key sub_questions."
        )
        messages = [
            {"role": "system", "content": self._PCQD_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        structured_sub_qs: List[Dict[str, Any]] = []
        raw = ""
        validation_rejections = []
        try:
            # Avoid response_format=json_object — guided decoding can kill vLLM.
            infer_ret = self.llm_model.infer(
                messages=messages,
                temperature=self.pcrag_config.qd_llm_temperature,
            )
            response_obj = infer_ret[0] if isinstance(infer_ret, tuple) and len(infer_ret) >= 1 else infer_ret
            raw = self._extract_llm_text(response_obj)

            candidates = [raw]
            left, right = raw.find("{"), raw.rfind("}")
            if 0 <= left < right:
                candidates.append(raw[left:right + 1])

            for cand in candidates:
                try:
                    parsed = json.loads(cand)
                except Exception:
                    continue
                sq = parsed.get("sub_questions", []) if isinstance(parsed, dict) else []
                if not isinstance(sq, list):
                    continue
                for item in sq:
                    if isinstance(item, str):
                        q = item.strip()
                        if q:
                            structured_sub_qs.append(
                                {
                                    "question": q,
                                    "grounded_entities": [],
                                    "supporting_hint_ids": [],
                                    "confidence": 0.5,
                                }
                            )
                    elif isinstance(item, dict):
                        q = str(item.get("question", "")).strip()
                        if not q:
                            continue
                        ge = item.get("grounded_entities", [])
                        if not isinstance(ge, list):
                            ge = []
                        sh = item.get("supporting_hint_ids", [])
                        if not isinstance(sh, list):
                            sh = []
                        invalid_hint_reference = any(
                            isinstance(x, bool) or not str(x).strip().isdigit()
                            or int(x) >= len(path_hints) for x in sh)
                        conf = item.get("confidence", 0.5)
                        try:
                            conf = float(conf)
                        except Exception:
                            conf = 0.5
                        structured_sub_qs.append(
                            {
                                "question": q,
                                "grounded_entities": [str(x) for x in ge if str(x).strip()],
                                "supporting_hint_ids": [int(x) for x in sh if str(x).strip().isdigit()],
                                "invalid_hint_reference": invalid_hint_reference,
                                "confidence": min(1.0, max(0.0, conf)),
                            }
                        )
                if structured_sub_qs:
                    break
        except Exception as e:
            logger.exception("[PCQD] decompose with hints failed for %r: %s", query[:60], e)
            raise

        if not structured_sub_qs:
            fallback_qs = self._decompose_query(query, max_sub_questions=max_sub_questions)
            structured_sub_qs = [
                {
                    "question": q,
                    "grounded_entities": [],
                    "supporting_hint_ids": [],
                    "confidence": 0.5,
                }
                for q in fallback_qs
            ]

        if getattr(self.pcrag_config, "improvement_stage", 0) >= 1:
            validated = []
            for item in structured_sub_qs:
                ids = item["supporting_hint_ids"]
                if item.get("invalid_hint_reference") or any(type(i) is not int or i < 0 or i >= len(path_hints) for i in ids):
                    validation_rejections.append({"question": item["question"], "reason": "invalid_hint_reference"})
                    continue
                entities = item["grounded_entities"]
                if entities and not ids:
                    # Repair omitted citations only using exact supported labels.
                    ids = [i for i, h in enumerate(path_hints) if any(
                        self._normalize_text(ent) in self._normalize_text(str(h.get("evidence", "")))
                        for ent in entities)]
                referenced = " ".join(str(path_hints[i].get("evidence", "")) for i in ids)
                if any(not self._normalize_text(ent) or
                       self._normalize_text(ent) not in self._normalize_text(referenced)
                       or re.fullmatch(r"(?:entity-)?[0-9a-fA-F]{32}", ent)
                       for ent in entities):
                    validation_rejections.append({"question": item["question"], "reason": "unsupported_entity_binding"})
                    continue
                item["supporting_hint_ids"] = ids
                item["evidence_validated"] = True
                validated.append(item)
            structured_sub_qs = validated
        structured_sub_qs = structured_sub_qs[:max_sub_questions]
        if not hasattr(self, "_pcqd_diagnostics_by_query"):
            self._pcqd_diagnostics_by_query = {}
        self._pcqd_diagnostics_by_query[query] = {"raw_response": raw,
                                                "validation_rejections": validation_rejections,
                                                "validated_sub_questions": structured_sub_qs}
        if self.pcrag_config.qd_cache_decompositions:
            self._pcqd_cache[cache_key] = structured_sub_qs
        return structured_sub_qs

    def _build_pcqd_path_hints(
        self,
        base_sorted_doc_ids: np.ndarray,
        base_sorted_doc_scores: np.ndarray,
        seed_distribution: Dict[str, float],
        bridges_by_seed: Dict[str, List[str]],
    ) -> Tuple[List[Dict[str, Any]], float]:
        corrected = getattr(self.pcrag_config, "improvement_stage", 0) >= 1
        self._last_pcqd_hint_stats = {"strict_accepted": 0, "relaxed_accepted": 0,
                                      "unsupported_rejected": 0, "score_rejected": 0}
        if not seed_distribution:
            return [], 0.0

        top_n = min(self.pcrag_config.pcqd_ground_top_docs, len(base_sorted_doc_ids))
        if top_n <= 0:
            return [], 0.0

        top_doc_ids = base_sorted_doc_ids[:top_n].tolist()
        top_doc_scores = np.asarray(min_max_normalize(base_sorted_doc_scores[:top_n]), dtype=np.float32)

        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        top_doc_context: List[Tuple[int, str, Set[str], str, float]] = []
        for rank, doc_id in enumerate(top_doc_ids):
            if doc_id >= len(self.passage_node_keys):
                continue
            passage_key = self.passage_node_keys[doc_id]
            doc_entities = self.chunk_to_entities.get(passage_key, set())
            row = self.chunk_embedding_store.get_row(passage_key)
            evidence_text = str(row.get("content", "")) if isinstance(row, dict) else ""
            path_score = float(top_doc_scores[rank]) if rank < len(top_doc_scores) else 0.0
            top_doc_context.append((doc_id, passage_key, doc_entities, evidence_text, path_score))

        def _add_hint(
            seed: str,
            bridge: str,
            passage_key: str,
            evidence_text: str,
            path_score: float,
            strict: bool,
            hint_type: str = "adhoc",
        ):
            if strict and not self.pcrag_config.pcqd_disable_path_filtering:
                if path_score < self.pcrag_config.pcqd_path_score_threshold:
                    self._last_pcqd_hint_stats["score_rejected"] += 1
                    return
                if not self._entity_supported_in_evidence(bridge, evidence_text):
                    self._last_pcqd_hint_stats["unsupported_rejected"] += 1
                    return

            source_label = self._grounded_entity_surface(seed)
            bridge_label = self._grounded_entity_surface(bridge)
            if corrected and (not source_label or not bridge_label or
                              not self._entity_supported_in_evidence(bridge, evidence_text)):
                self._last_pcqd_hint_stats["unsupported_rejected"] += 1
                return
            self._last_pcqd_hint_stats["strict_accepted" if strict else "relaxed_accepted"] += 1
            snippet = evidence_text[:400]
            if corrected:
                location = evidence_text.casefold().find(bridge_label.casefold())
                if location >= 0:
                    snippet = evidence_text[max(0, location - 160): location + len(bridge_label) + 240]

            key = (seed, bridge)
            if key not in grouped:
                grouped[key] = {
                    "_seed_key": seed,
                    "_bridge_key": bridge,
                    "source_entity": source_label,
                    "candidate_answer_or_bridge": bridge_label,
                    "evidence": snippet,
                    "confidence": 0.0,
                    "votes": 0,
                    "max_path_score": 0.0,
                    "support_passages": [],
                    "hint_type": hint_type if strict else "relaxed",
                }
            grouped[key]["votes"] += 1
            grouped[key]["max_path_score"] = max(grouped[key]["max_path_score"], path_score)
            grouped[key]["confidence"] = max(
                grouped[key]["confidence"],
                0.7 * path_score + 0.3 * float(seed_distribution.get(seed, 0.0)),
            )
            grouped[key]["support_passages"].append(passage_key)


        # Pass 1: use seed->bridge candidates from EBA.
        for _, passage_key, doc_entities, evidence_text, path_score in top_doc_context:
            for seed, bridge_entities in sorted(bridges_by_seed.items()):
                for bridge in bridge_entities:
                    if bridge not in doc_entities:
                        continue
                    _add_hint(seed, bridge, passage_key, evidence_text, path_score, strict=True)

        # Pass 2: if no hints survived, mine bridge candidates directly from top docs.
        if not grouped:
            seed_keys = set(seed_distribution.keys())
            for _, passage_key, doc_entities, evidence_text, path_score in top_doc_context:
                doc_seed_entities = sorted(seed_keys & doc_entities)
                if not doc_seed_entities:
                    continue
                bridge_pool = sorted(
                    doc_entities - seed_keys,
                    key=lambda e: (-float(self.entity_idf.get(e, 1.0)), e),
                )
                bridge_pool = bridge_pool[: max(1, self.pcrag_config.pcqd_entity_top_k)]
                for seed in doc_seed_entities:
                    for bridge in bridge_pool:
                        _add_hint(seed, bridge, passage_key, evidence_text, path_score, strict=True)

        # Pass 3: controlled relaxation to avoid total fallback.
        if not grouped and not self.pcrag_config.pcqd_disable_path_filtering:
            relaxed_docs = top_doc_context[:1]
            seed_keys = set(seed_distribution.keys())
            for _, passage_key, doc_entities, evidence_text, path_score in relaxed_docs:
                doc_seed_entities = sorted(seed_keys & doc_entities)
                if not doc_seed_entities:
                    doc_seed_entities = sorted(seed_keys)[:1]
                bridge_pool = sorted(
                    doc_entities - seed_keys,
                    key=lambda e: (-float(self.entity_idf.get(e, 1.0)), e),
                )
                bridge_pool = bridge_pool[: max(1, self.pcrag_config.pcqd_entity_top_k)]
                for seed in doc_seed_entities:
                    for bridge in bridge_pool:
                        _add_hint(seed, bridge, passage_key, evidence_text, path_score, strict=False)

        hints = list(grouped.values())
        pre_vote_bridges = defaultdict(set)
        for hint in hints:
            pre_vote_bridges[hint["_seed_key"]].add(hint["_bridge_key"])
        pre_vote_conflict = sum(len(b) > 1 for b in pre_vote_bridges.values()) / max(1, len(pre_vote_bridges))
        if hints and self.pcrag_config.pcqd_enable_bridge_voting:
            per_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for h in hints:
                per_source[str(h.get("_seed_key", ""))].append(h)

            voted_hints: List[Dict[str, Any]] = []
            for src in sorted(per_source):
                cands = per_source[src]
                cands.sort(
                    key=lambda x: (
                        -int(x.get("votes", 0)),
                        -float(x.get("confidence", 0.0)),
                        -float(x.get("max_path_score", 0.0)),
                        -float(self.entity_idf.get(str(x.get("_bridge_key", "")), 1.0)),
                        str(x.get("_bridge_key", "")),
                    ),
                )
                keep = cands[: self.pcrag_config.pcqd_max_bridges_per_source]
                voted_hints.extend(keep)
            hints = voted_hints

        hints.sort(key=lambda x: (
            -float(x["confidence"]),
            -int(x["votes"]),
            str(x.get("_seed_key", "")),
            str(x.get("_bridge_key", "")),
        ))

        if self.pcrag_config.pcqd_entity_top_k > 0:
            hints = hints[: self.pcrag_config.pcqd_entity_top_k]

        source_to_bridges_selected: Dict[str, Set[str]] = defaultdict(set)
        for h in hints:
            src = str(h.get("_seed_key", ""))
            br = str(h.get("_bridge_key", ""))
            if src and br:
                source_to_bridges_selected[src].add(br)

        for i, h in enumerate(hints):
            h["hint_id"] = i
            h.pop("_seed_key", None)
            h.pop("_bridge_key", None)

        total_sources = sum(1 for _, bridges in source_to_bridges_selected.items() if len(bridges) >= 1)
        conflict_sources = sum(1 for _, bridges in source_to_bridges_selected.items() if len(bridges) > 1)
        conflict_rate = float(conflict_sources / max(1, total_sources))
        if corrected:
            conflict_rate = float(pre_vote_conflict)
        self._last_pcqd_hint_stats["conflict_before_voting"] = float(pre_vote_conflict)
        self._last_pcqd_hint_stats["selected_hints"] = len(hints)
        return hints, conflict_rate

    def _build_qd_track_ranking(
        self,
        query: str,
        hops: int,
        sub_questions: List[str],
        base_sorted_doc_ids: np.ndarray,
        original_seed_entities: Set[str],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        sub_qs = [q.strip() for q in sub_questions if isinstance(q, str) and q.strip()]
        if not sub_qs:
            return None, None, {
                "used": False,
                "seed_count": 0,
                "sub_doc_count": 0,
                "sub_retrieval_mode": self.pcrag_config.qd_sub_retrieval_mode,
                "sub_query_ppr_used": 0,
                "sub_query_ppr_failed": 0,
                "sub_seq_rewrites": 0,
            }

        sub_doc_ids_all: List[int] = []
        local_doc_weights = defaultdict(float)
        ppr_used_count = 0
        ppr_failed_count = 0
        seq_rewrite_count = 0
        retrieval_mode = self.pcrag_config.qd_sub_retrieval_mode
        sequential_anchor_surface = ""
        for sub_q in sub_qs:
            retrieval_q = sub_q
            if self.pcrag_config.qd_enable_sequential_dependency and sequential_anchor_surface:
                rewritten = self._rewrite_sub_question_with_anchor(sub_q, sequential_anchor_surface)
                if rewritten != sub_q:
                    retrieval_q = rewritten
                    seq_rewrite_count += 1

            sub_ids: Optional[np.ndarray] = None
            if retrieval_mode == "query_ppr":
                try:
                    sub_ppr_ids, _sub_ppr_scores, used_query_ppr = self._query_embedding_ppr_fallback(retrieval_q)
                    if used_query_ppr:
                        sub_ids = sub_ppr_ids
                        ppr_used_count += 1
                    else:
                        ppr_failed_count += 1
                except Exception as e:
                    ppr_failed_count += 1
                    logger.warning("[QD] query_ppr failed for sub-question %r: %s", retrieval_q[:60], e)

            if sub_ids is None:
                if retrieval_mode == "query_ppr" and not self.pcrag_config.qd_sub_query_ppr_fallback_to_dpr:
                    continue
                try:
                    sub_dpr_ids, _ = self.dense_passage_retrieval(retrieval_q)
                    sub_ids = sub_dpr_ids
                except Exception as e:
                    logger.warning("[QD] DPR failed for sub-question %r: %s", retrieval_q[:60], e)
                    continue

            local_ids = sub_ids[: self.pcrag_config.qd_sub_retrieval_top_k].tolist()
            sub_doc_ids_all.extend(local_ids)
            for rank, doc_id in enumerate(local_ids):
                local_doc_weights[doc_id] += 1.0 / (1.0 + rank) / len(sub_qs)
            if self.pcrag_config.qd_enable_sequential_dependency:
                sequential_anchor_surface = self._extract_anchor_entity_from_docs(
                    sub_ids,
                    top_n=self.pcrag_config.qd_sequential_top_docs_for_anchor,
                    original_seed_entities=original_seed_entities,
                )

        if not sub_doc_ids_all:
            return None, None, {
                "used": False,
                "seed_count": 0,
                "sub_doc_count": 0,
                "sub_retrieval_mode": retrieval_mode,
                "sub_query_ppr_used": ppr_used_count,
                "sub_query_ppr_failed": ppr_failed_count,
                "sub_seq_rewrites": seq_rewrite_count,
            }

        sub_doc_ids_unique = list(dict.fromkeys(sub_doc_ids_all))
        sub_entities = self._collect_entities_from_top_docs(sub_doc_ids_unique)
        scored_seeds = self._select_iterative_bridge_seeds(
            query=query,
            candidate_entities=sub_entities,
            original_seed_entities=original_seed_entities,
        )
        if not scored_seeds:
            return None, None, {
                "used": False,
                "seed_count": 0,
                "sub_doc_count": len(sub_doc_ids_unique),
                "sub_retrieval_mode": retrieval_mode,
                "sub_query_ppr_used": ppr_used_count,
                "sub_query_ppr_failed": ppr_failed_count,
                "sub_seq_rewrites": seq_rewrite_count,
            }

        seed_distribution = self._softmax(dict(scored_seeds), temperature=0.7)
        num_nodes = len(self.graph.vs["name"])
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)

        for ent_key, prob in seed_distribution.items():
            v_idx = self.node_name_to_vertex_idx.get(ent_key)
            if v_idx is not None:
                phrase_weights[v_idx] += float(prob)

        sub_scores = np.zeros(len(self.passage_node_keys), dtype=np.float32)
        for rank, doc_id in enumerate(sub_doc_ids_unique):
            if doc_id >= len(self.passage_node_keys):
                continue
            sub_scores[doc_id] = (local_doc_weights[doc_id] if getattr(self.pcrag_config, "improvement_stage", 0) >= 1
                                  else 1.0 / (1.0 + rank))
        sub_scores_norm = np.asarray(min_max_normalize(sub_scores), dtype=np.float32)

        for doc_id in sub_doc_ids_unique:
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            pidx = self.node_name_to_vertex_idx.get(pkey)
            if pidx is None:
                continue
            passage_weights[pidx] = float(sub_scores_norm[doc_id]) * self.global_config.passage_node_weight

        node_weights = phrase_weights + passage_weights
        if float(np.sum(node_weights)) <= 0:
            return None, None, {
                "used": False,
                "seed_count": len(seed_distribution),
                "sub_doc_count": len(sub_doc_ids_unique),
                "sub_retrieval_mode": retrieval_mode,
                "sub_query_ppr_used": ppr_used_count,
                "sub_query_ppr_failed": ppr_failed_count,
                "sub_seq_rewrites": seq_rewrite_count,
            }

        damping, _ = self._qcappr_params(hops)
        track_ids, track_scores = self.run_ppr(node_weights, damping=damping)
        return track_ids, track_scores, {
            "used": True,
            "seed_count": len(seed_distribution),
            "sub_doc_count": len(sub_doc_ids_unique),
            "sub_retrieval_mode": retrieval_mode,
            "sub_query_ppr_used": ppr_used_count,
            "sub_query_ppr_failed": ppr_failed_count,
            "sub_seq_rewrites": seq_rewrite_count,
        }

    @staticmethod
    def _track_scores_to_full(num_passages: int, sorted_ids: np.ndarray, sorted_scores: np.ndarray) -> np.ndarray:
        full = np.zeros(num_passages, dtype=np.float32)
        norm_scores = np.asarray(min_max_normalize(sorted_scores), dtype=np.float32)
        full[sorted_ids] = norm_scores
        return full

    def _query_decomposition_retrieval(
        self,
        query: str,
        hops: int,
        base_sorted_doc_ids: np.ndarray,
        base_sorted_doc_scores: np.ndarray,
        seed_entities: Set[str],
        seed_distribution: Dict[str, float],
        bridges_by_seed: Dict[str, List[str]],
        prefetched_qd: Optional[
            Tuple[List[str], Optional[List[Dict[str, Any]]], List[Dict[str, Any]], float]
        ] = None,
        prefetched_static_track: Optional[
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]
        ] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if hops < self.pcrag_config.qd_min_hops:
            return base_sorted_doc_ids, base_sorted_doc_scores, {
                "qd_used": False,
                "qd_sub_questions": 0,
                "qd_sub_query_ppr_used": 0,
                "qd_sub_query_ppr_failed": 0,
                "qd_seq_rewrites": 0,
                "pcqd_used": False,
                "pcqd_sub_questions": 0,
                "pcqd_path_hints": 0,
                "pcqd_entity_replacement_num": 0.0,
                "pcqd_entity_replacement_den": 0.0,
                "pcqd_entity_replacement_rate": 0.0,
                "pcqd_fallback": False,
                "pcqd_conflict_rate": 0.0,
            }

        max_sub_questions = self._qd_subquestion_limit(hops)
        static_sub_questions = (
            prefetched_qd[0]
            if prefetched_qd is not None
            else self._decompose_query(query, max_sub_questions)
        )
        if not static_sub_questions:
            return base_sorted_doc_ids, base_sorted_doc_scores, {
                "qd_used": False,
                "qd_sub_questions": 0,
                "qd_sub_query_ppr_used": 0,
                "qd_sub_query_ppr_failed": 0,
                "qd_seq_rewrites": 0,
                "pcqd_used": False,
                "pcqd_sub_questions": 0,
                "pcqd_path_hints": 0,
                "pcqd_entity_replacement_num": 0.0,
                "pcqd_entity_replacement_den": 0.0,
                "pcqd_entity_replacement_rate": 0.0,
                "pcqd_fallback": False,
                "pcqd_conflict_rate": 0.0,
            }

        static_ids, static_scores, static_stats = (
            prefetched_static_track
            if prefetched_static_track is not None
            else self._build_qd_track_ranking(
                query=query,
                hops=hops,
                sub_questions=static_sub_questions,
                base_sorted_doc_ids=base_sorted_doc_ids,
                original_seed_entities=set(seed_entities),
            )
        )
        static_used = bool(static_stats.get("used", False) and static_ids is not None and static_scores is not None)
        if not static_used:
            return base_sorted_doc_ids, base_sorted_doc_scores, {
                "qd_used": False,
                "qd_sub_questions": len(static_sub_questions),
                "qd_sub_query_ppr_used": int(static_stats.get("sub_query_ppr_used", 0)),
                "qd_sub_query_ppr_failed": int(static_stats.get("sub_query_ppr_failed", 0)),
                "qd_seq_rewrites": int(static_stats.get("sub_seq_rewrites", 0)),
                "pcqd_used": False,
                "pcqd_sub_questions": 0,
                "pcqd_path_hints": 0,
                "pcqd_entity_replacement_num": 0.0,
                "pcqd_entity_replacement_den": 0.0,
                "pcqd_entity_replacement_rate": 0.0,
                "pcqd_fallback": False,
                "pcqd_conflict_rate": 0.0,
            }

        # Old behavior: base + static QD merge
        if not self.pcrag_config.use_path_conditioned_qd:
            merged_ids, merged_scores = self._merge_rankings(
                num_passages=len(self.passage_node_keys),
                base_sorted_ids=base_sorted_doc_ids,
                base_sorted_scores=base_sorted_doc_scores,
                extra_sorted_ids=static_ids,
                extra_sorted_scores=static_scores,
                alpha=self.pcrag_config.qd_merge_alpha,
            )
            return merged_ids, merged_scores, {
                "qd_used": True,
                "qd_sub_questions": len(static_sub_questions),
                "qd_sub_query_ppr_used": int(static_stats.get("sub_query_ppr_used", 0)),
                "qd_sub_query_ppr_failed": int(static_stats.get("sub_query_ppr_failed", 0)),
                "qd_seq_rewrites": int(static_stats.get("sub_seq_rewrites", 0)),
                "pcqd_used": False,
                "pcqd_sub_questions": 0,
                "pcqd_path_hints": 0,
                "pcqd_entity_replacement_num": 0.0,
                "pcqd_entity_replacement_den": 0.0,
                "pcqd_entity_replacement_rate": 0.0,
                "pcqd_fallback": False,
                "pcqd_conflict_rate": 0.0,
            }

        # PC-QD path hints
        if prefetched_qd is not None:
            path_hints, conflict_rate = prefetched_qd[2], prefetched_qd[3]
        else:
            path_hints, conflict_rate = self._build_pcqd_path_hints(
                base_sorted_doc_ids=base_sorted_doc_ids,
                base_sorted_doc_scores=base_sorted_doc_scores,
                seed_distribution=seed_distribution,
                bridges_by_seed=bridges_by_seed,
            )
        pcqd_fallback = len(path_hints) == 0
        pcqd_structured_sub_qs: List[Dict[str, Any]] = []
        if path_hints:
            pcqd_structured_sub_qs = (
                prefetched_qd[1]
                if prefetched_qd is not None and prefetched_qd[1] is not None
                else self._decompose_query_with_hints(
                    query=query,
                    path_hints=path_hints,
                    max_sub_questions=max_sub_questions,
                )
            )

        pcqd_questions: List[str] = []
        replacement_num = 0.0
        replacement_den = 0.0
        for item in pcqd_structured_sub_qs:
            q = str(item.get("question", "")).strip()
            if not q:
                continue
            pcqd_questions.append(q)
            replacement_den += 1.0
            grounded_entities = item.get("grounded_entities", [])
            if not isinstance(grounded_entities, list):
                grounded_entities = []
            q_norm = self._normalize_text(q)
            replaced = any(self._normalize_text(str(ent)) in q_norm for ent in grounded_entities if str(ent).strip())
            if replaced:
                replacement_num += 1.0

        pcqd_ids = None
        pcqd_scores = None
        pcqd_used = False
        pcqd_stats: Dict[str, Any] = {}
        if (getattr(self.pcrag_config, "improvement_stage", 0) >= 1
                and self.pcrag_config.pcqd_rewrite_mode == "append"):
            pcqd_questions = list(dict.fromkeys(static_sub_questions + pcqd_questions))
        if pcqd_questions:
            pcqd_ids, pcqd_scores, pcqd_stats = self._build_qd_track_ranking(
                query=query,
                hops=hops,
                sub_questions=pcqd_questions,
                base_sorted_doc_ids=base_sorted_doc_ids,
                original_seed_entities=set(seed_entities),
            )
            pcqd_used = bool(pcqd_stats.get("used", False) and pcqd_ids is not None and pcqd_scores is not None)
            if not pcqd_used:
                pcqd_fallback = True
        else:
            pcqd_fallback = True

        # If PC-QD failed and static fallback is disabled, keep base ranking.
        if pcqd_fallback and not self.pcrag_config.pcqd_fallback_to_static and not self.pcrag_config.pcqd_include_static_qd:
            return base_sorted_doc_ids, base_sorted_doc_scores, {
                "qd_used": True,
                "qd_sub_questions": len(static_sub_questions),
                "qd_sub_query_ppr_used": int(static_stats.get("sub_query_ppr_used", 0)),
                "qd_sub_query_ppr_failed": int(static_stats.get("sub_query_ppr_failed", 0)),
                "qd_seq_rewrites": int(static_stats.get("sub_seq_rewrites", 0)),
                "pcqd_used": False,
                "pcqd_sub_questions": 0,
                "pcqd_path_hints": len(path_hints),
                "pcqd_entity_replacement_num": replacement_num,
                "pcqd_entity_replacement_den": replacement_den,
                "pcqd_entity_replacement_rate": float(replacement_num / max(1.0, replacement_den)),
                "pcqd_fallback": True,
                "pcqd_conflict_rate": conflict_rate,
            }

        # 3-track fusion: base + static + path-conditioned
        num_passages = len(self.passage_node_keys)
        base_full = self._track_scores_to_full(num_passages, base_sorted_doc_ids, base_sorted_doc_scores)
        static_full = self._track_scores_to_full(num_passages, static_ids, static_scores) if static_used else np.zeros(num_passages, dtype=np.float32)
        pcqd_full = (
            self._track_scores_to_full(num_passages, pcqd_ids, pcqd_scores)
            if pcqd_used and pcqd_ids is not None and pcqd_scores is not None
            else np.zeros(num_passages, dtype=np.float32)
        )

        w_base = float(self.pcrag_config.pcqd_weight_base)
        w_static = float(self.pcrag_config.pcqd_weight_static) if self.pcrag_config.pcqd_include_static_qd else 0.0
        w_path = float(self.pcrag_config.pcqd_weight_path) if pcqd_used else 0.0
        if not static_used:
            w_static = 0.0
        if pcqd_fallback and self.pcrag_config.pcqd_fallback_to_static:
            w_path = 0.0

        if self.pcrag_config.pcqd_adaptive_fusion:
            if pcqd_used:
                hint_quality = min(
                    1.0,
                    float(len(path_hints)) / max(1e-6, float(self.pcrag_config.pcqd_adaptive_hint_ref)),
                )
                repl_rate = float(replacement_num / max(1.0, replacement_den))
                signal = 0.5 * hint_quality + 0.5 * repl_rate
                path_gain = float(self.pcrag_config.pcqd_adaptive_path_gain)
                w_path = w_path * (1.0 + path_gain * signal)
                w_base = w_base * max(0.05, 1.0 - 0.5 * path_gain * signal)
            if pcqd_fallback:
                w_base += float(self.pcrag_config.pcqd_adaptive_base_boost)

        w_sum = w_base + w_static + w_path
        if w_sum <= 0:
            w_base, w_static, w_path, w_sum = 1.0, 0.0, 0.0, 1.0
        w_base, w_static, w_path = w_base / w_sum, w_static / w_sum, w_path / w_sum

        fused = w_base * base_full + w_static * static_full + w_path * pcqd_full
        fused_ids = np.argsort(fused)[::-1]
        fused_scores = fused[fused_ids]

        return fused_ids, fused_scores, {
            "qd_used": True,
            "qd_sub_questions": len(static_sub_questions),
            "qd_sub_query_ppr_used": int(static_stats.get("sub_query_ppr_used", 0)) + int(pcqd_stats.get("sub_query_ppr_used", 0)),
            "qd_sub_query_ppr_failed": int(static_stats.get("sub_query_ppr_failed", 0)) + int(pcqd_stats.get("sub_query_ppr_failed", 0)),
            "qd_seq_rewrites": int(static_stats.get("sub_seq_rewrites", 0)) + int(pcqd_stats.get("sub_seq_rewrites", 0)),
            "pcqd_used": bool(pcqd_used),
            "pcqd_sub_questions": len(pcqd_questions),
            "pcqd_path_hints": len(path_hints),
            "pcqd_entity_replacement_num": replacement_num,
            "pcqd_entity_replacement_den": replacement_den,
            "pcqd_entity_replacement_rate": float(replacement_num / max(1.0, replacement_den)),
            "pcqd_fallback": bool(pcqd_fallback),
            "pcqd_conflict_rate": float(conflict_rate),
        }

    def _apply_path_set_optimization(
        self,
        query: str,
        base_sorted_doc_ids: np.ndarray,
        base_sorted_doc_scores: np.ndarray,
        ctx: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        top_doc_ids = base_sorted_doc_ids[: self.pcrag_config.path_candidate_docs].tolist()
        top_doc_scores = min_max_normalize(base_sorted_doc_scores[: self.pcrag_config.path_candidate_docs])

        seed_entities: Set[str] = set(ctx.get("seed_entities", []))
        if not seed_entities:
            return base_sorted_doc_ids, base_sorted_doc_scores, {
                "path_candidates": 0,
                "selected_paths": 0,
                "selected_path_candidates": [],
            }

        seed_distribution: Dict[str, float] = dict(ctx.get("seed_distribution", {}))
        bridges_by_seed: Dict[str, List[str]] = dict(ctx.get("bridges_by_seed", {}))

        candidates: List[PathCandidate] = []
        for rank, (doc_local_idx, doc_score) in enumerate(zip(top_doc_ids, top_doc_scores)):
            if doc_local_idx >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_local_idx]
            doc_entities = self.chunk_to_entities.get(pkey, set())
            if not doc_entities:
                continue

            # Direct two-node path: seed -> passage
            for seed in sorted(seed_entities):
                if seed not in doc_entities:
                    continue

                rel = float(doc_score)
                conn = 1.0
                cons = float(seed_distribution.get(seed, 0.0))
                comp = len({seed} & seed_entities) / max(1, len(seed_entities))

                candidates.append(
                    PathCandidate(
                        nodes=[seed, pkey],
                        passage_key=pkey,
                        covered_entities={seed},
                        score_relevance=rel,
                        score_connectivity=conn,
                        score_consistency=cons,
                        score_completeness=comp,
                        score_total=0.0,
                        metadata={"type": "direct", "rank": rank},
                    )
                )

            # Three-node path: seed -> bridge -> passage
            for seed, bridge_entities in sorted(bridges_by_seed.items()):
                for b in bridge_entities:
                    if b not in doc_entities:
                        continue

                    rel = float(doc_score)
                    conn = 1.2
                    seed_prob = float(seed_distribution.get(seed, 0.0))
                    idf_b = float(self.entity_idf.get(b, 1.0))
                    cons = 0.5 * seed_prob + 0.5 * idf_b
                    covered = {seed, b} & seed_entities
                    comp = len(covered) / max(1, len(seed_entities))

                    candidates.append(
                        PathCandidate(
                            nodes=[seed, b, pkey],
                            passage_key=pkey,
                            covered_entities=covered if covered else {seed},
                            score_relevance=rel,
                            score_connectivity=conn,
                            score_consistency=cons,
                            score_completeness=comp,
                            score_total=0.0,
                            metadata={"type": "bridge", "rank": rank},
                        )
                    )

        if not candidates:
            return base_sorted_doc_ids, base_sorted_doc_scores, {
                "path_candidates": 0,
                "selected_paths": 0,
                "selected_path_candidates": [],
            }

        # Normalize components and compute total score.
        candidates = normalize_candidate_scores(candidates)
        for c in candidates:
            c.score_total = (
                self.pcrag_config.path_score_relevance_weight * c.score_relevance
                + self.pcrag_config.path_score_connectivity_weight * c.score_connectivity
                + self.pcrag_config.path_score_consistency_weight * c.score_consistency
                + self.pcrag_config.path_score_completeness_weight * c.score_completeness
            )

        selected = greedy_path_set_selection(
            candidates=candidates,
            target_entities=seed_entities,
            set_size=self.pcrag_config.path_set_size,
            diversity_lambda=self.pcrag_config.path_set_diversity_lambda,
        )

        if not selected:
            return base_sorted_doc_ids, base_sorted_doc_scores, {
                "path_candidates": len(candidates),
                "selected_paths": 0,
                "selected_path_candidates": [],
            }

        path_passage_scores = aggregate_path_set_to_passage_scores(selected)

        # Merge path-set score back to global passage ranking.
        full_ppr_scores = np.zeros(len(self.passage_node_keys), dtype=np.float32)
        full_ppr_scores[base_sorted_doc_ids] = base_sorted_doc_scores
        full_ppr_scores = np.asarray(min_max_normalize(full_ppr_scores), dtype=np.float32)

        for pkey, pscore in path_passage_scores.items():
            pidx = self.passage_key_to_local_idx.get(pkey)
            if pidx is None:
                continue
            full_ppr_scores[pidx] = 0.7 * full_ppr_scores[pidx] + 0.3 * float(pscore)

        final_sorted_ids = np.argsort(full_ppr_scores)[::-1]
        final_sorted_scores = full_ppr_scores[final_sorted_ids]

        selected_meta: List[Dict[str, Any]] = []
        for p in selected:
            pkey = str(p.passage_key)
            covered = sorted(self.chunk_to_entities.get(pkey, set()))
            selected_meta.append(
                {
                    "passage_key": pkey,
                    "covered_entities": covered,
                    "score_total": float(p.score_total),
                    "score_relevance": float(p.score_relevance),
                }
            )

        return final_sorted_ids, final_sorted_scores, {
            "path_candidates": len(candidates),
            "selected_paths": len(selected),
            "selected_path_candidates": selected_meta,
        }


# Public alias (preferred name)
PathCondRAG = PCRAG
