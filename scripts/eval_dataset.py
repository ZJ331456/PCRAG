#!/usr/bin/env python3
"""PCRAG dataset evaluation entry."""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PCRAG_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PCRAG_ROOT)

# pathcondrag package
sys.path.insert(0, os.path.join(PCRAG_ROOT, "src"))

from pathcondrag import PathCondRAG, PathCondRAGConfig, PCRAG, PCRAGConfig
from eval_utils import (
    build_docs_from_full_corpus,
    build_docs_from_samples,
    ensure_gold_docs_in_corpus,
    get_gold_answers,
    get_gold_docs,
    get_benchmark_hops,
    load_json,
    sample_indices,
)
from stratified_eval import evaluate_stratified, print_stratified_results, save_stratified_results

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "sk-local-dummy"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pathcondrag.eval")


def _select_samples(samples: List[Dict[str, Any]], sample_size: int, sample_seed: int, sample_index: int):
    total = len(samples)
    if sample_index >= 0:
        if sample_index >= total:
            raise ValueError(f"sample_index={sample_index} out of range (total={total})")
        selected_indices = [sample_index]
    else:
        idxs = sample_indices(total=total, sample_size=sample_size, seed=sample_seed)
        selected_indices = idxs if idxs is not None else list(range(total))
    selected_samples = [samples[i] for i in selected_indices]
    return selected_samples, selected_indices


def run_eval(dataset: str, args: argparse.Namespace) -> str:
    dataset_file_stem = "nq_rear" if dataset == "nq" else dataset
    data_path = args.data_path or os.path.join(PROJECT_ROOT, "datasets", f"{dataset_file_stem}.json")
    corpus_path = args.corpus_path or os.path.join(PROJECT_ROOT, "datasets", f"{dataset_file_stem}_corpus.json")
    hop_source = getattr(args, "hop_source", "estimated")
    if hop_source not in ("estimated", "benchmark"):
        raise ValueError(f"Unknown hop_source={hop_source!r}")

    llm_name = args.llm_name or os.environ.get("HIPPO_LLM_NAME", "qwen3-8b")
    llm_base_url = args.llm_base_url or os.environ.get("HIPPO_LLM_BASE_URL", "http://127.0.0.1:8035/v1")
    embedding_model_name = args.embedding_model_name or os.environ.get("HIPPO_EMBEDDING_MODEL_NAME", "/root/models/NV-Embed-v2")
    embedding_base_url = args.embedding_base_url or os.environ.get("HIPPO_EMBEDDING_BASE_URL", "")

    eval_subdir = args.eval_subdir or os.environ.get("HIPPO_EVAL_SUBDIR", "eval_results_pcrag")
    save_dir = args.save_dir or os.path.join(PROJECT_ROOT, "outputs", eval_subdir, dataset)

    logger.info("[1/4] Loading dataset: %s", dataset)
    samples_all = load_json(data_path)
    corpus = load_json(corpus_path)

    # Validate the entire benchmark split before sampling. A subset must not
    # conceal missing MuSiQue decomposition labels elsewhere in the file.
    benchmark_hops_all = get_benchmark_hops(samples_all, dataset) if hop_source == "benchmark" else None

    samples, selected_indices = _select_samples(
        samples=samples_all,
        sample_size=max(0, int(args.sample_size)),
        sample_seed=int(args.sample_seed),
        sample_index=int(args.sample_index),
    )

    queries = [s["question"] for s in samples]
    benchmark_hops = (
        [benchmark_hops_all[i] for i in selected_indices]
        if benchmark_hops_all is not None
        else None
    )
    if benchmark_hops is not None:
        if dataset == "musique":
            hop_provenance = {
                "kind": "per_question_oracle",
                "field": "question_decomposition length",
                "oracle": True,
                "validated_samples": len(samples_all),
            }
        elif dataset in ("hotpotqa", "2wikimultihopqa"):
            hop_provenance = {
                "kind": "dataset_level_prior",
                "value": 2,
                "oracle": False,
                "note": "Question type and support count are not serial hop labels.",
            }
        else:
            hop_provenance = {
                "kind": "dataset_level_prior",
                "value": 1,
                "oracle": False,
                "note": "Single-hop benchmark prior; support count is not a hop label.",
            }
    else:
        hop_provenance = {"kind": "online_estimator", "oracle": False}
    hop_type_distribution = (
        dict(Counter(str(s["type"]) for s in samples))
        if dataset in ("hotpotqa", "2wikimultihopqa")
        else {}
    )
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset)

    if args.corpus_mode == "sample_only":
        docs = build_docs_from_samples(samples=samples, dataset_name=dataset)
    else:
        docs = build_docs_from_full_corpus(corpus)
        if args.max_corpus_docs > 0:
            docs = docs[: args.max_corpus_docs]

    if args.ensure_gold_in_corpus:
        docs = ensure_gold_docs_in_corpus(docs=docs, gold_docs=gold_docs)

    logger.info(
        "Samples=%d (selected=%s), Corpus docs for indexing=%d (mode=%s)",
        len(samples),
        selected_indices,
        len(docs),
        args.corpus_mode,
    )

    cfg_kwargs = dict(
        save_dir=save_dir,
        dataset=dataset,
        corpus_len=len(docs),
        llm_name=llm_name,
        llm_base_url=llm_base_url,
        embedding_model_name=embedding_model_name,
        embedding_base_url=embedding_base_url,
        force_index_from_scratch=bool(args.force_index_from_scratch),
        force_openie_from_scratch=bool(args.force_openie_from_scratch),
        retrieval_top_k=int(args.retrieval_top_k),
        linking_top_k=int(args.linking_top_k),
        qa_top_k=int(args.qa_top_k),
        max_new_tokens=args.max_new_tokens,
        max_qa_steps=int(args.max_qa_steps),
        embedding_batch_size=int(args.embedding_batch_size),
        llm_prefetch_workers=int(getattr(args, "llm_prefetch_workers", 1)),
        use_enhanced_hop_estimation=not args.no_enhanced_hop_estimation,
        use_iterative_retrieval=bool(args.use_iterative_retrieval),
        use_qcappr=not args.no_qcappr,
        use_eba=not args.no_eba,
        use_path_set_optimization=not args.no_path_set_opt,
        use_entity_idf_index=not args.no_entity_idf_index,
        use_bridge_cache_index=not args.no_bridge_cache_index,
        empty_rerank_fallback=args.empty_rerank_fallback,
    )

    # Memory-safe synonymy on small hosts (same defaults as /root/eval HippoRAG2 runners).
    # Set PCRAG_SAFE_SYNONYMY=0 to use BaseConfig defaults (topk=2047).
    if os.environ.get("PCRAG_SAFE_SYNONYMY", "1").strip().lower() not in ("0", "false", "no"):
        cfg_kwargs["synonymy_edge_topk"] = int(os.environ.get("SYNONYMY_EDGE_TOPK", "50"))
        cfg_kwargs["synonymy_edge_query_batch_size"] = int(os.environ.get("SYNONYMY_EDGE_QUERY_BATCH", "128"))
        cfg_kwargs["synonymy_edge_key_batch_size"] = int(os.environ.get("SYNONYMY_EDGE_KEY_BATCH", "1024"))
        logger.info(
            "Safe synonymy enabled: topk=%s query_bs=%s key_bs=%s",
            cfg_kwargs["synonymy_edge_topk"],
            cfg_kwargs["synonymy_edge_query_batch_size"],
            cfg_kwargs["synonymy_edge_key_batch_size"],
        )

    if args.path_set_size is not None:
        cfg_kwargs["path_set_size"] = int(args.path_set_size)
    if args.path_candidate_docs is not None:
        cfg_kwargs["path_candidate_docs"] = int(args.path_candidate_docs)
    if args.path_set_diversity_lambda is not None:
        cfg_kwargs["path_set_diversity_lambda"] = float(args.path_set_diversity_lambda)
    if args.eba_bridge_weight is not None:
        cfg_kwargs["eba_bridge_weight"] = float(args.eba_bridge_weight)
    if args.qcappr_hub_penalty_gamma is not None:
        cfg_kwargs["qcappr_hub_penalty_gamma"] = float(args.qcappr_hub_penalty_gamma)
    if args.eba_bridge_semantic_threshold is not None:
        cfg_kwargs["eba_bridge_semantic_threshold"] = float(args.eba_bridge_semantic_threshold)
    if args.hop_force_max is not None:
        cfg_kwargs["hop_force_max"] = int(args.hop_force_max)
    if args.hop_multi_min_signals is not None:
        cfg_kwargs["hop_multi_min_signals"] = int(args.hop_multi_min_signals)
    if args.hop_keyword_multi_min is not None:
        cfg_kwargs["hop_keyword_multi_min"] = int(args.hop_keyword_multi_min)
    if args.hop_entity_multi_min is not None:
        cfg_kwargs["hop_entity_multi_min"] = int(args.hop_entity_multi_min)
    if args.hop_diversity_multi_min is not None:
        cfg_kwargs["hop_diversity_multi_min"] = float(args.hop_diversity_multi_min)
    # Single-hop detection (new scoring-based parameters)
    if args.use_hop_scoring_detection:
        cfg_kwargs["use_hop_scoring_detection"] = True
    if args.hop_single_keyword_max is not None:
        cfg_kwargs["hop_single_keyword_max"] = int(args.hop_single_keyword_max)
    if args.hop_single_entity_max is not None:
        cfg_kwargs["hop_single_entity_max"] = int(args.hop_single_entity_max)
    if args.hop_single_diversity_max is not None:
        cfg_kwargs["hop_single_diversity_max"] = float(args.hop_single_diversity_max)
    if args.hop_single_min_score is not None:
        cfg_kwargs["hop_single_min_score"] = int(args.hop_single_min_score)
    if args.no_hop_use_dpr_coverage_signal:
        cfg_kwargs["hop_use_dpr_coverage_signal"] = False
    if args.hop_single_dpr_coverage_threshold is not None:
        cfg_kwargs["hop_single_dpr_coverage_threshold"] = float(args.hop_single_dpr_coverage_threshold)
    if args.qcappr_single_hop_damping is not None:
        cfg_kwargs["qcappr_single_hop_damping"] = float(args.qcappr_single_hop_damping)
    if args.qcappr_two_hop_damping is not None:
        cfg_kwargs["qcappr_two_hop_damping"] = float(args.qcappr_two_hop_damping)
    if args.qcappr_multi_hop_damping is not None:
        cfg_kwargs["qcappr_multi_hop_damping"] = float(args.qcappr_multi_hop_damping)
    if args.query_ppr_seed_top_k is not None:
        cfg_kwargs["query_ppr_seed_top_k"] = int(args.query_ppr_seed_top_k)
    if args.query_ppr_damping is not None:
        cfg_kwargs["query_ppr_damping"] = float(args.query_ppr_damping)
    if args.query_ppr_passage_weight is not None:
        cfg_kwargs["query_ppr_passage_weight"] = float(args.query_ppr_passage_weight)
    if args.no_facts_enable_weak_entity_seed:
        cfg_kwargs["no_facts_enable_weak_entity_seed"] = True
    if args.no_facts_weak_seed_top_k is not None:
        cfg_kwargs["no_facts_weak_seed_top_k"] = int(args.no_facts_weak_seed_top_k)
    if args.no_facts_weak_seed_min_score is not None:
        cfg_kwargs["no_facts_weak_seed_min_score"] = float(args.no_facts_weak_seed_min_score)
    if args.no_facts_weak_ngram_max_n is not None:
        cfg_kwargs["no_facts_weak_ngram_max_n"] = int(args.no_facts_weak_ngram_max_n)
    if args.iterative_round1_top_docs is not None:
        cfg_kwargs["iterative_round1_top_docs"] = int(args.iterative_round1_top_docs)
    if args.iterative_round2_seed_top_k is not None:
        cfg_kwargs["iterative_round2_seed_top_k"] = int(args.iterative_round2_seed_top_k)
    if args.iterative_merge_alpha is not None:
        cfg_kwargs["iterative_merge_alpha"] = float(args.iterative_merge_alpha)
    if args.iterative_min_seed_entities is not None:
        cfg_kwargs["iterative_min_seed_entities"] = int(args.iterative_min_seed_entities)
    if args.iterative_damping is not None:
        cfg_kwargs["iterative_damping"] = float(args.iterative_damping)
    if args.iterative_seed_mode is not None:
        cfg_kwargs["iterative_seed_mode"] = args.iterative_seed_mode

    # Query Decomposition
    if args.use_query_decomposition:
        cfg_kwargs["use_query_decomposition"] = True
    if args.qd_min_hops is not None:
        cfg_kwargs["qd_min_hops"] = int(args.qd_min_hops)
    if args.qd_max_sub_questions is not None:
        cfg_kwargs["qd_max_sub_questions"] = int(args.qd_max_sub_questions)
    if args.qd_sub_retrieval_top_k is not None:
        cfg_kwargs["qd_sub_retrieval_top_k"] = int(args.qd_sub_retrieval_top_k)
    if args.qd_sub_retrieval_mode is not None:
        cfg_kwargs["qd_sub_retrieval_mode"] = args.qd_sub_retrieval_mode
    if args.no_qd_sub_query_ppr_fallback_to_dpr:
        cfg_kwargs["qd_sub_query_ppr_fallback_to_dpr"] = False
    if args.qd_enable_sequential_dependency:
        cfg_kwargs["qd_enable_sequential_dependency"] = True
    if args.qd_sequential_top_docs_for_anchor is not None:
        cfg_kwargs["qd_sequential_top_docs_for_anchor"] = int(args.qd_sequential_top_docs_for_anchor)
    if args.qd_merge_alpha is not None:
        cfg_kwargs["qd_merge_alpha"] = float(args.qd_merge_alpha)
    if args.qd_llm_temperature is not None:
        cfg_kwargs["qd_llm_temperature"] = float(args.qd_llm_temperature)
    if args.use_path_conditioned_qd:
        cfg_kwargs["use_path_conditioned_qd"] = True
    if args.no_pcqd_include_static_qd:
        cfg_kwargs["pcqd_include_static_qd"] = False
    if args.no_pcqd_fallback_to_static:
        cfg_kwargs["pcqd_fallback_to_static"] = False
    if args.pcqd_ground_top_docs is not None:
        cfg_kwargs["pcqd_ground_top_docs"] = int(args.pcqd_ground_top_docs)
    if args.pcqd_entity_top_k is not None:
        cfg_kwargs["pcqd_entity_top_k"] = int(args.pcqd_entity_top_k)
    if args.pcqd_path_score_threshold is not None:
        cfg_kwargs["pcqd_path_score_threshold"] = float(args.pcqd_path_score_threshold)
    if args.pcqd_rewrite_mode is not None:
        cfg_kwargs["pcqd_rewrite_mode"] = args.pcqd_rewrite_mode
    if args.pcqd_disable_path_filtering:
        cfg_kwargs["pcqd_disable_path_filtering"] = True
    if args.pcqd_disable_entity_grounding:
        cfg_kwargs["pcqd_disable_entity_grounding"] = True
    if args.no_pcqd_bridge_voting:
        cfg_kwargs["pcqd_enable_bridge_voting"] = False
    if args.pcqd_max_bridges_per_source is not None:
        cfg_kwargs["pcqd_max_bridges_per_source"] = int(args.pcqd_max_bridges_per_source)
    if args.pcqd_weight_base is not None:
        cfg_kwargs["pcqd_weight_base"] = float(args.pcqd_weight_base)
    if args.pcqd_weight_static is not None:
        cfg_kwargs["pcqd_weight_static"] = float(args.pcqd_weight_static)
    if args.pcqd_weight_path is not None:
        cfg_kwargs["pcqd_weight_path"] = float(args.pcqd_weight_path)
    if args.pcqd_adaptive_fusion:
        cfg_kwargs["pcqd_adaptive_fusion"] = True
    if args.pcqd_adaptive_hint_ref is not None:
        cfg_kwargs["pcqd_adaptive_hint_ref"] = float(args.pcqd_adaptive_hint_ref)
    if args.pcqd_adaptive_path_gain is not None:
        cfg_kwargs["pcqd_adaptive_path_gain"] = float(args.pcqd_adaptive_path_gain)
    if args.pcqd_adaptive_base_boost is not None:
        cfg_kwargs["pcqd_adaptive_base_boost"] = float(args.pcqd_adaptive_base_boost)

    # MPCE (optional)
    if args.use_mpce:
        cfg_kwargs["use_mpce"] = True
    if args.mpce_candidate_top_k is not None:
        cfg_kwargs["mpce_candidate_top_k"] = int(args.mpce_candidate_top_k)
    if args.mpce_consensus_min_paths is not None:
        cfg_kwargs["mpce_consensus_min_paths"] = int(args.mpce_consensus_min_paths)
    if args.mpce_gamma is not None:
        cfg_kwargs["mpce_gamma"] = float(args.mpce_gamma)
    if args.mpce_boost_cap is not None:
        cfg_kwargs["mpce_boost_cap"] = float(args.mpce_boost_cap)
    if args.no_mpce_only_on_two_hop:
        cfg_kwargs["mpce_only_on_two_hop"] = False
    if args.no_mpce_use_entity_consensus:
        cfg_kwargs["mpce_use_entity_consensus"] = False
    if args.mpce_entity_consensus_top_k is not None:
        cfg_kwargs["mpce_entity_consensus_top_k"] = int(args.mpce_entity_consensus_top_k)
    if args.mpce_entity_consensus_weight is not None:
        cfg_kwargs["mpce_entity_consensus_weight"] = float(args.mpce_entity_consensus_weight)

    if hop_source == "benchmark":
        if dataset == "musique":
            cfg_kwargs["hop_force_max"] = max(4, int(cfg_kwargs.get("hop_force_max", 3)))
            cfg_kwargs["qd_max_sub_questions"] = max(4, int(cfg_kwargs.get("qd_max_sub_questions", 3)))
        elif dataset in ("nq", "popqa"):
            # hops=1 must not activate QD/PCQD for the single-hop benchmarks.
            cfg_kwargs["qd_min_hops"] = 2

    config = PathCondRAGConfig(**cfg_kwargs)

    logger.info("[2/4] Building index")
    rag = PathCondRAG(global_config=config)
    rag.index(docs=docs)
    if benchmark_hops is not None:
        rag.set_query_hop_overrides(hops=benchmark_hops, source="benchmark")
        logger.info("Benchmark hop policy: %s; selected distribution=%s", hop_provenance,
                    dict(Counter(str(h) for h in benchmark_hops)))

    logger.info("[3/4] Running evaluation mode=%s", args.eval_mode)
    retrieval_metrics: Dict[str, Any] = {}
    qa_metrics: Dict[str, Any] = {}
    retrieval_diagnostics: Dict[str, Any] = {}
    output_rows: List[Dict[str, Any]] = []

    has_answers = any(len(a) > 0 for a in gold_answers)
    predicted_answers_for_stratified = None

    if args.eval_mode == "rag_qa" and has_answers:
        if getattr(args, "rag_qa_mode", "full") == "dpr_only":
            logger.info("rag_qa_mode=dpr_only: using rag_qa_dpr() (dense retrieval + QA)")
            query_solutions, _, _, retrieval_metrics, qa_metrics = rag.rag_qa_dpr(
                queries=queries,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
        else:
            query_solutions, _, _, retrieval_metrics, qa_metrics = rag.rag_qa(
                queries=queries,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
        output_rows = [q.to_dict() for q in query_solutions]
        predicted_answers_for_stratified = [q.answer if q.answer is not None else "" for q in query_solutions]
    else:
        retrieval_results, retrieval_metrics = rag.retrieve(
            queries=queries,
            num_to_retrieve=int(args.retrieval_top_k),
            gold_docs=gold_docs,
        )
        output_rows = [r.to_dict() for r in retrieval_results]

    out_path = args.output or os.path.join(save_dir, "evaluation_results.json")

    # Stratified evaluation if enabled
    if args.stratified_eval:
        logger.info("Running stratified evaluation by query hop complexity...")
        if args.eval_mode == "rag_qa" and has_answers:
            retrieval_results_for_stratified = query_solutions
        else:
            retrieval_results_for_stratified = retrieval_results

        stratified_results = evaluate_stratified(
            queries=queries,
            retrieval_results=retrieval_results_for_stratified,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
            global_config=config,
            predicted_answers=predicted_answers_for_stratified,
        )
        print_stratified_results(stratified_results)

        if args.stratified_output:
            stratified_path = args.stratified_output
        else:
            stratified_path = os.path.splitext(out_path)[0] + "_stratified.json"
        save_stratified_results(stratified_results, stratified_path)

    if hasattr(rag, "get_retrieval_diagnostics"):
        try:
            retrieval_diagnostics = rag.get_retrieval_diagnostics()
        except Exception as exc:
            logger.exception("Failed to collect retrieval diagnostics: %s", exc)
            retrieval_diagnostics = {}

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "hop_source": hop_source,
                "hop_provenance": hop_provenance,
                "hop_distribution": (
                    dict(Counter(str(h) for h in benchmark_hops))
                    if benchmark_hops is not None
                    else retrieval_diagnostics.get("hop_counter", {})
                ),
                "hop_type_distribution": hop_type_distribution,
                "data_path": data_path,
                "corpus_path": corpus_path,
                "sample_size_requested": int(args.sample_size),
                "sample_seed": int(args.sample_seed),
                "sample_index": int(args.sample_index),
                "sample_size_effective": len(samples),
                "selected_indices": selected_indices,
                "corpus_mode": args.corpus_mode,
                "indexed_docs": len(docs),
                "eval_mode": args.eval_mode,
                "rag_qa_mode": getattr(args, "rag_qa_mode", "full"),
                "retrieval_metrics": retrieval_metrics,
                "qa_metrics": qa_metrics,
                "retrieval_diagnostics": retrieval_diagnostics,
                "runtime_config": asdict(config),
                "results": output_rows,
                "runtime": {
                    "llm_name": llm_name,
                    "llm_base_url": llm_base_url,
                    "embedding_model_name": embedding_model_name,
                    "embedding_base_url": embedding_base_url,
                    "save_dir": save_dir,
                    "pcrag_config": asdict(config),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("[4/4] Done. Results saved to: %s", out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PCRAG on supported datasets")
    parser.add_argument("--dataset", required=True, choices=["musique", "hotpotqa", "2wikimultihopqa", "nq", "popqa"])
    parser.add_argument("--data_path", default="")
    parser.add_argument("--corpus_path", default="")
    parser.add_argument("--hop_source", choices=["estimated", "benchmark"], default="estimated",
                        help="estimated preserves prior behavior; benchmark uses documented dataset labels/priors (MuSiQue is oracle).")

    parser.add_argument("--sample_size", type=int, default=0, help="0 means all")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_index", type=int, default=-1)

    parser.add_argument("--corpus_mode", choices=["full", "sample_only"], default="full")
    parser.add_argument("--max_corpus_docs", type=int, default=0)
    parser.add_argument("--ensure_gold_in_corpus", action="store_true", default=True)
    parser.add_argument("--no_ensure_gold_in_corpus", dest="ensure_gold_in_corpus", action="store_false")

    parser.add_argument("--eval_mode", choices=["rag_qa", "retrieve"], default="retrieve")
    parser.add_argument(
        "--rag_qa_mode",
        choices=["full", "dpr_only"],
        default="full",
        help="rag_qa path: full=graph-enhanced rag_qa(); dpr_only=dense passage retrieval + QA (rag_qa_dpr).",
    )
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--max_qa_steps", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--embedding_batch_size", type=int, default=2)
    parser.add_argument("--llm_prefetch_workers", type=int, default=1,
                        help="Bounded concurrent LLM prefetch across queries; 1 keeps serial behavior.")

    parser.add_argument("--save_dir", default="")
    parser.add_argument("--eval_subdir", default="")
    parser.add_argument("--output", default="")

    parser.add_argument("--llm_name", default="")
    parser.add_argument("--llm_base_url", default="")
    parser.add_argument("--embedding_model_name", default="")
    parser.add_argument("--embedding_base_url", default="")

    parser.add_argument("--force_index_from_scratch", action="store_true", default=False)
    parser.add_argument("--force_openie_from_scratch", action="store_true", default=False)

    # Path-centric module toggles
    parser.add_argument("--no_enhanced_hop_estimation", action="store_true", default=False)
    parser.add_argument("--hop_force_max", type=int, default=None)
    parser.add_argument("--hop_multi_min_signals", type=int, default=None)
    parser.add_argument("--hop_keyword_multi_min", type=int, default=None)
    parser.add_argument("--hop_entity_multi_min", type=int, default=None)
    parser.add_argument("--hop_diversity_multi_min", type=float, default=None)
    # Single-hop detection scoring (opt-in gate + parameters)
    parser.add_argument("--use_hop_scoring_detection", action="store_true", default=False)
    parser.add_argument("--hop_single_keyword_max", type=int, default=None)
    parser.add_argument("--hop_single_entity_max", type=int, default=None)
    parser.add_argument("--hop_single_diversity_max", type=float, default=None)
    parser.add_argument("--hop_single_min_score", type=int, default=None)
    parser.add_argument("--no_hop_use_dpr_coverage_signal", action="store_true", default=False)
    parser.add_argument("--hop_single_dpr_coverage_threshold", type=float, default=None)
    parser.add_argument("--use_iterative_retrieval", action="store_true", default=False)
    parser.add_argument("--no_qcappr", action="store_true", default=False)
    parser.add_argument("--no_eba", action="store_true", default=False)
    parser.add_argument("--no_path_set_opt", action="store_true", default=False)

    # Index innovation toggles
    parser.add_argument("--no_entity_idf_index", action="store_true", default=False)
    parser.add_argument("--no_bridge_cache_index", action="store_true", default=False)

    # Ablation knobs
    parser.add_argument("--path_set_size", type=int, default=None)
    parser.add_argument("--path_candidate_docs", type=int, default=None)
    parser.add_argument("--path_set_diversity_lambda", type=float, default=None)
    parser.add_argument("--eba_bridge_weight", type=float, default=None)
    parser.add_argument("--eba_bridge_semantic_threshold", type=float, default=None)
    parser.add_argument("--qcappr_hub_penalty_gamma", type=float, default=None)
    parser.add_argument("--qcappr_single_hop_damping", type=float, default=None)
    parser.add_argument("--qcappr_two_hop_damping", type=float, default=None)
    parser.add_argument("--qcappr_multi_hop_damping", type=float, default=None)
    parser.add_argument("--query_ppr_seed_top_k", type=int, default=None)
    parser.add_argument("--query_ppr_damping", type=float, default=None)
    parser.add_argument("--query_ppr_passage_weight", type=float, default=None)
    parser.add_argument("--no_facts_enable_weak_entity_seed", action="store_true", default=False)
    parser.add_argument("--no_facts_weak_seed_top_k", type=int, default=None)
    parser.add_argument("--no_facts_weak_seed_min_score", type=float, default=None)
    parser.add_argument("--no_facts_weak_ngram_max_n", type=int, default=None)
    parser.add_argument("--iterative_round1_top_docs", type=int, default=None)
    parser.add_argument("--iterative_round2_seed_top_k", type=int, default=None)
    parser.add_argument("--iterative_merge_alpha", type=float, default=None)
    parser.add_argument("--iterative_min_seed_entities", type=int, default=None)
    parser.add_argument("--iterative_damping", type=float, default=None)
    parser.add_argument("--iterative_seed_mode", choices=["idf_novel", "idf_only", "sim_idf"], default=None,
                        help="Bridge seed selection strategy for iterative retrieval")

    # Query Decomposition
    parser.add_argument("--use_query_decomposition", action="store_true", default=False)
    parser.add_argument("--qd_min_hops", type=int, default=None)
    parser.add_argument("--qd_max_sub_questions", type=int, default=None)
    parser.add_argument("--qd_sub_retrieval_top_k", type=int, default=None)
    parser.add_argument("--qd_sub_retrieval_mode", choices=["dpr", "query_ppr"], default=None)
    parser.add_argument("--no_qd_sub_query_ppr_fallback_to_dpr", action="store_true", default=False)
    parser.add_argument("--qd_enable_sequential_dependency", action="store_true", default=False)
    parser.add_argument("--qd_sequential_top_docs_for_anchor", type=int, default=None)
    parser.add_argument("--qd_merge_alpha", type=float, default=None)
    parser.add_argument("--qd_llm_temperature", type=float, default=None)
    parser.add_argument("--use_path_conditioned_qd", action="store_true", default=False)
    parser.add_argument("--no_pcqd_include_static_qd", action="store_true", default=False)
    parser.add_argument("--no_pcqd_fallback_to_static", action="store_true", default=False)
    parser.add_argument("--pcqd_ground_top_docs", type=int, default=None)
    parser.add_argument("--pcqd_entity_top_k", type=int, default=None)
    parser.add_argument("--pcqd_path_score_threshold", type=float, default=None)
    parser.add_argument("--pcqd_rewrite_mode", choices=["replace", "append"], default=None)
    parser.add_argument("--pcqd_disable_path_filtering", action="store_true", default=False)
    parser.add_argument("--pcqd_disable_entity_grounding", action="store_true", default=False)
    parser.add_argument("--no_pcqd_bridge_voting", action="store_true", default=False)
    parser.add_argument("--pcqd_max_bridges_per_source", type=int, default=None)
    parser.add_argument("--pcqd_weight_base", type=float, default=None)
    parser.add_argument("--pcqd_weight_static", type=float, default=None)
    parser.add_argument("--pcqd_weight_path", type=float, default=None)
    parser.add_argument("--pcqd_adaptive_fusion", action="store_true", default=False)
    parser.add_argument("--pcqd_adaptive_hint_ref", type=float, default=None)
    parser.add_argument("--pcqd_adaptive_path_gain", type=float, default=None)
    parser.add_argument("--pcqd_adaptive_base_boost", type=float, default=None)

    # MPCE (optional, default off)
    parser.add_argument("--use_mpce", action="store_true", default=False)
    parser.add_argument("--mpce_candidate_top_k", type=int, default=None)
    parser.add_argument("--mpce_consensus_min_paths", type=int, default=None)
    parser.add_argument("--mpce_gamma", type=float, default=None)
    parser.add_argument("--mpce_boost_cap", type=float, default=None)
    parser.add_argument("--no_mpce_only_on_two_hop", action="store_true", default=False)
    parser.add_argument("--no_mpce_use_entity_consensus", action="store_true", default=False)
    parser.add_argument("--mpce_entity_consensus_top_k", type=int, default=None)
    parser.add_argument("--mpce_entity_consensus_weight", type=float, default=None)

    # Evaluation options
    parser.add_argument("--stratified_eval", action="store_true", default=False,
                        help="Enable stratified evaluation by query hop complexity")
    parser.add_argument("--stratified_output", default="",
                        help="Output path for stratified metrics JSON (defaults to <output>_stratified.json)")

    parser.add_argument("--empty_rerank_fallback", choices=["dpr", "unfiltered", "query_ppr"], default="dpr")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_eval(dataset=args.dataset, args=args)


if __name__ == "__main__":
    main()
