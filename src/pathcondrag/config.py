from dataclasses import dataclass, field
from typing import Literal

from .utils.config_utils import BaseConfig


@dataclass
class PCRAGConfig(BaseConfig):
    """Configuration for Path-Centric RAG (PC3 + optional MPCE)."""

    # Hop estimation
    use_enhanced_hop_estimation: bool = field(default=True)
    hop_force_max: int = field(default=3)
    hop_multi_min_signals: int = field(default=2)
    hop_keyword_multi_min: int = field(default=5)
    hop_entity_multi_min: int = field(default=5)
    hop_diversity_multi_min: float = field(default=8.0)
    use_hop_scoring_detection: bool = field(default=False)
    hop_single_keyword_max: int = field(default=1)
    hop_single_entity_max: int = field(default=5)
    hop_single_diversity_max: float = field(default=8.0)
    hop_single_min_score: int = field(default=3)
    hop_use_dpr_coverage_signal: bool = field(default=True)
    hop_single_dpr_coverage_threshold: float = field(default=0.5)

    # PC3 core toggles
    use_qcappr: bool = field(default=True)
    use_eba: bool = field(default=True)
    use_path_set_optimization: bool = field(default=True)

    # QCAPPR params
    qcappr_hub_penalty_gamma: float = field(default=0.35)
    qcappr_single_hop_damping: float = field(default=0.85)
    qcappr_two_hop_damping: float = field(default=0.50)
    qcappr_multi_hop_damping: float = field(default=0.45)
    qcappr_single_hop_temperature: float = field(default=0.35)
    qcappr_two_hop_temperature: float = field(default=0.7)
    qcappr_multi_hop_temperature: float = field(default=1.0)

    # EBA params
    eba_first_hop_k: int = field(default=20)
    eba_bridge_top_k: int = field(default=8)
    eba_bridge_weight: float = field(default=0.25)
    eba_bridge_semantic_threshold: float = field(default=0.1)

    # Path-set optimization params
    path_candidate_docs: int = field(default=80)
    path_set_size: int = field(default=12)
    path_score_relevance_weight: float = field(default=0.55)
    path_score_connectivity_weight: float = field(default=0.15)
    path_score_consistency_weight: float = field(default=0.15)
    path_score_completeness_weight: float = field(default=0.15)
    path_set_diversity_lambda: float = field(default=0.35)

    # MPCE (optional, default off)
    use_mpce: bool = field(default=False)
    mpce_candidate_top_k: int = field(default=60)
    mpce_consensus_min_paths: int = field(default=2)
    mpce_gamma: float = field(default=0.25)
    mpce_boost_cap: float = field(default=0.20)
    mpce_only_on_two_hop: bool = field(default=True)
    mpce_use_entity_consensus: bool = field(default=True)
    mpce_entity_consensus_top_k: int = field(default=3)
    mpce_entity_consensus_weight: float = field(default=0.5)

    # Index-side toggles (kept)
    use_entity_idf_index: bool = field(default=True)
    use_bridge_cache_index: bool = field(default=True)
    bridge_cache_top_k_per_entity: int = field(default=32)
    bridge_cache_entity_limit: int = field(default=120000)

    # Safety / fallback
    empty_rerank_fallback: Literal["dpr", "unfiltered", "query_ppr"] = field(default="dpr")
    query_ppr_seed_top_k: int = field(default=64)
    query_ppr_damping: float = field(default=0.45)
    query_ppr_passage_weight: float = field(default=0.25)
    no_facts_enable_weak_entity_seed: bool = field(default=False)
    no_facts_weak_seed_top_k: int = field(default=32)
    no_facts_weak_seed_min_score: float = field(default=0.15)
    no_facts_weak_ngram_max_n: int = field(default=5)

    # Iterative retrieval
    use_iterative_retrieval: bool = field(default=False)
    iterative_round1_top_docs: int = field(default=1)
    iterative_round2_seed_top_k: int = field(default=5)
    iterative_seed_mode: str = field(default="idf_novel")
    iterative_merge_alpha: float = field(default=0.55)
    iterative_min_seed_entities: int = field(default=1)
    iterative_damping: float = field(default=-1.0)

    # Query Decomposition
    use_query_decomposition: bool = field(default=False)
    qd_min_hops: int = field(default=2)
    qd_max_sub_questions: int = field(default=3)
    qd_sub_retrieval_top_k: int = field(default=3)
    qd_sub_retrieval_mode: str = field(default="dpr")
    qd_sub_query_ppr_fallback_to_dpr: bool = field(default=True)
    qd_enable_sequential_dependency: bool = field(default=False)
    qd_sequential_top_docs_for_anchor: int = field(default=1)
    qd_merge_alpha: float = field(default=0.30)
    qd_llm_temperature: float = field(default=0.0)
    qd_cache_decompositions: bool = field(default=True)
    # In-flight QD/PCQD generation requests per retrieval process.  Use 1
    # to retain serial generation; the default is 8.
    llm_prefetch_workers: int = field(default=8)

    # Path-conditioned QD
    use_path_conditioned_qd: bool = field(default=False)
    pcqd_include_static_qd: bool = field(default=True)
    pcqd_fallback_to_static: bool = field(default=True)
    pcqd_ground_top_docs: int = field(default=5)
    pcqd_entity_top_k: int = field(default=3)
    pcqd_path_score_threshold: float = field(default=0.6)
    pcqd_rewrite_mode: str = field(default="replace")
    pcqd_disable_path_filtering: bool = field(default=False)
    pcqd_disable_entity_grounding: bool = field(default=False)
    pcqd_enable_bridge_voting: bool = field(default=True)
    pcqd_max_bridges_per_source: int = field(default=1)
    pcqd_weight_base: float = field(default=0.4)
    pcqd_weight_static: float = field(default=0.2)
    pcqd_weight_path: float = field(default=0.4)
    pcqd_adaptive_fusion: bool = field(default=False)
    pcqd_adaptive_hint_ref: float = field(default=3.0)
    pcqd_adaptive_path_gain: float = field(default=0.5)
    pcqd_adaptive_base_boost: float = field(default=0.2)

    def __post_init__(self):
        super().__post_init__()
        self.qcappr_hub_penalty_gamma = max(0.0, float(self.qcappr_hub_penalty_gamma))
        self.eba_bridge_weight = max(0.0, float(self.eba_bridge_weight))
        self.hop_force_max = min(4, max(1, int(self.hop_force_max)))
        self.hop_multi_min_signals = min(3, max(1, int(self.hop_multi_min_signals)))
        self.path_set_size = max(1, int(self.path_set_size))
        self.path_candidate_docs = max(self.path_set_size, int(self.path_candidate_docs))
        self.eba_first_hop_k = max(1, int(self.eba_first_hop_k))
        self.eba_bridge_top_k = max(1, int(self.eba_bridge_top_k))
        self.query_ppr_seed_top_k = max(1, int(self.query_ppr_seed_top_k))
        self.iterative_round1_top_docs = max(1, int(self.iterative_round1_top_docs))
        self.iterative_round2_seed_top_k = max(1, int(self.iterative_round2_seed_top_k))
        self.iterative_min_seed_entities = max(1, int(self.iterative_min_seed_entities))
        self.iterative_merge_alpha = min(1.0, max(0.0, float(self.iterative_merge_alpha)))
        self.query_ppr_passage_weight = max(0.0, float(self.query_ppr_passage_weight))
        self.no_facts_weak_seed_top_k = max(1, int(self.no_facts_weak_seed_top_k))
        self.no_facts_weak_seed_min_score = max(0.0, float(self.no_facts_weak_seed_min_score))
        self.no_facts_weak_ngram_max_n = max(1, int(self.no_facts_weak_ngram_max_n))
        self.mpce_candidate_top_k = max(1, int(self.mpce_candidate_top_k))
        self.mpce_consensus_min_paths = max(1, int(self.mpce_consensus_min_paths))
        self.mpce_gamma = max(0.0, float(self.mpce_gamma))
        self.mpce_boost_cap = max(0.0, float(self.mpce_boost_cap))
        self.mpce_entity_consensus_top_k = max(1, int(self.mpce_entity_consensus_top_k))
        self.mpce_entity_consensus_weight = min(1.0, max(0.0, float(self.mpce_entity_consensus_weight)))
        self.bridge_cache_top_k_per_entity = max(1, int(self.bridge_cache_top_k_per_entity))
        self.bridge_cache_entity_limit = max(1, int(self.bridge_cache_entity_limit))
        self.qd_min_hops = max(1, int(self.qd_min_hops))
        self.qd_max_sub_questions = max(1, int(self.qd_max_sub_questions))
        self.llm_prefetch_workers = min(8, max(1, int(self.llm_prefetch_workers)))
        self.qd_sub_retrieval_top_k = max(1, int(self.qd_sub_retrieval_top_k))
        self.qd_sequential_top_docs_for_anchor = max(1, int(self.qd_sequential_top_docs_for_anchor))
        if self.qd_sub_retrieval_mode not in {"dpr", "query_ppr"}:
            self.qd_sub_retrieval_mode = "dpr"
        self.qd_merge_alpha = min(1.0, max(0.0, float(self.qd_merge_alpha)))
        self.pcqd_ground_top_docs = max(1, int(self.pcqd_ground_top_docs))
        self.pcqd_entity_top_k = max(1, int(self.pcqd_entity_top_k))
        self.pcqd_max_bridges_per_source = max(1, int(self.pcqd_max_bridges_per_source))
        self.pcqd_path_score_threshold = min(1.0, max(0.0, float(self.pcqd_path_score_threshold)))
        if self.pcqd_rewrite_mode not in {"replace", "append"}:
            self.pcqd_rewrite_mode = "replace"
        self.pcqd_weight_base = max(0.0, float(self.pcqd_weight_base))
        self.pcqd_weight_static = max(0.0, float(self.pcqd_weight_static))
        self.pcqd_weight_path = max(0.0, float(self.pcqd_weight_path))
        self.pcqd_adaptive_hint_ref = max(1e-6, float(self.pcqd_adaptive_hint_ref))
        self.pcqd_adaptive_path_gain = max(0.0, float(self.pcqd_adaptive_path_gain))
        self.pcqd_adaptive_base_boost = max(0.0, float(self.pcqd_adaptive_base_boost))


# Public alias (preferred name)
PathCondRAGConfig = PCRAGConfig
