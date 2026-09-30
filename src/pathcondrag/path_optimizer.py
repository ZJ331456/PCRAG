from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class PathCandidate:
    """A lightweight path unit used in path-set optimization."""

    nodes: List[str]
    passage_key: str
    covered_entities: Set[str]
    score_relevance: float
    score_connectivity: float
    score_consistency: float
    score_completeness: float
    score_total: float
    metadata: Dict = field(default_factory=dict)


def _safe_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-8:
        return [1.0 for _ in values]
    return ((arr - vmin) / (vmax - vmin)).tolist()


def normalize_candidate_scores(candidates: List[PathCandidate]) -> List[PathCandidate]:
    """Normalize each score component to [0, 1] so cross-query comparisons are stable."""
    if not candidates:
        return candidates

    rel = _safe_norm([c.score_relevance for c in candidates])
    conn = _safe_norm([c.score_connectivity for c in candidates])
    cons = _safe_norm([c.score_consistency for c in candidates])
    comp = _safe_norm([c.score_completeness for c in candidates])

    for i, c in enumerate(candidates):
        c.score_relevance = float(rel[i])
        c.score_connectivity = float(conn[i])
        c.score_consistency = float(cons[i])
        c.score_completeness = float(comp[i])
    return candidates


def greedy_path_set_selection(
    candidates: List[PathCandidate],
    target_entities: Set[str],
    set_size: int,
    diversity_lambda: float,
) -> List[PathCandidate]:
    """
    Greedy path-set optimization:
    select a set of paths maximizing both path quality and incremental evidence coverage.
    """
    if not candidates:
        return []

    set_size = max(1, int(set_size))
    diversity_lambda = max(0.0, min(1.0, float(diversity_lambda)))

    selected: List[PathCandidate] = []
    used_passages: Set[str] = set()
    covered_entities: Set[str] = set()

    target_total = max(1, len(target_entities))

    remaining = candidates[:]
    while remaining and len(selected) < set_size:
        best_idx = -1
        best_gain = -1e9

        for idx, cand in enumerate(remaining):
            if cand.passage_key in used_passages:
                continue

            incremental = len((cand.covered_entities - covered_entities) & target_entities)
            inc_cov = incremental / target_total
            objective = (1.0 - diversity_lambda) * cand.score_total + diversity_lambda * inc_cov

            if objective > best_gain:
                best_gain = objective
                best_idx = idx

        if best_idx < 0:
            break

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        used_passages.add(chosen.passage_key)
        covered_entities |= chosen.covered_entities

    return selected


def aggregate_path_set_to_passage_scores(
    selected_paths: List[PathCandidate],
) -> Dict[str, float]:
    """Aggregate selected path scores to final passage ranking scores."""
    score_map: Dict[str, float] = {}
    for p in selected_paths:
        prev = score_map.get(p.passage_key)
        if prev is None:
            score_map[p.passage_key] = p.score_total
        else:
            score_map[p.passage_key] = max(prev, p.score_total)
    return score_map
