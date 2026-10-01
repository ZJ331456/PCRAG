"""Stratified evaluation by query hop complexity."""

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from pathcondrag.evaluation.retrieval_eval import RetrievalRecall
from pathcondrag.evaluation.qa_eval import QAExactMatch, QAF1Score
from pathcondrag.utils.misc_utils import QuerySolution


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy scalars/containers to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_builtin(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def estimate_query_hops_simple(query: str) -> int:
    """
    Lightweight hop estimation for stratification.
    Uses keyword clues only (no model dependency).
    """
    q = f" {query.lower()} "
    clues = [
        " and ", " after ", " before ", " then ", " which ", " whose ",
        " where ", " when ", " first ", " second ", " former ", " latter ",
        " that ", " who ",
    ]
    score = sum(1 for c in clues if c in q)
    if score <= 1:
        return 1
    elif score <= 4:
        return 2
    else:
        return 3


def stratify_by_hops(
    queries: List[str],
    retrieval_results: List[QuerySolution],
    gold_docs: List[List[str]],
    gold_answers: List[List[str]],
    query_hops: Optional[List[int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Group by supplied benchmark labels, or the legacy keyword heuristic.

    Explicit labels keep three-hop and four-hop questions separate. ``indices``
    preserves the alignment of predicted answers with each group's examples.
    """
    if query_hops is not None:
        if len(query_hops) != len(queries):
            raise ValueError("query_hops must have one label per selected query")
        if any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or not 1 <= h <= 4
               for h in query_hops):
            raise ValueError("query_hops labels must be integers from 1 through 4")
        group_names = {1: "single_hop", 2: "two_hop", 3: "three_hop", 4: "four_hop"}
    else:
        # Preserve the original keyword-based report for estimated runs.
        group_names = {1: "single_hop", 2: "two_hop", 3: "multi_hop"}
    grouped = {
        name: {"queries": [], "retrieval_results": [], "gold_docs": [],
               "gold_answers": [], "indices": []}
        for name in group_names.values()
    }

    for i, query in enumerate(queries):
        hops = query_hops[i] if query_hops is not None else estimate_query_hops_simple(query)
        target = grouped[group_names[hops]]
        target["queries"].append(query)
        target["retrieval_results"].append(retrieval_results[i])
        target["gold_docs"].append(gold_docs[i])
        target["gold_answers"].append(gold_answers[i])
        target["indices"].append(i)

    return grouped


def evaluate_stratified(
    queries: List[str],
    retrieval_results: List[QuerySolution],
    gold_docs: List[List[str]],
    gold_answers: List[List[str]],
    global_config: Any,
    predicted_answers: Optional[List[str]] = None,
    k_list: List[int] = None,
    query_hops: Optional[List[int]] = None,
    hop_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate hop groups without changing retrieval or QA metric formulas.

    ``query_hops`` must follow the selected query order. When supplied, the
    report records their provenance and exposes separate 1/2/3/4-hop groups.
    Without labels, the legacy single/two/multi-hop grouping is retained.
    """
    if k_list is None:
        k_list = [1, 2, 5, 10, 20, 30, 50, 100]
    
    # Overall retrieval evaluation
    retrieval_eval = RetrievalRecall(global_config=global_config)
    overall_retrieval, _ = retrieval_eval.calculate_metric_scores(
        gold_docs=gold_docs,
        retrieved_docs=[r.docs for r in retrieval_results],
        k_list=k_list,
    )

    # Optional QA evaluation (only when valid predicted answers are provided)
    can_eval_qa = (
        predicted_answers is not None
        and len(predicted_answers) == len(queries)
        and any(len(x) > 0 for x in gold_answers)
    )
    if can_eval_qa:
        qa_em_eval = QAExactMatch(global_config=global_config)
        qa_f1_eval = QAF1Score(global_config=global_config)
        overall_em, _ = qa_em_eval.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
        )
        overall_f1, _ = qa_f1_eval.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
        )
        overall_qa = {
            "ExactMatch": overall_em.get("ExactMatch", 0.0),
            "F1": overall_f1.get("F1", 0.0),
        }
    else:
        overall_qa = {}
    
    # Stratified evaluation
    stratified_data = stratify_by_hops(
        queries, retrieval_results, gold_docs, gold_answers, query_hops=query_hops,
    )
    
    result = {
        "overall": {
            "count": len(queries),
            "retrieval": overall_retrieval,
            "qa": overall_qa,
        }
    }
    
    if query_hops is not None:
        result["stratification"] = {
            "hop_source": "benchmark",
            "hop_provenance": hop_provenance or {},
            "hop_distribution": dict(Counter(str(h) for h in query_hops)),
        }

    for hop_type, data in stratified_data.items():
        count = len(data["queries"])
        if count == 0:
            result[hop_type] = {"count": 0, "retrieval": {}, "qa": {}}
            continue
        
        # Retrieval metrics
        strat_retrieval, _ = retrieval_eval.calculate_metric_scores(
            gold_docs=data["gold_docs"],
            retrieved_docs=[r.docs for r in data["retrieval_results"]],
            k_list=k_list,
        )
        
        # QA metrics (optional)
        if can_eval_qa:
            hop_indices = data["indices"]
            hop_pred_answers = [predicted_answers[i] for i in hop_indices]
            qa_em_eval = QAExactMatch(global_config=global_config)
            qa_f1_eval = QAF1Score(global_config=global_config)
            hop_em, _ = qa_em_eval.calculate_metric_scores(
                gold_answers=data["gold_answers"],
                predicted_answers=hop_pred_answers,
            )
            hop_f1, _ = qa_f1_eval.calculate_metric_scores(
                gold_answers=data["gold_answers"],
                predicted_answers=hop_pred_answers,
            )
            strat_qa = {
                "ExactMatch": hop_em.get("ExactMatch", 0.0),
                "F1": hop_f1.get("F1", 0.0),
            }
        else:
            strat_qa = {}
        
        result[hop_type] = {
            "count": count,
            "percentage": f"{100.0 * count / len(queries):.1f}%",
            "retrieval": strat_retrieval,
            "qa": strat_qa,
        }
    
    return _to_builtin(result)


def print_stratified_results(results: Dict[str, Any]):
    """Pretty print stratified evaluation results."""
    print("\n" + "=" * 80)
    print("STRATIFIED EVALUATION RESULTS")
    print("=" * 80)
    
    for hop_type in ["overall", "single_hop", "two_hop", "three_hop", "four_hop", "multi_hop"]:
        if hop_type not in results:
            continue
        
        data = results[hop_type]
        count = data.get("count", 0)
        
        print(f"\n{hop_type.upper().replace('_', ' ')}: {count} queries")
        if "percentage" in data:
            print(f"  Percentage: {data['percentage']}")
        
        if count == 0:
            continue
        
        # Retrieval metrics
        retrieval = data.get("retrieval", {})
        if retrieval:
            print(f"  Retrieval:")
            for k in [1, 5, 10]:
                key = f"Recall@{k}"
                if key in retrieval:
                    print(f"    {key}: {retrieval[key]:.4f}")
        
        # QA metrics
        qa = data.get("qa", {})
        if qa:
            print(f"  QA:")
            if "ExactMatch" in qa:
                print(f"    EM: {qa['ExactMatch']:.4f}")
            elif "EM" in qa:
                print(f"    EM: {qa['EM']:.4f}")
            if "F1" in qa:
                print(f"    F1: {qa['F1']:.4f}")
    
    print("\n" + "=" * 80)


def save_stratified_results(results: Dict[str, Any], output_path: str):
    """Save stratified results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nStratified results saved to: {output_path}")
