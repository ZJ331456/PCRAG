"""Stratified evaluation by query hop complexity."""

import json
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
) -> Dict[str, Dict[str, Any]]:
    """
    Stratify evaluation results by query hop complexity.
    
    Returns:
        {
            "single_hop": {"queries": [...], "retrieval_results": [...], ...},
            "two_hop": {...},
            "multi_hop": {...},
            "overall": {...}
        }
    """
    single_hop_data = {"queries": [], "retrieval_results": [], "gold_docs": [], "gold_answers": []}
    two_hop_data = {"queries": [], "retrieval_results": [], "gold_docs": [], "gold_answers": []}
    multi_hop_data = {"queries": [], "retrieval_results": [], "gold_docs": [], "gold_answers": []}
    
    for i, query in enumerate(queries):
        hops = estimate_query_hops_simple(query)
        
        if hops == 1:
            target = single_hop_data
        elif hops == 2:
            target = two_hop_data
        else:
            target = multi_hop_data
        
        target["queries"].append(query)
        target["retrieval_results"].append(retrieval_results[i])
        target["gold_docs"].append(gold_docs[i])
        target["gold_answers"].append(gold_answers[i])
    
    return {
        "single_hop": single_hop_data,
        "two_hop": two_hop_data,
        "multi_hop": multi_hop_data,
    }


def evaluate_stratified(
    queries: List[str],
    retrieval_results: List[QuerySolution],
    gold_docs: List[List[str]],
    gold_answers: List[List[str]],
    global_config: Any,
    predicted_answers: Optional[List[str]] = None,
    k_list: List[int] = None,
) -> Dict[str, Any]:
    """
    Perform stratified evaluation by hop complexity.
    
    Returns:
        {
            "overall": {...},
            "single_hop": {"count": N, "retrieval": {...}, "qa": {...}},
            "two_hop": {...},
            "multi_hop": {...}
        }
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
    stratified_data = stratify_by_hops(queries, retrieval_results, gold_docs, gold_answers)
    
    result = {
        "overall": {
            "count": len(queries),
            "retrieval": overall_retrieval,
            "qa": overall_qa,
        }
    }
    
    if can_eval_qa:
        idx_groups = {"single_hop": [], "two_hop": [], "multi_hop": []}
        for i, query in enumerate(queries):
            hops = estimate_query_hops_simple(query)
            if hops == 1:
                idx_groups["single_hop"].append(i)
            elif hops == 2:
                idx_groups["two_hop"].append(i)
            else:
                idx_groups["multi_hop"].append(i)
    else:
        idx_groups = {}

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
            hop_indices = idx_groups.get(hop_type, [])
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
    
    for hop_type in ["overall", "single_hop", "two_hop", "multi_hop"]:
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
