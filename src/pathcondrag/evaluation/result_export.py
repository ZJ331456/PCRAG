"""Evaluation-only exports. Gold information never enters retrieval planning."""


def detailed_result(solution, sample, query_index, gold_docs, benchmark_hops=None,
                    result_top_k=10, candidate_output_top_k=200):
    row = solution.to_dict(top_k=result_top_k)
    candidates = list(solution.docs[:candidate_output_top_k])
    scores = solution.doc_scores
    gold = list(dict.fromkeys(gold_docs))
    ranks = {doc: rank for rank, doc in reversed(list(enumerate(solution.docs, 1)))}
    metrics = {
        f"Recall@{k}": sum(doc in solution.docs[:k] for doc in gold) / len(gold) if gold else 0.0
        for k in (1, 2, 5, 10, 20, 200)
    }
    row.update(
        query_index=int(query_index), sample_id=sample.get("id", sample.get("_id", query_index)),
        benchmark_hops=benchmark_hops, gold_docs=gold,
        candidate_docs=candidates,
        candidate_doc_scores=[float(v) for v in scores[:candidate_output_top_k]] if scores is not None else [],
        retrieval_metrics=metrics,
        gold_document_ranks=[{"doc": doc, "rank": ranks.get(doc)} for doc in gold],
        all_gold_in_top5=bool(gold) and all(doc in solution.docs[:5] for doc in gold),
        all_gold_in_top10=bool(gold) and all(doc in solution.docs[:10] for doc in gold),
    )
    return row
