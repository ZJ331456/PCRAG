from typing import List
import os
import numpy as np
import torch
from tqdm import tqdm


def retrieve_knn(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047, query_batch_size=1000,
                 key_batch_size=10000):
    """
    Retrieve the top-k nearest neighbors for each query id from the key ids.

    Returns:
        dict[query_id] -> (neighbor_key_indices: np.ndarray[int32], scores: np.ndarray[float32])
        where neighbor_key_indices index into ``key_ids``.
    """
    knn_device = os.environ.get("HIPPORAG_KNN_DEVICE", "cpu").strip().lower()
    if knn_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if len(key_vecs) == 0: return {}

    query_vecs = torch.tensor(query_vecs, dtype=torch.float32)
    query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)

    key_vecs = torch.tensor(key_vecs, dtype=torch.float32)
    key_vecs = torch.nn.functional.normalize(key_vecs, dim=1)

    results = {}

    def get_batches(vecs, batch_size):
        for i in range(0, len(vecs), batch_size):
            yield vecs[i:i + batch_size], i

    for query_batch, query_batch_start_idx in tqdm(
            get_batches(vecs=query_vecs, batch_size=query_batch_size),
            total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,
            desc="KNN for Queries"
    ):
        query_batch = query_batch.clone().detach()
        query_batch = query_batch.to(device)

        batch_topk_sim_scores = []
        batch_topk_indices = []

        offset_keys = 0

        for key_batch, key_batch_start_idx in get_batches(vecs=key_vecs, batch_size=key_batch_size):
            key_batch = key_batch.to(device)
            actual_key_batch_size = key_batch.size(0)

            similarity = torch.mm(query_batch, key_batch.T)

            topk_sim_scores, topk_indices = torch.topk(similarity, min(k, actual_key_batch_size), dim=1, largest=True,
                                                       sorted=True)

            topk_indices += offset_keys

            batch_topk_sim_scores.append(topk_sim_scores)
            batch_topk_indices.append(topk_indices)

            del similarity
            if device.type == "cuda":
                key_batch = key_batch.cpu()
                torch.cuda.empty_cache()

            offset_keys += actual_key_batch_size

        batch_topk_sim_scores = torch.cat(batch_topk_sim_scores, dim=1)
        batch_topk_indices = torch.cat(batch_topk_indices, dim=1)

        final_topk_sim_scores, final_topk_indices = torch.topk(batch_topk_sim_scores,
                                                               min(k, batch_topk_sim_scores.size(1)), dim=1,
                                                               largest=True, sorted=True)
        final_topk_sim_scores = final_topk_sim_scores.cpu()
        final_topk_indices = final_topk_indices.cpu()
        batch_topk_indices = batch_topk_indices.cpu()

        for i in range(final_topk_indices.size(0)):
            query_relative_idx = query_batch_start_idx + i
            query_idx = query_ids[query_relative_idx]

            final_topk_indices_i = final_topk_indices[i]
            final_topk_sim_scores_i = final_topk_sim_scores[i]

            key_rel = batch_topk_indices[i][final_topk_indices_i].numpy().astype(np.int32, copy=False)
            scores = final_topk_sim_scores_i.numpy().astype(np.float32, copy=False)
            results[query_idx] = (key_rel, scores)

        del batch_topk_sim_scores, batch_topk_indices, final_topk_sim_scores, final_topk_indices
        if device.type == "cuda":
            query_batch = query_batch.cpu()
            torch.cuda.empty_cache()

    del query_vecs, key_vecs
    return results
