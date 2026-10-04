"""Infer per-question hop counts from unlabeled multi-hop QA samples.

Aligned with HiGraAgent (EACL Findings 2026) Appendix B.3 Table 3 on their
released ``*-v1.json`` eval sets. Empirically their ``hop_num`` equals:

* **MuSiQue**: ``len(question_decomposition)`` (also encoded in ``id`` as ``Nhop__...``)
* **HotpotQA**: ``len(supporting_facts)``  — sentence-level gold supports
  (unique titles are almost always 2; hop is NOT unique-title count)
* **2WikiMultihopQA**: ``len(supporting_facts)`` — sentence-level gold supports
  (bridge_comparison often has 4 entries)

Paper text says "number of gold passages"; the shipped labels match
**supporting-fact entry counts** for Hotpot/2Wiki and **decomposition length**
for MuSiQue.

This module does *not* need a pre-existing ``hop_num`` field. It only reads
gold structural fields present in the original / HiGra-exported samples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

DatasetName = str  # hotpotqa | 2wikimultihopqa | musique | auto


def _as_supporting_facts_list(sample: Dict[str, Any]) -> List[Any]:
    """Normalize supporting_facts from raw Hotpot/2Wiki or HiGra wrappers."""
    if "supporting_facts" in sample:
        sf = sample["supporting_facts"]
        if isinstance(sf, dict):
            titles = sf.get("title") or []
            sents = sf.get("sent_id") or sf.get("sent_ids") or [None] * len(titles)
            return list(zip(titles, sents))
        if isinstance(sf, list):
            return sf
    di = sample.get("dataset_information") or {}
    if isinstance(di, dict):
        extra = di.get("extra_info") or {}
        if isinstance(extra, dict) and isinstance(extra.get("supporting_facts"), list):
            return extra["supporting_facts"]
    return []


def _musique_decomposition(sample: Dict[str, Any]) -> Optional[List[Any]]:
    qd = sample.get("question_decomposition")
    if isinstance(qd, list) and qd:
        return qd
    di = sample.get("dataset_information") or {}
    if isinstance(di, dict):
        extra = di.get("extra_info") or {}
        if isinstance(extra, dict) and isinstance(extra.get("question_decomposition"), list):
            return extra["question_decomposition"]
    return None


def detect_dataset(sample: Dict[str, Any]) -> str:
    """Best-effort dataset detector for a single sample."""
    di = sample.get("dataset_information") or {}
    if isinstance(di, dict):
        name = (di.get("dataset_name") or "").lower()
        if "musique" in name:
            return "musique"
        if "hotpot" in name:
            return "hotpotqa"
        if "2wiki" in name or "wiki" in name:
            return "2wikimultihopqa"
    sid = str(sample.get("id") or sample.get("_id") or "")
    if "hop__" in sid:
        return "musique"
    if _musique_decomposition(sample) is not None:
        return "musique"
    if "supporting_facts" in sample or (
        isinstance(di, dict)
        and isinstance((di.get("extra_info") or {}).get("supporting_facts"), list)
    ):
        # Hotpot vs 2Wiki: 2Wiki usually has evidences / type in comparison family
        extra = {}
        if isinstance(di, dict):
            extra = di.get("extra_info") or {}
        typ = str(sample.get("type") or extra.get("type") or "").lower()
        if typ in {"comparison", "bridge_comparison", "compositional", "inference"} or "evidences" in sample or "evidences" in extra:
            return "2wikimultihopqa"
        if sample.get("level") is not None or extra.get("level") is not None:
            return "hotpotqa"
        return "hotpotqa"
    raise ValueError("Cannot detect dataset for sample; pass dataset= explicitly")


def infer_hop(sample: Dict[str, Any], dataset: DatasetName = "auto") -> int:
    """Infer hop count for one unlabeled sample.

    Parameters
    ----------
    sample:
        One QA dict (Hotpot / 2Wiki / MuSiQue / HiGra-wrapped).
    dataset:
        ``hotpotqa`` | ``2wikimultihopqa`` | ``musique`` | ``auto``.

    Returns
    -------
    int
        Hop count in ``{2,3,4,...}`` (MuSiQue max 4 in the official set).

    Raises
    ------
    ValueError
        If the gold fields needed for that dataset are missing.
    """
    name = detect_dataset(sample) if dataset in (None, "", "auto") else dataset
    name = name.lower().replace("-", "").replace("_", "")
    # normalize aliases
    if name in {"hotpot", "hotpotqa"}:
        name = "hotpotqa"
    elif name in {"2wiki", "2wikimultihop", "2wikimultihopqa", "wikimultihopqa"}:
        name = "2wikimultihopqa"
    elif name in {"musique", "musiqueans", "musiquev1"}:
        name = "musique"
    else:
        raise ValueError(f"Unknown dataset={dataset!r}")

    if name == "musique":
        qd = _musique_decomposition(sample)
        if qd is not None:
            hops = len(qd)
            if hops < 1:
                raise ValueError("MuSiQue question_decomposition is empty")
            return hops
        sid = str(sample.get("id") or "")
        if "hop__" in sid:
            prefix = sid.split("hop")[0]
            if prefix.isdigit():
                return int(prefix)
        raise ValueError(
            "MuSiQue hop requires question_decomposition (or id like '2hop__...')"
        )

    # Hotpot + 2Wiki: len(supporting_facts entries)
    sf = _as_supporting_facts_list(sample)
    if not sf:
        raise ValueError(
            f"{name} hop requires supporting_facts "
            "(HiGra hotpot-v1.json strips them — join original HotpotQA by id)"
        )
    return len(sf)


def infer_hops(
    samples: Sequence[Dict[str, Any]],
    dataset: DatasetName = "auto",
) -> List[int]:
    """Infer hops for a list of samples (order-preserving)."""
    if dataset == "auto" and samples:
        dataset = detect_dataset(samples[0])
    return [infer_hop(s, dataset=dataset) for s in samples]


def bucket_hops(hops: Sequence[int]) -> Dict[str, int]:
    """Bucket into paper Table-3 style: 2 / 3 / 4 / 5+."""
    out = {"2": 0, "3": 0, "4": 0, "5+": 0}
    for h in hops:
        key = "5+" if int(h) >= 5 else str(int(h))
        if key not in out:
            out[key] = 0
        out[key] += 1
    return out


PAPER_TABLE3 = {
    "hotpotqa": {"2": 550, "3": 300, "4": 121, "5+": 29},
    "2wikimultihopqa": {"2": 500, "3": 2, "4": 496, "5+": 2},
    "musique": {"2": 550, "3": 350, "4": 100, "5+": 0},
}


def compare_to_paper_table3(dataset: str, hops: Sequence[int]) -> Dict[str, Any]:
    buckets = bucket_hops(hops)
    paper = PAPER_TABLE3[dataset]
    return {
        "buckets": buckets,
        "paper_table3": paper,
        "exact_match": buckets == paper,
        "diff": {k: buckets.get(k, 0) - paper[k] for k in paper},
    }
