"""Evaluation helpers for PropRAG dataset scripts."""

import json
import random
from typing import Any, Dict, List, Optional


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_indices(total: int, sample_size: int, seed: int) -> Optional[List[int]]:
    if sample_size <= 0 or sample_size >= total:
        return None
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), sample_size))


def get_benchmark_hops(samples: List[Dict[str, Any]], dataset_name: str) -> List[int]:
    """Return benchmark hop labels in sample order, never using answer content.

    MuSiQue supplies an oracle decomposition for every question. The other
    datasets only support a dataset-level prior, not a per-question hop count.
    """
    if dataset_name == "musique":
        hops = []
        for idx, sample in enumerate(samples):
            decomposition = sample.get("question_decomposition")
            if (
                not isinstance(decomposition, list)
                or len(decomposition) not in (2, 3, 4)
                or any(
                    not isinstance(step, dict)
                    or not isinstance(step.get("question"), str)
                    or not step["question"].strip()
                    for step in decomposition
                )
            ):
                raise ValueError(
                    f"MuSiQue sample index {idx} lacks a valid 2/3/4-step question_decomposition"
                )
            hops.append(len(decomposition))
        return hops
    if dataset_name in ("hotpotqa", "2wikimultihopqa"):
        return [2] * len(samples)
    if dataset_name in ("nq", "popqa"):
        return [1] * len(samples)
    raise ValueError(f"No benchmark hop policy for dataset={dataset_name!r}")


def _answer_alias_list(value: Any) -> List[str]:
    """Decode PopQA's JSON-encoded aliases without iterating over characters."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = [value]
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return [str(alias) for alias in value if alias is not None and str(alias)]


def get_gold_docs(samples: List[Dict[str, Any]], dataset_name: str) -> List[List[str]]:
    """
    Same extraction policy as hipporag/eval/ori_eval_utils.py.
    """
    gold_docs: List[List[str]] = []
    for sample in samples:
        if "supporting_facts" in sample:  # hotpotqa, 2wiki
            gold_title = set(item[0] for item in sample["supporting_facts"])
            gold_title_and_content_list = [
                item for item in sample["context"] if item[0] in gold_title
            ]
            if dataset_name.startswith("hotpotqa"):
                gold_doc = [
                    item[0] + "\n" + "".join(item[1]) for item in gold_title_and_content_list
                ]
            else:
                gold_doc = [
                    item[0] + "\n" + " ".join(item[1]) for item in gold_title_and_content_list
                ]
        elif "contexts" in sample:  # nq-like
            contexts = sample["contexts"]
            supporting_contexts = [
                item for item in contexts if item.get("is_supporting", False)
            ]
            if not supporting_contexts:
                supporting_contexts = contexts
            gold_doc = [
                item["title"] + "\n" + item["text"]
                for item in supporting_contexts
            ]
        else:
            assert "paragraphs" in sample, (
                f"`paragraphs` should exist in sample (dataset={dataset_name})"
            )
            gold_paragraphs = []
            for item in sample["paragraphs"]:
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [
                item["title"] + "\n" + (item.get("text") or item.get("paragraph_text", ""))
                for item in gold_paragraphs
            ]

        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def get_gold_answers(samples: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Same extraction policy as hipporag/eval/ori_eval_utils.py.
    """
    gold_answers: List[List[str]] = []
    for sample in samples:
        if "answer" in sample or "gold_ans" in sample:
            gold_ans = sample.get("answer") or sample.get("gold_ans")
        elif "reference" in sample:
            gold_ans = sample["reference"]
        elif "obj" in sample:
            ans_set = {sample["obj"]}
            if "possible_answers" in sample:
                ans_set.update(_answer_alias_list(sample["possible_answers"]))
            if "o_wiki_title" in sample:
                ans_set.add(sample["o_wiki_title"])
            if "o_aliases" in sample:
                ans_set.update(_answer_alias_list(sample["o_aliases"]))
            gold_ans = list(ans_set)
        elif "answers" in sample:
            gold_ans = sample["answers"]
        else:
            gold_ans = []

        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        elif not isinstance(gold_ans, list):
            gold_ans = [str(gold_ans)]

        ans_set = set(gold_ans)
        if "answer_aliases" in sample:
            ans_set.update(sample["answer_aliases"])

        gold_answers.append(list(ans_set))

    return gold_answers


def build_docs_from_full_corpus(corpus: List[Dict[str, Any]]) -> List[str]:
    docs: List[str] = []
    for doc in corpus:
        title = doc.get("title", "")
        text = doc.get("text") or doc.get("paragraph_text") or ""
        docs.append(f"{title}\n{text}")
    return docs


def build_docs_from_samples(samples: List[Dict[str, Any]], dataset_name: str) -> List[str]:
    """
    Build a tiny corpus only from sampled examples (fast smoke-test mode).
    """
    docs_map: Dict[str, None] = {}
    for sample in samples:
        if "supporting_facts" in sample and "context" in sample:
            for title, sentences in sample["context"]:
                if dataset_name.startswith("hotpotqa"):
                    text = "".join(sentences)
                else:
                    text = " ".join(sentences)
                docs_map[f"{title}\n{text}"] = None
        elif "paragraphs" in sample:
            for para in sample["paragraphs"]:
                title = para.get("title", "")
                text = para.get("text") or para.get("paragraph_text") or ""
                docs_map[f"{title}\n{text}"] = None
        elif "contexts" in sample:
            for ctx in sample["contexts"]:
                title = ctx.get("title", "")
                text = ctx.get("text") or ctx.get("paragraph_text") or ""
                docs_map[f"{title}\n{text}"] = None
    return list(docs_map.keys())


def ensure_gold_docs_in_corpus(docs: List[str], gold_docs: List[List[str]]) -> List[str]:
    present = set(docs)
    out = list(docs)
    for one_query_gold_docs in gold_docs:
        for gd in one_query_gold_docs:
            if gd not in present:
                out.append(gd)
                present.add(gd)
    return out
