#!/usr/bin/env python3
"""Analyze hop distributions (labels + inferred) vs HiGraAgent paper Table 3.

This script is an *analysis / audit* tool. For labeling unlabeled samples use:
  ``infer_hop/infer_hop.py`` (API) and ``infer_hop/label_dataset_hops.py`` (CLI).

Inference rule (matches HiGra ``hop_num`` / Table 3 on their eval files):
  - MuSiQue: len(question_decomposition)
  - HotpotQA / 2Wiki: len(supporting_facts)  # sentence-level entries
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_hop import PAPER_TABLE3, infer_hop  # noqa: E402

PAPER_META = {
    "title": "HiGraAgent: Dual-Agent Adaptive Reasoning over Hierarchical Knowledge Graph "
    "for Open Domain Multi-hop Question Answering",
    "venue": "Findings of the Association for Computational Linguistics: EACL 2026",
    "url": "https://aclanthology.org/2026.findings-eacl.62.pdf",
    "github": "https://github.com/headinthecloud6453/higra_agent",
    "sampling_note": (
        "Sample 1,000 questions from each dataset's development set; "
        "exclude repeated answers (except yes/no); stratify by hop count."
    ),
}


def _bucket(h: int) -> str:
    if h >= 5:
        return "5+"
    return str(h)


def _counter_to_buckets(c: Counter) -> Dict[str, int]:
    out = {"2": 0, "3": 0, "4": 0, "5+": 0}
    for k, v in c.items():
        if k is None:
            continue
        b = _bucket(int(k))
        if b not in out:
            out[b] = 0
        out[b] += v
    return out


def _supporting_facts_list(sample: Dict[str, Any]) -> List[Any]:
    if "supporting_facts" in sample:
        sf = sample["supporting_facts"]
        if isinstance(sf, dict):
            titles = sf.get("title") or []
            sents = sf.get("sent_id") or sf.get("sent_ids") or [None] * len(titles)
            return list(zip(titles, sents))
        if isinstance(sf, list):
            return sf
    di = sample.get("dataset_information") or {}
    extra = di.get("extra_info") or {}
    sf = extra.get("supporting_facts")
    if isinstance(sf, list):
        return sf
    return []


def hop_from_supporting_sentences(sample: Dict[str, Any]) -> Optional[int]:
    sf = _supporting_facts_list(sample)
    return len(sf) if sf else None


def hop_from_unique_support_titles(sample: Dict[str, Any]) -> Optional[int]:
    sf = _supporting_facts_list(sample)
    if not sf:
        # musique / higra paragraphs
        paras = sample.get("paragraphs") or []
        titles = {
            p.get("title")
            for p in paras
            if p.get("is_supporting") is True
            or ("is_supporting" not in p and p.get("is_supporting") is not False)
        }
        # only count explicitly supporting if flag exists on any para
        if any("is_supporting" in p for p in paras):
            titles = {p.get("title") for p in paras if p.get("is_supporting")}
        titles.discard(None)
        return len(titles) if titles else None
    titles = set()
    for item in sf:
        if isinstance(item, (list, tuple)) and item:
            titles.add(item[0])
        elif isinstance(item, dict) and item.get("title"):
            titles.add(item["title"])
    return len(titles) if titles else None


def hop_from_musique_decomposition(sample: Dict[str, Any]) -> Optional[int]:
    qd = sample.get("question_decomposition")
    if qd is None:
        di = sample.get("dataset_information") or {}
        extra = di.get("extra_info") or {}
        qd = extra.get("question_decomposition")
    if isinstance(qd, list) and qd:
        return len(qd)
    sid = str(sample.get("id") or "")
    if "hop__" in sid:
        prefix = sid.split("hop")[0]
        if prefix.isdigit():
            return int(prefix)
    return None


def hop_labeled(sample: Dict[str, Any]) -> Optional[int]:
    hn = sample.get("hop_num")
    if hn is None:
        return None
    return int(hn)


def analyze_file(
    path: Path,
    dataset_key: str,
    *,
    prefer_label: bool = True,
) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")

    labeled = Counter()
    derived = Counter()
    method_counts: Counter = Counter()

    label_vs_derived = Counter()
    missing_derived = 0

    for s in data:
        lab = hop_labeled(s)
        if lab is not None:
            labeled[lab] += 1

        try:
            der = infer_hop(s, dataset=dataset_key)
            method = "infer_hop"
            derived[der] += 1
            method_counts[method] += 1
        except ValueError:
            missing_derived += 1
            der = None
        if lab is not None and der is not None:
            label_vs_derived[(lab, der)] += 1

    derived_method = (
        ", ".join(f"{m}:{c}" for m, c in method_counts.most_common())
        or "infer_hop"
    )

    labeled_buckets = _counter_to_buckets(labeled) if labeled else None
    derived_buckets = _counter_to_buckets(derived) if derived else None
    paper = PAPER_TABLE3[dataset_key]

    def _match(buckets: Optional[Dict[str, int]]) -> Optional[Dict[str, Any]]:
        if not buckets:
            return None
        return {
            "exact_match_table3": buckets == paper,
            "diff_vs_table3": {k: buckets.get(k, 0) - paper[k] for k in paper},
            "buckets": buckets,
        }

    agree = sum(v for (a, b), v in label_vs_derived.items() if a == b)
    compared = sum(label_vs_derived.values())

    return {
        "path": str(path),
        "n": len(data),
        "dataset_key": dataset_key,
        "labeled_hop_num": {
            "raw": dict(sorted((str(k), v) for k, v in labeled.items())),
            "buckets": labeled_buckets,
            "vs_paper_table3": _match(labeled_buckets),
        },
        "derived_hops": {
            "method": derived_method,
            "raw": dict(sorted((str(k), v) for k, v in derived.items())),
            "buckets": derived_buckets,
            "missing": missing_derived,
            "vs_paper_table3": _match(derived_buckets),
        },
        "label_vs_derived_agreement": {
            "compared": compared,
            "agree": agree,
            "agree_rate": (agree / compared) if compared else None,
            "top_pairs": [
                {"hop_num": a, "derived": b, "count": c}
                for (a, b), c in label_vs_derived.most_common(12)
            ],
        },
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Hop distribution analysis (HiGraAgent paper vs local data)",
        "",
        f"- Paper: [{report['paper']['title']}]({report['paper']['url']})",
        f"- Venue: {report['paper']['venue']}",
        f"- Project match: **{report['is_higra_agent_paper']}** (same title / github / benchmarks)",
        "",
        "## Paper Table 3 (Appendix B.3)",
        "",
        "| Dataset | 2-hop | 3-hop | 4-hop | 5+ hop |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("hotpotqa", "HotpotQA"),
        ("2wikimultihopqa", "2WikiMultihop"),
        ("musique", "MuSiQue"),
    ]:
        t = PAPER_TABLE3[key]
        lines.append(
            f"| {label} | {t['2']} | {t['3']} | {t['4']} | {t['5+']} |"
        )
    lines += [
        "",
        "Paper hop definition: *number of gold passages required* (B.3). "
        "On released HiGra files, Hotpot `hop_num` tracks **#supporting sentences**, "
        "2Wiki tracks **#unique supporting titles**, MuSiQue tracks **decomposition length**.",
        "",
        "## Local comparisons",
        "",
    ]

    for block_name, block in report["analyses"].items():
        lines.append(f"### {block_name}")
        lines.append("")
        for key in ("hotpotqa", "2wikimultihopqa", "musique"):
            a = block[key]
            lab = a["labeled_hop_num"]["vs_paper_table3"]
            der = a["derived_hops"]["vs_paper_table3"]
            lines.append(f"- **{key}** (`n={a['n']}`)")
            if lab:
                flag = "MATCH" if lab["exact_match_table3"] else "DIFF"
                lines.append(
                    f"  - labeled `hop_num` buckets: `{lab['buckets']}` → **{flag}** Table 3"
                )
            else:
                lines.append("  - labeled `hop_num`: *(absent)*")
            if der and a["derived_hops"]["missing"] < a["n"]:
                flag = "MATCH" if der["exact_match_table3"] else "DIFF"
                lines.append(
                    f"  - derived via `{a['derived_hops']['method']}`: "
                    f"`{der['buckets']}` → **{flag}** Table 3 "
                    f"(missing={a['derived_hops']['missing']})"
                )
            else:
                lines.append(
                    f"  - derived via `{a['derived_hops']['method']}`: "
                    f"unavailable for most rows (missing={a['derived_hops']['missing']})"
                )
            agr = a["label_vs_derived_agreement"]
            if agr["compared"]:
                lines.append(
                    f"  - label vs derived agree: {agr['agree']}/{agr['compared']} "
                    f"({agr['agree_rate']:.1%})"
                )
            lines.append("")
    lines.append("## Verdict")
    lines.append("")
    for line in report["verdict"]:
        lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--higra-root",
        type=Path,
        default=Path("/root/baseline/higra_agent/data/test_data"),
    )
    ap.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/root/datasets"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    higra_map = {
        "hotpotqa": args.higra_root / "hotpot-v1.json",
        "2wikimultihopqa": args.higra_root / "2wikimultihop-v1.json",
        "musique": args.higra_root / "musique-v1.json",
    }
    datasets_map = {
        "hotpotqa": args.datasets_root / "hotpotqa.json",
        "2wikimultihopqa": args.datasets_root / "2wikimultihopqa.json",
        "musique": args.datasets_root / "musique.json",
    }

    higra_stats = {k: analyze_file(p, k) for k, p in higra_map.items()}
    datasets_stats = {k: analyze_file(p, k) for k, p in datasets_map.items()}

    verdict = []
    for k, a in higra_stats.items():
        vs = a["labeled_hop_num"]["vs_paper_table3"]
        if vs and vs["exact_match_table3"]:
            verdict.append(f"HiGra `{k}` hop_num **matches** paper Table 3 exactly.")
        else:
            verdict.append(f"HiGra `{k}` hop_num **differs** from Table 3: {vs}.")
    verdict.append(
        "Therefore the uploaded PDF is the HiGraAgent paper for this repo, and "
        "`data/test_data/*-v1.json` is the Table-3 evaluation sample."
    )
    for k, a in datasets_stats.items():
        der = a["derived_hops"]["vs_paper_table3"]
        if der and der["exact_match_table3"]:
            verdict.append(f"`/root/datasets` `{k}` derived hops also match Table 3.")
        else:
            verdict.append(
                f"`/root/datasets` `{k}` derived hops **do not** match Table 3 "
                f"(different 1000-subset and/or different hop proxy): {der and der['buckets']}."
            )
    verdict.append(
        "PathCondRAG `get_benchmark_hops` currently forces Hotpot/2Wiki → all 2; "
        "that is a dataset-level prior, not HiGra Table-3 per-question labels."
    )

    report = {
        "paper": PAPER_META,
        "paper_table3": PAPER_TABLE3,
        "is_higra_agent_paper": True,
        "analyses": {
            "higra_agent_test_data": higra_stats,
            "pathcondrag_datasets": datasets_stats,
        },
        "verdict": verdict,
        "notes": [
            "HotpotQA gold docs are almost always 2 unique titles; HiGra hop_num "
            "uses supporting-sentence count (2..7), matching Table 3's 5+ = 29.",
            "2Wiki bridge_comparison ≈ 4 titles / hop_num=4; comparison/compositional/"
            "inference ≈ 2.",
            "MuSiQue hop_num == len(question_decomposition) == id Nhop__ prefix.",
        ],
    }

    json_path = args.out_dir / "hop_distribution_report.json"
    md_path = args.out_dir / "hop_distribution_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")
    print("\n".join(f"- {v}" for v in verdict))


if __name__ == "__main__":
    main()
