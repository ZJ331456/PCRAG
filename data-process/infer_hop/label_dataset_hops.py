#!/usr/bin/env python3
"""Label hop counts on unlabeled QA JSON and compare to HiGraAgent Table 3.

This is the *derivation* entrypoint (not just analysis of existing hop_num).

Examples
--------
# MuSiQue / 2Wiki HiGra files (have the needed gold fields):
python data-process/infer_hop/label_dataset_hops.py \\
  --input /root/baseline/higra_agent/data/test_data/musique-v1.json \\
  --dataset musique

python data-process/infer_hop/label_dataset_hops.py \\
  --input /root/baseline/higra_agent/data/test_data/2wikimultihop-v1.json \\
  --dataset 2wikimultihopqa

# Hotpot HiGra strips supporting_facts — join original HotpotQA by id:
python data-process/infer_hop/label_dataset_hops.py \\
  --input /root/baseline/higra_agent/data/test_data/hotpot-v1.json \\
  --dataset hotpotqa \\
  --lookup data-process/raw/hotpotqa/hotpot_dev_distractor_v1.json

# PathCondRAG /root/datasets (already have supporting_facts / decomposition):
python data-process/infer_hop/label_dataset_hops.py \\
  --input /root/datasets/hotpotqa.json --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script from repo root or data-process/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer_hop import (  # noqa: E402
    PAPER_TABLE3,
    compare_to_paper_table3,
    infer_hop,
    infer_hops,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    """Map id -> sample for Hotpot/2Wiki join (provides supporting_facts)."""
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"lookup must be a JSON list: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for s in data:
        sid = s.get("_id") or s.get("id")
        if sid is not None:
            out[str(sid)] = s
    return out


def _merge_supporting_facts(
    sample: Dict[str, Any], lookup: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a shallow copy with supporting_facts filled from lookup if missing."""
    if sample.get("supporting_facts"):
        return sample
    di = sample.get("dataset_information") or {}
    extra = di.get("extra_info") if isinstance(di, dict) else None
    if isinstance(extra, dict) and extra.get("supporting_facts"):
        return sample
    sid = str(sample.get("id") or sample.get("_id") or "")
    src = lookup.get(sid)
    if src is None:
        # try question match
        q = (sample.get("question") or "").strip().lower()
        for cand in lookup.values():
            if (cand.get("question") or "").strip().lower() == q:
                src = cand
                break
    if src is None or not src.get("supporting_facts"):
        return sample
    merged = dict(sample)
    merged["supporting_facts"] = src["supporting_facts"]
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Unlabeled (or HiGra) QA JSON list")
    ap.add_argument(
        "--dataset",
        choices=["auto", "hotpotqa", "2wikimultihopqa", "musique"],
        default="auto",
    )
    ap.add_argument(
        "--lookup",
        type=Path,
        default=None,
        help="Optional original Hotpot/2Wiki JSON to supply supporting_facts by id",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write samples with inferred_hop (and hop_num if missing) to this JSON",
    )
    ap.add_argument(
        "--compare-paper",
        action="store_true",
        default=True,
        help="Compare hop buckets to HiGraAgent Table 3 (default on)",
    )
    ap.add_argument("--no-compare-paper", action="store_false", dest="compare_paper")
    args = ap.parse_args()

    samples: List[Dict[str, Any]] = _load_json(args.input)
    if not isinstance(samples, list):
        raise SystemExit(f"--input must be a JSON list: {args.input}")

    lookup = _index_lookup(args.lookup) if args.lookup else {}
    prepared = [_merge_supporting_facts(s, lookup) if lookup else s for s in samples]

    dataset = args.dataset
    hops: List[int] = []
    errors = 0
    for i, s in enumerate(prepared):
        try:
            hops.append(infer_hop(s, dataset=dataset))
        except ValueError as e:
            errors += 1
            if errors <= 5:
                print(f"[error] idx={i} id={s.get('id') or s.get('_id')}: {e}", file=sys.stderr)
            hops.append(-1)

    ok = [h for h in hops if h > 0]
    print(f"[infer] file={args.input} n={len(samples)} ok={len(ok)} errors={errors}")
    print(f"[infer] raw_counts={dict(sorted(Counter(ok).items()))}")

    # resolve dataset name for table3
    ds = dataset
    if ds == "auto" and ok:
        # re-detect from first successful
        from infer_hop import detect_dataset

        ds = detect_dataset(prepared[0])

    if args.compare_paper and ds in PAPER_TABLE3 and not errors:
        cmp = compare_to_paper_table3(ds, hops)
        print(f"[table3] dataset={ds} buckets={cmp['buckets']}")
        print(f"[table3] paper   ={cmp['paper_table3']}")
        print(f"[table3] exact_match={cmp['exact_match']} diff={cmp['diff']}")
    elif args.compare_paper and errors:
        print(
            "[table3] skipped exact compare because some samples failed inference. "
            "For HiGra hotpot-v1.json pass --lookup pointing at full HotpotQA "
            "(with supporting_facts), e.g. hotpot_dev_distractor_v1.json.",
            file=sys.stderr,
        )
        if ds in PAPER_TABLE3 and ok:
            cmp = compare_to_paper_table3(ds, ok)
            print(f"[table3-partial] on {len(ok)} ok samples: {cmp}")

    # agreement with existing hop_num if present
    labeled = [(s.get("hop_num"), h) for s, h in zip(samples, hops) if s.get("hop_num") is not None and h > 0]
    if labeled:
        agree = sum(1 for a, b in labeled if int(a) == int(b))
        print(f"[vs hop_num] agree={agree}/{len(labeled)} ({agree/len(labeled):.1%})")

    if args.out:
        out_rows = []
        for s, h in zip(samples, hops):
            row = dict(s)
            if h > 0:
                row["inferred_hop"] = h
                row.setdefault("hop_num", h)
            out_rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
