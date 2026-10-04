#!/usr/bin/env python3
"""Download original HotpotQA / 2WikiMultihopQA / MuSiQue into data-process/raw/.

Uses HuggingFace mirrors (official CMU Hotpot host is often unreachable).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw"
HOTPOT = RAW / "hotpotqa"
WIKI = RAW / "2wikimultihopqa"
MUSIQUE = RAW / "musique"


def curl_download(url: str, dest: Path, min_bytes: int = 1000) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > min_bytes:
        print(f"[skip] {dest} ({dest.stat().st_size:,} bytes)")
        return
    partial = dest.with_suffix(dest.suffix + ".partial")
    print(f"[get] {url}\n  -> {dest}", flush=True)
    cmd = [
        "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
        "-C", "-", "-o", str(partial), url,
    ]
    subprocess.check_call(cmd)
    if partial.stat().st_size < min_bytes:
        raise RuntimeError(f"download too small: {partial}")
    partial.replace(dest)
    print(f"[ok] {dest} ({dest.stat().st_size:,} bytes)", flush=True)


def dump_hotpot_from_hf() -> None:
    from datasets import load_dataset

    HOTPOT.mkdir(parents=True, exist_ok=True)
    mapping = {
        ("distractor", "train"): HOTPOT / "hotpot_train_v1.1.json",
        ("distractor", "validation"): HOTPOT / "hotpot_dev_distractor_v1.json",
        ("fullwiki", "validation"): HOTPOT / "hotpot_dev_fullwiki_v1.json",
        ("fullwiki", "test"): HOTPOT / "hotpot_test_fullwiki_v1.json",
    }
    for (cfg, split), dest in mapping.items():
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"[skip] {dest.name} ({dest.stat().st_size:,})")
            continue
        print(f"[hf] hotpot_qa/{cfg} split={split}", flush=True)
        ds = load_dataset("hotpot_qa", cfg, split=split)
        rows = []
        for ex in ds:
            sf = ex["supporting_facts"]
            if isinstance(sf, dict):
                pairs = list(zip(sf["title"], sf["sent_id"]))
            else:
                pairs = sf
            ctx = ex["context"]
            if isinstance(ctx, dict):
                context = list(zip(ctx["title"], ctx["sentences"]))
            else:
                context = ctx
            rows.append(
                {
                    "_id": ex["id"],
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "supporting_facts": pairs,
                    "context": context,
                    "type": ex.get("type"),
                    "level": ex.get("level"),
                }
            )
        dest.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] {dest.name} n={len(rows)} size={dest.stat().st_size:,}", flush=True)


def dump_2wiki() -> None:
    base = "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main"
    for name in ("train.json", "dev.json", "test.json"):
        curl_download(f"{base}/{name}?download=true", WIKI / name, min_bytes=10_000)


def dump_musique() -> None:
    base = "https://huggingface.co/datasets/bdsaglam/musique/resolve/main"
    # Official HF mirror ships train+dev only (test answers withheld).
    files = [
        "musique_ans_v1.0_train.jsonl",
        "musique_ans_v1.0_dev.jsonl",
        "musique_full_v1.0_train.jsonl",
        "musique_full_v1.0_dev.jsonl",
    ]
    for name in files:
        curl_download(f"{base}/{name}?download=true", MUSIQUE / name, min_bytes=10_000)
    print("[note] MuSiQue test splits are not public on HF; request via StonyBrookNLP/musique if needed.")


def _musique_via_hf(wanted: str) -> None:
    """Fallback: export from HF dataset configs if direct file missing."""
    from datasets import load_dataset

    # bdsaglam/musique configs vary; try answerable / default
    dest = MUSIQUE / wanted
    if dest.exists() and dest.stat().st_size > 10_000:
        return
    kind = "answerable" if "_ans_" in wanted else "default"
    split = "train" if "train" in wanted else ("validation" if "dev" in wanted else "test")
    print(f"[hf] bdsaglam/musique config={kind} split={split}", flush=True)
    try:
        ds = load_dataset("bdsaglam/musique", kind, split=split)
    except Exception:
        ds = load_dataset("bdsaglam/musique", split=split)
    MUSIQUE.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(dict(ex), ensure_ascii=False) + "\n")
    print(f"[ok] {dest.name} via HF n≈lines size={dest.stat().st_size:,}", flush=True)


def summarize() -> None:
    print("\n======= SUMMARY =======")
    for d in (HOTPOT, WIKI, MUSIQUE):
        if not d.exists():
            continue
        print(f"\n{d.relative_to(RAW)}/")
        for p in sorted(d.iterdir()):
            if p.is_file() and not p.name.endswith(".partial"):
                print(f"  {p.name:45s} {p.stat().st_size:>12,} bytes")
                if p.suffix == ".json":
                    try:
                        n = len(json.loads(p.read_text(encoding="utf-8")))
                        print(f"    n={n}")
                    except Exception as e:
                        print(f"    (json count failed: {e})")
                elif p.suffix == ".jsonl":
                    n = sum(1 for _ in p.open(encoding="utf-8"))
                    print(f"    n={n}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"RAW={RAW}", flush=True)
    dump_hotpot_from_hf()
    dump_2wiki()
    dump_musique()
    summarize()
    readme = RAW / "README.md"
    readme.write_text(
        """# Original multi-hop QA datasets

| Dataset | Dir | Upstream |
|---------|-----|----------|
| HotpotQA | `hotpotqa/` | HF `hotpot_qa` (official CMU format fields) |
| 2WikiMultihopQA | `2wikimultihopqa/` | HF `voidful/2WikiMultihopQA` |
| MuSiQue | `musique/` | HF `bdsaglam/musique` (Ans + Full) |

Re-run: `python PathCondRAG/data-process/download/download_original_datasets.py`
""",
        encoding="utf-8",
    )
    print("[done]")


if __name__ == "__main__":
    main()
