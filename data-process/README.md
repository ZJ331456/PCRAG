# data-process

| Subdir | Role |
|--------|------|
| [`download/`](download/) | Download original HotpotQA / 2Wiki / MuSiQue → `raw/` |
| [`infer_hop/`](infer_hop/) | Infer / label / analyze hop counts (HiGra Table 3 aligned) |
| [`raw/`](raw/) | Official raw dumps (Hotpot / 2Wiki / MuSiQue / FanOutQA) |

Datasets live under `data-process/raw/{hotpotqa,2wikimultihopqa,musique,fanoutqa}/`.

Derived evaluation JSON lives in `/root/datasets/` (`*_v1`-style default files plus `*_v2` advantage-aligned 1000-question subsets). See [`raw/README.md`](raw/README.md) for v1/v2 definitions and sampling justification.

## Download

```bash
python data-process/download/download_original_datasets.py
```

FanOutQA is not part of that downloader: it needs the official package plus the dated Wikipedia cache. See [`raw/fanoutqa/README.md`](raw/fanoutqa/README.md) for layout, construction steps, and paper-facing justification. Derived eval files are `/root/datasets/fanoutqa.json` and `/root/datasets/fanoutqa_corpus.json`.

## Infer hops

```bash
# API: from infer_hop import infer_hop
python data-process/infer_hop/label_dataset_hops.py \
  --input /path/to/qa.json --dataset musique

# Hotpot HiGra (needs original supporting_facts):
python data-process/infer_hop/label_dataset_hops.py \
  --input /root/baseline/higra_agent/data/test_data/hotpot-v1.json \
  --dataset hotpotqa \
  --lookup data-process/raw/hotpotqa/hotpot_dev_distractor_v1.json

# Audit vs paper Table 3:
python data-process/infer_hop/analyze_hop_distribution.py
```

FanOutQA `hop_num` uses the MuSiQue definition `len(question_decomposition)` and is already written into `/root/datasets/fanoutqa.json`.