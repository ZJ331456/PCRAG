# data-process

| Subdir | Role |
|--------|------|
| [`download/`](download/) | Download original HotpotQA / 2Wiki / MuSiQue → `raw/` |
| [`infer_hop/`](infer_hop/) | Infer / label / analyze hop counts (HiGra Table 3 aligned) |

Datasets live under `data-process/raw/{hotpotqa,2wikimultihopqa,musique}/`.

## Download

```bash
python data-process/download/download_original_datasets.py
```

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
