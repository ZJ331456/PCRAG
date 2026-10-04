# Original multi-hop QA datasets

| Dataset | Dir | Upstream |
|---------|-----|----------|
| HotpotQA | `hotpotqa/` | HF `hotpot_qa` (classic JSON fields) |
| 2WikiMultihopQA | `2wikimultihopqa/` | HF `voidful/2WikiMultihopQA` |
| MuSiQue | `musique/` | HF `bdsaglam/musique` (Ans + Full train/dev) |

MuSiQue **test** is not public on HF; request via [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique).

Re-download:

```bash
python PathCondRAG/data-process/download/download_original_datasets.py
# or (CMU Hotpot; often unreachable):
bash PathCondRAG/data-process/download/download_original_datasets.sh
```
