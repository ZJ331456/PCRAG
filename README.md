# PathCondRAG

Path-conditioned query decomposition for multi-hop RAG.

**Full PC3** + optional **MPCE** (default off).

## Package layout

```
src/pathcondrag/
  __init__.py
  pathcondrag.py       # main entry (PathCondRAG)
  config.py
  path_optimizer.py
  BaseRAG.py           # graph RAG engine
  embedding_model/
  embedding_store.py
  evaluation/
  information_extraction/
  llm/
  prompts/
  rerank.py
  utils/
scripts/
  eval_dataset.py
  run_pc3.sh           # USE_MPCE=0|1
```

## Usage

```python
from pathcondrag import PathCondRAG, PathCondRAGConfig
```

```bash
source /root/PathCondRAG/env_qwen3_nvembed.sh
pip install -e /root/PathCondRAG

SAMPLE_SIZE=2 CORPUS_MODE=sample_only EVAL_MODE=retrieve \
  bash /root/PathCondRAG/scripts/run_pc3.sh
```

Default models: Qwen3-8B (`qwen3-8b` @ :8035) + NV-Embed-v2 (`/root/models/NV-Embed-v2`).
