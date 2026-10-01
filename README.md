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

## Local Qwen3 concurrency and reproducibility

On this machine, online OpenIE can use 8 worker threads with a bounded limit of
8 active LLM HTTP requests. Set these variables **before starting Python**:

```bash
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8
export HIPPO_OPENIE_MAX_WORKERS=8
export PYTHONHASHSEED=0
```

For retrieval with Qwen3-Embedding-8B, pass
`--embedding_model_name /root/models/Qwen3-Embedding-8B
--embedding_batch_size 4 --llm_prefetch_workers 4` to `scripts/eval_dataset.py`.
Embedding batch size, retrieval prefetch workers, and the HTTP request limit
control different stages. The library keeps a conservative HTTP default of 4;
the settings above are for the measured local Qwen3 server.

Qwen3 thinking is forced off. Normal OpenIE requests keep NER's existing
512-to-1024 retry policy and the 2048-token triple extraction limit. A truncated
response is rejected; retries first reuse the decoding settings with a distinct
seed, then try frequency penalties 0.2 and 0.5 only if truncation persists.
Persistent failures stop indexing rather than silently inserting partial triples.
These recovery attempts are recorded in result metadata.

Temperature 0 does not make cold vLLM replies identical across runs. Reuse one
successful OpenIE index and LLM cache when comparing retrieval methods or
checking concurrency equivalence. Cold-cache throughput measurements and
fixed-cache equivalence checks answer different questions.

`--hop_source benchmark` uses MuSiQue's per-question 2/3/4-step labels and reports
their provenance. This is an oracle-conditioned experiment; disclose it and
align the information available to compared methods. NQ and PopQA use a
single-hop dataset prior. Stratified reports use the same supplied labels in
benchmark mode.
