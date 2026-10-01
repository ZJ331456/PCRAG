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
  run_*.sh            # experiment settings, execution order, logs
  experiment_tools.py # CLI for shell runner helpers
  utils/              # data preparation, validation, result reports
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

## Experiment script helpers

The `run_*.sh` scripts keep their existing environment variables and launch
commands. Their Python logic lives in `scripts/utils/`: `embedding.py` prepares
smoke samples and paired reports, `ablations.py` validates component ablations,
and `prefetch.py` / `safe_prefetch.py` check and summarize LLM concurrency runs.
`common.py` shares JSON and vLLM log readers. Shell scripts pass explicit,
quoted arguments through one entry point:

```bash
python scripts/experiment_tools.py --help
python scripts/experiment_tools.py prefetch-summary --help
```

Report and validation commands use the Python standard library. Model-related
dependencies load only when a command runs an evaluation or GPU cleanup.
Existing result filenames, JSON fields, and validation rules are preserved.

## Local Qwen3 concurrency and reproducibility

On this machine, online OpenIE can use 8 worker threads with a bounded limit of
8 active LLM HTTP requests. Set these variables **before starting Python**:

```bash
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8
export PYTHONHASHSEED=0
```

For retrieval with Qwen3-Embedding-8B, pass
`--embedding_model_name /root/models/Qwen3-Embedding-8B
--embedding_batch_size 4 --openie_max_workers 8 --llm_prefetch_workers 8`
to `scripts/eval_dataset.py`.
Embedding batch size, retrieval prefetch workers, and the HTTP request limit
control different stages. The process HTTP ceiling now defaults to 8;
set a lower `PATHCONDRAG_LLM_MAX_IN_FLIGHT` to bound both stages more tightly.
The settings above are for the measured local Qwen3 server. The machine-specific
`env_qwen3_nvembed.sh` also defaults the HTTP ceiling to 8. Retrieval prefetch
now defaults to 8; set `--llm_prefetch_workers 1` for serial generation.

`--openie_max_workers` accepts 1 through 8 and applies to both NER and triple
extraction. The two phases run sequentially; documents within each phase run
concurrently. Explicit CLI/config values override the legacy
`HIPPO_OPENIE_MAX_WORKERS`, `HIPPO_OPENIE_NER_WORKERS`, and
`HIPPO_OPENIE_TRIPLE_WORKERS` settings. Without an explicit value, those
environment variables and the existing 8-worker default remain supported.
Effective HTTP concurrency is bounded by both the stage's worker count and
`PATHCONDRAG_LLM_MAX_IN_FLIGHT`; setting 8 workers with an HTTP ceiling of 4
still allows only 4 active requests. Reusing complete OpenIE results makes no
new indexing LLM requests. Passage/entity/fact embedding and graph construction
do not use the chat LLM.

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
