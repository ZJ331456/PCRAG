#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/pathcondrag_new_index_10_4}"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
mkdir -p "${OUT_ROOT}/logs"
exec 9>"${ROOT}/outputs/.recall_top5_improvements.lock"
flock -n 9 || { echo "Another retrieval experiment is running." >&2; exit 1; }
exec > >(tee -a "${OUT_ROOT}/logs/run.log") 2>&1

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8
export HIPPORAG_LLM_MAX_IN_FLIGHT=8
export HIPPO_OPENIE_MAX_WORKERS=8
export HIPPO_OPENIE_NER_WORKERS=8
export HIPPO_OPENIE_TRIPLE_WORKERS=8
export HIPPO_OPENIE_NER_MAX_TOKENS=512
export HIPPO_OPENIE_TRIPLE_MAX_TOKENS=2048
export HIPPO_OPENIE_QUALITY_MAX_RETRIES=2
export PATHCONDRAG_SHARED_KNN_DEVICE=cuda

"${RAG_PYTHON}" -B -u "${ROOT}/scripts/new_index_compare.py" --out-root "${OUT_ROOT}" "$@"
