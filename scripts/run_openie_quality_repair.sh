#!/usr/bin/env bash
# Repair the frozen shared index into an isolated, validated new directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/openie_quality_repair_qwen3_b4_20261003}"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
mkdir -p "${OUT_ROOT}/logs"
if [[ -d "${OUT_ROOT}/runtime_deps/transformers" ]]; then
  export PYTHONPATH="${OUT_ROOT}/runtime_deps${PYTHONPATH:+:${PYTHONPATH}}"
fi
exec 9>"${ROOT}/outputs/.openie_quality_repair.lock"
flock -n 9 || { echo "An OpenIE repair is already running." >&2; exit 1; }
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

"${RAG_PYTHON}" -u "${ROOT}/scripts/repair_openie_index.py" --phase run --out-root "${OUT_ROOT}" "$@"
echo "[done] repaired index validated: ${OUT_ROOT}/repaired_index"
