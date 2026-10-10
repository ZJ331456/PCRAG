#!/usr/bin/env bash
# Compare legacy and dependency-aware scoring on the same frozen NV2 indexes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_OUT="${ROOT}/outputs/nv2_dependency_scoring_subset_10_10"
for argument in "$@"; do
  if [[ "${argument}" == "--smoke" ]]; then
    DEFAULT_OUT="${DEFAULT_OUT}_smoke"
    break
  fi
done
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT}}"
INDEX_ROOT="${INDEX_ROOT:-${ROOT}/outputs/3multi_hop_datasets_results_with_nv2_10_8}"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
RUN_LOG="${RUN_LOG:-${OUT_ROOT}/logs/run.log}"

mkdir -p "${OUT_ROOT}/logs" "$(dirname "${RUN_LOG}")"
exec 9>"${ROOT}/outputs/.recall_top5_improvements.lock"
flock -n 9 || { echo "Another retrieval experiment is running." >&2; exit 1; }
exec > >(tee -a "${RUN_LOG}") 2>&1
trap 'status=$?; echo "[failed] exit=${status} line=${LINENO}; inspect ${RUN_LOG}" >&2; exit "${status}"' ERR

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
export PATHCONDRAG_NVEMBED_OOM_SPLIT=false
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8
export HIPPORAG_LLM_MAX_IN_FLIGHT=8
export HIPPO_OPENIE_NER_WORKERS=8
export HIPPO_OPENIE_TRIPLE_WORKERS=8
export HIPPORAG_KNN_DEVICE=cpu
export PATHCONDRAG_SHARED_KNN_DEVICE=cpu

OPTIONS=(--index-root "${INDEX_ROOT}" --out-root "${OUT_ROOT}"
  --python "${RAG_PYTHON}" --llm-base-url "${LLM_BASE_URL}")
if [[ -n "${VLLM_LOG:-}" ]]; then
  OPTIONS+=(--vllm-log "${VLLM_LOG}")
fi
echo "[run] out=${OUT_ROOT} indexes=${INDEX_ROOT}/shared_indexes"
echo "[run] datasets=3 paired_cases=6 embedding=NV-Embed-v2 batch=4 llm_workers=8 max_tokens=2048 thinking=false"
"${RAG_PYTHON}" -B -u "${ROOT}/scripts/nv2_dependency_scoring_subset.py" "${OPTIONS[@]}" "$@"
