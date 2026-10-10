#!/usr/bin/env bash
# --methods legacy_dependency_joint: reuse NV2 indexes and retrieve this case only.
# Without --methods: build indexes, then run the original five methods per dataset.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_OUT="${ROOT}/outputs/3multi_hop_datasets_results_with_nv2_10_8"
SMOKE=false
JOINT_ONLY=false
for argument in "$@"; do
  if [[ "${argument}" == "--smoke" ]]; then
    SMOKE=true
  elif [[ "${argument}" == "legacy_dependency_joint" || "${argument}" == "--methods=legacy_dependency_joint" ]]; then
    JOINT_ONLY=true
  fi
done
if [[ "${SMOKE}" == true && "${JOINT_ONLY}" == false ]]; then
  DEFAULT_OUT="${DEFAULT_OUT}_smoke"
fi
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT}}"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/root/models/NV-Embed-v2}"
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-nvembed}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
DEFAULT_LOG="${OUT_ROOT}/logs/run_nv2.log"
if [[ "${JOINT_ONLY}" == true ]]; then
  DEFAULT_LOG="${OUT_ROOT}/logs/run_legacy_dependency_joint_full.log"
  if [[ "${SMOKE}" == true ]]; then
    DEFAULT_LOG="${OUT_ROOT}/_legacy_dependency_joint_smoke/logs/run.log"
  fi
fi
RUN_LOG="${RUN_LOG:-${DEFAULT_LOG}}"

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
export HIPPO_EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL}"
export HIPPO_EMBEDDING_BASE_URL=""
export PATHCONDRAG_NVEMBED_OOM_SPLIT="${PATHCONDRAG_NVEMBED_OOM_SPLIT:-false}"
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8
export HIPPORAG_LLM_MAX_IN_FLIGHT=8
export HIPPO_OPENIE_NER_WORKERS=8
export HIPPO_OPENIE_TRIPLE_WORKERS=8
export HIPPORAG_KNN_DEVICE=cpu
export PATHCONDRAG_SHARED_KNN_DEVICE=cpu

OPTIONS=(--out-root "${OUT_ROOT}" --python "${RAG_PYTHON}"
  --llm-base-url "${LLM_BASE_URL}" --embedding-model "${EMBEDDING_MODEL}"
  --embedding-provider "${EMBEDDING_PROVIDER}" --embedding-batch-size "${EMBEDDING_BATCH_SIZE}")
if [[ -n "${VLLM_LOG:-}" ]]; then
  OPTIONS+=(--vllm-log "${VLLM_LOG}")
fi
echo "[run] out=${OUT_ROOT} embedding=${EMBEDDING_MODEL} batch=${EMBEDDING_BATCH_SIZE}"
if [[ "${JOINT_ONLY}" == true ]]; then
  echo "[run] indexes=reuse datasets=3 retrieval_cases=3 method=legacy_dependency_joint llm_workers=8 max_tokens=2048 thinking=false"
else
  echo "[run] indexes=3 retrieval_cases=15 llm_workers=8 max_tokens=2048 thinking=false"
fi
"${RAG_PYTHON}" -B -u "${ROOT}/scripts/multi_dataset_retrieval_nv2.py" "${OPTIONS[@]}" "$@"
