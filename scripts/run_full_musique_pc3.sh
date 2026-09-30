#!/usr/bin/env bash
# Full MuSiQue: retrieval + QA, full PC3, MPCE OFF.
# Model pair aligned with baseline/HippoRAG (qwen3-8b + NV-Embed-v2).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/env_qwen3_nvembed.sh"

export PYTHONUNBUFFERED=1
export HIPPO_OPENIE_NER_MAX_TOKENS="${HIPPO_OPENIE_NER_MAX_TOKENS:-512}"
export HIPPO_OPENIE_TRIPLE_MAX_TOKENS="${HIPPO_OPENIE_TRIPLE_MAX_TOKENS:-2048}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-2}"

OUT_DIR="${OUT_DIR:-${ROOT}/outputs/pc3_musique_full_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/logs}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"
LOG="${LOG:-${LOG_DIR}/full_musique_pc3_qwen3_nvembed.log}"

curl -sf -o /dev/null "${HIPPO_LLM_BASE_URL}/models"
echo "[PathCondRAG] full musique PC3 (MPCE off)"
echo "  llm=${HIPPO_LLM_NAME} @ ${HIPPO_LLM_BASE_URL}"
echo "  emb=${HIPPO_EMBEDDING_MODEL_NAME} batch=${EMBEDDING_BATCH_SIZE} max_seq=2048"
echo "  max_new_tokens=${MAX_NEW_TOKENS} ner=${HIPPO_OPENIE_NER_MAX_TOKENS} triple=${HIPPO_OPENIE_TRIPLE_MAX_TOKENS}"
echo "  out=${OUT_DIR}"
echo "  log=${LOG}"

SAMPLE_SIZE=0 \
CORPUS_MODE=full \
EVAL_MODE=rag_qa \
USE_MPCE=0 \
FORCE_INDEX="${FORCE_INDEX:-1}" \
DATASET=musique \
OUT_DIR="${OUT_DIR}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  bash "${ROOT}/scripts/run_pc3.sh" 2>&1 | tee "${LOG}"

echo "[done] log=${LOG} result=${OUT_DIR}/result.json"
