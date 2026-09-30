#!/usr/bin/env bash
# Default model pair for PathCondRAG on this machine.
#   LLM : vLLM OpenAI API @ :8035  (served-model-name: qwen3-8b)
#   Emb : in-process NV-Embed-v2   (no embedding HTTP server)
#
# Usage: source /root/PathCondRAG/env_qwen3_nvembed.sh

export HIPPO_LLM_NAME="${HIPPO_LLM_NAME:-qwen3-8b}"
export HIPPO_LLM_BASE_URL="${HIPPO_LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
export HIPPO_EMBEDDING_MODEL_NAME="${HIPPO_EMBEDDING_MODEL_NAME:-/root/models/NV-Embed-v2}"
export HIPPO_EMBEDDING_BASE_URL="${HIPPO_EMBEDDING_BASE_URL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-dummy}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-2}"
export HIPPO_OPENIE_MAX_WORKERS="${HIPPO_OPENIE_MAX_WORKERS:-8}"
export TOKENIZERS_PARALLELISM=false

echo "LLM: ${HIPPO_LLM_NAME} @ ${HIPPO_LLM_BASE_URL}"
echo "Emb: ${HIPPO_EMBEDDING_MODEL_NAME} (batch=${EMBEDDING_BATCH_SIZE})"
