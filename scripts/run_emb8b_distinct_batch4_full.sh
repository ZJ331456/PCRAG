#!/usr/bin/env bash
# Compare HippoRAG2 and PathCondRAG on the same Qwen3-Embedding-8B source index.
# Reuse document embeddings and graph; encode queries with batch size 4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPPO_ROOT="${HIPPO_ROOT:-/root/baseline/HippoRAG}"
SOURCE="${SOURCE:-${ROOT}/outputs/emb_ablation_qwen3emb8b_full/hipporag2_musique}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/emb_ablation_qwen3emb8b_distinct_b4_full}"
HIPPO_DIR="${OUT_ROOT}/hipporag2_musique"
PATH_DIR="${OUT_ROOT}/pathcondrag"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"
LLM_NAME="${LLM_NAME:-qwen3-8b}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
EMB_PATH="/root/models/Qwen3-Embedding-8B"
MODEL_DIR="qwen3-8b__root_models_Qwen3-Embedding-8B"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HIPPORAG_KNN_DEVICE="${HIPPORAG_KNN_DEVICE:-cpu}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ "${EMBEDDING_BATCH_SIZE}" != "4" ]]; then
  echo "Expected embedding batch size 4 for both systems." >&2
  exit 2
fi
test -f "${SOURCE}/${MODEL_DIR}/graph.pickle"
test -f "${SOURCE}/openie_results_ner_${LLM_NAME}.json"
curl -sf -o /dev/null "${LLM_BASE_URL}/models"

mkdir -p "${HIPPO_DIR}" "${PATH_DIR}/index"
rsync -a --exclude='*.lock' "${SOURCE}/${MODEL_DIR}/" "${HIPPO_DIR}/${MODEL_DIR}/"
cp -a "${SOURCE}/openie_results_ner_${LLM_NAME}.json" "${HIPPO_DIR}/"
if [[ -d "${SOURCE}/llm_cache" ]]; then
  rsync -a "${SOURCE}/llm_cache/" "${HIPPO_DIR}/llm_cache/"
fi

echo "[1/2] HippoRAG2, Qwen3-Embedding-8B, batch 4, rag_qa"
conda run --no-capture-output -n rag python -u "${HIPPO_ROOT}/main.py" \
  --dataset musique \
  --datasets_dir /root/datasets \
  --rag_type hipporag \
  --llm_name "${LLM_NAME}" \
  --llm_base_url "${LLM_BASE_URL}" \
  --embedding_name "${EMB_PATH}" \
  --embedding_provider transformers \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --force_index_from_scratch false \
  --sample_size 0 \
  --save_dir "${OUT_ROOT}/hipporag2"

echo "[2/2] PathCondRAG, Qwen3-Embedding-8B, batch 4, rag_qa"
rsync -a --exclude='*.lock' "${HIPPO_DIR}/${MODEL_DIR}/" "${PATH_DIR}/index/${MODEL_DIR}/"
cp -a "${HIPPO_DIR}/openie_results_ner_${LLM_NAME}.json" "${PATH_DIR}/index/"

conda run --no-capture-output -n rag python -u "${ROOT}/scripts/eval_dataset.py" \
  --dataset musique \
  --data_path /root/datasets/musique.json \
  --corpus_path /root/datasets/musique_corpus.json \
  --sample_size 0 --sample_seed 42 \
  --corpus_mode full --eval_mode rag_qa \
  --retrieval_top_k 200 --qa_top_k 5 --max_qa_steps 1 --max_new_tokens 2048 \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --save_dir "${PATH_DIR}/index" --output "${PATH_DIR}/result.json" \
  --llm_name "${LLM_NAME}" --llm_base_url "${LLM_BASE_URL}" \
  --embedding_model_name "${EMB_PATH}" --embedding_base_url "" \
  --stratified_eval --stratified_output "${PATH_DIR}/stratified.json" \
  --hop_force_max 2 --hop_multi_min_signals 2 \
  --use_iterative_retrieval \
  --iterative_round1_top_docs 1 --iterative_round2_seed_top_k 5 \
  --iterative_seed_mode idf_novel --iterative_merge_alpha 0.45 \
  --iterative_min_seed_entities 1 \
  --use_query_decomposition --use_path_conditioned_qd \
  --qd_min_hops 2 --qd_max_sub_questions 3 --qd_sub_retrieval_top_k 3 \
  --pcqd_ground_top_docs 5 --pcqd_entity_top_k 3 \
  --pcqd_path_score_threshold 0.60 \
  --pcqd_weight_base 0.40 --pcqd_weight_static 0.20 --pcqd_weight_path 0.40

python "${ROOT}/scripts/experiment_tools.py" embedding-distinct-summary \
  --output-dir "${OUT_ROOT}"
