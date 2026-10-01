#!/usr/bin/env bash
# Re-validate HippoRAG2 vs PathCondRAG-PC3 on Qwen3-Embedding-8B after fixing
# task-specific query instructions. Reuses existing index (no OpenIE / re-embed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPPO_ROOT="${HIPPO_ROOT:-/root/baseline/HippoRAG}"
# shellcheck disable=SC1091
source "${ROOT}/env_qwen3_nvembed.sh"

export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HIPPORAG_KNN_DEVICE="${HIPPORAG_KNN_DEVICE:-cpu}"
export TOKENIZERS_PARALLELISM=false

CONDA_ENV="${CONDA_ENV:-rag}"
LLM_NAME="${HIPPO_LLM_NAME:-qwen3-8b}"
LLM_BASE_URL="${HIPPO_LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
EMB_PATH="/root/models/Qwen3-Embedding-8B"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-2}"
DATASETS_DIR="${DATASETS_DIR:-/root/datasets}"

SRC_HIPPO="${SRC_HIPPO:-${ROOT}/outputs/emb_ablation_qwen3emb8b_full/hipporag2_musique}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/emb_ablation_qwen3emb8b_instrfix}"
HIPPO_SAVE_PREFIX="${OUT_ROOT}/hipporag2"
HIPPO_DIR="${HIPPO_SAVE_PREFIX}_musique"
PCR_DIR="${OUT_ROOT}/pathcondrag"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-${LOG_DIR}/emb8b_instrfix_retrieve_${TS}.log}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${PCR_DIR}/index"

if [[ ! -d "${SRC_HIPPO}/qwen3-8b__root_models_Qwen3-Embedding-8B" ]]; then
  echo "[ERROR] missing Hippo working dir under ${SRC_HIPPO}" >&2
  exit 1
fi
if [[ ! -f "${SRC_HIPPO}/openie_results_ner_${LLM_NAME}.json" ]]; then
  echo "[ERROR] missing OpenIE under ${SRC_HIPPO}" >&2
  exit 1
fi

echo "================================================================="
echo " Qwen3-Emb-8B instr-fix retrieve revalidation"
echo " SRC=${SRC_HIPPO}"
echo " OUT=${OUT_ROOT}"
echo " LOG=${LOG}"
echo "================================================================="

curl -sf -o /dev/null "${LLM_BASE_URL}/models"

# ---------- seed Hippo save_dir from existing index (no rebuild) ----------
echo "[seed] copy Hippo index -> ${HIPPO_DIR}"
rm -rf "${HIPPO_DIR}"
mkdir -p "${HIPPO_DIR}"
cp -a "${SRC_HIPPO}/openie_results_ner_${LLM_NAME}.json" "${HIPPO_DIR}/"
rsync -a --exclude='*.lock' \
  "${SRC_HIPPO}/qwen3-8b__root_models_Qwen3-Embedding-8B/" \
  "${HIPPO_DIR}/qwen3-8b__root_models_Qwen3-Embedding-8B/"
# optional llm_cache speeds recognition-memory rerank
if [[ -d "${SRC_HIPPO}/llm_cache" ]]; then
  rsync -a "${SRC_HIPPO}/llm_cache/" "${HIPPO_DIR}/llm_cache/"
fi

# ---------- 1) HippoRAG2 retrieve-only ----------
echo ""
echo "[1/2] HippoRAG2 retrieve-only -> ${HIPPO_DIR}/metrics_retrieve.json"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u "${ROOT}/scripts/experiment_tools.py" embedding-eval-hippo-instrfix \
  --hippo-root "${HIPPO_ROOT}" --datasets-dir "${DATASETS_DIR}" \
  --save-dir "${HIPPO_DIR}" --llm "${LLM_NAME}" --llm-base-url "${LLM_BASE_URL}" \
  --embedding "${EMB_PATH}" --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"

# ---------- 2) PathCondRAG-PC3 retrieve-only, reuse same Hippo index ----------
echo ""
echo "[2/2] PathCondRAG-PC3 retrieve-only -> ${PCR_DIR}/result.json"
rm -rf "${PCR_DIR}/index"
mkdir -p "${PCR_DIR}/index"
cp -a "${HIPPO_DIR}/openie_results_ner_${LLM_NAME}.json" "${PCR_DIR}/index/"
rsync -a --exclude='*.lock' \
  "${HIPPO_DIR}/qwen3-8b__root_models_Qwen3-Embedding-8B/" \
  "${PCR_DIR}/index/qwen3-8b__root_models_Qwen3-Embedding-8B/"

PC3_ARGS=(
  --hop_force_max 2 --hop_multi_min_signals 2
  --use_iterative_retrieval
  --iterative_round1_top_docs 1 --iterative_round2_seed_top_k 5
  --iterative_seed_mode idf_novel --iterative_merge_alpha 0.45
  --iterative_min_seed_entities 1
  --use_query_decomposition --use_path_conditioned_qd
  --qd_min_hops 2 --qd_max_sub_questions 3 --qd_sub_retrieval_top_k 3
  --pcqd_ground_top_docs 5 --pcqd_entity_top_k 3
  --pcqd_path_score_threshold 0.60
  --pcqd_weight_base 0.40 --pcqd_weight_static 0.20 --pcqd_weight_path 0.40
)

conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u "${ROOT}/scripts/eval_dataset.py" \
  --dataset musique \
  --data_path "${DATASETS_DIR}/musique.json" \
  --corpus_path "${DATASETS_DIR}/musique_corpus.json" \
  --sample_size 0 \
  --sample_seed 42 \
  --corpus_mode full \
  --eval_mode retrieve \
  --retrieval_top_k 200 \
  --qa_top_k 5 \
  --max_qa_steps 1 \
  --max_new_tokens 2048 \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --save_dir "${PCR_DIR}/index" \
  --output "${PCR_DIR}/result.json" \
  --llm_name "${LLM_NAME}" \
  --llm_base_url "${LLM_BASE_URL}" \
  --embedding_model_name "${EMB_PATH}" \
  --embedding_base_url "" \
  --stratified_eval --stratified_output "${PCR_DIR}/stratified.json" \
  "${PC3_ARGS[@]}"

python "${ROOT}/scripts/experiment_tools.py" embedding-instrfix-summary \
  --output-dir "${OUT_ROOT}" --source-hippo "${SRC_HIPPO}" \
  --embedding "${EMB_PATH}" --llm "${LLM_NAME}"

echo "[done] ${OUT_ROOT}"
echo "LOG=${LOG}"
