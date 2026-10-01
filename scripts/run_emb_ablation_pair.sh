#!/usr/bin/env bash
# Embedding ablation: HippoRAG2 then PathCondRAG-PC3 (reuse Hippo index).
# Emb params aligned with NV-Embed-v2 stack: batch=2, max_seq=2048, normalized, in-process.
#
# Usage:
#   MODE=smoke EMB_TAG=8b bash scripts/run_emb_ablation_pair.sh
#   MODE=full  EMB_TAG=0.6b bash scripts/run_emb_ablation_pair.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPPO_ROOT="${HIPPO_ROOT:-/root/baseline/HippoRAG}"
# shellcheck disable=SC1091
source "${ROOT}/env_qwen3_nvembed.sh"

export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HIPPORAG_KNN_DEVICE="${HIPPORAG_KNN_DEVICE:-cpu}"
export TOKENIZERS_PARALLELISM=false
export HIPPO_OPENIE_MAX_WORKERS="${HIPPO_OPENIE_MAX_WORKERS:-8}"

MODE="${MODE:-smoke}"          # smoke | full
EMB_TAG="${EMB_TAG:-8b}"       # 8b | 0.6b
SAMPLE_SEED="${SAMPLE_SEED:-42}"
CONDA_ENV="${CONDA_ENV:-rag}"
LLM_NAME="${HIPPO_LLM_NAME:-qwen3-8b}"
LLM_BASE_URL="${HIPPO_LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

case "${EMB_TAG}" in
  8b|8B)
    EMB_PATH="/root/models/Qwen3-Embedding-8B"
    TAG="qwen3emb8b"
    ;;
  0.6b|0.6B|06b)
    EMB_PATH="/root/models/Qwen3-Embedding-0.6B"
    TAG="qwen3emb06b"
    ;;
  *)
    echo "[ERROR] unknown EMB_TAG=${EMB_TAG} (use 8b or 0.6b)" >&2
    exit 1
    ;;
esac

if [[ ! -d "${EMB_PATH}" ]]; then
  echo "[ERROR] missing embedding model: ${EMB_PATH}" >&2
  exit 1
fi

OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/emb_ablation_${TAG}_${MODE}}"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/logs}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
LOG="${LOG:-${LOG_DIR}/emb_ablation_${TAG}_${MODE}.log}"

# Hippo main appends _${dataset} to non-default save_dir
HIPPO_SAVE_PREFIX="${OUT_ROOT}/hipporag2"
HIPPO_DIR="${HIPPO_SAVE_PREFIX}_musique"
PCR_DIR="${OUT_ROOT}/pathcondrag"

if [[ "${MODE}" == "smoke" ]]; then
  SAMPLE_SIZE="${SAMPLE_SIZE:-2}"
  CORPUS_MODE=sample_only
  FORCE_HIPPO_INDEX=true
else
  SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
  CORPUS_MODE=full
  FORCE_HIPPO_INDEX=true
fi

DATASETS_DIR="${DATASETS_DIR:-/root/datasets}"
SMOKE_DATA_DIR="${OUT_ROOT}/smoke_data"

prepare_smoke_data() {
  mkdir -p "${SMOKE_DATA_DIR}"
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${ROOT}/scripts/experiment_tools.py" embedding-prepare-smoke \
    --datasets-dir "${DATASETS_DIR}" --output-dir "${SMOKE_DATA_DIR}" \
    --sample-size "${SAMPLE_SIZE}" --sample-seed "${SAMPLE_SEED}"
}

echo "================================================================="
echo " Emb ablation pair  MODE=${MODE} TAG=${TAG}"
echo " LLM=${LLM_NAME} @ ${LLM_BASE_URL}"
echo " EMB=${EMB_PATH}  batch=${EMBEDDING_BATCH_SIZE} max_seq=2048"
echo " out=${OUT_ROOT}"
echo "================================================================="

curl -sf -o /dev/null "${LLM_BASE_URL}/models"

HIPPO_DATA_DIR="${DATASETS_DIR}"
PCR_SAMPLE_SIZE="${SAMPLE_SIZE}"
PCR_CORPUS_MODE="${CORPUS_MODE}"
if [[ "${MODE}" == "smoke" ]]; then
  prepare_smoke_data
  HIPPO_DATA_DIR="${SMOKE_DATA_DIR}"
  # Hippo smoke uses the tiny corpus; PathCond indexes the same via data_path/corpus_path
  PCR_SAMPLE_SIZE=0   # already the only samples in smoke json
  PCR_CORPUS_MODE=full
fi

# ---------- 1) HippoRAG2 ----------
echo ""
echo "[1/2] HippoRAG2 indexing+rag_qa  -> ${HIPPO_DIR}"
rm -rf "${HIPPO_DIR}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u "${HIPPO_ROOT}/main.py" \
  --dataset musique \
  --datasets_dir "${HIPPO_DATA_DIR}" \
  --rag_type hipporag \
  --llm_name "${LLM_NAME}" \
  --llm_base_url "${LLM_BASE_URL}" \
  --embedding_name "${EMB_PATH}" \
  --embedding_provider transformers \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --force_index_from_scratch "${FORCE_HIPPO_INDEX}" \
  --sample_size 0 \
  --save_dir "${HIPPO_SAVE_PREFIX}"

python "${ROOT}/scripts/experiment_tools.py" embedding-print-metrics \
  --result "${HIPPO_DIR}/metrics.json" --method hippo

# ---------- 2) PathCondRAG PC3 reuse Hippo index ----------
echo ""
echo "[2/2] PathCondRAG-PC3 reuse Hippo index -> ${PCR_DIR}"
rm -rf "${PCR_DIR}"
mkdir -p "${PCR_DIR}/index"
# seed openie + working dir (embeddings/graph)
cp -a "${HIPPO_DIR}/openie_results_ner_${LLM_NAME}.json" "${PCR_DIR}/index/" 2>/dev/null || \
  cp -a "${HIPPO_DIR}/"openie_results_ner_*.json "${PCR_DIR}/index/"
# working dir name: {llm}_{emb_with_slashes_as_underscores}
EMB_LABEL="${EMB_PATH//\//_}"
WD_NAME="${LLM_NAME}_${EMB_LABEL}"
if [[ -d "${HIPPO_DIR}/${WD_NAME}" ]]; then
  rsync -a --exclude='*.lock' "${HIPPO_DIR}/${WD_NAME}/" "${PCR_DIR}/index/${WD_NAME}/"
else
  # fallback: copy whatever model dir exists
  src_wd=$(find "${HIPPO_DIR}" -maxdepth 1 -type d -name "${LLM_NAME}_*" | head -1)
  echo "[warn] expected ${WD_NAME}, using ${src_wd}"
  rsync -a --exclude='*.lock' "${src_wd}/" "${PCR_DIR}/index/$(basename "${src_wd}")/"
fi

DATA_PATH="${HIPPO_DATA_DIR}/musique.json"
CORPUS_PATH="${HIPPO_DATA_DIR}/musique_corpus.json"

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
  python "${ROOT}/scripts/eval_dataset.py" \
  --dataset musique \
  --data_path "${DATA_PATH}" \
  --corpus_path "${CORPUS_PATH}" \
  --sample_size "${PCR_SAMPLE_SIZE}" \
  --sample_seed "${SAMPLE_SEED}" \
  --corpus_mode "${PCR_CORPUS_MODE}" \
  --eval_mode rag_qa \
  --retrieval_top_k 200 \
  --qa_top_k 5 \
  --max_qa_steps 1 \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --save_dir "${PCR_DIR}/index" \
  --output "${PCR_DIR}/result.json" \
  --llm_name "${LLM_NAME}" \
  --llm_base_url "${LLM_BASE_URL}" \
  --embedding_model_name "${EMB_PATH}" \
  --embedding_base_url "" \
  --stratified_eval --stratified_output "${PCR_DIR}/stratified.json" \
  "${PC3_ARGS[@]}"

python "${ROOT}/scripts/experiment_tools.py" embedding-print-metrics \
  --result "${PCR_DIR}/result.json" --method pcr

# drop a pair summary
python "${ROOT}/scripts/experiment_tools.py" embedding-pair-summary \
  --output-dir "${OUT_ROOT}" --mode "${MODE}" --tag "${TAG}" \
  --embedding "${EMB_PATH}" --llm "${LLM_NAME}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"

echo "[done] pair ${TAG}/${MODE} -> ${OUT_ROOT}"
