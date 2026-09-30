#!/usr/bin/env bash
# =============================================================================
# PathCondRAG runner — full PC3, optional MPCE (default OFF)
#
# Full PC3 = hop control + iterative retrieval + path-conditioned QD
# MPCE     = set USE_MPCE=1 to enable (best γ=0.40 from prior sweeps)
#
# Examples:
#   # smoke (2 samples, sample-only corpus, retrieve)
#   SAMPLE_SIZE=2 CORPUS_MODE=sample_only EVAL_MODE=retrieve \
#     bash /root/PathCondRAG/scripts/run_pc3.sh
#
#   # full MuSiQue RAG-QA with MPCE
#   SAMPLE_SIZE=0 CORPUS_MODE=full EVAL_MODE=rag_qa USE_MPCE=1 \
#     DATASET=musique bash /root/PathCondRAG/scripts/run_pc3.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/env_qwen3_nvembed.sh"

DATASET="${DATASET:-musique}"
DATA_PATH="${DATA_PATH:-/root/datasets/${DATASET}.json}"
CORPUS_PATH="${CORPUS_PATH:-/root/datasets/${DATASET}_corpus.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-2}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
CORPUS_MODE="${CORPUS_MODE:-sample_only}"
EVAL_MODE="${EVAL_MODE:-retrieve}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-200}"
QA_TOP_K="${QA_TOP_K:-5}"
USE_MPCE="${USE_MPCE:-0}"
FORCE_INDEX="${FORCE_INDEX:-0}"
STRATIFIED_EVAL="${STRATIFIED_EVAL:-1}"
CONDA_ENV="${CONDA_ENV:-rag}"

TS="$(date +%Y%m%d_%H%M%S)"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/outputs}"
TAG="pc3"
[[ "${USE_MPCE}" == "1" ]] && TAG="pc3_mpce"
OUT_DIR="${OUT_DIR:-${SAVE_ROOT}/${TAG}_${DATASET}_${TS}}"
mkdir -p "${OUT_DIR}"

# Full PC3
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

MPCE_ARGS=()
if [[ "${USE_MPCE}" == "1" ]]; then
  MPCE_ARGS=(
    --use_mpce --mpce_gamma 0.40 --mpce_boost_cap 0.30
    --mpce_candidate_top_k 60 --mpce_consensus_min_paths 2
    --mpce_entity_consensus_top_k 3 --mpce_entity_consensus_weight 0.5
  )
fi

COMMON=(
  --dataset "${DATASET}"
  --data_path "${DATA_PATH}"
  --corpus_path "${CORPUS_PATH}"
  --sample_size "${SAMPLE_SIZE}"
  --sample_seed "${SAMPLE_SEED}"
  --corpus_mode "${CORPUS_MODE}"
  --eval_mode "${EVAL_MODE}"
  --retrieval_top_k "${RETRIEVAL_TOP_K}"
  --qa_top_k "${QA_TOP_K}"
  --max_qa_steps 1
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}"
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}"
  --save_dir "${OUT_DIR}/index"
  --output "${OUT_DIR}/result.json"
  --llm_name "${HIPPO_LLM_NAME}"
  --llm_base_url "${HIPPO_LLM_BASE_URL}"
  --embedding_model_name "${HIPPO_EMBEDDING_MODEL_NAME}"
  --embedding_base_url "${HIPPO_EMBEDDING_BASE_URL}"
)
[[ "${STRATIFIED_EVAL}" == "1" ]] && COMMON+=(--stratified_eval --stratified_output "${OUT_DIR}/stratified.json")
[[ "${FORCE_INDEX}" == "1" ]] && COMMON+=(--force_index_from_scratch --force_openie_from_scratch)

echo "================================================================="
echo " PathCondRAG  tag=${TAG}  USE_MPCE=${USE_MPCE}"
echo " dataset=${DATASET}  sample=${SAMPLE_SIZE}  corpus=${CORPUS_MODE}"
echo " eval=${EVAL_MODE}  out=${OUT_DIR}"
echo "================================================================="

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  python "${ROOT}/scripts/eval_dataset.py"
  "${COMMON[@]}" "${PC3_ARGS[@]}" "${MPCE_ARGS[@]}"
)
printf '+ '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

echo ""
echo "[done] result: ${OUT_DIR}/result.json"
