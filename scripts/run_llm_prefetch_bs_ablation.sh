#!/usr/bin/env bash
# Ablate retrieval-phase LLM prefetch concurrency (llm_prefetch_workers).
# Reuses Qwen3-Embedding-8B index; retrieve-only; full MuSiQue by default.
#
# Cases (order):
#   1) pathcondrag_bs8            hop_source=benchmark  workers=8
#   2) pathcondrag_bs8_fixedhop   hop_source=estimated   workers=8  (legacy hop_force_max=2)
#   3) pathcondrag_bs4            hop_source=benchmark  workers=4
#   4) pathcondrag_bs2            hop_source=benchmark  workers=2
#
# Usage:
#   SAMPLE_SIZE=2 bash scripts/run_llm_prefetch_bs_ablation.sh          # smoke
#   SAMPLE_SIZE=0 bash scripts/run_llm_prefetch_bs_ablation.sh          # full
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_INDEX="${SOURCE_INDEX:-${ROOT}/outputs/emb_ablation_qwen3emb8b_distinct_b4_retrieve/pathcondrag/index}"
MODEL_DIR="qwen3-8b__root_models_Qwen3-Embedding-8B"
EMB_PATH="/root/models/Qwen3-Embedding-8B"
LLM_NAME="qwen3-8b"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
CONDA_ENV="${CONDA_ENV:-rag}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"
VLLM_LOG="${VLLM_LOG:-/root/eval/logs/vllm_qwen3.log}"
MAX_IN_FLIGHT="${PATHCONDRAG_LLM_MAX_IN_FLIGHT:-8}"

MODE_TAG="full"
if (( SAMPLE_SIZE > 0 )); then
  MODE_TAG="smoke${SAMPLE_SIZE}"
fi
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/llm_prefetch_bs_ablation_qwen3emb8b_${MODE_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/logs}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-${LOG_DIR}/llm_prefetch_bs_ablation_${MODE_TAG}_${TS}.log}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TOKENIZERS_PARALLELISM=false
export HIPPORAG_KNN_DEVICE="${HIPPORAG_KNN_DEVICE:-cpu}"
export PATHCONDRAG_LLM_MAX_IN_FLIGHT="${MAX_IN_FLIGHT}"
export PYTHONUNBUFFERED=1

test -f "${SOURCE_INDEX}/${MODEL_DIR}/graph.pickle"
curl -fsS -o /dev/null "${LLM_BASE_URL}/models"

source_sha="$(sha256sum "${SOURCE_INDEX}/${MODEL_DIR}/graph.pickle" | cut -d' ' -f1)"
echo "================================================================="
echo " LLM prefetch bs ablation  MODE=${MODE_TAG} sample_size=${SAMPLE_SIZE}"
echo " SOURCE=${SOURCE_INDEX}"
echo " OUT=${OUT_ROOT}"
echo " EMB=${EMB_PATH} emb_batch=${EMBEDDING_BATCH_SIZE}"
echo " llm_max_in_flight=${PATHCONDRAG_LLM_MAX_IN_FLIGHT}"
echo " graph_sha=${source_sha}"
echo "================================================================="

# Shared PC3 retrieve knobs (only hop policy + llm_prefetch_workers vary).
PC3_COMMON=(
  --use_iterative_retrieval
  --iterative_round1_top_docs 1 --iterative_round2_seed_top_k 5
  --iterative_seed_mode idf_novel --iterative_merge_alpha 0.45
  --iterative_min_seed_entities 1
  --use_query_decomposition --use_path_conditioned_qd
  --qd_min_hops 2 --qd_sub_retrieval_top_k 3
  --pcqd_ground_top_docs 5 --pcqd_entity_top_k 3
  --pcqd_path_score_threshold 0.60
  --pcqd_weight_base 0.40 --pcqd_weight_static 0.20 --pcqd_weight_path 0.40
)

run_case() {
  local name="$1"
  local workers="$2"
  local hop_mode="$3"   # benchmark | fixed
  local case_dir="${OUT_ROOT}/${name}"
  local index_dir="${case_dir}/index"
  local result_path="${case_dir}/result.json"

  if [[ -f "${result_path}" ]]; then
    echo "[skip] ${name} already has ${result_path}"
    return 0
  fi

  echo ""
  echo "-----------------------------------------------------------------"
  echo "[case] ${name} workers=${workers} hop=${hop_mode} $(date '+%F %T')"
  echo "-----------------------------------------------------------------"
  rm -rf "${case_dir}"
  mkdir -p "${index_dir}"
  rsync -a --exclude='*.lock' "${SOURCE_INDEX}/" "${index_dir}/"
  local copied_sha
  copied_sha="$(sha256sum "${index_dir}/${MODEL_DIR}/graph.pickle" | cut -d' ' -f1)"
  if [[ "${copied_sha}" != "${source_sha}" ]]; then
    echo "[error] graph checksum mismatch for ${name}" >&2
    exit 1
  fi

  local hop_args=()
  if [[ "${hop_mode}" == "benchmark" ]]; then
    hop_args=(--hop_source benchmark --hop_force_max 4 --qd_max_sub_questions 4)
  else
    # Legacy fixed-hop PC3: heuristic estimate capped at 2.
    hop_args=(--hop_source estimated --hop_force_max 2 --hop_multi_min_signals 2 --qd_max_sub_questions 3)
  fi

  local log_start log_end start_ts end_ts
  log_start="$(stat -c %s "${VLLM_LOG}" 2>/dev/null || echo 0)"
  start_ts="$(date +%s)"

  conda run --no-capture-output -n "${CONDA_ENV}" python -u "${ROOT}/scripts/eval_dataset.py" \
    --dataset musique \
    --data_path /root/datasets/musique.json \
    --corpus_path /root/datasets/musique_corpus.json \
    --sample_size "${SAMPLE_SIZE}" --sample_seed "${SAMPLE_SEED}" \
    --corpus_mode full --eval_mode retrieve \
    --retrieval_top_k 200 --qa_top_k 5 --max_qa_steps 1 --max_new_tokens 2048 \
    --embedding_model_name "${EMB_PATH}" --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
    --llm_name "${LLM_NAME}" --llm_base_url "${LLM_BASE_URL}" \
    --llm_prefetch_workers "${workers}" \
    --save_dir "${index_dir}" --output "${result_path}" \
    --stratified_eval --stratified_output "${case_dir}/stratified.json" \
    "${hop_args[@]}" \
    "${PC3_COMMON[@]}"

  end_ts="$(date +%s)"
  log_end="$(stat -c %s "${VLLM_LOG}" 2>/dev/null || echo 0)"

  python "${ROOT}/scripts/experiment_tools.py" prefetch-report \
    --case-dir "${case_dir}" --result "${result_path}" --name "${name}" \
    --workers "${workers}" --hop-mode "${hop_mode}" --elapsed "$((end_ts-start_ts))" \
    --sample-size "${SAMPLE_SIZE}" --vllm-log "${VLLM_LOG}" \
    --log-start "${log_start}" --log-end "${log_end}"
}

# Order: 8 (real hop) -> 8 (fixed hop) -> 4 -> 2
# Optional: CASES="pathcondrag_bs8 pathcondrag_bs8_fixedhop" to subset.
ALL_CASES=(
  "pathcondrag_bs8|8|benchmark"
  "pathcondrag_bs8_fixedhop|8|fixed"
  "pathcondrag_bs4|4|benchmark"
  "pathcondrag_bs2|2|benchmark"
)
for spec in "${ALL_CASES[@]}"; do
  IFS='|' read -r cname cworkers chop <<< "${spec}"
  if [[ -n "${CASES:-}" && " ${CASES} " != *" ${cname} "* ]]; then
    echo "[skip] ${cname} (not in CASES)"
    continue
  fi
  run_case "${cname}" "${cworkers}" "${chop}"
done

python "${ROOT}/scripts/experiment_tools.py" prefetch-summary --out-root "${OUT_ROOT}"

echo "[done] all cases -> ${OUT_ROOT}"
echo "LOG hint: ${LOG}"
