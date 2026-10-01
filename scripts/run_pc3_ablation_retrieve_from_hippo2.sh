#!/usr/bin/env bash
# PC3 component ablation (retrieve-only) on the Hippo2-reused index.
# Index is NOT rebuilt; each case writes its own result JSON.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/env_qwen3_nvembed.sh"

export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HIPPORAG_KNN_DEVICE="${HIPPORAG_KNN_DEVICE:-cpu}"
export TOKENIZERS_PARALLELISM=false

INDEX_DIR="${INDEX_DIR:-${ROOT}/outputs/pc3_musique_full_from_hipppo2_index/index}"
DATA_PATH="${DATA_PATH:-/root/datasets/musique.json}"
CORPUS_PATH="${CORPUS_PATH:-/root/datasets/musique_corpus.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-200}"
CONDA_ENV="${CONDA_ENV:-rag}"
CASE_REGEX="${CASE_REGEX:-}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/pc3_ablation_from_hippo2_${TS}}"
RESULT_DIR="${OUT_ROOT}/results"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/logs}"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
LOG="${LOG:-${LOG_DIR}/pc3_ablation_from_hippo2_${TS}.log}"

if [[ ! -f "${INDEX_DIR}/qwen3-8b__root_models_NV-Embed-v2/graph.pickle" ]]; then
  echo "[ERROR] missing graph at ${INDEX_DIR}/qwen3-8b__root_models_NV-Embed-v2/graph.pickle" >&2
  exit 1
fi

curl -sf -o /dev/null "${HIPPO_LLM_BASE_URL}/models"

COMMON=(
  --dataset musique
  --data_path "${DATA_PATH}"
  --corpus_path "${CORPUS_PATH}"
  --sample_size "${SAMPLE_SIZE}"
  --sample_seed "${SAMPLE_SEED}"
  --corpus_mode full
  --eval_mode retrieve
  --retrieval_top_k "${RETRIEVAL_TOP_K}"
  --qa_top_k 5
  --max_qa_steps 1
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}"
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE:-2}"
  --save_dir "${INDEX_DIR}"
  --llm_name "${HIPPO_LLM_NAME}"
  --llm_base_url "${HIPPO_LLM_BASE_URL}"
  --embedding_model_name "${HIPPO_EMBEDDING_MODEL_NAME}"
  --embedding_base_url "${HIPPO_EMBEDDING_BASE_URL}"
)

PC3_HOP=(--hop_force_max 2 --hop_multi_min_signals 2)
PC3_ITER=(
  --use_iterative_retrieval
  --iterative_round1_top_docs 1 --iterative_round2_seed_top_k 5
  --iterative_seed_mode idf_novel --iterative_merge_alpha 0.45
  --iterative_min_seed_entities 1
)
PC3_STATIC_QD=(
  --use_query_decomposition
  --qd_min_hops 2 --qd_max_sub_questions 3 --qd_sub_retrieval_top_k 3
)
PC3_PCQD=(
  --use_query_decomposition --use_path_conditioned_qd
  --qd_min_hops 2 --qd_max_sub_questions 3 --qd_sub_retrieval_top_k 3
  --pcqd_ground_top_docs 5 --pcqd_entity_top_k 3
  --pcqd_path_score_threshold 0.60
  --pcqd_weight_base 0.40 --pcqd_weight_static 0.20 --pcqd_weight_path 0.40
)
PC3_FULL=("${PC3_HOP[@]}" "${PC3_ITER[@]}" "${PC3_PCQD[@]}")

run_case() {
  local name="$1"; shift
  if [[ -n "${CASE_REGEX}" && ! "${name}" =~ ${CASE_REGEX} ]]; then
    echo "[skip] ${name} (CASE_REGEX=${CASE_REGEX})"
    return 0
  fi
  local out="${RESULT_DIR}/${name}.json"
  if [[ "${SKIP_EXISTING:-1}" == "1" && -f "${out}" ]]; then
    if python "${ROOT}/scripts/experiment_tools.py" ablation-pc3-result-complete \
      --result "${out}"
    then
      echo "[skip] ${name} already has retrieval_metrics -> ${out}"
      return 0
    fi
  fi
  echo ""
  echo "================================================================="
  echo "[case] ${name}"
  echo "  out=${out}"
  echo "  extra: $*"
  echo "================================================================="
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${ROOT}/scripts/eval_dataset.py" \
    "${COMMON[@]}" --output "${out}" "$@"
  python "${ROOT}/scripts/experiment_tools.py" ablation-pc3-print-result \
    --result "${out}" --name "${name}"
}

echo "================================================================="
echo " PC3 ablation (retrieve-only) on Hippo2 index"
echo " index=${INDEX_DIR}"
echo " out=${OUT_ROOT}"
echo " log=${LOG}"
echo "================================================================="

# 1) Hippo-like: all PC3 innovations off
run_case abl_hipporag \
  --no_qcappr --no_eba --no_path_set_opt \
  --no_entity_idf_index --no_bridge_cache_index

# 2) Core graph enhancers only (QCAPPR+EBA, defaults on; no pathset/iter/QD)
run_case abl_core_minimal \
  "${PC3_HOP[@]}" --no_path_set_opt

# 3) hop + iterative (no QD)
run_case abl_no_qd \
  "${PC3_HOP[@]}" "${PC3_ITER[@]}"

# 4) hop + iterative + static QD (no path-conditioned)
run_case abl_static_qd \
  "${PC3_HOP[@]}" "${PC3_ITER[@]}" "${PC3_STATIC_QD[@]}"

# 5) full PC3 minus path-set
run_case abl_wo_pathset \
  "${PC3_FULL[@]}" --no_path_set_opt

# 6) full PC3 minus EBA
run_case abl_wo_eba \
  "${PC3_FULL[@]}" --no_eba

# 7) full PC3 minus QCAPPR
run_case abl_wo_qcappr \
  "${PC3_FULL[@]}" --no_qcappr

# 8) full PC3 minus iterative
run_case abl_wo_iter \
  "${PC3_HOP[@]}" "${PC3_PCQD[@]}"

# 9) full PC3 (reference)
run_case abl_pc3 \
  "${PC3_FULL[@]}"

# summary table
python "${ROOT}/scripts/experiment_tools.py" ablation-pc3-summary \
  --result-dir "${RESULT_DIR}"

echo "[done] results=${RESULT_DIR}"
