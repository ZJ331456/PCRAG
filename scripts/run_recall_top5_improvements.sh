#!/usr/bin/env bash
# Seven retrieve-only cases, cloned from one untouched HippoRAG2 index.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPPO_ROOT="${HIPPO_ROOT:-/root/baseline/HippoRAG}"
SOURCE_INDEX="${SOURCE_INDEX:-${ROOT}/outputs/emb_ablation_qwen3emb8b_distinct_b4_retrieve/hipporag2_musique}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
CONDA_ENV="${CONDA_ENV:-rag}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
VLLM_LOG="${VLLM_LOG:-/root/eval/logs/vllm_qwen3.log}"
DEFAULT_OUT_ROOT="${ROOT}/outputs/pathcondrag_new_innvotion_10_1"
if (( SAMPLE_SIZE > 0 )); then
  DEFAULT_OUT_ROOT="${DEFAULT_OUT_ROOT}_smoke${SAMPLE_SIZE}_$(date +%Y%m%d_%H%M%S)"
fi
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
TOOLS="${ROOT}/scripts/experiment_tools.py"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HIPPORAG_KNN_DEVICE=cpu
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8
export HIPPORAG_LLM_MAX_IN_FLIGHT=8
export HIPPO_OPENIE_MAX_WORKERS=8
export HIPPO_OPENIE_NER_WORKERS=8
export HIPPO_OPENIE_TRIPLE_WORKERS=8

mkdir -p "${OUT_ROOT}/logs"
exec 9>"${ROOT}/outputs/.recall_top5_improvements.lock"
flock -n 9 || { echo "Another seven-case run is already active." >&2; exit 1; }
exec > >(tee -a "${OUT_ROOT}/logs/run.log") 2>&1
trap 'status=$?; echo "[failed] exit=${status} line=${LINENO}; inspect ${OUT_ROOT}/logs/run.log" >&2; exit "${status}"' ERR
test -f "${VLLM_LOG}"
curl -fsS -o /dev/null "${LLM_BASE_URL}/models"

prepare_args=(
  --out-root "${OUT_ROOT}" --source-index "${SOURCE_INDEX}"
  --data-path /root/datasets/musique.json --corpus-path /root/datasets/musique_corpus.json
  --sample-size "${SAMPLE_SIZE}" --sample-seed "${SAMPLE_SEED}" --cases "${CASES:-}"
)
if [[ -n "${SAMPLE_INDICES_FILE:-}" ]]; then
  prepare_args+=(--sample-indices-file "${SAMPLE_INDICES_FILE}")
fi
python "${TOOLS}" improvement-prepare "${prepare_args[@]}"

echo "[run] OUT_ROOT=${OUT_ROOT} SAMPLE_SIZE=${SAMPLE_SIZE} seed=${SAMPLE_SEED}"
echo "[run] Qwen3-Embedding-8B batch=4; qwen3-8b workers=8; max_new_tokens=2048; thinking disabled in clients"
echo "[run] source=${SOURCE_INDEX}; all cases use identical selected_indices.json and initial cache snapshot"

COMMON=(
  --dataset musique --sample_size "${SAMPLE_SIZE}" --sample_seed "${SAMPLE_SEED}"
  --sample_indices_file "${OUT_ROOT}/selected_indices.json"
  --llm_name qwen3-8b --llm_base_url "${LLM_BASE_URL}"
  --embedding_batch_size 4 --openie_max_workers 8 --llm_prefetch_workers 8
  --eval_mode retrieve --retrieval_top_k 200 --result_top_k 10 --candidate_output_top_k 200
  --reuse_index
)
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
  --hop_source benchmark --hop_force_max 4 --qd_max_sub_questions 4
)
ALL_CASES=(
  "hipporag2|-1"
  "pathcondrag_original|0"
  "exp1_correctness|1"
  "exp2_evidence_candidates|2"
  "exp3_prefix_coverage|3"
  "exp4_dependency_binding|4"
  "exp5_verified_beam|5"
)

run_case() {
  local name="$1" stage="$2"
  local case_dir="${OUT_ROOT}/cases/${name}"
  if python "${TOOLS}" improvement-ready --out-root "${OUT_ROOT}" --name "${name}"; then
    return 0
  fi
  python "${TOOLS}" improvement-initialize-case --out-root "${OUT_ROOT}" --name "${name}"
  local log_start log_end start end
  log_start="$(stat -c %s "${VLLM_LOG}")"
  start="$(date +%s)"
  echo "[case] ${name} stage=${stage} started=$(date '+%F %T')"
  if (( stage < 0 )); then
    conda run --no-capture-output -n "${CONDA_ENV}" python -u "${HIPPO_ROOT}/main.py" \
      "${COMMON[@]}" --datasets_dir /root/datasets --rag_type hipporag \
      --embedding_name /root/models/Qwen3-Embedding-8B --embedding_provider transformers \
      --save_dir_exact --save_dir "${case_dir}/index" --output "${case_dir}/result.json" \
      2>&1 | tee "${OUT_ROOT}/logs/${name}.log"
  else
    conda run --no-capture-output -n "${CONDA_ENV}" python -u "${ROOT}/scripts/eval_dataset.py" \
      "${COMMON[@]}" "${PC3_COMMON[@]}" \
      --data_path /root/datasets/musique.json --corpus_path /root/datasets/musique_corpus.json \
      --corpus_mode full --qa_top_k 5 --max_qa_steps 1 --max_new_tokens 2048 \
      --embedding_model_name /root/models/Qwen3-Embedding-8B \
      --improvement_stage "${stage}" \
      --save_dir "${case_dir}/index" --output "${case_dir}/result.json" \
      --stratified_eval --stratified_output "${case_dir}/stratified.json" \
      2>&1 | tee "${OUT_ROOT}/logs/${name}.log"
  fi
  end="$(date +%s)"
  log_end="$(stat -c %s "${VLLM_LOG}")"
  python "${TOOLS}" improvement-report \
    --out-root "${OUT_ROOT}" --name "${name}" --elapsed "$((end-start))" \
    --vllm-log "${VLLM_LOG}" --log-start "${log_start}" --log-end "${log_end}"
}

for spec in "${ALL_CASES[@]}"; do
  IFS='|' read -r name stage <<< "${spec}"
  if [[ -n "${CASES:-}" && " ${CASES} " != *" ${name} "* ]]; then continue; fi
  run_case "${name}" "${stage}"
done
python "${TOOLS}" improvement-summary --out-root "${OUT_ROOT}"
echo "[done] all selected cases validated: ${OUT_ROOT}"
