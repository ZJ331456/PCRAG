#!/usr/bin/env bash
# Four contribution-control families, eleven cases, on the existing Hippo index.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_ROOT="${REFERENCE_ROOT:-${ROOT}/outputs/pathcondrag_new_innvotion_10_1}"
SOURCE_INDEX="${SOURCE_INDEX:-${REFERENCE_ROOT}/shared_hipporag2_index}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
DEFAULT_OUT_ROOT="${REFERENCE_ROOT}"
if (( SAMPLE_SIZE > 0 )); then
  DEFAULT_OUT_ROOT="${ROOT}/outputs/exp4_contribution_smoke${SAMPLE_SIZE}_$(date +%Y%m%d_%H%M%S)"
fi
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
VLLM_LOG="${VLLM_LOG:-/root/eval/logs/vllm_qwen3.log}"
PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
TOOLS="${ROOT}/scripts/experiment_tools.py"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8
export HIPPORAG_LLM_MAX_IN_FLIGHT=8
export HIPPORAG_KNN_DEVICE=cpu
export HIPPO_OPENIE_MAX_WORKERS=8
export HIPPO_OPENIE_NER_WORKERS=8
export HIPPO_OPENIE_TRIPLE_WORKERS=8
mkdir -p "${OUT_ROOT}/logs"
exec 9>"${ROOT}/outputs/.recall_top5_improvements.lock"
flock -n 9 || { echo "Another retrieval experiment is running." >&2; exit 1; }
exec > >(tee -a "${OUT_ROOT}/logs/run2.log") 2>&1
trap 'status=$?; echo "[failed-ablation] exit=${status} line=${LINENO}; inspect logs/run2.log" >&2; exit "${status}"' ERR
test -f "${VLLM_LOG}"
curl -fsS -o /dev/null "${LLM_BASE_URL}/models"

python "${TOOLS}" exp4-ablation-prepare \
  --out-root "${OUT_ROOT}" --reference-root "${REFERENCE_ROOT}" --source-index "${SOURCE_INDEX}" \
  --sample-size "${SAMPLE_SIZE}" --sample-seed "${SAMPLE_SEED:-42}" \
  --sample-indices-file "${SAMPLE_INDICES_FILE:-}" --cases "${CASES:-}"
echo "[run-ablation] OUT_ROOT=${OUT_ROOT}; existing source=${SOURCE_INDEX}; embedding batch=4; LLM workers=8; thinking=false; max_new_tokens=2048"
COMMON=(
  --dataset musique --data_path /root/datasets/musique.json --corpus_path /root/datasets/musique_corpus.json
  --sample_size "${SAMPLE_SIZE}" --sample_seed "${SAMPLE_SEED:-42}"
  --sample_indices_file "${OUT_ROOT}/ablation_selected_indices.json"
  --corpus_mode full --reuse_index --eval_mode retrieve
  --retrieval_top_k 200 --result_top_k 10 --candidate_output_top_k 200
  --embedding_model_name /root/models/Qwen3-Embedding-8B --embedding_batch_size 4
  --llm_name qwen3-8b --llm_base_url "${LLM_BASE_URL}" --llm_prefetch_workers 8 --openie_max_workers 8
  --max_new_tokens 2048 --qa_top_k 5 --max_qa_steps 1
  --use_iterative_retrieval --iterative_round1_top_docs 1 --iterative_round2_seed_top_k 5
  --iterative_seed_mode idf_novel --iterative_merge_alpha 0.45 --iterative_min_seed_entities 1
  --use_query_decomposition --use_path_conditioned_qd --qd_min_hops 2 --qd_sub_retrieval_top_k 3
  --pcqd_ground_top_docs 5 --pcqd_entity_top_k 3 --pcqd_path_score_threshold 0.60
  --pcqd_weight_base 0.40 --pcqd_weight_static 0.20 --pcqd_weight_path 0.40
  --hop_source benchmark --hop_force_max 4 --qd_max_sub_questions 4
  --evidence_ablation_inputs_file "${OUT_ROOT}/ablation_inputs.json"
)
SPECS="$(python "${TOOLS}" exp4-ablation-list --cases "${CASES:-}")"
while IFS='|' read -r name mode binding selection stage; do
  case_dir="${OUT_ROOT}/cases/${name}"
  if python "${TOOLS}" exp4-ablation-ready --out-root "${OUT_ROOT}" --name "${name}"; then continue; fi
  python "${TOOLS}" exp4-ablation-initialize-case --out-root "${OUT_ROOT}" --name "${name}"
  log_start="$(stat -c %s "${VLLM_LOG}")"
  start="$(date +%s)"
  echo "[case-ablation] ${name} mode=${mode} binding=${binding} selection=${selection} stage=${stage} started=$(date '+%F %T')"
  "${PYTHON}" -u "${ROOT}/scripts/eval_dataset.py" "${COMMON[@]}" \
    --improvement_stage "${stage}" --evidence_ablation_mode "${mode}" \
    --evidence_binding_mode "${binding}" --evidence_selection_mode "${selection}" \
    --save_dir "${case_dir}/index" --output "${case_dir}/result.json" \
    --stratified_eval --stratified_output "${case_dir}/stratified.json" \
    2>&1 | tee "${OUT_ROOT}/logs/${name}.log"
  end="$(date +%s)"
  log_end="$(stat -c %s "${VLLM_LOG}")"
  python "${TOOLS}" exp4-ablation-report --out-root "${OUT_ROOT}" --name "${name}" \
    --elapsed "$((end-start))" --vllm-log "${VLLM_LOG}" --log-start "${log_start}" --log-end "${log_end}"
done <<< "${SPECS}"
python "${TOOLS}" exp4-ablation-summary --out-root "${OUT_ROOT}"
echo "[done-ablation] all selected contribution controls validated: ${OUT_ROOT}"
