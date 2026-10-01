#!/usr/bin/env bash
# Compare serial and bounded LLM prefetch with the same Qwen3 index and questions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_INDEX="${ROOT}/outputs/emb_ablation_qwen3emb8b_distinct_b4_retrieve/pathcondrag/index"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/hop_oracle_llm_prefetch_$(date '+%Y%m%d_%H%M%S')}"
SAMPLE_SIZE="${SAMPLE_SIZE:-24}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
WORKER_LEVELS="${WORKER_LEVELS:-1 2 4}"
VLLM_LOG="${VLLM_LOG:-/root/eval/logs/vllm_qwen3.log}"
MODEL_DIR="qwen3-8b__root_models_Qwen3-Embedding-8B"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TOKENIZERS_PARALLELISM=false
export HIPPORAG_KNN_DEVICE="${HIPPORAG_KNN_DEVICE:-cpu}"
export PATHCONDRAG_LLM_MAX_IN_FLIGHT="${PATHCONDRAG_LLM_MAX_IN_FLIGHT:-4}"
export SAMPLE_SIZE SAMPLE_SEED

test -f "${SOURCE_INDEX}/${MODEL_DIR}/graph.pickle"
curl -fsS -o /dev/null http://127.0.0.1:8035/v1/models
if ! [[ "${SAMPLE_SIZE}" =~ ^[0-9]+$ ]] || (( SAMPLE_SIZE < 1 || SAMPLE_SIZE > 1000 )); then
  echo "[error] SAMPLE_SIZE must be between 1 and 1000" >&2
  exit 1
fi
if [[ " ${WORKER_LEVELS} " != *" 1 "* ]]; then
  echo "[error] WORKER_LEVELS must include serial baseline 1" >&2
  exit 1
fi
for workers in ${WORKER_LEVELS}; do
  if ! [[ "${workers}" =~ ^[1-8]$ ]]; then
    echo "[error] worker level must be an integer from 1 through 8" >&2
    exit 1
  fi
done
mkdir -p "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${RUN_ROOT}/run_root.txt"
source_sha="$(sha256sum "${SOURCE_INDEX}/${MODEL_DIR}/graph.pickle" | cut -d' ' -f1)"

echo "[setup] run_root=${RUN_ROOT} sample_size=${SAMPLE_SIZE} seed=${SAMPLE_SEED} worker_levels=${WORKER_LEVELS}"
echo "[setup] source_graph_sha=${source_sha} llm_max_in_flight=${PATHCONDRAG_LLM_MAX_IN_FLIGHT}"
echo "[setup] vllm_log=${VLLM_LOG}"

for workers in ${WORKER_LEVELS}; do
  case_dir="${RUN_ROOT}/workers_${workers}"
  index_dir="${case_dir}/index"
  result_path="${case_dir}/result.json"
  mkdir -p "${case_dir}"

  if test -e "${result_path}"; then
    echo "[error] result already exists; choose a new RUN_ROOT: ${result_path}" >&2
    exit 1
  fi

  echo "[copy] workers=${workers} copying original index and cache"
  mkdir -p "${index_dir}"
  rsync -a --exclude='*.lock' "${SOURCE_INDEX}/" "${index_dir}/"
  copied_sha="$(sha256sum "${index_dir}/${MODEL_DIR}/graph.pickle" | cut -d' ' -f1)"
  if test "${copied_sha}" != "${source_sha}"; then
    echo "[error] graph checksum mismatch for workers=${workers}" >&2
    exit 1
  fi

  log_start="$(stat -c %s "${VLLM_LOG}")"
  start_ts="$(date +%s)"
  echo "[start] workers=${workers} $(date '+%F %T %Z')"
  conda run --no-capture-output -n rag python -u "${ROOT}/scripts/eval_dataset.py" \
    --dataset musique \
    --data_path /root/datasets/musique.json \
    --corpus_path /root/datasets/musique_corpus.json \
    --sample_size "${SAMPLE_SIZE}" --sample_seed "${SAMPLE_SEED}" \
    --corpus_mode full --eval_mode retrieve \
    --retrieval_top_k 200 --qa_top_k 5 --max_qa_steps 1 --max_new_tokens 2048 \
    --embedding_model_name /root/models/Qwen3-Embedding-8B --embedding_batch_size 4 \
    --llm_name qwen3-8b --llm_base_url http://127.0.0.1:8035/v1 \
    --hop_source benchmark --hop_force_max 4 --qd_max_sub_questions 4 \
    --use_iterative_retrieval --use_query_decomposition --use_path_conditioned_qd \
    --llm_prefetch_workers "${workers}" \
    --save_dir "${index_dir}" --output "${result_path}"
  end_ts="$(date +%s)"
  log_end="$(stat -c %s "${VLLM_LOG}")"
  python "${ROOT}/scripts/experiment_tools.py" safe-prefetch-report \
    --result "${result_path}" --case-dir "${case_dir}" \
    --elapsed "$((end_ts-start_ts))" --sample-size "${SAMPLE_SIZE}" \
    --vllm-log "${VLLM_LOG}" --log-start "${log_start}" --log-end "${log_end}"
done

python "${ROOT}/scripts/experiment_tools.py" safe-prefetch-compare \
  --run-root "${RUN_ROOT}" --samples "/root/datasets/musique.json"
