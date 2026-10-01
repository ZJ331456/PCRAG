#!/usr/bin/env bash
# Seven leave-one-component-out retrieval ablations of the existing Qwen3 PC3 run.
# Each case starts from a private copy of the PathCondRAG index used by that run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIR_ROOT="${ROOT}/outputs/emb_ablation_qwen3emb8b_distinct_b4_retrieve"
SOURCE_INDEX="${PAIR_ROOT}/pathcondrag/index"
SOURCE_RESULT="${PAIR_ROOT}/pathcondrag/result.json"
OUT_ROOT="${PAIR_ROOT}/pathcondrag-ablation"
MODEL_DIR="qwen3-8b__root_models_Qwen3-Embedding-8B"
EMBEDDING_MODEL="/root/models/Qwen3-Embedding-8B"
LLM_NAME="qwen3-8b"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8035/v1}"
CONDA_ENV="${CONDA_ENV:-rag}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HIPPORAG_KNN_DEVICE="${HIPPORAG_KNN_DEVICE:-cpu}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${OUT_ROOT}/results" "${OUT_ROOT}/indexes" "${OUT_ROOT}/logs"

if [[ ! -f "${SOURCE_INDEX}/${MODEL_DIR}/graph.pickle" || ! -f "${SOURCE_RESULT}" ]]; then
  echo "[error] Missing the completed PathCondRAG index/result under ${PAIR_ROOT}" >&2
  exit 1
fi
if ! curl -sf -o /dev/null "${LLM_BASE_URL}/models"; then
  echo "[error] LLM endpoint unavailable: ${LLM_BASE_URL}" >&2
  exit 1
fi

python "${ROOT}/scripts/experiment_tools.py" ablation-qwen-validate-source \
  --source-result "${SOURCE_RESULT}" --source-index "${SOURCE_INDEX}" \
  --embedding-model "${EMBEDDING_MODEL}"

SOURCE_GRAPH_SHA="$(sha256sum "${SOURCE_INDEX}/${MODEL_DIR}/graph.pickle" | cut -d' ' -f1)"
printf '%s\n' "${SOURCE_GRAPH_SHA}" > "${OUT_ROOT}/source_graph.sha256"

COMMON=(
  --dataset musique
  --data_path /root/datasets/musique.json
  --corpus_path /root/datasets/musique_corpus.json
  --sample_size 0 --sample_seed 42
  --corpus_mode full --eval_mode retrieve
  --retrieval_top_k 200 --qa_top_k 5 --max_qa_steps 1 --max_new_tokens 2048
  --embedding_batch_size 4
  --llm_name "${LLM_NAME}" --llm_base_url "${LLM_BASE_URL}"
  --embedding_model_name "${EMBEDDING_MODEL}" --embedding_base_url ""
  --hop_force_max 2 --hop_multi_min_signals 2
  --iterative_round1_top_docs 1 --iterative_round2_seed_top_k 5
  --iterative_seed_mode idf_novel --iterative_merge_alpha 0.45
  --iterative_min_seed_entities 1
  --qd_min_hops 2 --qd_max_sub_questions 3 --qd_sub_retrieval_top_k 3
  --pcqd_ground_top_docs 5 --pcqd_entity_top_k 3
  --pcqd_path_score_threshold 0.60
  --pcqd_weight_base 0.40 --pcqd_weight_static 0.20 --pcqd_weight_path 0.40
)

run_case() {
  local name="$1"; shift
  local result="${OUT_ROOT}/results/${name}.json"
  local index="${OUT_ROOT}/indexes/${name}"

  if [[ -f "${result}" ]] && python "${ROOT}/scripts/experiment_tools.py" \
    ablation-qwen-result-complete --result "${result}"
  then
    echo "[skip] ${name}: completed result exists"
    return 0
  fi

  echo "[case] ${name}: cloning the completed PathCondRAG index"
  mkdir -p "${index}"
  rsync -a --exclude='*.lock' "${SOURCE_INDEX}/" "${index}/"
  local copied_sha
  copied_sha="$(sha256sum "${index}/${MODEL_DIR}/graph.pickle" | cut -d' ' -f1)"
  if [[ "${copied_sha}" != "${SOURCE_GRAPH_SHA}" ]]; then
    echo "[error] Index graph differs from the PathCondRAG source in ${name}" >&2
    exit 1
  fi

  echo "[case] ${name}: retrieval started $(date '+%F %T %Z')"
  conda run --no-capture-output -n "${CONDA_ENV}" python -u "${ROOT}/scripts/eval_dataset.py" \
    "${COMMON[@]}" --save_dir "${index}" --output "${result}" "$@"

  python "${ROOT}/scripts/experiment_tools.py" ablation-qwen-print-result \
    --result "${result}" --name "${name}"
}

echo "[source] ${SOURCE_INDEX}"
echo "[output] ${OUT_ROOT}"
echo "[model] ${EMBEDDING_MODEL}; embedding_batch_size=4; retrieve-only"

run_case wo_qcappr --use_iterative_retrieval --use_query_decomposition --use_path_conditioned_qd --no_qcappr
run_case wo_eba --use_iterative_retrieval --use_query_decomposition --use_path_conditioned_qd --no_eba
run_case wo_idf --use_iterative_retrieval --use_query_decomposition --use_path_conditioned_qd --no_entity_idf_index
run_case wo_iter --use_query_decomposition --use_path_conditioned_qd
run_case wo_qd --use_iterative_retrieval
run_case wo_pcqd --use_iterative_retrieval --use_query_decomposition
run_case wo_pathset --use_iterative_retrieval --use_query_decomposition --use_path_conditioned_qd --no_path_set_opt

python "${ROOT}/scripts/experiment_tools.py" ablation-qwen-summary \
  --out-root "${OUT_ROOT}" --source-result "${SOURCE_RESULT}"
