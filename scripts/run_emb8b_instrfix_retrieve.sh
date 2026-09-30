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
conda run --no-capture-output -n "${CONDA_ENV}" python -u - <<PY
import json, logging, os, sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("HIPPORAG_KNN_DEVICE", "cpu")

HIPPO_ROOT = Path("${HIPPO_ROOT}")
sys.path.insert(0, str(HIPPO_ROOT / "src"))
sys.path.insert(0, str(HIPPO_ROOT))

from main import get_gold_docs, get_gold_answers
from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig

logging.basicConfig(level=logging.INFO)
dataset_dir = Path("${DATASETS_DIR}")
samples = json.loads((dataset_dir / "musique.json").read_text())
corpus = json.loads((dataset_dir / "musique_corpus.json").read_text())
docs = [f"{d['title']}\n{d['text']}" for d in corpus]
queries = [s["question"] for s in samples]
gold_docs = get_gold_docs(samples, "musique")

save_dir = "${HIPPO_DIR}"
config = BaseConfig(
    save_dir=save_dir,
    dataset="musique",
    llm_name="${LLM_NAME}",
    llm_base_url="${LLM_BASE_URL}",
    embedding_model_name="${EMB_PATH}",
    embedding_provider="transformers",
    embedding_batch_size=int("${EMBEDDING_BATCH_SIZE}"),
    force_index_from_scratch=False,
    force_openie_from_scratch=False,
    retrieval_top_k=200,
    linking_top_k=5,
    qa_top_k=5,
    max_new_tokens=2048,
    temperature=0.0,
    openie_mode="online",
    synonymy_edge_topk=50,
    synonymy_edge_query_batch_size=128,
    synonymy_edge_key_batch_size=1024,
    rerank_dspy_file_path=str(HIPPO_ROOT / "src" / "hipporag" / "prompts" / "dspy_prompts" / "filter_llama3.3-70B-Instruct.json"),
)
with HippoRAG(global_config=config) as rag:
    rag.index(docs)  # reuse existing graph/embeddings
    retrieval_results, retrieval_metrics = rag.retrieve(queries=queries, gold_docs=gold_docs)

metrics = {
    "dataset": "musique",
    "method": "hipporag2",
    "eval_mode": "retrieve",
    "note": "instrfix: Qwen3 uses Hippo task prompts via ST prompt=",
    "n_samples": len(samples),
    "n_docs": len(docs),
    "llm_name": "${LLM_NAME}",
    "embedding_name": "${EMB_PATH}",
    "retrieval_metrics": retrieval_metrics or {},
}
out = Path(save_dir) / "metrics_retrieve.json"
out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print("[hippo]", json.dumps(retrieval_metrics, indent=2, ensure_ascii=False))
print(f"[saved] {out}")
PY

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

python - <<PY
import json
from pathlib import Path
out = Path("${OUT_ROOT}")
h = json.loads((out / "hipporag2_musique" / "metrics_retrieve.json").read_text())
p = json.loads((out / "pathcondrag" / "result.json").read_text())
old = {}
old_path = Path("${SRC_HIPPO}").parent / "pair_summary.json"
if old_path.is_file():
    old = json.loads(old_path.read_text())
summary = {
    "note": "instrfix retrieve-only; index reused from emb_ablation_qwen3emb8b_full",
    "embedding": "${EMB_PATH}",
    "llm": "${LLM_NAME}",
    "hipporag2_retrieve": h.get("retrieval_metrics"),
    "pathcondrag_pc3_retrieve": p.get("retrieval_metrics"),
    "baseline_before_fix": {
        "hipporag2": (old.get("hipporag2") or {}).get("retrieval"),
        "pathcondrag_pc3": (old.get("pathcondrag_pc3") or {}).get("retrieval"),
    },
}
(out / "pair_summary_retrieve.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("[summary]", json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "[done] ${OUT_ROOT}"
echo "LOG=${LOG}"
