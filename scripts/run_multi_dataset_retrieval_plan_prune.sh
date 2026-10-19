#!/usr/bin/env bash
# Full three-dataset retrieval: exp4 + plan_prune on frozen shared_indexes.
# Writes cases/<dataset>/exp4_dependency_binding_plan_prune.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/3multi_hop_datasets_results_10_5}"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
mkdir -p "${OUT_ROOT}/logs"
exec 9>"${ROOT}/outputs/.recall_top5_improvements.lock"
flock -n 9 || { echo "Another retrieval experiment is running." >&2; exit 1; }
exec > >(tee -a "${OUT_ROOT}/logs/run_plan_prune_full.log") 2>&1

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

"${RAG_PYTHON}" -B -u "${ROOT}/scripts/multi_dataset_retrieval_plan_prune.py" --out-root "${OUT_ROOT}" "$@"
