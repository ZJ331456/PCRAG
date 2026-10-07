#!/usr/bin/env bash
# Reuse the three frozen public indexes for exp4 + plan_prune + dag_package.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/3multi_hop_datasets_results_10_5}"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
RUN_LOG="${RUN_LOG:-${OUT_ROOT}/logs/run_plan_prune_dag_package_full.log}"
mkdir -p "${OUT_ROOT}/logs" "$(dirname "${RUN_LOG}")"
exec 9>"${ROOT}/outputs/.recall_top5_improvements.lock"
flock -n 9 || { echo "Another retrieval experiment is running." >&2; exit 1; }
exec > >(tee -a "${RUN_LOG}") 2>&1

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

"${RAG_PYTHON}" -B -u \
  "${ROOT}/scripts/multi_dataset_retrieval_plan_prune_dag_package.py" \
  --out-root "${OUT_ROOT}" "$@"
