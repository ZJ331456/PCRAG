#!/usr/bin/env bash
# Frozen-candidate CPU selection with the existing local LLM API; no reindexing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/3multi_hop_datasets_results_10_5}"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
mkdir -p "${OUT_ROOT}/logs"
exec 9>"${ROOT}/outputs/.recall_top5_improvements.lock"
flock -n 9 || { echo "Another retrieval experiment is running." >&2; exit 1; }
exec > >(tee -a "${OUT_ROOT}/logs/run_swap_verification_trials.log") 2>&1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8
"${RAG_PYTHON}" -B -u "${ROOT}/scripts/exp4_swap_verification_trials.py" --out-root "${OUT_ROOT}" "$@"
