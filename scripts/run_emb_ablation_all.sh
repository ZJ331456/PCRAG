#!/usr/bin/env bash
# Run embedding ablation: smoke (optional) then full 8B pair -> 0.6B pair.
# Unloads 8B by process exit before starting 0.6B.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/logs}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${MASTER_LOG:-${LOG_DIR}/emb_ablation_master_${TS}.log}"

run_pair() {
  local mode="$1" tag="$2"
  echo ""
  echo "######################################################################"
  echo "# START $(date '+%F %T')  MODE=${mode} EMB_TAG=${tag}"
  echo "######################################################################"
  MODE="${mode}" EMB_TAG="${tag}" \
    bash "${ROOT}/scripts/run_emb_ablation_pair.sh"
  # encourage GPU release between pairs
  conda run --no-capture-output -n rag \
    python "${ROOT}/scripts/experiment_tools.py" embedding-release-memory || true
}

echo "Master log: ${MASTER_LOG}"
{
  echo "emb ablation master ${TS}"
  if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    run_pair smoke 8b
    run_pair smoke 0.6b
    echo "##### SMOKE DONE $(date '+%F %T') #####"
  fi
  if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    echo "[smoke-only] exit"
    exit 0
  fi
  run_pair full 8b
  run_pair full 0.6b
  echo "##### FULL DONE $(date '+%F %T') #####"
} 2>&1 | tee -a "${MASTER_LOG}"

echo "[done] ${MASTER_LOG}"
