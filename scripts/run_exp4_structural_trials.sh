#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec bash "${ROOT}/scripts/run_exp4_round2.sh" \
  --mode all --run-tag structural_support \
  --baseline-profile plan_prune --selection-policy hotpot_structural \
  --sampling-policy representative \
  --exclude-run-tags structure_representative,failure_focused,bridge_support \
  --screen-size 60 --confirm-size 30 --screen-seed 942 --confirm-seed 1042 \
  --variants dag_package,structural_recovery,structural_package \
  --protected-metrics Recall@1,Recall@2,Recall@5,Recall@10,Recall@20,Recall@200 \
  --require-raw-guard "$@"
