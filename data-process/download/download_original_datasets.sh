#!/usr/bin/env bash
# Download original HotpotQA / 2WikiMultihopQA / MuSiQue into data-process/raw/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="${ROOT}/../raw"
mkdir -p "${RAW}/hotpotqa" "${RAW}/2wikimultihopqa" "${RAW}/musique"
LOG="${RAW}/download_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

download() {
  local url="$1" dest="$2"
  if [[ -f "${dest}" && $(stat -c%s "${dest}") -gt 1000 ]]; then
    echo "[skip] exists $(du -h "${dest}" | cut -f1)  ${dest}"
    return 0
  fi
  echo "[get] ${url}"
  echo "  -> ${dest}"
  mkdir -p "$(dirname "${dest}")"
  # resume-friendly
  curl -L --fail --retry 5 --retry-delay 3 -C - -o "${dest}.partial" "${url}"
  mv "${dest}.partial" "${dest}"
  echo "[ok] $(du -h "${dest}" | cut -f1)  ${dest}"
}

echo "================================================================="
echo " Downloading original multi-hop QA datasets -> ${RAW}"
echo " started $(date '+%F %T')"
echo "================================================================="

# ---------- HotpotQA (official CMU) ----------
# https://hotpotqa.github.io/
HP="http://curtis.ml.cmu.edu/datasets/hotpot"
download "${HP}/hotpot_train_v1.1.json"            "${RAW}/hotpotqa/hotpot_train_v1.1.json"
download "${HP}/hotpot_dev_distractor_v1.json"     "${RAW}/hotpotqa/hotpot_dev_distractor_v1.json"
download "${HP}/hotpot_dev_fullwiki_v1.json"       "${RAW}/hotpotqa/hotpot_dev_fullwiki_v1.json"
download "${HP}/hotpot_test_fullwiki_v1.json"      "${RAW}/hotpotqa/hotpot_test_fullwiki_v1.json"

# ---------- 2WikiMultihopQA (HF mirror of Alab-NII release) ----------
# Official repo: https://github.com/Alab-NII/2wikimultihop
W2="https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main"
download "${W2}/train.json?download=true" "${RAW}/2wikimultihopqa/train.json"
download "${W2}/dev.json?download=true"   "${RAW}/2wikimultihopqa/dev.json"
download "${W2}/test.json?download=true"  "${RAW}/2wikimultihopqa/test.json"

# ---------- MuSiQue (HF mirror of official Ans + Full) ----------
# Official: https://github.com/StonyBrookNLP/musique
MQ="https://huggingface.co/datasets/bdsaglam/musique/resolve/main"
download "${MQ}/musique_ans_v1.0_train.jsonl?download=true"  "${RAW}/musique/musique_ans_v1.0_train.jsonl"
download "${MQ}/musique_ans_v1.0_dev.jsonl?download=true"    "${RAW}/musique/musique_ans_v1.0_dev.jsonl"
download "${MQ}/musique_ans_v1.0_test.jsonl?download=true"   "${RAW}/musique/musique_ans_v1.0_test.jsonl"
download "${MQ}/musique_full_v1.0_train.jsonl?download=true" "${RAW}/musique/musique_full_v1.0_train.jsonl"
download "${MQ}/musique_full_v1.0_dev.jsonl?download=true"   "${RAW}/musique/musique_full_v1.0_dev.jsonl"
download "${MQ}/musique_full_v1.0_test.jsonl?download=true"  "${RAW}/musique/musique_full_v1.0_test.jsonl"

echo ""
echo "================================================================="
echo " Summary"
echo "================================================================="
du -sh "${RAW}/hotpotqa" "${RAW}/2wikimultihopqa" "${RAW}/musique" 2>/dev/null || true
find "${RAW}" -type f \( -name '*.json' -o -name '*.jsonl' \) -printf '%10s  %p\n' | sort

# quick counts
source /root/anaconda3/etc/profile.d/conda.sh
conda activate rag
RAW="${RAW}" python - <<'PY'
import json
import os
from pathlib import Path
raw = Path(os.environ["RAW"])

def n_json(p):
    d = json.loads(p.read_text())
    return len(d) if isinstance(d, list) else None

def n_jsonl(p):
    return sum(1 for _ in p.open())

print("\n[counts]")
for p in sorted((raw/"hotpotqa").glob("*.json")):
    print(f"  hotpotqa/{p.name}: {n_json(p)}")
for p in sorted((raw/"2wikimultihopqa").glob("*.json")):
    print(f"  2wiki/{p.name}: {n_json(p)}")
for p in sorted((raw/"musique").glob("*.jsonl")):
    print(f"  musique/{p.name}: {n_jsonl(p)}")
PY

# README pointer
cat > "${RAW}/README.md" <<'EOF'
# Original multi-hop QA datasets

Downloaded for PathCondRAG hop labeling / experiments.

| Dataset | Path | Source |
|---------|------|--------|
| HotpotQA | `hotpotqa/` | https://hotpotqa.github.io/ (CMU) |
| 2WikiMultihopQA | `2wikimultihopqa/` | https://github.com/Alab-NII/2wikimultihop (HF mirror voidful) |
| MuSiQue | `musique/` | https://github.com/StonyBrookNLP/musique (HF mirror bdsaglam) |

Re-download: `bash PathCondRAG/data-process/download/download_original_datasets.sh`
# Prefer Python (HF mirrors): `python PathCondRAG/data-process/download/download_original_datasets.py`
EOF

echo "[done] $(date '+%F %T')  log=${LOG}"
