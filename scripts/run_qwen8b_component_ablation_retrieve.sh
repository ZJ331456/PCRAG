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

SOURCE_RESULT="${SOURCE_RESULT}" SOURCE_INDEX="${SOURCE_INDEX}" EMBEDDING_MODEL="${EMBEDDING_MODEL}" python - <<'PY'
import json
import os
from pathlib import Path

d = json.loads(Path(os.environ["SOURCE_RESULT"]).read_text())
c = d["runtime_config"]
expected = {
    "embedding_model_name": os.environ["EMBEDDING_MODEL"],
    "embedding_batch_size": 4,
    "retrieval_top_k": 200,
    "hop_force_max": 2,
    "hop_multi_min_signals": 2,
    "use_qcappr": True,
    "use_eba": True,
    "use_entity_idf_index": True,
    "use_bridge_cache_index": True,
    "use_iterative_retrieval": True,
    "use_query_decomposition": True,
    "use_path_conditioned_qd": True,
    "use_path_set_optimization": True,
}
for key, value in expected.items():
    if c.get(key) != value:
        raise SystemExit(f"Source result mismatch: {key}={c.get(key)!r}, expected {value!r}")
if d.get("sample_size_effective") != 1000 or d.get("indexed_docs") != 11656:
    raise SystemExit("Source result is not the completed 1000-question full-corpus run")
manifest = json.loads((Path(os.environ["SOURCE_INDEX"]) / "qwen3-8b__root_models_Qwen3-Embedding-8B" / "index_manifest.json").read_text())
if manifest["embedding"]["model_name"] != os.environ["EMBEDDING_MODEL"]:
    raise SystemExit("Source index embedding does not match the requested Qwen3-Embedding-8B")
print("[verified] Existing PathCondRAG Qwen3 index and full-run configuration")
PY

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

  if [[ -f "${result}" ]] && RESULT="${result}" NAME="${name}" python - <<'PY'
import json
import os
from pathlib import Path
d = json.loads(Path(os.environ["RESULT"]).read_text())
c = d.get("runtime_config", {})
ok = (d.get("sample_size_effective") == 1000
      and d.get("eval_mode") == "retrieve"
      and d.get("retrieval_metrics", {}).get("Recall@20") is not None
      and c.get("embedding_batch_size") == 4
      and c.get("embedding_model_name") == "/root/models/Qwen3-Embedding-8B")
raise SystemExit(0 if ok else 1)
PY
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

  RESULT="${result}" NAME="${name}" python - <<'PY'
import json
import os
from pathlib import Path
d = json.loads(Path(os.environ["RESULT"]).read_text())
r = d["retrieval_metrics"]
c = d["runtime_config"]
keys = ("Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20")
print("[done]", os.environ["NAME"], {k: r.get(k) for k in keys},
      "active=", {k: c[k] for k in ("use_qcappr", "use_eba", "use_entity_idf_index",
                                     "use_iterative_retrieval", "use_query_decomposition",
                                     "use_path_conditioned_qd", "use_path_set_optimization")},
      flush=True)
PY
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

OUT_ROOT="${OUT_ROOT}" SOURCE_RESULT="${SOURCE_RESULT}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
full = json.loads(Path(os.environ["SOURCE_RESULT"]).read_text())
keys = ["Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"]
summary = {
    "source_index": str(root.parent / "pathcondrag" / "index"),
    "embedding_model": "/root/models/Qwen3-Embedding-8B",
    "embedding_batch_size": 4,
    "full_reference_existing_result": {k: full["retrieval_metrics"].get(k) for k in keys},
    "ablations": {},
}
for name in ("wo_qcappr", "wo_eba", "wo_idf", "wo_iter", "wo_qd", "wo_pcqd", "wo_pathset"):
    d = json.loads((root / "results" / f"{name}.json").read_text())
    summary["ablations"][name] = {
        "retrieval": {k: d["retrieval_metrics"].get(k) for k in keys},
        "delta_vs_existing_full": {
            k: round(d["retrieval_metrics"][k] - full["retrieval_metrics"][k], 4) for k in keys
        },
        "module_usage": d.get("retrieval_diagnostics", {}).get("module_usage", {}),
    }
path = root / "summary.json"
path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"[summary] {path}", flush=True)
PY
