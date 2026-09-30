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
  RUN_RESULT="${result_path}" RUN_CASE="${case_dir}" RUN_ELAPSED="$((end_ts-start_ts))" \
    VLLM_LOG="${VLLM_LOG}" LOG_START="${log_start}" LOG_END="${log_end}" python - <<'PY'
import json
import os
import re
from pathlib import Path

result = json.loads(Path(os.environ["RUN_RESULT"]).read_text())
expected = int(os.environ.get("SAMPLE_SIZE", "24"))
if not (len(result["results"]) == len(result["selected_indices"]) == result["sample_size_effective"] == expected):
    raise SystemExit(f"Unexpected completed sample count: {result['sample_size_effective']} != {expected}")
if result.get("hop_source") != "benchmark":
    raise SystemExit("Benchmark hop source was not recorded")
with open(os.environ["VLLM_LOG"], "rb") as f:
    f.seek(int(os.environ["LOG_START"]))
    log = f.read(int(os.environ["LOG_END"]) - int(os.environ["LOG_START"])).decode("utf-8", "replace")
status = {}
for code in re.findall(r'POST /v1/chat/completions HTTP/1\.1" (\d{3})', log):
    status[code] = status.get(code, 0) + 1
report = {
    "seconds": int(os.environ["RUN_ELAPSED"]),
    "retrieval_seconds": result.get("retrieval_seconds"),
    "llm_request_stats": result.get("llm_request_stats", {}),
    "retrieval_metrics": result["retrieval_metrics"],
    "hop_distribution": result.get("hop_distribution"),
    "hop_counter": result["retrieval_diagnostics"].get("hop_counter"),
    "qd_used": result["retrieval_diagnostics"].get("qd_used_count"),
    "pcqd_used": result["retrieval_diagnostics"].get("pcqd_used_count"),
    "http_status_in_log": status,
}
Path(os.environ["RUN_CASE"], "report.json").write_text(json.dumps(report, indent=2))
print("[done]", Path(os.environ["RUN_CASE"]).name, report, flush=True)
if any(code != "200" for code in status):
    raise SystemExit("Non-200 LLM response detected in vLLM log")
if report["llm_request_stats"].get("failures", 0):
    raise SystemExit("Client reported terminal LLM request failure")
if report["llm_request_stats"].get("http_attempts", 0) > 0 and not status:
    raise SystemExit("Client made LLM requests but no HTTP statuses were found in the vLLM log")
PY
done

RUN_ROOT="${RUN_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
cases = sorted(root.glob("workers_*/result.json"))
summary = {}
baseline_path = root / "workers_1" / "result.json"
if not baseline_path.exists():
    raise SystemExit("Missing workers_1 baseline")
baseline = json.loads(baseline_path.read_text())
samples = json.loads(Path("/root/datasets/musique.json").read_text())
for path in cases:
    result = json.loads(path.read_text())
    if not (len(result["results"]) == len(result["selected_indices"]) == result["sample_size_effective"]):
        raise SystemExit(f"Invalid result length: {path}")
    all_gold_top5 = 0
    for row, sample_index in zip(result["results"], result["selected_indices"]):
        sample = samples[sample_index]
        gold = {
            p["title"] + "\n" + (p.get("text") or p.get("paragraph_text", ""))
            for p in sample["paragraphs"] if p.get("is_supporting") is not False
        }
        all_gold_top5 += int(bool(gold) and gold.issubset(set(row["docs"][:5])))
    summary[path.parent.name] = {
        "seconds": json.loads((path.parent / "report.json").read_text())["seconds"],
        "retrieval_seconds": result.get("retrieval_seconds"),
        "Recall@1": result["retrieval_metrics"]["Recall@1"],
        "Recall@2": result["retrieval_metrics"]["Recall@2"],
        "Recall@5": result["retrieval_metrics"]["Recall@5"],
        "Recall@10": result["retrieval_metrics"]["Recall@10"],
        "all_gold_top5": all_gold_top5 / len(result["results"]),
        "qd_used": result["retrieval_diagnostics"].get("qd_used_count"),
        "pcqd_used": result["retrieval_diagnostics"].get("pcqd_used_count"),
        "llm_request_stats": result.get("llm_request_stats", {}),
    }
    if result["selected_indices"] != baseline["selected_indices"]:
        raise SystemExit("Cases used different question subsets")
    summary[path.parent.name]["same_ordered_top5_vs_workers_1"] = sum(
        a["docs"][:5] == b["docs"][:5]
        for a, b in zip(result["results"], baseline["results"])
    )
    summary[path.parent.name]["same_top5_scores_vs_workers_1"] = sum(
        a["doc_scores"][:5] == b["doc_scores"][:5]
        for a, b in zip(result["results"], baseline["results"])
    )
    summary[path.parent.name]["different_question_indices_vs_workers_1"] = [
        idx for idx, (a, b) in enumerate(zip(result["results"], baseline["results"]))
        if a["docs"][:5] != b["docs"][:5]
    ]
(root / "comparison.json").write_text(json.dumps(summary, indent=2))
print("[comparison]", summary, flush=True)
PY
