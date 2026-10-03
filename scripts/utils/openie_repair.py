"""Versioned, checkpointed OpenIE repair and source-compatible graph rebuild.

The HippoRAG package is imported read-only. Its graph builder keeps the source
edge schema and Unicode normalization; all repair behavior lives in this repo.
"""

import argparse
import copy
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pathcondrag.information_extraction.openie_openai import OpenIE
from pathcondrag.utils.openie_quality import merge_triples, validate_triples

ROOT = Path(__file__).resolve().parents[2]
REPAIR_VERSION = "pathcondrag_openie_quality_v2"
SOURCE_DEFAULT = ROOT / "outputs/pathcondrag_new_innvotion_10_1/shared_hipporag2_index"
OUT_DEFAULT = ROOT / "outputs/openie_quality_repair_qwen3_b4_20261003"
ASSETS = ("openie_state.json", "index_manifest.json", "chunk_metadata.json", "graph.pickle",
          "chunk_embeddings/vdb_chunk.parquet", "entity_embeddings/vdb_entity.parquet",
          "fact_embeddings/vdb_fact.parquet")
LOG = logging.getLogger("openie.repair")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_id(value, prefix):
    return prefix + hashlib.md5(value.encode("utf-8")).hexdigest()


def normalize(value):
    """Exactly the source index's unicode_alnum_casefold_v1 contract."""
    return " ".join("".join(c if c.isalnum() or c.isspace() else " "
                            for c in value.casefold()).split())


def baseline_modules(args):
    sys.path.insert(0, str(Path(args.hippo_root) / "src"))
    from hipporag.HippoRAG import HippoRAG
    from hipporag.llm.openai_gpt import CacheOpenAI
    from hipporag.utils.config_utils import BaseConfig
    return HippoRAG, CacheOpenAI, BaseConfig


def baseline_code_hashes(root):
    root = Path(root).resolve()
    listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
                             check=True, capture_output=True).stdout
    return {name.decode(): sha256(root / name.decode())
            for name in listing.split(b"\0") if name and (root / name.decode()).is_file()}


def source_layout(args):
    source = Path(args.source_index).resolve()
    manifests = list(source.glob("*/index_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Source must contain exactly one model/index_manifest.json")
    model_dir = manifests[0].parent.name
    manifest = read_json(manifests[0])
    if manifest.get("text_normalization") != "unicode_alnum_casefold_v1":
        raise ValueError("This compatible repair runner requires a Unicode HippoRAG index")
    if manifest.get("embedding", {}).get("model_name") != "/root/models/Qwen3-Embedding-8B":
        raise ValueError("Unexpected embedding model; do not silently change index identity")
    state = read_json(source / model_dir / "openie_state.json")
    return source, model_dir, manifest, state


def initialize(args):
    source, model_dir, manifest, state = source_layout(args)
    out = Path(args.out_root).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Repair output must be separate from the frozen source index")
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repair artifacts must stay under PathCondRAG/outputs")
    out.mkdir(parents=True, exist_ok=True)
    snapshot_path = out / "source_snapshot.json"
    if snapshot_path.exists():
        snapshot = read_json(snapshot_path)
        if snapshot["source_index"] != str(source) or snapshot["repair_version"] != REPAIR_VERSION:
            raise ValueError("Existing repair directory belongs to another source/version")
        for name in ("openie_state.json", "index_manifest.json", "chunk_metadata.json"):
            if sha256(source / model_dir / name) != snapshot["asset_sha256"][name]:
                raise ValueError("Source identity changed before repair: " + name)
        if baseline_code_hashes(args.hippo_root) != snapshot["baseline_code_sha256"]:
            raise ValueError("Baseline source changed since this repair began")
    else:
        snapshot = {"source_index": str(source), "model_dir": model_dir,
                    "repair_version": REPAIR_VERSION,
                    "asset_sha256": {name: sha256(source / model_dir / name) for name in ASSETS},
                    "baseline_code_sha256": baseline_code_hashes(args.hippo_root)}
        write_json(snapshot_path, snapshot)
    index = out / "repaired_index"
    cache = index / "llm_cache"
    cache.mkdir(parents=True, exist_ok=True)
    source_cache = source / "llm_cache/qwen3-8b_cache.sqlite"
    destination_cache = cache / source_cache.name
    if not destination_cache.exists():
        with sqlite3.connect(f"file:{source_cache}?mode=ro", uri=True) as original:
            with sqlite3.connect(destination_cache) as destination:
                original.backup(destination)
    return source, model_dir, manifest, state, out, index, snapshot


def make_config(args, manifest, index, *, build=False):
    _, _, BaseConfig = baseline_modules(args)
    embedding = manifest["embedding"]
    identity = manifest["openie"]["identity"]
    return BaseConfig(
        save_dir=str(index), llm_name=identity["model_name"], llm_base_url=args.llm_base_url,
        embedding_model_name=embedding["model_name"], embedding_provider=embedding["provider"],
        embedding_batch_size=4, embedding_max_seq_len=embedding["max_sequence_length"],
        embedding_model_dtype=embedding["dtype"], embedding_return_as_normalized=embedding["normalized"],
        openie_mode="online", openie_max_workers=8, llm_prefetch_workers=8, openie_ner_max_tokens=512,
        openie_triple_max_tokens=2048, max_new_tokens=2048, temperature=identity["temperature"],
        seed=identity["seed"], response_format=identity["response_format"],
        force_index_from_scratch=build, force_openie_from_scratch=False,
        synonymy_edge_topk=manifest["graph_construction"]["synonymy_edge_topk"],
        synonymy_edge_sim_threshold=manifest["graph_construction"]["synonymy_edge_sim_threshold"],
        synonymy_edge_query_batch_size=128, synonymy_edge_key_batch_size=16384,
        is_directed_graph=manifest["is_directed_graph"], retrieval_top_k=200,
    )


def split_row(row):
    report = validate_triples(row["extracted_triples"])
    valid, invalid = [], list(report.invalid_triples)
    for triple in report.valid_triples:
        if all(normalize(field) for field in triple):
            valid.append(triple)
        else:
            invalid.append(triple)
    metadata = row.get("openie_metadata", {}).get("triples", {})
    defective = bool(invalid or not row["extracted_triples"]
                     or metadata.get("openie_skipped") or metadata.get("error")
                     or metadata.get("quality_status") in ("failed", "partial"))
    return valid, invalid, defective


def recovered_rows(state, out):
    rows, pending, summaries = [], [], []
    for original in state["docs"]:
        valid, invalid, defective = split_row(original)
        checkpoint = out / "repairs" / (original["idx"] + ".json")
        if defective and checkpoint.exists():
            saved = read_json(checkpoint)
            if saved.get("repair_version") != REPAIR_VERSION or saved.get("original_sha256") != row_hash(original):
                raise ValueError("Stale repair checkpoint: " + original["idx"])
            if saved["complete"] and not split_row(saved["row"])[2]:
                rows.append(saved["row"])
                summaries.append(saved["summary"])
                continue
        if defective:
            pending.append(original)
        cleaned = copy.deepcopy(original)
        cleaned["extracted_triples"] = valid
        rows.append(cleaned)
    return rows, pending, summaries


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def repair_one(extractor, original, out):
    valid, invalid, _ = split_row(original)
    if invalid:
        context = ("Defective extracted records to repair:\n" + json.dumps(invalid, ensure_ascii=False)
                   + "\nReturn only source-supported corrections for these records. "
                   "Preserve negatives, dates, amounts and qualifications; do not add unrelated facts.")
    else:
        context = ("The previous extraction of this whole passage failed; recover its explicit relations. "
                   "The title and original passage are available above.\nPrevious error: "
                   + str(original.get("openie_metadata", {}).get("triples", {}).get("openie_skip_reason"))
                   + "\nRe-extract directly from the original passage. Cover its explicit relationships, "
                   "including parenthetical facts. Split a shared predicate over coordinated people into "
                   "separate facts for each person. Keep specific relationship words in the predicate, "
                   "with the corresponding named entity as object; do not emit conjunctions as relations. "
                   "Use short, contiguous support quotes, rather than repeating an entire table per fact.")
    result = extractor.triple_extraction(original["idx"], original["passage"],
                                         original["extracted_entities"], repair_context=context)
    report = validate_triples(result.triples)
    accepted = [triple for triple in report.valid_triples if all(normalize(x) for x in triple)]
    complete = (result.metadata.get("quality_status") in ("success", "empty_valid")
                and not report.invalid_triples and len(accepted) == len(report.valid_triples))
    complete = complete and not (result.metadata.get("error") or result.metadata.get("openie_skipped"))
    complete = complete and (result.metadata.get("finish_reason") == "stop"
                             or result.metadata.get("window_recovery_complete") is True)
    if result.metadata.get("support_quote_matches") is not None:
        complete = complete and all(result.metadata["support_quote_matches"])
    for window in result.metadata.get("window_recovery", []):
        meta = window["metadata"]
        complete = complete and meta.get("quality_status") in ("success", "empty_valid")
        complete = complete and meta.get("finish_reason") == "stop"
        complete = complete and all(meta.get("support_quote_matches", []))
    # These existing failed passages all contain explicit relations. A purported
    # no-facts recovery is not enough to publish a repaired, still-empty chunk.
    if not original["extracted_triples"] and not accepted:
        complete = False
    updated = copy.deepcopy(original)
    updated["extracted_triples"] = merge_triples(valid, accepted)
    updated.setdefault("openie_metadata", {})["triples"] = result.metadata
    updated.setdefault("openie_responses", {})["triples"] = result.response
    updated["openie_quality_repair"] = {
        "version": REPAIR_VERSION, "original_sha256": row_hash(original),
        "original_metadata": original.get("openie_metadata", {}).get("triples", {}),
        "invalid_original_records": invalid, "valid_original_count": len(valid),
        "recovered_count": len(accepted), "complete": complete,
    }
    if complete:
        updated["openie_metadata"]["triples"].pop("openie_skipped", None)
        updated["openie_metadata"]["triples"].pop("openie_skip_reason", None)
        updated["openie_metadata"]["triples"]["quality_status"] = (
            "success" if updated["extracted_triples"] else "empty_valid")
    summary = {"idx": original["idx"], "title": original["passage"].split("\n")[0],
               "before_count": len(original["extracted_triples"]),
               "invalid_count": len(invalid), "after_count": len(updated["extracted_triples"]),
               "recovered_count": len(accepted), "complete": complete,
               "quality_status": result.metadata.get("quality_status"),
               "attempt_count": result.metadata.get("openie_attempt_count"),
               "window_attempt_count": result.metadata.get("window_recovery_attempt_count", 0)}
    write_json(out / "repairs" / (original["idx"] + ".json"), {
        "repair_version": REPAIR_VERSION, "original_sha256": row_hash(original),
        "complete": complete, "summary": summary, "row": updated,
    })
    LOG.info("[repaired] %s complete=%s triples=%s->%s", summary["title"], complete,
             summary["before_count"], summary["after_count"])
    return summary


def repair(args, smoke=False):
    source, model_dir, manifest, state, out, index, snapshot = initialize(args)
    _, pending, _ = recovered_rows(state, out)
    if smoke:
        short_empty = next((row for row in pending if not row["extracted_triples"]
                            and len(row["passage"]) < 400), None)
        nonempty = next((row for row in pending if split_row(row)[1]), None)
        pending = [row for row in (short_empty, nonempty) if row is not None]
    _, CacheOpenAI, _ = baseline_modules(args)
    llm = CacheOpenAI.from_experiment_config(make_config(args, manifest, index))
    extractor = OpenIE(llm, max_workers=8, respect_env_workers=False,
                       quality_max_retries=2, guided_recovery=True)
    started = time.monotonic()
    server_log = Path(args.vllm_log)
    log_start = server_log.stat().st_size if server_log.is_file() else None
    write_json(out / "repair_run_started.json", {
        "pid": os.getpid(), "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pending_chunks": len(pending), "smoke": smoke,
        "server_log_start": log_start, "workers": 8, "triple_max_tokens": 2048,
        "extractor_sha256": sha256(ROOT / "src/pathcondrag/information_extraction/openie_openai.py"),
    })
    LOG.info("[repair] pending=%s workers=8 max_tokens=2048 thinking=false smoke=%s", len(pending), smoke)
    completed = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(repair_one, extractor, row, out): row["idx"]
                   for row in pending}
        for future in as_completed(futures):
            completed.append(future.result())
    stats = llm.get_request_stats()
    if hasattr(llm, "close"):
        llm.close()
    run_report = {"smoke": smoke, "seconds": time.monotonic() - started,
                  "attempted_chunks": len(pending), "rows": completed,
                  "llm_request_stats": stats, "all_attempted_complete": all(x["complete"] for x in completed)}
    http_statuses = {}
    if log_start is not None:
        with server_log.open("rb") as stream:
            stream.seek(log_start)
            log_segment = stream.read().decode("utf-8", "replace")
        for status in re.findall(r'POST /v1/chat/completions HTTP/1\.1" (\d{3})', log_segment):
            http_statuses[status] = http_statuses.get(status, 0) + 1
    run_report["http_status_in_server_log"] = http_statuses
    write_json(out / ("smoke_report.json" if smoke else "repair_run_report.json"), run_report)
    if stats.get("failures") or any(status != "200" for status in http_statuses) or not run_report["all_attempted_complete"]:
        raise RuntimeError("Unresolved extraction failures; checkpoints kept, no repaired index published")
    if smoke:
        return
    rows, remaining, summaries = recovered_rows(state, out)
    if remaining:
        raise RuntimeError(f"{len(remaining)} defective chunks still unresolved")
    payload = copy.deepcopy(state)
    payload["docs"] = rows
    payload["provenance"]["quality_repair"] = {
        "schema": REPAIR_VERSION, "scope": "validated_base_openie_plus_source_grounded_local_repairs",
        "base_prompt_schema": manifest["openie"]["identity"]["prompt_schema"],
        "repair_extractor": "pathcondrag.information_extraction.openie_openai.OpenIE",
        "extractor_sha256": sha256(ROOT / "src/pathcondrag/information_extraction/openie_openai.py"),
        "validation_sha256": sha256(ROOT / "src/pathcondrag/utils/openie_quality.py"),
        "source_openie_sha256": snapshot["asset_sha256"]["openie_state.json"],
        "repaired_chunks": len(summaries), "thinking": False, "triple_max_tokens": 2048,
        "support_check": "original_quote_match; semantic_entailment_not_guaranteed",
    }
    write_json(out / "repaired_openie.json", payload)
    write_json(out / "repair_summary.json", {"version": REPAIR_VERSION, "total_chunks": len(rows),
               "repaired_chunks": len(summaries), "empty_chunks": sum(not r["extracted_triples"] for r in rows),
               "invalid_records": sum(len(split_row(r)[1]) for r in rows), "rows": summaries})


def required_contents(rows):
    entities, facts = set(), set()
    for row in rows:
        metadata = row.get("openie_metadata") or {}
        for stage in ("ner", "triples"):
            status = metadata.get(stage) or {}
            if (status.get("error") or status.get("openie_skipped")
                    or status.get("quality_status") in ("failed", "partial")):
                raise ValueError("Incomplete OpenIE stage in publishable index: " + row["idx"])
        if row.get("openie_quality_repair", {}).get("complete") is False:
            raise ValueError("Incomplete repair in publishable index: " + row["idx"])
        validation = validate_triples(row["extracted_triples"])
        if validation.invalid_triples:
            raise ValueError("Invalid triples in publishable OpenIE: " + row["idx"])
        for triple in validation.valid_triples:
            normalized = tuple(normalize(field) for field in triple)
            if not all(normalized):
                raise ValueError("Normalized empty triple: " + row["idx"])
            entities.update((normalized[0], normalized[2]))
            facts.add(str(normalized))
    return entities, facts


def build(args):
    import pandas as pd
    import torch
    source, model_dir, manifest, _, out, index, _ = initialize(args)
    payload = read_json(out / "repaired_openie.json")
    entities, facts = required_contents(payload["docs"])
    destination = index / model_dir
    destination.mkdir(parents=True, exist_ok=True)
    if (out / "build_complete.json").exists():
        marker = read_json(out / "build_complete.json")
        if marker.get("repaired_openie_sha256") == sha256(out / "repaired_openie.json"):
            LOG.info("[build] previously completed for this exact OpenIE; skipping reconstruction")
            return
        LOG.info("[build] OpenIE changed since the previous build; reconstructing graph/stores")
    for name in ("chunk_embeddings/vdb_chunk.parquet", "chunk_metadata.json"):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / model_dir / name, target)
    pruned = {}
    for namespace, needed in (("entity", entities), ("fact", facts)):
        name = f"{namespace}_embeddings/vdb_{namespace}.parquet"
        frame = pd.read_parquet(source / model_dir / name)
        kept = frame[frame["content"].isin(needed)].copy()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        kept.to_parquet(target, index=False)
        pruned[namespace] = {"removed": len(frame) - len(kept), "reused": len(kept),
                             "new_required": len(needed) - len(kept)}
    write_json(destination / "openie_state.json", payload)
    write_json(index / "openie_results_ner_qwen3-8b.json", payload)
    revised_manifest = copy.deepcopy(manifest)
    revised_manifest["openie"] = payload["provenance"]
    write_json(destination / "index_manifest.json", revised_manifest)
    HippoRAG, _, _ = baseline_modules(args)
    rag = HippoRAG(global_config=make_config(args, manifest, index, build=True))
    original_synonymy = rag.add_synonymy_edges

    def release_model_then_synonyms(query_node_keys=None):
        # No more embedding calls occur during this build. Free the 8B model
        # before the unchanged float32 cosine KNN to keep GPU memory bounded.
        if hasattr(rag.embedding_model, "model"):
            rag.embedding_model.model = None
        gc.collect()
        torch.cuda.empty_cache()
        return original_synonymy(query_node_keys)

    rag.add_synonymy_edges = release_model_then_synonyms
    os.environ["HIPPORAG_KNN_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    try:
        rag.index([row["passage"] for row in payload["docs"]])
        write_json(out / "build_complete.json", {"seconds": time.monotonic() - started,
                   "pruned_vectors": pruned, "graph_nodes": rag.graph.vcount(),
                   "graph_edges": rag.graph.ecount(), "baseline_builder_unmodified": True,
                   "repaired_openie_sha256": sha256(out / "repaired_openie.json")})
    finally:
        rag.close()


def validate(args):
    import igraph as ig
    import numpy as np
    import pyarrow.parquet as pq
    from utils.openie_graph_validation import validate_graph_contributions
    source, model_dir, _, _, out, index, snapshot = initialize(args)
    destination = index / model_dir
    state = read_json(destination / "openie_state.json")
    payload_path = out / "repaired_openie.json"
    marker = read_json(out / "build_complete.json")
    expected_payload = read_json(payload_path)
    if (marker.get("repaired_openie_sha256") != sha256(payload_path)
            or state.get("docs") != expected_payload.get("docs")
            or state.get("provenance") != expected_payload.get("provenance")):
        raise ValueError("Built index is not bound to the current completed repair")
    entities, facts = required_contents(state["docs"])
    chunks = {row["idx"] for row in state["docs"]}
    if len(chunks) != len(state["docs"]):
        raise ValueError("Duplicate chunk IDs")
    for row in state["docs"]:
        if row["idx"] != md5_id(row["passage"], "chunk-"):
            raise ValueError("Chunk passage changed")
    sets = {}
    vector_reports = {}
    for namespace, needed in (("chunk", {row["passage"] for row in state["docs"]}),
                              ("entity", entities), ("fact", facts)):
        path = destination / f"{namespace}_embeddings/vdb_{namespace}.parquet"
        ids, contents, count = set(), set(), 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            rows = batch.to_pydict()
            for key, content, vector in zip(rows["hash_id"], rows["content"], rows["embedding"]):
                if key != md5_id(content, namespace + "-") or key in ids:
                    raise ValueError("Invalid/duplicate vector identity")
                values = np.asarray(vector)
                if values.shape != (4096,) or not np.isfinite(values).all():
                    raise ValueError("Invalid vector dimensions or non-finite values")
                if not np.isclose(np.linalg.norm(values), 1.0, atol=0.02):
                    raise ValueError("Embedding does not meet the normalized-vector contract")
                ids.add(key)
                contents.add(content)
                count += 1
        if contents != needed:
            raise ValueError(f"{namespace} vectors do not match current OpenIE")
        sets[namespace] = ids
        vector_reports[namespace] = {"count": count, "dimension": 4096,
                                     "finite": True, "normalized": True}
    graph = ig.Graph.Read_Pickle(str(destination / "graph.pickle"))
    if "hipporag_edge_schema" not in graph.attributes() or graph["hipporag_edge_schema"] != 1:
        raise ValueError("Source-aware graph edge schema missing")
    if set(graph.vs["name"]) != sets["chunk"] | sets["entity"]:
        raise ValueError("Graph/store node identity mismatch")
    if md5_id("", "entity-") in sets["entity"]:
        raise ValueError("Empty entity persisted")
    edge_report = validate_graph_contributions(graph, state["docs"], normalize)
    if set(read_json(destination / "chunk_metadata.json")) != chunks:
        raise ValueError("Chunk metadata coverage mismatch")
    for name, expected in snapshot["asset_sha256"].items():
        if sha256(source / model_dir / name) != expected:
            raise ValueError("Frozen source index was modified: " + name)
    if baseline_code_hashes(args.hippo_root) != snapshot["baseline_code_sha256"]:
        raise ValueError("HippoRAG original source was modified")
    report = {"complete": True, "source_index_unchanged": True, "baseline_code_unchanged": True,
              "repaired_index": str(index), "vectors": vector_reports,
              "empty_chunks": sum(not row["extracted_triples"] for row in state["docs"]),
              "invalid_records": 0, "empty_entity_present": False,
              "graph_nodes": graph.vcount(), "graph_edges": graph.ecount(),
              "edge_validation": edge_report,
              "asset_sha256": {name: sha256(destination / name) for name in ASSETS}}
    write_json(out / "validation_report.json", report)
    LOG.info("[validated] graph/store/OpenIE consistent; source and HippoRAG code unchanged")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "repair", "build", "validate", "run"), default="run")
    parser.add_argument("--source-index", default=str(SOURCE_DEFAULT))
    parser.add_argument("--out-root", default=str(OUT_DEFAULT))
    parser.add_argument("--hippo-root", default="/root/baseline/HippoRAG")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8035/v1")
    parser.add_argument("--vllm-log", default="/root/eval/logs/vllm_qwen3.log")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("PATHCONDRAG_LLM_MAX_IN_FLIGHT", "8")
    os.environ.setdefault("HIPPORAG_LLM_MAX_IN_FLIGHT", "8")
    if args.phase == "run":
        # Separate processes release all model references between stages.
        for phase in ("repair", "build", "validate"):
            command = [sys.executable, str(ROOT / "scripts/repair_openie_index.py"),
                       "--phase", phase, "--source-index", args.source_index,
                       "--out-root", args.out_root, "--hippo-root", args.hippo_root,
                       "--llm-base-url", args.llm_base_url, "--vllm-log", args.vllm_log]
            subprocess.run(command, check=True)
    elif args.phase == "smoke":
        repair(args, smoke=True)
    elif args.phase == "repair":
        repair(args)
    elif args.phase == "build":
        build(args)
    else:
        validate(args)
    return 0
