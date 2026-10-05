"""Strict, model-free preparation and reporting for the seven retrieval cases."""

import hashlib
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from .common import http_status_counts, read_json, write_json
from eval_utils import get_benchmark_hops, get_gold_docs


MODEL_DIR = "qwen3-8b__root_models_Qwen3-Embedding-8B"
EMBEDDING_MODEL = "/root/models/Qwen3-Embedding-8B"
DEFAULT_EMBEDDING_PROVIDER = "transformers"
DEFAULT_EMBEDDING_BATCH_SIZE = 4
CASES = {
    "hipporag2": None,
    "pathcondrag_original": 0,
    "exp1_correctness": 1,
    "exp2_evidence_candidates": 2,
    "exp3_prefix_coverage": 3,
    "exp4_dependency_binding": 4,
    "exp5_verified_beam": 5,
}
ASSETS = (
    "graph.pickle",
    "chunk_embeddings/vdb_chunk.parquet",
    "entity_embeddings/vdb_entity.parquet",
    "fact_embeddings/vdb_fact.parquet",
)
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def model_dir_name(embedding_model):
    return "qwen3-8b__" + str(embedding_model).strip("/").replace("/", "_")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_hashes(index, model_dir=None):
    return {name: sha256(Path(index) / (model_dir or MODEL_DIR) / name) for name in ASSETS}


def cache_hashes(directory):
    directory = Path(directory)
    return {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.name.endswith(".lock")
    }


def passage_text(item):
    return item["title"] + "\n" + (item.get("text") or item.get("paragraph_text", ""))


def dataset_name_for(data_path=None, sample=None, dataset_name=None):
    """Resolve the benchmark while retaining the historical MuSiQue default."""
    if dataset_name:
        require(dataset_name in DATASETS, f"unsupported retrieval dataset: {dataset_name}")
        return dataset_name
    if data_path and Path(data_path).stem in DATASETS:
        return Path(data_path).stem
    sample = sample or {}
    if "question_decomposition" in sample or "paragraphs" in sample:
        return "musique"
    if "evidences" in sample:
        return "2wikimultihopqa"
    if "supporting_facts" in sample and "context" in sample:
        return "hotpotqa"
    raise ValueError("cannot identify retrieval dataset from its path/schema")


def sample_identity(sample, index):
    """Match detailed_result's id/_id/query-index export policy exactly."""
    return sample.get("id", sample.get("_id", index))


def benchmark_hop_policy(dataset_name):
    """State whether a supplied hop value is an oracle label or dataset prior."""
    return {
        "scope": "per_question_decomposition" if dataset_name == "musique" else "dataset_prior",
        "source": "question_decomposition_length" if dataset_name == "musique" else "dataset_prior:2",
        "uses_gold_relations": False,
        "baseline_export_scope": "per_question_id" if dataset_name == "musique" else "unavailable",
    }


def gold_docs(sample, dataset_name=None):
    name = dataset_name_for(sample=sample, dataset_name=dataset_name)
    return set(get_gold_docs([sample], name)[0])


def validated_dataset(data_path, corpus_path, dataset_name=None):
    data, corpus = read_json(data_path), read_json(corpus_path)
    require(isinstance(data, list) and bool(data), "dataset must be a non-empty list")
    require(isinstance(corpus, list) and bool(corpus), "corpus must be a non-empty list")
    require(all(isinstance(sample, dict) for sample in data), "dataset contains non-object samples")
    name = dataset_name_for(data_path, data[0], dataset_name)
    require(all(isinstance(p, dict) and isinstance(p.get("title"), str)
                and isinstance(p.get("text") or p.get("paragraph_text", ""), str)
                for p in corpus), "corpus contains invalid title/text passages")
    docs = set(passage_text(p) for p in corpus)
    # Use the same reader as the actual retrieval entry point. Gold relations
    # are used only to validate exports; they never enter retrieval planning.
    if name == "musique":
        for i, sample in enumerate(data):
            decomposition = sample.get("question_decomposition")
            require(isinstance(decomposition, list) and len(decomposition) in (2, 3, 4)
                    and all(isinstance(step, dict) and isinstance(step.get("question"), str)
                            and bool(step["question"].strip()) for step in decomposition),
                    f"dataset[{i}] lacks valid 2/3/4-hop annotations")
    hops = get_benchmark_hops(data, name)
    ids = set()
    for i, sample in enumerate(data):
        require(isinstance(sample.get("question"), str) and bool(sample["question"].strip()),
                f"dataset[{i}] lacks a question")
        if name == "musique":
            match = re.match(r"([234])hop\d*(?:__|_)", str(sample.get("id", "")))
            require(match is not None and int(match[1]) == hops[i], f"dataset[{i}] hop ID mismatch")
        else:
            require(isinstance(sample.get("supporting_facts"), list)
                    and bool(sample["supporting_facts"])
                    and all(isinstance(fact, (list, tuple)) and len(fact) == 2
                            and isinstance(fact[0], str) for fact in sample["supporting_facts"]),
                    f"dataset[{i}] lacks valid supporting facts")
            require(isinstance(sample.get("context"), list), f"dataset[{i}] lacks context passages")
            require(all(isinstance(item, (list, tuple)) and len(item) == 2
                        and isinstance(item[0], str) and isinstance(item[1], list)
                        and all(isinstance(sentence, str) for sentence in item[1])
                        for item in sample["context"]), f"dataset[{i}] has invalid context passages")
            support_titles = {fact[0] for fact in sample["supporting_facts"]}
            require(support_titles <= {item[0] for item in sample["context"]},
                    f"dataset[{i}] supporting titles absent from context")
        identity = sample_identity(sample, i)
        require(isinstance(identity, (str, int)) and not isinstance(identity, bool),
                f"dataset[{i}] has invalid sample ID")
        require(identity not in ids, f"duplicate sample ID: {identity}")
        ids.add(identity)
        gold = gold_docs(sample, name)
        require(bool(gold) and gold <= docs, f"dataset[{i}] gold passages absent from corpus")
    return data, docs, hops


def fresh_index_path(root, dataset_name=None, shared_output_root=None):
    """Declare exactly one dataset's persistent shared-index location."""
    if shared_output_root:
        shared_root = Path(shared_output_root).resolve()
        require(dataset_name in DATASETS, "a shared output root requires an explicit dataset")
        require(root == shared_root / "metadata" / dataset_name,
                "shared dataset metadata must be in SHARED_OUTPUT_ROOT/metadata/DATASET")
        return shared_root / "shared_indexes" / dataset_name
    return root / "shared_hipporag2_index"


def load_indices(path, total):
    indices = read_json(path)
    if isinstance(indices, dict):
        indices = indices.get("selected_indices")
    require(isinstance(indices, list) and bool(indices), "indices must be a non-empty JSON list")
    require(all(type(i) is int and 0 <= i < total for i in indices), "indices are out of range")
    require(len(set(indices)) == len(indices), "indices contain duplicates")
    return indices


def index_identity(source, docs, embedding_model=None, embedding_provider=None, model_dir=None):
    embedding_model = embedding_model or EMBEDDING_MODEL
    embedding_provider = embedding_provider or DEFAULT_EMBEDDING_PROVIDER
    model_dir = model_dir or model_dir_name(embedding_model)
    identity = read_json(Path(source) / model_dir / "index_manifest.json")
    require(identity.get("embedding", {}).get("model_name") == embedding_model,
            f"source index embedding model is not {embedding_model}")
    require(identity.get("embedding", {}).get("provider") == embedding_provider,
            "source embedding provider mismatch")
    require(identity.get("openie", {}).get("identity", {}).get("model_name") == "qwen3-8b",
            "source extraction model mismatch")
    chunk_ids = set(read_json(Path(source) / model_dir / "chunk_metadata.json"))
    expected_ids = {"chunk-" + hashlib.md5(doc.encode()).hexdigest() for doc in docs}
    require(chunk_ids == expected_ids, "source index does not contain exactly the declared corpus")


def prepare(args):
    out, source = Path(args.out_root).resolve(), Path(args.source_index).resolve()
    fresh = bool(getattr(args, "build_shared_index", False))
    requested_dataset = getattr(args, "dataset", None)
    shared_output_root = getattr(args, "shared_output_root", None)
    if fresh:
        expected = fresh_index_path(out, requested_dataset, shared_output_root)
        require(source == expected, f"fresh source_index must equal OUT_ROOT declared shared index: {expected}")
    else:
        require(source != out and source not in out.parents and out not in source.parents,
                "source index and output directory must be separate")
    data, docs, hops = validated_dataset(args.data_path, args.corpus_path, requested_dataset)
    dataset = dataset_name_for(args.data_path, data[0], requested_dataset)
    require(0 <= args.sample_size <= len(data), "sample_size must be between 0 and dataset size")
    if args.sample_indices_file:
        indices = load_indices(args.sample_indices_file, len(data))
        require(not args.sample_size or len(indices) == args.sample_size, "sample_size/indices mismatch")
    elif not args.sample_size:
        indices = list(range(len(data)))
    elif args.sample_size == 2:
        # Keep the MuSiQue smoke's 2/4-hop coverage, and select two real
        # questions for datasets without a 4-hop label in the supplied subset.
        indices = [hops.index(2), hops.index(4)] if 2 in hops and 4 in hops else [0, 1]
    else:
        indices = sorted(random.Random(args.sample_seed).sample(range(len(data)), args.sample_size))
    names = args.cases.split() if args.cases else list(CASES)
    require(bool(names) and len(names) == len(set(names)) and all(n in CASES for n in names),
            "CASES contains unknown or duplicate case names")
    names = [n for n in CASES if n in names]
    embedding_model = getattr(args, "embedding_model", None) or EMBEDDING_MODEL
    embedding_provider = getattr(args, "embedding_provider", None) or DEFAULT_EMBEDDING_PROVIDER
    embedding_batch_size = int(getattr(args, "embedding_batch_size", None) or DEFAULT_EMBEDDING_BATCH_SIZE)
    model_dir = model_dir_name(embedding_model)
    if fresh:
        hashes, initial_cache = {}, {}
    else:
        index_identity(source, docs, embedding_model=embedding_model,
                       embedding_provider=embedding_provider, model_dir=model_dir)
        hashes = asset_hashes(source, model_dir)
        cache_dir = source / "llm_cache"
        initial_cache = cache_hashes(cache_dir) if cache_dir.is_dir() else {}
    manifest = {
        "schema_version": 2,
        "index_build_mode": "fresh" if fresh else "external_snapshot",
        "index_build_status": "pending" if fresh else "ready",
        "source_index": str(source), "model_dir": model_dir,
        "embedding_provider": embedding_provider,
        "data_path": str(Path(args.data_path).resolve()),
        "corpus_path": str(Path(args.corpus_path).resolve()),
        "data_sha256": sha256(args.data_path), "corpus_sha256": sha256(args.corpus_path),
        "source_asset_sha256": hashes, "initial_cache_sha256": initial_cache,
        "indexed_docs": len(docs), "dataset_size": len(data),
        "sample_size_requested": args.sample_size, "sample_seed": args.sample_seed,
        "selected_indices": indices, "selected_sample_ids": [sample_identity(data[i], i) for i in indices],
        "benchmark_hops": [hops[i] for i in indices],
        "hop_distribution": dict(Counter(str(hops[i]) for i in indices)),
        "dataset_hop_distribution": dict(Counter(map(str, hops))),
        "cases": names,
        "runtime": {"embedding_model_name": embedding_model,
                    "embedding_provider": embedding_provider,
                    "embedding_batch_size": embedding_batch_size,
                    "llm_name": "qwen3-8b", "llm_prefetch_workers": 8, "openie_max_workers": 8,
                    "max_new_tokens": 2048, "enable_thinking": False,
                    "retrieval_top_k": 200, "result_top_k": 10, "candidate_output_top_k": 200},
    }
    # Omit these additions for historical MuSiQue callers so frozen manifests
    # from the seven-case experiment remain resumable without rewriting them.
    if requested_dataset or dataset != "musique" or shared_output_root:
        manifest["dataset"] = dataset
        manifest["benchmark_hop_policy"] = benchmark_hop_policy(dataset)
        manifest["gold_support_count_distribution"] = dict(Counter(
            str(len(gold_docs(data[i], dataset))) for i in indices))
    if shared_output_root:
        manifest["shared_output_root"] = str(Path(shared_output_root).resolve())
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    snapshot = out / "initial_llm_cache"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        comparable = dict(existing)
        if fresh:
            for key in ("source_asset_sha256", "initial_cache_sha256", "index_build_status"):
                comparable[key] = manifest[key]
            comparable.pop("index_build_validation", None)
        require(comparable == manifest, "existing manifest differs; use a new OUT_ROOT")
        if existing["index_build_status"] == "ready":
            require(cache_hashes(snapshot) == existing["initial_cache_sha256"], "initial LLM cache snapshot changed")
            if fresh:
                require(index_ready(args) == 0, "fresh shared index validation failed")
        else:
            require(not snapshot.exists(), "pending index must not have a cache snapshot")
        require(load_indices(out / "selected_indices.json", len(data)) == indices, "saved indices changed")
    else:
        require(not snapshot.exists(), "partial cache snapshot exists; use a new OUT_ROOT")
        if fresh:
            require(not source.exists() or not any(source.iterdir()), "fresh shared index directory is not empty")
            require(not (out / "index_build_result.json").exists(), "fresh index build result already exists")
        else:
            cache_dir = source / "llm_cache"
            if cache_dir.is_dir():
                shutil.copytree(cache_dir, snapshot, ignore=shutil.ignore_patterns("*.lock"))
            else:
                snapshot.mkdir(parents=True)
            require(cache_hashes(snapshot) == initial_cache, "source LLM cache changed during snapshot")
        write_json(out / "selected_indices.json", indices)
        write_json(manifest_path, manifest)
    print(f"[prepared] n={len(indices)} hops={manifest['hop_distribution']} cases={names}", flush=True)


def manifest_at(root):
    return read_json(Path(root) / "manifest.json")


def index_ready(args):
    root, manifest = Path(args.out_root).resolve(), manifest_at(args.out_root)
    if manifest.get("index_build_status") != "ready":
        return 1
    require(asset_hashes(manifest["source_index"], manifest["model_dir"]) == manifest["source_asset_sha256"],
            "shared source index changed")
    require(cache_hashes(root / "initial_llm_cache") == manifest["initial_cache_sha256"],
            "shared cache snapshot changed")
    if manifest.get("index_build_mode") == "fresh":
        validation = manifest.get("index_build_validation") or {}
        require(validation.get("result_sha256") == sha256(root / "index_build_result.json")
                and validation.get("report_sha256") == sha256(root / "index_build_report.json"),
                "index build result/report changed")
        require(validation.get("openie_sha256")
                == sha256(Path(manifest["source_index"]) / "openie_results_ner_qwen3-8b.json"),
                "fresh OpenIE results changed")
    return 0


def freeze_index(args):
    strict = getattr(args, 'openie_strict', True)
    require(type(strict) is bool, "openie_strict must be a boolean")
    root, manifest = Path(args.out_root).resolve(), manifest_at(args.out_root)
    require(manifest.get("index_build_mode") == "fresh", "freeze-index requires fresh build mode")
    if manifest.get("index_build_status") == "ready":
        require(index_ready(args) == 0, "frozen shared index validation failed")
        print("[index-ready] shared Hippo index already validated", flush=True)
        return
    require(manifest.get("index_build_status") == "pending", "invalid shared index state")
    source = Path(manifest["source_index"]).resolve()
    require(source == fresh_index_path(root, manifest.get("dataset"), manifest.get("shared_output_root")),
            "fresh shared index path mismatch")
    require(sha256(manifest["data_path"]) == manifest["data_sha256"], "dataset changed during index build")
    require(sha256(manifest["corpus_path"]) == manifest["corpus_sha256"], "corpus changed during index build")
    _, docs, _ = validated_dataset(manifest["data_path"], manifest["corpus_path"], manifest.get("dataset"))
    index_identity(source, docs,
                   embedding_model=manifest["runtime"]["embedding_model_name"],
                   embedding_provider=manifest.get("embedding_provider") or manifest["runtime"].get("embedding_provider"),
                   model_dir=manifest["model_dir"])
    index_manifest = read_json(source / manifest['model_dir'] / 'index_manifest.json')
    quality = (index_manifest.get('openie') or {}).get('quality_profile') or {}
    validation_scope = quality.get('validation_scope') or quality.get('semantic_scope') or 'baseline_structural'
    result = read_json(root / "index_build_result.json")
    require(result.get("eval_mode") == "index_only" and result.get("index_build_complete") is True,
            "index_only did not report successful index completion")
    require(result.get("indexed_docs") == len(docs) and result.get("openie_document_count") == len(docs),
            "fresh index/OpenIE document coverage is incomplete")
    if strict:
        require(result.get("openie_failure_count") == 0, "fresh OpenIE has failed documents")
    else:
        require((result.get('runtime_config') or {}).get('openie_strict') is False,
                'Tolerant freezing requires an explicit tolerant build report')
    openie_path = source / "openie_results_ner_qwen3-8b.json"
    openie_rows = read_json(openie_path).get("docs")
    require(isinstance(openie_rows, list) and len(openie_rows) == len(docs),
            "fresh OpenIE artifact has incomplete document coverage")
    require({row.get("passage") for row in openie_rows} == docs,
            "fresh OpenIE artifact does not cover the full corpus")
    if quality:
        # Choose the declared contract instead of imposing a semantic audit on
        # structural-only extraction or treating a structural record as audited.
        from .fresh_index_validation import _validated_rows
        _validated_rows(openie_rows, docs, quality, strict=strict)
    for row in openie_rows:
        passage = row["passage"]
        require(row.get("idx") == "chunk-" + hashlib.md5(passage.encode()).hexdigest(),
                "fresh OpenIE artifact chunk identity mismatch")
        require(isinstance(row.get("extracted_entities"), list)
                and isinstance(row.get("extracted_triples"), list), "fresh OpenIE extraction output is invalid")
        metadata = row.get("openie_metadata") or {}
        publication = metadata.get('publication') or {}
        if not strict and publication.get('skipped_from_graph') is True:
            require(publication.get('openie_strict') is False and row['extracted_triples'] == [],
                    'Tolerated failed extraction retained graph facts')
            continue
        for stage in ("ner", "triples"):
            stage_meta = metadata.get(stage) or {}
            finished_ok = (
                stage_meta.get("finish_reason") == "stop"
                and not stage_meta.get("openie_skipped") and not stage_meta.get("error")
                and stage_meta.get("quality_status") not in ("failed", "partial")
            )
            require(finished_ok, f"fresh OpenIE {stage} did not finish successfully: {row['idx']}")
    if not strict:
        excluded = sum(((row.get('openie_metadata') or {}).get('publication') or {}).get('skipped_from_graph') is True
                       for row in openie_rows)
        require(result.get('openie_failure_count') == excluded,
                'Tolerant build failure count differs from explicitly excluded chunks')
    config = result.get("runtime_config") or {}
    for key in ("embedding_model_name", "llm_name", "embedding_batch_size", "openie_max_workers",
                "llm_prefetch_workers", "max_new_tokens"):
        require(config.get(key) == manifest["runtime"][key], f"fresh index runtime_config.{key} mismatch")
    require(config.get("force_index_from_scratch") is True and config.get("force_openie_from_scratch") is True,
            "fresh index must rebuild both graph and OpenIE")
    require(config.get("openie_mode") == "online", "fresh OpenIE must use online LLM extraction")
    stats = result.get("llm_request_stats") or {}
    require((not strict or stats.get("failures") == 0) and stats.get("max_in_flight") == 8,
            "fresh indexing LLM failures/concurrency mismatch")
    require(isinstance(stats.get("http_attempts"), int) and stats["http_attempts"] > 0,
            "fresh OpenIE build did not make actual LLM HTTP requests")
    require(args.log_start >= 0 and args.log_end >= args.log_start, "vLLM index-build log rotated/truncated")
    status = http_status_counts(args.vllm_log, args.log_start, args.log_end)
    require(bool(status) and (not strict or not any(code != "200" for code in status)),
            f"fresh indexing missing statuses or non-200 HTTP responses: {status}")
    hashes = asset_hashes(source, manifest["model_dir"])
    require((source / "llm_cache").is_dir(), "fresh indexing LLM cache is missing")
    snapshot = root / "initial_llm_cache"
    require(not snapshot.exists(), "pending fresh index already has a cache snapshot")
    current_cache = cache_hashes(source / "llm_cache")
    require(bool(current_cache), "fresh indexing LLM cache is empty")
    shutil.copytree(source / "llm_cache", snapshot, ignore=shutil.ignore_patterns("*.lock"))
    require(cache_hashes(snapshot) == current_cache, "fresh LLM cache changed during freeze")
    output = {
        "validated": True, "index_build_mode": "fresh", "indexed_docs": len(docs),
        "openie_document_count": result["openie_document_count"],
        "openie_failure_count": result.get('openie_failure_count', 0), 'openie_strict': strict,
        'validation_scope': validation_scope,
        'quality_profile': quality,
        'semantic_verified': False if validation_scope == 'structural' else None,
        "seconds": args.elapsed, "runtime_config": config, "llm_request_stats": stats,
        "http_status_in_log": status, "source_asset_sha256": hashes,
        "initial_cache_sha256": current_cache,
        "vllm_log_slice": {"path": args.vllm_log, "start": args.log_start, "end": args.log_end},
    }
    write_json(root / "index_build_report.json", output)
    manifest.update({"index_build_status": "ready", "source_asset_sha256": hashes,
                     "initial_cache_sha256": current_cache,
                     "index_build_validation": {"result_sha256": sha256(root / "index_build_result.json"),
                                                "report_sha256": sha256(root / "index_build_report.json"),
                                                "openie_sha256": sha256(openie_path)}})
    write_json(root / "manifest.json", manifest)
    print(f"[index-ready] fresh Hippo graph/OpenIE validated: docs={len(docs)}, http={stats['http_attempts']}", flush=True)


def initialize_case(args):
    root, name = Path(args.out_root).resolve(), args.name
    manifest = manifest_at(root)
    require(manifest.get("index_build_status") == "ready", "shared Hippo index is pending; freeze it before retrieval")
    require(index_ready(args) == 0, "shared Hippo index integrity check failed")
    require(name in manifest["cases"], f"case not selected: {name}")
    source = Path(manifest["source_index"])
    require(asset_hashes(source, manifest["model_dir"]) == manifest["source_asset_sha256"],
            "source graph/embeddings changed")
    case = root / "cases" / name
    require(not (case / "validated.ok").exists(), "validated case must not be overwritten")
    require(not case.exists(), f"incomplete case exists: {case}; inspect it and select a new OUT_ROOT")
    case.mkdir(parents=True)
    index = case / "index"
    shutil.copytree(source, index, ignore=shutil.ignore_patterns("*.lock", "metrics*.json", "eval_results*"))
    llm_cache = index / "llm_cache"
    if llm_cache.exists():
        shutil.rmtree(llm_cache)
    shutil.copytree(root / "initial_llm_cache", llm_cache)
    require(asset_hashes(index, manifest["model_dir"]) == manifest["source_asset_sha256"],
            "copied graph/embeddings differ")
    require(cache_hashes(index / "llm_cache") == manifest["initial_cache_sha256"], "copied LLM cache differs")
    write_json(case / "before.json", {"asset_sha256": asset_hashes(index, manifest["model_dir"]),
                                      "initial_cache_sha256": cache_hashes(index / "llm_cache")})
    print(f"[initialized] {name}: isolated Hippo index and identical cache snapshot", flush=True)


def ready(args):
    case = Path(args.out_root) / "cases" / args.name
    if not (case / "validated.ok").is_file():
        return 1
    marker = read_json(case / "validated.ok")
    require(marker == {"result_sha256": sha256(case / "result.json"),
                       "report_sha256": sha256(case / "report.json"),
                       "manifest_sha256": sha256(Path(args.out_root) / "manifest.json")},
            f"validated files changed: {args.name}")
    require(asset_hashes(case / "index", manifest_at(args.out_root)["model_dir"])
            == manifest_at(args.out_root)["source_asset_sha256"],
            f"validated index changed: {args.name}")
    print(f"[skip] {args.name}: previously validated", flush=True)
    return 0


def recall(gold, docs):
    return len(gold & set(docs)) / len(gold)


def validate_result(result, manifest, name, data, corpus, expected_stage=None):
    indices = manifest["selected_indices"]
    expected_candidates = min(200, len(corpus))
    if manifest.get("dataset"):
        require(result.get("dataset") == manifest["dataset"], "result dataset mismatch")
        require(result.get("eval_mode") == "retrieve" and not result.get("qa_metrics"),
                "multi-dataset experiment must perform retrieval only")
        indexed_count = result.get("n_docs") if name == "hipporag2" else result.get("indexed_docs")
        require(indexed_count == len(corpus), "retrieval index does not cover the declared full corpus")
    require(result.get("selected_indices") == indices, "selected indices mismatch")
    require(result.get("sample_size_effective") == len(indices), "sample count mismatch")
    require(result.get("result_top_k") == 10 and result.get("candidate_output_top_k") == 200,
            "result/candidate export budgets mismatch")
    config = result.get("runtime_config") or {}
    for key in ("embedding_batch_size", "llm_prefetch_workers", "openie_max_workers"):
        require(config.get(key) == manifest["runtime"][key], f"runtime_config.{key} mismatch")
    require(config.get("embedding_model_name") == manifest["runtime"]["embedding_model_name"],
            "runtime embedding model mismatch")
    require(config.get("llm_name") == "qwen3-8b", "runtime LLM model mismatch")
    require(config.get("max_new_tokens") == 2048, "max_new_tokens must remain 2048")
    stage = CASES[name] if expected_stage is None else expected_stage
    if stage is not None:
        require(config.get("improvement_stage") == stage, "improvement stage mismatch")
        require(result.get("hop_source") == "benchmark", "hop_source must be benchmark")
        require(result.get("hop_distribution") == manifest["hop_distribution"], "hop distribution mismatch")
    rows = result.get("results")
    require(isinstance(rows, list) and len(rows) == len(indices), "result rows mismatch")
    measurements = []
    for position, (row, index) in enumerate(zip(rows, indices)):
        sample, hop = data[index], manifest["benchmark_hops"][position]
        require(row.get("query_index") == index and row.get("sample_id") == sample_identity(sample, index),
                f"row {position}: sample identity mismatch")
        dataset = dataset_name_for(sample=sample, dataset_name=manifest.get("dataset"))
        # Native Hippo only exports MuSiQue hops parsed from its question ID.
        # Hotpot/2Wiki have no per-question hop field in that baseline output;
        # the PathCondRAG reader still supplies the existing dataset prior.
        exported_hop = None if name == "hipporag2" and dataset != "musique" else hop
        require(row.get("benchmark_hops") == exported_hop and row.get("question") == sample["question"],
                f"row {position}: question/hop mismatch")
        docs, candidates = row.get("docs"), row.get("candidate_docs")
        require(isinstance(docs, list) and len(docs) == 10 and len(set(docs)) == 10,
                f"row {position}: need 10 unique result documents")
        require(isinstance(candidates, list) and len(candidates) == expected_candidates
                and len(set(candidates)) == expected_candidates,
                f"row {position}: need {expected_candidates} unique candidate documents")
        require(docs == candidates[:10], f"row {position}: result/candidate prefixes differ")
        require(set(candidates) <= corpus, f"row {position}: document absent from corpus")
        for key, count in (("doc_scores", 10), ("candidate_doc_scores", expected_candidates)):
            scores = row.get(key)
            require(isinstance(scores, list) and len(scores) == count
                    and all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores),
                    f"row {position}: invalid {key}")
        gold = gold_docs(sample, manifest.get("dataset"))
        require(set(row.get("gold_docs") or []) == gold, f"row {position}: exported gold docs mismatch")
        metrics = {f"Recall@{k}": recall(gold, candidates[:k]) for k in (1, 2, 5, 10, 20, 200)}
        published_row = row.get("retrieval_metrics") or {}
        require(all(isinstance(published_row.get(key), (int, float))
                    and abs(published_row[key] - value) <= 0.00011 for key, value in metrics.items()),
                f"row {position}: per-question recall differs")
        ranks = row.get("gold_document_ranks")
        expected_ranks = {doc: candidates.index(doc) + 1 if doc in candidates else None for doc in gold}
        require(isinstance(ranks, list) and len(ranks) == len(gold)
                and all(isinstance(item, dict) and "doc" in item and "rank" in item for item in ranks)
                and {item["doc"]: item["rank"] for item in ranks} == expected_ranks,
                f"row {position}: gold document ranks differ")
        all5, all10 = gold <= set(docs[:5]), gold <= set(docs)
        require(row.get("all_gold_in_top5") is all5 and row.get("all_gold_in_top10") is all10,
                f"row {position}: full-chain flags mismatch")
        require(isinstance(row.get("retrieval_trace"), dict), f"row {position}: missing retrieval trace")
        measurements.append({"query_index": index, "hops": hop,
                             "metrics": metrics, "all_gold_top5": all5, "all_gold_top10": all10})
    metrics = {key: sum(m["metrics"][key] for m in measurements) / len(measurements)
               for key in measurements[0]["metrics"]}
    published = result.get("retrieval_metrics") or {}
    for key, value in metrics.items():
        require(isinstance(published.get(key), (int, float)) and abs(published[key] - value) <= 0.00011,
                f"{key}: published metric differs from macro exact-string recall")
    return measurements, metrics


def report(args):
    root, name = Path(args.out_root), args.name
    manifest = manifest_at(root)
    require(name in manifest["cases"], "case not selected")
    require(sha256(manifest["data_path"]) == manifest["data_sha256"], "dataset changed")
    require(sha256(manifest["corpus_path"]) == manifest["corpus_sha256"], "corpus changed")
    data, corpus, _ = validated_dataset(manifest["data_path"], manifest["corpus_path"], manifest.get("dataset"))
    case = root / "cases" / name
    result = read_json(case / "result.json")
    measurements, metrics = validate_result(result, manifest, name, data, corpus)
    before = read_json(case / "before.json")
    after = asset_hashes(case / "index", manifest["model_dir"])
    require(before["asset_sha256"] == after == manifest["source_asset_sha256"], "graph/embeddings changed")
    require(asset_hashes(manifest["source_index"], manifest["model_dir"]) == manifest["source_asset_sha256"],
            "source index changed")
    require(before["initial_cache_sha256"] == manifest["initial_cache_sha256"], "initial cache mismatch")
    require(args.log_start >= 0 and args.log_end >= args.log_start, "vLLM log rotated/truncated")
    status = http_status_counts(args.vllm_log, args.log_start, args.log_end)
    require(not any(c != "200" for c in status), f"non-200 HTTP responses: {status}")
    stats = result.get("llm_request_stats")
    require(isinstance(stats, dict) and "failures" in stats, "LLM request failure accounting is missing")
    require(stats.get("failures") == 0, f"LLM request failures: {stats}")
    require(stats.get("max_in_flight") == 8, "HTTP in-flight ceiling must be 8")
    require(not stats.get("http_attempts") or bool(status), "HTTP attempts have no vLLM status accounting")
    stratified = {}
    for hop in sorted({m["hops"] for m in measurements}):
        subset = [m for m in measurements if m["hops"] == hop]
        if subset:
            stratified[str(hop)] = {
                "n_samples": len(subset),
                "retrieval_metrics": {key: sum(m["metrics"][key] for m in subset) / len(subset) for key in metrics},
                "all_gold_top5": sum(m["all_gold_top5"] for m in subset) / len(subset),
                "all_gold_top10": sum(m["all_gold_top10"] for m in subset) / len(subset),
            }
    output = {
        "name": name, "improvement_stage": CASES[name], "validated": True,
        "seconds": args.elapsed, "retrieval_seconds": result.get("retrieval_seconds"),
        "n_samples": len(measurements), "retrieval_metrics": metrics,
        "all_gold_top5": sum(m["all_gold_top5"] for m in measurements) / len(measurements),
        "all_gold_top10": sum(m["all_gold_top10"] for m in measurements) / len(measurements),
        "hop_distribution": manifest["hop_distribution"], "stratified": stratified,
        "llm_request_stats": stats, "http_status_in_log": status,
        "vllm_log_slice": {"path": args.vllm_log, "start": args.log_start, "end": args.log_end},
        "asset_sha256_before": before["asset_sha256"], "asset_sha256_after": after,
        "per_question": measurements,
    }
    if manifest.get("dataset"):
        output["dataset"] = manifest["dataset"]
        output["benchmark_hop_policy"] = manifest["benchmark_hop_policy"]
        output["gold_support_count_distribution"] = manifest["gold_support_count_distribution"]
    write_json(case / "report.json", output)
    write_json(case / "validated.ok", {"result_sha256": sha256(case / "result.json"),
                                      "report_sha256": sha256(case / "report.json"),
                                      "manifest_sha256": sha256(root / "manifest.json")})
    print(f"[validated] {name} n={len(measurements)} sec={args.elapsed} R@5={metrics['Recall@5']:.4f}", flush=True)


def summary(args):
    root, manifest = Path(args.out_root), manifest_at(args.out_root)
    reports = {}
    for name in manifest["cases"]:
        args.name = name
        require(ready(args) == 0, f"case is incomplete: {name}")
        reports[name] = read_json(root / "cases" / name / "report.json")
    compact = {
        name: {key: value for key, value in report.items() if key != "per_question"}
        for name, report in reports.items()
    }
    write_json(root / "comparison.json", {"manifest": str(root / "manifest.json"), "cases": compact})
    table = ["| Case | R@1 | R@2 | R@5 | R@10 | All gold@5 | Seconds | HTTP attempts |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in reports.items():
        values = [f"{item['retrieval_metrics'][f'Recall@{k}']:.4f}" for k in (1, 2, 5, 10)]
        table.append(f"| {name} | " + " | ".join(values)
                     + f" | {item['all_gold_top5']:.4f} | {item['seconds']} | {item['llm_request_stats'].get('http_attempts', 0)} |")
    (root / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    write_json(root / "completed.ok", {"cases": list(reports), "manifest_sha256": sha256(root / "manifest.json")})
    print("\n".join(table), flush=True)


def register_commands(subparsers):
    parser = subparsers.add_parser("improvement-prepare", help="validate corpus/index and freeze sample/cache identity")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--shared-output-root",
                        help="persistent multi-dataset root containing metadata/ and shared_indexes/")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-indices-file")
    parser.add_argument("--cases", default="")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--build-shared-index", action="store_true",
                        help="prepare an empty OUT_ROOT/shared_hipporag2_index for fresh OpenIE/index build")
    parser.set_defaults(handler=prepare)
    parser = subparsers.add_parser("improvement-index-ready", help="check frozen shared index integrity")
    parser.add_argument("--out-root", required=True)
    parser.set_defaults(handler=index_ready)
    parser = subparsers.add_parser("improvement-freeze-index", help="validate fresh OpenIE/index and freeze its cache")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--elapsed", type=int, required=True)
    parser.add_argument("--vllm-log", required=True)
    parser.add_argument("--log-start", type=int, required=True)
    parser.add_argument("--log-end", type=int, required=True)
    parser.add_argument("--openie-strict", choices=("true", "false"), default="true",
                        type=lambda value: value.lower(), dest="openie_strict_choice")
    def freeze_cli(args):
        args.openie_strict = args.openie_strict_choice == "true"
        return freeze_index(args)
    parser.set_defaults(handler=freeze_cli)
    for command, handler in (("improvement-initialize-case", initialize_case), ("improvement-ready", ready)):
        parser = subparsers.add_parser(command)
        parser.add_argument("--out-root", required=True)
        parser.add_argument("--name", choices=list(CASES), required=True)
        parser.set_defaults(handler=handler)
    parser = subparsers.add_parser("improvement-report", help="strict post-run integrity and retrieval validation")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--name", choices=list(CASES), required=True)
    parser.add_argument("--elapsed", type=int, required=True)
    parser.add_argument("--vllm-log", required=True)
    parser.add_argument("--log-start", type=int, required=True)
    parser.add_argument("--log-end", type=int, required=True)
    parser.set_defaults(handler=report)
    parser = subparsers.add_parser("improvement-summary", help="summarize only validated completed cases")
    parser.add_argument("--out-root", required=True)
    parser.set_defaults(handler=summary)
