"""Read-only publication checks for a freshly built shared retrieval index.

The validator never loads an embedding model or calls an LLM. It checks the
actual artifacts, including the source verification records for recovered
extractions, before the experiment runner freezes the shared index.
"""

import hashlib
import json
from pathlib import Path

from pathcondrag.utils.openie_quality import validate_triples

from .openie_graph_validation import validate_graph_contributions


ASSETS = (
    "openie_state.json", "index_manifest.json", "chunk_metadata.json", "graph.pickle",
    "chunk_embeddings/vdb_chunk.parquet", "entity_embeddings/vdb_entity.parquet",
    "fact_embeddings/vdb_fact.parquet",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _key(value, namespace):
    return namespace + "-" + hashlib.md5(value.encode("utf-8")).hexdigest()


def unicode_normalize(value):
    """Match the native unicode_alnum_casefold_v1 normalization exactly."""
    return " ".join("".join(character if character.isalnum() or character.isspace() else " "
                            for character in value.casefold()).split())


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_counts(metadata, triples, chunk_id):
    """Bind the final triples to the final completed source-only verdict."""
    audits = metadata.get("semantic_verification_history")
    _require(isinstance(audits, list) and bool(audits),
             f"Recovered extraction lacks semantic audits: {chunk_id}")
    counts = {"verification_passes": len(audits), "accepted_assertions_across_passes": 0,
              "rejected_assertions_across_passes": 0}
    final_accepted = None
    for audit in audits:
        _require(isinstance(audit, dict) and audit.get("complete") is True
                 and not audit.get("error") and audit.get("n_unverified") == 0,
                 f"Incomplete recovered source audit: {chunk_id}")
        supported, checks = audit.get("supported"), audit.get("checks")
        _require(isinstance(supported, list) and all(type(flag) is bool for flag in supported)
                 and isinstance(checks, list) and len(checks) == len(supported)
                 and audit.get("n_input") == len(supported),
                 f"Invalid recovered audit verdicts: {chunk_id}")
        accepted = []
        for position, (check, flag) in enumerate(zip(checks, supported)):
            _require(isinstance(check, dict) and check.get("index") == position
                     and check.get("supported") is flag,
                     f"Recovered audit order mismatch: {chunk_id}")
            triple = check.get("triple")
            _require(isinstance(triple, list) and not validate_triples([triple]).invalid_triples,
                     f"Invalid recovered audit triple: {chunk_id}")
            if flag:
                accepted.append(tuple(triple))
        accepted_count, rejected_count = sum(supported), len(supported) - sum(supported)
        _require(audit.get("n_accepted") == accepted_count
                 and audit.get("n_rejected") == rejected_count,
                 f"Recovered audit counts mismatch: {chunk_id}")
        counts["accepted_assertions_across_passes"] += accepted_count
        counts["rejected_assertions_across_passes"] += rejected_count
        final_accepted = set(accepted)
    _require(final_accepted == {tuple(triple) for triple in triples},
             f"Final recovered facts differ from accepted source verdicts: {chunk_id}")
    counts["accepted_relations_final"] = len(triples)
    return counts


def _validated_rows(rows, docs, quality_profile):
    _require(isinstance(rows, list) and len(rows) == len(docs), "Fresh OpenIE coverage mismatch")
    entities, facts, chunks, passages = set(), set(), set(), set()
    counts = {"empty_chunks": 0, "legitimate_empty_chunks": 0, "recovered_chunks": 0,
              "audited_chunks": 0, "verification_passes": 0,
              "accepted_relations_final": 0, "accepted_assertions_across_passes": 0,
              "rejected_assertions_across_passes": 0}
    for row in rows:
        _require(isinstance(row, dict), "Invalid OpenIE row")
        passage, chunk_id = row.get("passage"), row.get("idx")
        _require(isinstance(passage, str) and passage in docs
                 and chunk_id == _key(passage, "chunk") and chunk_id not in chunks,
                 "Invalid, extraneous or duplicate fresh OpenIE chunk")
        chunks.add(chunk_id)
        passages.add(passage)
        ner = row.get("extracted_entities")
        _require(isinstance(ner, list) and all(isinstance(entity, str) for entity in ner),
                 f"Invalid fresh NER entities: {chunk_id}")
        metadata = row.get("openie_metadata")
        _require(isinstance(metadata, dict), f"Missing fresh extraction metadata: {chunk_id}")
        for stage in ("ner", "triples"):
            stage_metadata = metadata.get(stage)
            _require(isinstance(stage_metadata, dict) and stage_metadata.get("finish_reason") == "stop"
                     and not stage_metadata.get("error") and not stage_metadata.get("openie_skipped")
                     and stage_metadata.get("quality_status") not in ("failed", "partial")
                     and stage_metadata.get("complete") is not False,
                     f"Incomplete fresh {stage} metadata: {chunk_id}")
        triples = row.get("extracted_triples")
        _require(isinstance(triples, list), f"Invalid fresh triples: {chunk_id}")
        validation = validate_triples(triples)
        _require(not validation.invalid_triples, f"Structurally invalid fresh triples: {chunk_id}")
        triple_metadata = metadata["triples"]
        recovered = bool(triple_metadata.get("quality_recovered")
                         or triple_metadata.get("source_verified_schema"))
        if recovered:
            counts["recovered_chunks"] += 1
            _require(triple_metadata.get("source_verified_schema")
                     == "pathcondrag_fresh_source_verified_openie_v1"
                     and triple_metadata.get("semantic_verified") is True
                     and triple_metadata.get("complete") is True
                     and triple_metadata.get("requires_semantic_verification") is False,
                     f"Recovered extraction is not completely source verified: {chunk_id}")
            audit_counts = _audit_counts(triple_metadata, triples, chunk_id)
            counts["audited_chunks"] += 1
            for name, value in audit_counts.items():
                counts[name] += value
            verifier = quality_profile.get("fresh_recovery_semantic_verifier")
            _require(isinstance(verifier, str) and bool(verifier)
                     and triple_metadata.get("semantic_verifier_contract") == verifier
                     and all(audit.get("contract_version") == verifier
                             for audit in triple_metadata["semantic_verification_history"]),
                     f"Recovered source-verifier contract mismatch: {chunk_id}")
        else:
            _require(not triple_metadata.get("requires_semantic_verification"),
                     f"Unverified fresh relations remain: {chunk_id}")
        if not triples:
            counts["empty_chunks"] += 1
            _require(recovered and triple_metadata.get("source_no_supported_relations") is True
                     and triple_metadata.get("semantic_verified") is True,
                     f"Empty extraction lacks explicit source-only confirmation: {chunk_id}")
            counts["legitimate_empty_chunks"] += 1
        for triple in triples:
            normalized = tuple(unicode_normalize(field) for field in triple)
            _require(all(normalized), f"Empty normalized fresh triple field: {chunk_id}")
            entities.update((normalized[0], normalized[2]))
            facts.add(str(normalized))
    _require(passages == docs, "Fresh OpenIE passage set differs from the corpus")
    return {"chunk": passages, "entity": entities, "fact": facts}, chunks, counts


def _validate_vectors(directory, namespace, needed):
    import numpy as np
    import pyarrow.parquet as pq

    path = directory / f"{namespace}_embeddings/vdb_{namespace}.parquet"
    ids, contents = set(), set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        values = batch.to_pydict()
        _require({"hash_id", "content", "embedding"} <= set(values),
                 f"Invalid {namespace} vector-store columns")
        for key, content, vector in zip(values["hash_id"], values["content"], values["embedding"]):
            _require(isinstance(content, str) and key == _key(content, namespace) and key not in ids,
                     f"Invalid or duplicate {namespace} vector identity")
            array = np.asarray(vector)
            _require(array.shape == (4096,) and np.isfinite(array).all(),
                     f"Invalid {namespace} vector dimension or finite values")
            _require(np.isclose(np.linalg.norm(array), 1.0, atol=0.02),
                     f"Non-normalized {namespace} vector")
            ids.add(key)
            contents.add(content)
    _require(contents == needed, f"{namespace} vector contents differ from current OpenIE")
    return ids, {"count": len(ids), "dimension": 4096, "finite": True, "normalized": True}


def validate_fresh_index(out, index, docs):
    """Validate fresh artifacts and write OUT/fresh_index_validation.json."""
    import igraph as ig

    out, index, docs = Path(out).resolve(), Path(index).resolve(), set(docs)
    _require(bool(docs) and all(isinstance(doc, str) for doc in docs), "Fresh corpus must contain text")
    candidates = list(index.glob("*/index_manifest.json"))
    _require(len(candidates) == 1, "Fresh index must contain exactly one model manifest")
    directory = candidates[0].parent
    manifest, state = _read(candidates[0]), _read(directory / "openie_state.json")
    top = _read(index / "openie_results_ner_qwen3-8b.json")
    _require(manifest.get("text_normalization") == "unicode_alnum_casefold_v1",
             "Fresh index normalization is incompatible")
    _require(state.get("docs") == top.get("docs") and state.get("provenance") == top.get("provenance"),
             "Model-state and top-level OpenIE artifacts differ")
    provenance = state.get("provenance")
    _require(isinstance(provenance, dict) and manifest.get("openie") == provenance,
             "Fresh index manifest and OpenIE provenance differ")
    profile = provenance.get("quality_profile")
    _require(isinstance(profile, dict)
             and str(profile.get("extractor", "")).endswith(".SourceVerifiedOpenIE"),
             "Fresh index was not built with source-verified PathCondRAG extraction")
    needed, chunks, counts = _validated_rows(state.get("docs"), docs, profile)
    vector_ids, vector_reports = {}, {}
    for namespace in ("chunk", "entity", "fact"):
        vector_ids[namespace], vector_reports[namespace] = _validate_vectors(
            directory, namespace, needed[namespace])
    _require(vector_ids["chunk"] == chunks, "Chunk vector identities differ from source rows")
    _require(set(_read(directory / "chunk_metadata.json")) == chunks,
             "Chunk metadata does not cover the exact corpus")
    _require(_key("", "entity") not in vector_ids["entity"], "Empty entity persisted in graph inputs")
    graph = ig.Graph.Read_Pickle(str(directory / "graph.pickle"))
    names = graph.vs["name"] if "name" in graph.vs.attribute_names() else []
    _require(len(names) == len(set(names))
             and set(names) == vector_ids["chunk"] | vector_ids["entity"],
             "Fresh graph vertex identities differ from the vector stores")
    edges = validate_graph_contributions(graph, state["docs"], unicode_normalize)
    # Only an execution record actually exported by the build is reported.
    # Runtime defaults or source code alone do not prove which path ran.
    execution_path = directory / "shared_build_execution.json"
    execution = _read(execution_path) if execution_path.is_file() else None
    trace = execution.get("shared_knn_execution", execution) if isinstance(execution, dict) else execution
    if trace is not None:
        _require(isinstance(trace, dict) and trace.get("device") == "cuda"
                 and trace.get("dtype") == "float32" and trace.get("allow_tf32") is False
                 and trace.get("query_batch_size") == 1000 and trace.get("key_batch_size") == 16384
                 and trace.get("native_cosine_topk_and_threshold_unchanged") is True
                 and trace.get("complete") is True,
                 "Observed shared KNN execution profile is inconsistent")
    report = {"complete": True, "index": str(index), "indexed_docs": len(docs),
              "model_dir": directory.name, "invalid_records": 0,
              "quality_profile": profile, "vectors": vector_reports,
              "graph_nodes": graph.vcount(), "graph_edges": graph.ecount(),
              "edge_validation": edges, **counts,
              "shared_knn_execution": {"observed": trace is not None, "trace": trace},
              "asset_sha256": {name: _sha256(directory / name) for name in ASSETS},
              "top_openie_sha256": _sha256(index / "openie_results_ner_qwen3-8b.json")}
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "fresh_index_validation.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    print(f"[fresh-index-validated] docs={len(docs)} recovered={counts['recovered_chunks']} "
          f"legitimate_empty={counts['legitimate_empty_chunks']} graph/store/OpenIE consistent", flush=True)
    return report
