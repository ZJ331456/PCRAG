"""Path-owned completion report with explicit extraction validation scope."""

from dataclasses import asdict
from pathlib import Path
from .publication_policy import apply_publication_policy


def tolerant_index_build_report(rag, docs, config, seconds):
    """Retain coverage/storage checks and report the publication policy."""
    expected_docs = set(docs)
    indexed_docs = set(rag.chunk_embedding_store.get_all_texts())
    chunk_ids = set(rag.chunk_embedding_store.get_all_ids())
    rows = getattr(rag, '_openie_info', None)
    if not isinstance(rows, list):
        raise RuntimeError('Index construction did not retain its OpenIE rows')
    ids = [row.get('idx') for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != chunk_ids:
        raise RuntimeError('Index/OpenIE coverage is incomplete or contains duplicate chunk IDs')
    publication = getattr(rag, '_openie_publication_report', None)
    if publication is None:
        failed_ids = [row['idx'] for row in rows if
                      ((row.get('openie_metadata') or {}).get('publication') or {}).get('skipped_from_graph')]
        publication = apply_publication_policy(rows, failed_ids, strict=False)
    failed = set(publication['failed_chunk_ids'])
    for row in rows:
        if row['idx'] in failed:
            policy = (row.get('openie_metadata') or {}).get('publication') or {}
            if row.get('extracted_triples') != [] or policy.get('skipped_from_graph') is not True:
                raise RuntimeError('Failed extraction was not explicitly excluded from graph inputs')
    if indexed_docs != expected_docs:
        raise RuntimeError('Constructed passage index does not cover the complete corpus')
    if rag.graph.vcount() != len(chunk_ids) + len(rag.entity_embedding_store.get_all_ids()):
        raise RuntimeError('Constructed graph and vector stores have different node counts')
    if not Path(rag._graph_pickle_filename).is_file() or not Path(rag.index_manifest_path).is_file():
        raise RuntimeError('Constructed index is missing its graph or schema manifest')
    stats = rag.llm_model.get_request_stats()
    if stats.get('http_attempts', 0) <= 0:
        raise RuntimeError(f'Fresh index did not perform LLM requests: {stats}')
    profile = rag._quality_profile()
    mode = profile.get('validation_mode', profile.get('openie_validation_mode', 'source_verified'))
    runtime = asdict(config)
    runtime.update(openie_strict=publication.get('openie_strict', False),
                   openie_prompt_version=profile['prompt_version'],
                   openie_validation_mode=mode)
    return {
        'dataset': config.dataset, 'method': 'hipporag2', 'eval_mode': 'index_only',
        'save_dir': config.save_dir, 'indexed_docs': len(indexed_docs),
        'indexed_chunks': len(chunk_ids), 'corpus_docs': len(docs),
        'openie_document_count': len(rows), 'openie_failure_count': len(failed),
        'openie_empty_entities_count': sum(not row.get('extracted_entities') for row in rows),
        'openie_empty_triples_count': sum(not row.get('extracted_triples') for row in rows),
        'index_seconds': seconds, 'runtime_config': runtime, 'llm_request_stats': stats,
        'force_index_from_scratch': config.force_index_from_scratch,
        'force_openie_from_scratch': config.force_openie_from_scratch,
        'reuse_index': False, 'shared_index': True, 'index_build_complete': True,
        'all_openie_complete': not failed,
        'all_openie_verified': not failed and mode == 'source_verified',
        'semantic_verification_applied': mode == 'source_verified',
        'openie_validation_scope': 'structural' if mode == 'structural' else 'all_final_relations',
        'publication_report': publication,
    }
