"""Validate the source-aware HippoRAG graph against its current OpenIE rows.

This is independent of the graph builder: expected fact and passage
contributions are derived from the source rows, then compared with every
physical edge. It never edits the graph or imports an embedding model.
"""

import hashlib
import math
from collections import defaultdict
from numbers import Real

from pathcondrag.index.openie.openie_quality import validate_triples


EDGE_ATTRIBUTES = {
    'weight', 'edge_kind', 'fact_source_counts', 'synonym_score',
    'passage_source', 'source_key', 'target_key',
}


def _entity_key(text):
    return 'entity-' + hashlib.md5(text.encode('utf-8')).hexdigest()


def _finite_number(value, field, edge_key):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'Invalid finite numeric {field} for edge {edge_key}')
    return float(value)


def validate_graph_contributions(graph, rows, normalize):
    """Check exact fact counts, passage support, typed edges and their weights.

    Undirected edges have one sorted logical key. Source-aware directed graphs
    have reciprocal fact edges and forward passage edges, as in HippoRAG.
    Exact duplicate raw triples are counted once, matching the baseline's
    reformat step. Distinct raw relations that normalize to the same entity
    pair still each contribute a count. Fact self loops are omitted by the
    baseline's add_new_edges and are therefore omitted from expected edges.

    Synonym contributions are checked for valid endpoints, finite cosine
    scores and the shared index's threshold of 0.8; checking completeness or
    reproducing cosine scores requires embeddings and is outside this gate.
    """
    if ('hipporag_edge_schema' not in graph.attributes()
            or type(graph['hipporag_edge_schema']) is not int
            or graph['hipporag_edge_schema'] != 1):
        raise ValueError('Graph must declare hipporag_edge_schema=1')
    directed = graph.is_directed()

    def logical_key(left, right):
        return (left, right) if directed else tuple(sorted((left, right)))

    expected_facts = defaultdict(dict)
    expected_passages = {}
    chunk_ids, entity_ids = set(), set()
    ignored_self_loops = 0
    for row in rows:
        chunk_id = row['idx']
        if not isinstance(chunk_id, str) or chunk_id in chunk_ids:
            raise ValueError('Invalid or duplicate OpenIE chunk identity')
        chunk_ids.add(chunk_id)
        triples = row['extracted_triples']
        if validate_triples(triples).invalid_triples:
            raise ValueError(f'Invalid OpenIE triples for {chunk_id}')
        raw_seen, chunk_entities = set(), set()
        for triple in triples:
            raw_key = tuple(triple)
            if raw_key in raw_seen:
                continue
            raw_seen.add(raw_key)
            normalized = tuple(normalize(field) for field in triple)
            if any(not isinstance(field, str) or not field for field in normalized):
                raise ValueError(f'Normalized empty OpenIE triple for {chunk_id}')
            left, right = _entity_key(normalized[0]), _entity_key(normalized[2])
            entity_ids.update((left, right))
            chunk_entities.update((left, right))
            if left == right:
                ignored_self_loops += 1
                continue
            fact_keys = ((left, right), (right, left)) if directed else (logical_key(left, right),)
            for edge_key in fact_keys:
                counts = expected_facts[edge_key]
                counts[chunk_id] = counts.get(chunk_id, 0) + 1
        for entity_id in chunk_entities:
            expected_passages[logical_key(chunk_id, entity_id)] = chunk_id

    if 'name' not in graph.vs.attribute_names():
        if graph.vcount():
            raise ValueError('Graph vertices lack names')
        names = []
    else:
        names = graph.vs['name']
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError('Invalid or duplicate graph vertex names')
    if graph.ecount() and not EDGE_ATTRIBUTES.issubset(graph.es.attribute_names()):
        raise ValueError('Graph is missing source-aware edge attributes')
    allowed_endpoints = chunk_ids | entity_ids
    seen, synonym_edges = set(), 0
    for edge in graph.es:
        physical = (names[edge.source], names[edge.target])
        edge_key = logical_key(*physical)
        if physical[0] == physical[1]:
            raise ValueError(f'Unexpected graph self loop {edge_key}')
        if edge_key in seen:
            raise ValueError(f'Duplicate logical graph edge {edge_key}')
        seen.add(edge_key)
        if any(endpoint not in allowed_endpoints for endpoint in physical):
            raise ValueError(f'Unexpected graph endpoint {edge_key}')
        declared = (edge['source_key'], edge['target_key'])
        if (not all(isinstance(key, str) for key in declared)
                or declared != edge_key):
            raise ValueError(f'Edge metadata does not match canonical physical endpoints {edge_key}')

        counts = edge['fact_source_counts']
        if (not isinstance(counts, dict)
                or any(not isinstance(source, str) or type(count) is not int or count < 1
                       for source, count in counts.items())):
            raise ValueError(f'Invalid fact source counts for edge {edge_key}')
        if counts != expected_facts.get(edge_key, {}):
            raise ValueError(f'Fact source counts do not match current OpenIE for edge {edge_key}')
        passage_source = edge['passage_source']
        if passage_source != expected_passages.get(edge_key):
            raise ValueError(f'Passage source does not match current OpenIE for edge {edge_key}')
        if passage_source is not None and not isinstance(passage_source, str):
            raise ValueError(f'Invalid passage source for edge {edge_key}')

        score = _finite_number(edge['synonym_score'], 'synonym_score', edge_key)
        if score < 0 or (score > 0 and (score < 0.8 or score > 1.0 + 1e-6)):
            raise ValueError(f'Synonym score violates cosine/threshold bounds for edge {edge_key}')
        if score > 0:
            if any(endpoint not in entity_ids for endpoint in physical):
                raise ValueError(f'Synonym edge must connect two entities: {edge_key}')
            synonym_edges += 1

        kinds = []
        if counts:
            kinds.append('fact')
        if passage_source is not None:
            kinds.append('passage')
        if score > 0:
            kinds.append('synonym')
        if not kinds or edge['edge_kind'] != '+'.join(kinds):
            raise ValueError(f'Incorrect edge_kind for edge {edge_key}')
        weight = _finite_number(edge['weight'], 'weight', edge_key)
        expected_weight = max(float(sum(counts.values())),
                              1.0 if passage_source is not None else 0.0, score)
        if not math.isclose(weight, expected_weight, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'Incorrect composed weight for edge {edge_key}')

    missing = (set(expected_facts) | set(expected_passages)) - seen
    if missing:
        example = next(iter(missing))
        raise ValueError(f'Missing {len(missing)} expected fact/passage edges; example {example}')
    return {
        'edge_schema': 1, 'directed': directed,
        'validated_edges': graph.ecount(), 'expected_fact_edges': len(expected_facts),
        'expected_passage_edges': len(expected_passages), 'synonym_edges': synonym_edges,
        'fact_source_contributions': sum(sum(counts.values()) for counts in expected_facts.values()),
        'ignored_fact_self_loops': ignored_self_loops,
        'fact_provenance_exact': True, 'passage_provenance_exact': True,
        'weights_and_edge_kinds_valid': True,
    }
