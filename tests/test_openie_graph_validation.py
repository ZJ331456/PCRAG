"""Check repaired graph contributions against small graphs built by HippoRAG."""

import copy
import hashlib
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import igraph as ig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, '/root/baseline/HippoRAG/src')

from hipporag.HippoRAG import HippoRAG  # noqa: E402
from hipporag.utils.misc_utils import reformat_openie_results, text_processing  # noqa: E402
from utils.openie_graph_validation import validate_graph_contributions  # noqa: E402

baseline_module = importlib.import_module('hipporag.HippoRAG')


def hash_id(value, prefix):
    return prefix + hashlib.md5(value.encode()).hexdigest()


def fixture_rows():
    rows = []
    for passage, triples in (
            ('First source passage', [['Alice', 'born in', 'Paris'],
                                      ['Alice', 'works in', 'Paris'],
                                      ['Alice', 'is', 'Alice']]),
            ('Second source passage', [['Alice', 'visited', 'Paris'],
                                       ['Alice', 'visited', 'Berlin']])):
        rows.append({'idx': hash_id(passage, 'chunk-'), 'passage': passage,
                     'extracted_entities': ['Alice', 'Paris', 'Berlin'],
                     'extracted_triples': triples})
    return rows


def build_baseline_graph(rows, directed=False):
    """Use unchanged baseline edge code, with in-memory stores and no models."""
    rag = HippoRAG.__new__(HippoRAG)
    rag.graph = ig.Graph(directed=directed)
    rag.graph['hipporag_edge_schema'] = 1
    rag.node_to_node_stats = {}
    rag.ent_node_to_chunk_ids = {}
    rag._fact_edge_source_counts = {}
    rag._passage_edge_sources = {}
    rag._synonym_edge_scores = {}
    _, outputs = reformat_openie_results(rows)
    triples = [[text_processing(triple) for triple in outputs[row['idx']].triples] for row in rows]
    chunk_entities = [list({triple[i] for triple in group for i in (0, 2)}) for group in triples]
    entities = {entity for group in chunk_entities for entity in group}
    entity_rows = {hash_id(entity, 'entity-'): {'content': entity} for entity in entities}
    chunk_rows = {row['idx']: {'content': row['passage']} for row in rows}
    rag.entity_embedding_store = SimpleNamespace(get_all_id_to_rows=lambda: copy.deepcopy(entity_rows))
    rag.chunk_embedding_store = SimpleNamespace(get_all_id_to_rows=lambda: copy.deepcopy(chunk_rows))
    with patch.object(baseline_module, 'tqdm', lambda iterator, **kwargs: iterator):
        rag.add_fact_edges([row['idx'] for row in rows], triples)
        rag.add_passage_edges([row['idx'] for row in rows], chunk_entities)
    for first, second, score in (('Alice', 'Paris', 0.9), ('Paris', 'Berlin', 0.85)):
        left, right = hash_id(first.lower(), 'entity-'), hash_id(second.lower(), 'entity-')
        keys = ((left, right), (right, left)) if directed else (tuple(sorted((left, right))),)
        for key in keys:
            rag.node_to_node_stats[key] = max(rag.node_to_node_stats.get(key, 0), score)
            rag._synonym_edge_scores[key] = score
    rag.add_new_nodes()
    rag.add_new_edges()
    return rag.graph


def find_edge(graph, left, right):
    names = graph.vs['name']
    return next(edge for edge in graph.es
                if {names[edge.source], names[edge.target]} == {left, right})


class GraphContributionTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture_rows()
        self.graph = build_baseline_graph(self.rows)
        self.alice = hash_id('alice', 'entity-')
        self.paris = hash_id('paris', 'entity-')
        self.berlin = hash_id('berlin', 'entity-')

    def validate(self, graph=None, rows=None):
        return validate_graph_contributions(graph if graph is not None else self.graph,
                                            rows if rows is not None else self.rows, text_processing)

    def test_unchanged_baseline_graph_passes_exact_contribution_checks(self):
        report = self.validate()
        self.assertEqual(report['expected_fact_edges'], 2)
        self.assertEqual(report['expected_passage_edges'], 5)
        self.assertEqual(report['fact_source_contributions'], 4)
        self.assertEqual(report['ignored_fact_self_loops'], 1)
        self.assertEqual(report['synonym_edges'], 2)
        self.assertTrue(report['fact_provenance_exact'])
        report = self.validate(build_baseline_graph(self.rows, directed=True))
        self.assertEqual(report['expected_fact_edges'], 4)
        self.assertEqual(report['fact_source_contributions'], 8)

    def test_raw_duplicates_and_normalization_collisions_match_baseline_counts(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['extracted_triples'].append(['Alice', 'born in', 'Paris'])
        rows[0]['extracted_triples'].append([' ALICE ', 'BORN IN', ' PARIS '])
        graph = build_baseline_graph(rows)
        self.assertEqual(self.validate(graph, rows)['fact_source_contributions'], 5)

    def test_missing_fact_and_missing_passage_edges_are_detected(self):
        for endpoints in ((self.alice, self.paris), (self.rows[0]['idx'], self.alice)):
            graph = self.graph.copy()
            graph.delete_edges(find_edge(graph, *endpoints).index)
            with self.subTest(endpoints=endpoints), self.assertRaisesRegex(ValueError, 'Missing'):
                self.validate(graph)

    def test_stale_sources_and_wrong_counts_are_detected_even_with_recomposed_weights(self):
        for stale in (True, False):
            graph = self.graph.copy()
            edge = find_edge(graph, self.alice, self.paris)
            counts = dict(edge['fact_source_counts'])
            key = 'chunk-old-deleted' if stale else self.rows[0]['idx']
            counts[key] = counts.get(key, 0) + 1
            edge['fact_source_counts'] = counts
            edge['weight'] = float(sum(counts.values()))
            with self.subTest(stale=stale), self.assertRaisesRegex(ValueError, 'source counts do not match'):
                self.validate(graph)

    def test_wrong_weight_and_edge_kind_are_detected(self):
        edge = find_edge(self.graph, self.alice, self.paris)
        edge['weight'] += 0.1
        with self.assertRaisesRegex(ValueError, 'composed weight'):
            self.validate()
        edge['weight'] -= 0.1
        edge['edge_kind'] = 'fact'
        with self.assertRaisesRegex(ValueError, 'edge_kind'):
            self.validate()

    def test_wrong_passage_provenance_is_detected(self):
        find_edge(self.graph, self.rows[0]['idx'], self.alice)['passage_source'] = self.rows[1]['idx']
        with self.assertRaisesRegex(ValueError, 'Passage source'):
            self.validate()

    def test_swapped_canonical_metadata_is_detected(self):
        edge = find_edge(self.graph, self.alice, self.paris)
        edge['source_key'], edge['target_key'] = edge['target_key'], edge['source_key']
        with self.assertRaisesRegex(ValueError, 'canonical physical endpoints'):
            self.validate()

    def test_parallel_logical_edges_and_self_loops_are_detected(self):
        edge = find_edge(self.graph, self.alice, self.paris)
        self.graph.add_edge(edge.source, edge.target, **edge.attributes())
        with self.assertRaisesRegex(ValueError, 'Duplicate logical'):
            self.validate()
        graph = build_baseline_graph(self.rows)
        vertex = graph.vs.find(name=self.alice).index
        graph.add_edge(vertex, vertex, weight=1.0, edge_kind='fact',
                       fact_source_counts={self.rows[0]['idx']: 1}, synonym_score=0.0,
                       passage_source=None, source_key=self.alice, target_key=self.alice)
        with self.assertRaisesRegex(ValueError, 'self loop'):
            self.validate(graph)

    def test_synonym_score_must_be_finite_and_within_threshold_cosine_bounds(self):
        for score in (float('nan'), float('inf'), -0.1, 0.7, 1.01):
            graph = self.graph.copy()
            find_edge(graph, self.paris, self.berlin)['synonym_score'] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.validate(graph)

    def test_bool_count_and_untyped_edge_are_rejected(self):
        edge = find_edge(self.graph, self.alice, self.paris)
        counts = dict(edge['fact_source_counts'])
        counts[self.rows[1]['idx']] = True
        edge['fact_source_counts'] = counts
        with self.assertRaisesRegex(ValueError, 'Invalid fact source counts'):
            self.validate()
        graph = self.graph.copy()
        del graph.es['passage_source']
        with self.assertRaisesRegex(ValueError, 'source-aware edge attributes'):
            self.validate(graph)

    def test_missing_schema_and_malformed_source_triples_are_rejected(self):
        self.graph['hipporag_edge_schema'] = True
        with self.assertRaisesRegex(ValueError, 'hipporag_edge_schema=1'):
            self.validate()
        rows = copy.deepcopy(self.rows)
        rows[0]['extracted_triples'].append(['Alice', 'missing object', ''])
        with self.assertRaisesRegex(ValueError, 'Invalid OpenIE triples'):
            self.validate(build_baseline_graph(self.rows), rows)


if __name__ == '__main__':
    unittest.main()
