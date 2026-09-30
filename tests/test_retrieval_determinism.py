"""CPU-only tie-break checks for retrieval paths and PCQD hints."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathcondrag.path_optimizer import PathCandidate, greedy_path_set_selection  # noqa: E402
from pathcondrag.pathcondrag import PCRAG  # noqa: E402


class RetrievalDeterminismTests(unittest.TestCase):
    def test_equal_idf_iterative_seeds_use_entity_key(self):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(
            iterative_seed_mode="idf_novel",
            use_entity_idf_index=True,
            iterative_round2_seed_top_k=2,
        )
        rag.entity_key_to_local_idx = {"entity-c": 0, "entity-b": 1, "entity-a": 2}
        rag.entity_embeddings = np.zeros((3, 2), dtype=np.float32)
        rag.entity_idf = {key: 3.0 for key in rag.entity_key_to_local_idx}
        result = rag._select_iterative_bridge_seeds(
            query="q",
            candidate_entities={"entity-c", "entity-b", "entity-a"},
            original_seed_entities=set(),
        )
        self.assertEqual(result, [("entity-a", 3.0), ("entity-b", 3.0)])

    def test_equal_seed_scores_use_stable_softmax_order(self):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(use_entity_idf_index=False, use_qcappr=False)
        rag.global_config = SimpleNamespace(damping=0.5)
        rag.query_to_embedding = {"triple": {"q": np.array([1.0, 0.0])}}
        rag.entity_key_to_local_idx = {"entity-z": 0, "entity-a": 1}
        rag.entity_embeddings = np.array([[0.0, 1.0], [0.0, 1.0]])
        distribution = rag._build_seed_distribution("q", {"entity-z", "entity-a"}, 2)
        self.assertEqual(list(distribution), ["entity-a", "entity-z"])

    def test_pcqd_equal_idf_bridge_and_source_ties_are_stable(self):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(
            pcqd_ground_top_docs=1,
            pcqd_disable_path_filtering=False,
            pcqd_path_score_threshold=0.0,
            pcqd_entity_top_k=2,
            pcqd_enable_bridge_voting=True,
            pcqd_max_bridges_per_source=1,
        )
        rag.passage_node_keys = ["p0"]
        rag.chunk_to_entities = {"p0": {"source-z", "source-a", "bridge-z", "bridge-a"}}
        rag.chunk_embedding_store = SimpleNamespace(get_row=lambda key: {"content": "evidence"})
        rag.entity_idf = {"bridge-z": 3.0, "bridge-a": 3.0}
        rag._entity_supported_in_evidence = lambda bridge, text: True
        rag._entity_surface = lambda key: key
        hints, _ = rag._build_pcqd_path_hints(
            np.array([0]), np.array([1.0]),
            {"source-z": 0.5, "source-a": 0.5},
            {},
        )
        self.assertEqual(
            [(h["source_entity"], h["candidate_answer_or_bridge"]) for h in hints],
            [("source-a", "bridge-a"), ("source-z", "bridge-a")],
        )

    def test_pcqd_relaxed_fallback_uses_stable_first_seed(self):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(
            pcqd_ground_top_docs=1,
            pcqd_disable_path_filtering=False,
            pcqd_path_score_threshold=0.0,
            pcqd_entity_top_k=2,
            pcqd_enable_bridge_voting=True,
            pcqd_max_bridges_per_source=1,
        )
        rag.passage_node_keys = ["p0"]
        rag.chunk_to_entities = {"p0": {"bridge-z", "bridge-a"}}
        rag.chunk_embedding_store = SimpleNamespace(get_row=lambda key: {"content": "evidence"})
        rag.entity_idf = {"bridge-z": 3.0, "bridge-a": 3.0}
        rag._entity_supported_in_evidence = lambda bridge, text: True
        rag._entity_surface = lambda key: key
        hints, _ = rag._build_pcqd_path_hints(
            np.array([0]), np.array([1.0]),
            {"source-z": 0.5, "source-a": 0.5},
            {},
        )
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["source_entity"], "source-a")
        self.assertEqual(hints[0]["candidate_answer_or_bridge"], "bridge-a")

    def test_path_set_equal_objectives_do_not_depend_on_input_order(self):
        def candidate(key):
            return PathCandidate(
                nodes=["entity", key], passage_key=key, covered_entities={"entity"},
                score_relevance=1.0, score_connectivity=1.0,
                score_consistency=1.0, score_completeness=1.0,
                score_total=1.0,
            )

        for order in ([candidate("p-z"), candidate("p-a")],
                      [candidate("p-a"), candidate("p-z")]):
            selected = greedy_path_set_selection(order, {"entity"}, 1, 0.35)
            self.assertEqual(selected[0].passage_key, "p-a")


if __name__ == "__main__":
    unittest.main()
