"""CPU-only checks for bounded QD prefetch and explicit hop labels."""

import threading
import unittest
from types import SimpleNamespace

import numpy as np

from pathcondrag.pathcondrag import PCRAG


class RetrievalPrefetchTests(unittest.TestCase):
    def test_hop_override_validation_and_question_limit(self):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(qd_max_sub_questions=4)
        for bad in ([0], [5], [True], [2.5], ["2"]):
            with self.assertRaises(ValueError):
                rag.set_query_hop_overrides(bad, source="benchmark")
        rag.set_query_hop_overrides([2, 3, 4], source="benchmark")
        self.assertEqual([rag._qd_subquestion_limit(h) for h in [2, 3, 4]], [3, 3, 4])
        with self.assertRaisesRegex(ValueError, "count must match"):
            rag.retrieve(["only one query"])
        rag.clear_query_hop_overrides()
        self.assertEqual(rag._qd_subquestion_limit(2), 4)

    def test_graph_search_uses_four_hop_override_without_heuristic(self):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(
            use_qcappr=False,
            use_iterative_retrieval=False,
            use_query_decomposition=False,
        )
        rag.global_config = SimpleNamespace(passage_node_weight=0.5, damping=0.5)
        rag.graph = SimpleNamespace(vs={"name": ["entity", "passage"]})
        rag.node_name_to_vertex_idx = {"p0": 1}
        rag.passage_node_keys = ["p0"]
        rag._estimate_query_hops = lambda *args, **kwargs: self.fail("heuristic must not run")
        rag._extract_seed_entities_from_facts = lambda facts: set()
        rag._build_seed_distribution = lambda query, seeds, hops: {}
        rag._detect_bridge_entities = lambda query, seeds: {}
        rag.dense_passage_retrieval = lambda query: (np.array([0]), np.array([1.0]))
        rag.run_ppr = lambda weights, damping: (np.array([0]), np.array([1.0]))
        ids, scores, ctx = rag._path_graph_search(
            query="four hop query",
            query_fact_scores=np.array([1.0]),
            top_k_facts=[("s", "p", "o")],
            top_k_fact_indices=[0],
            hop_override=4,
        )
        self.assertEqual(ctx["hops"], 4)
        np.testing.assert_array_equal(ids, np.array([0]))

    def test_only_llm_generation_runs_in_bounded_workers(self):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(
            llm_prefetch_workers=3,
            use_query_decomposition=True,
            use_path_conditioned_qd=True,
            qd_min_hops=2,
            qd_max_sub_questions=4,
        )
        rag.set_query_hop_overrides([2, 3, 4], source="benchmark")
        main_thread = threading.current_thread()
        worker_barrier = threading.Barrier(3, timeout=5)
        graph_threads = []
        llm_threads = []

        rag.get_fact_scores = lambda query: np.array([1.0])
        rag.rerank_facts = lambda query, scores: ([0], [(query, "p", "o")], {})

        def graph_search(query, **kwargs):
            graph_threads.append(threading.current_thread())
            self.assertTrue(kwargs["defer_qd"])
            return (
                np.array([0]),
                np.array([1.0]),
                {
                    "hops": kwargs["hop_override"],
                    "_qd_deferred": True,
                    "seed_entities": [],
                    "seed_distribution": {},
                    "bridges_by_seed": {},
                },
            )

        def path_hints(**kwargs):
            graph_threads.append(threading.current_thread())
            return [{"source_entity": "x"}], 0.0

        def llm_generation(query, *args):
            llm_threads.append(threading.current_thread())
            worker_barrier.wait()
            if len(args) == 1:
                return [query]
            return [{"question": query}]

        def static_track(**kwargs):
            graph_threads.append(threading.current_thread())
            return np.array([0]), np.array([1.0]), {"used": True}

        rag._path_graph_search = graph_search
        rag._build_pcqd_path_hints = path_hints
        rag._decompose_query = llm_generation
        rag._decompose_query_with_hints = llm_generation
        rag._build_qd_track_ranking = static_track
        states = list(rag._iter_retrieval_states(["q0", "q1", "q2"]))
        self.assertEqual([s["query_idx"] for s in states], [0, 1, 2])
        self.assertEqual([s["base"][2]["hops"] for s in states], [2, 3, 4])
        self.assertTrue(all(thread is main_thread for thread in graph_threads))
        self.assertEqual(len(llm_threads), 6)
        self.assertTrue(all(thread is not main_thread for thread in llm_threads))
        self.assertEqual([s["static_sub_questions"] for s in states], [["q0"], ["q1"], ["q2"]])
        self.assertEqual([s["pcqd_sub_questions"][0]["question"] for s in states], ["q0", "q1", "q2"])

    def test_llm_transport_error_is_not_silently_converted_to_empty_qd(self):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(
            qd_cache_decompositions=True,
            qd_llm_temperature=0.0,
            pcqd_disable_entity_grounding=False,
        )

        def fail(**kwargs):
            raise RuntimeError("simulated HTTP failure")

        rag.llm_model = SimpleNamespace(infer=fail)
        with self.assertRaisesRegex(RuntimeError, "HTTP failure"):
            rag._decompose_query("question", 3)
        with self.assertRaisesRegex(RuntimeError, "HTTP failure"):
            rag._decompose_query_with_hints("question", [{"source_entity": "x"}], 3)

    def test_empty_and_failed_static_tracks_do_not_issue_pcqd_requests(self):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(
            llm_prefetch_workers=4,
            use_query_decomposition=True,
            use_path_conditioned_qd=True,
            qd_min_hops=2,
            qd_max_sub_questions=3,
        )
        rag.set_query_hop_overrides([2, 2, 2, 2], source="benchmark")
        pcqd_calls = []
        rag.get_fact_scores = lambda query: np.array([1.0])
        rag.rerank_facts = lambda query, scores: (
            ([], [], {}) if query == "no-facts" else ([0], [(query, "p", "o")], {})
        )
        rag._path_graph_search = lambda query, **kwargs: (
            np.array([0]),
            np.array([1.0]),
            {
                "hops": kwargs["hop_override"],
                "_qd_deferred": True,
                "seed_entities": [],
                "seed_distribution": {},
                "bridges_by_seed": {},
            },
        )
        rag._decompose_query = lambda query, limit: [] if query == "empty-static" else [query]
        rag._build_qd_track_ranking = lambda query, **kwargs: (
            (None, None, {"used": False})
            if query == "failed-static"
            else (np.array([0]), np.array([1.0]), {"used": True})
        )
        rag._build_pcqd_path_hints = lambda **kwargs: ([], 0.0)

        def unexpected_pcqd(*args):
            pcqd_calls.append(args)
            return []

        rag._decompose_query_with_hints = unexpected_pcqd
        states = list(rag._iter_retrieval_states(
            ["no-facts", "empty-static", "failed-static", "empty-hints"]
        ))
        self.assertNotIn("base", states[0])
        self.assertNotIn("static_track", states[1])
        self.assertFalse(states[2]["static_track"][2]["used"])
        self.assertEqual(states[3]["path_hints"], ([], 0.0))
        self.assertEqual(pcqd_calls, [])

    def test_prefetched_and_serial_fusion_match_for_identical_llm_outputs(self):
        rag = object.__new__(PCRAG)
        rag._query_hop_overrides = None
        rag.passage_node_keys = ["p0", "p1", "p2"]
        rag.pcrag_config = SimpleNamespace(
            qd_min_hops=2,
            qd_max_sub_questions=3,
            use_path_conditioned_qd=True,
            pcqd_fallback_to_static=True,
            pcqd_include_static_qd=True,
            pcqd_adaptive_fusion=False,
            pcqd_weight_base=0.4,
            pcqd_weight_static=0.2,
            pcqd_weight_path=0.4,
        )
        base_ids = np.array([0, 1, 2])
        base_scores = np.array([0.9, 0.5, 0.1])
        static_sub_questions = ["static subquestion"]
        path_hints = [{"source_entity": "x", "candidate_answer_or_bridge": "y"}]
        pcqd_sub_questions = [{"question": "path subquestion", "grounded_entities": ["y"]}]
        static_track = (np.array([1, 0, 2]), np.array([0.8, 0.4, 0.1]), {"used": True})
        path_track = (np.array([2, 1, 0]), np.array([0.8, 0.4, 0.1]), {"used": True})

        rag._decompose_query = lambda query, limit: static_sub_questions
        rag._decompose_query_with_hints = lambda **kwargs: pcqd_sub_questions
        rag._build_pcqd_path_hints = lambda **kwargs: (path_hints, 0.0)
        rag._build_qd_track_ranking = lambda sub_questions, **kwargs: (
            static_track if sub_questions == static_sub_questions else path_track
        )
        kwargs = dict(
            query="question",
            hops=2,
            base_sorted_doc_ids=base_ids,
            base_sorted_doc_scores=base_scores,
            seed_entities=set(),
            seed_distribution={},
            bridges_by_seed={},
        )
        serial_ids, serial_scores, serial_stats = rag._query_decomposition_retrieval(**kwargs)
        prefetched_ids, prefetched_scores, prefetched_stats = rag._query_decomposition_retrieval(
            **kwargs,
            prefetched_qd=(static_sub_questions, pcqd_sub_questions, path_hints, 0.0),
            prefetched_static_track=static_track,
        )
        np.testing.assert_array_equal(prefetched_ids, serial_ids)
        np.testing.assert_array_equal(prefetched_scores, serial_scores)
        self.assertEqual(prefetched_stats, serial_stats)


if __name__ == "__main__":
    unittest.main()
