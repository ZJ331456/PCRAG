"""Checks for evidence labels, invalid bindings and evaluation-only exports."""
import json
import unittest
from types import SimpleNamespace

import numpy as np

from pathcondrag.pathcondrag import PCRAG
from pathcondrag.utils.misc_utils import QuerySolution
from pathcondrag.evaluation.result_export import detailed_result


class ImprovementCorrectnessTests(unittest.TestCase):
    def make_rag(self, stage):
        rag = object.__new__(PCRAG)
        rag.pcrag_config = SimpleNamespace(improvement_stage=stage, qd_cache_decompositions=False,
                                         qd_llm_temperature=0.0, pcqd_disable_entity_grounding=False)
        rag.entity_embedding_store = SimpleNamespace(get_row=lambda key: {"content": "Alice Smith"})
        return rag

    def test_real_entity_text_replaces_hash_only_for_new_stages(self):
        entity = "entity-" + "a" * 32
        original = self.make_rag(0)
        corrected = self.make_rag(1)
        evidence = "Alice Smith wrote this book."
        self.assertFalse(original._entity_supported_in_evidence(entity, evidence))
        self.assertTrue(corrected._entity_supported_in_evidence(entity, evidence))
        self.assertEqual(corrected._grounded_entity_surface(entity), "Alice Smith")
        corrected.entity_embedding_store.get_row = lambda key: {}
        self.assertEqual(corrected._grounded_entity_surface(entity), "")

    def test_invalid_citations_and_unsupported_entities_are_rejected(self):
        rag = self.make_rag(1)
        items = [
            {"question": "Where was Alice Smith born?", "grounded_entities": ["Alice Smith"], "supporting_hint_ids": [0]},
            {"question": "Where was Mallory born?", "grounded_entities": ["Mallory"], "supporting_hint_ids": [0]},
            {"question": "Where was Alice Smith born?", "grounded_entities": ["Alice Smith"], "supporting_hint_ids": [-1]},
        ]
        rag.llm_model = SimpleNamespace(infer=lambda **kwargs: (json.dumps({"sub_questions": items}), {}))
        hints = [{"source_entity": "Book", "candidate_answer_or_bridge": "Alice Smith", "evidence": "Alice Smith wrote Book."}]
        questions = rag._decompose_query_with_hints("query", hints, 4)
        self.assertEqual(len(questions), 1)
        self.assertTrue(questions[0]["evidence_validated"])
        self.assertEqual({r["reason"] for r in rag._pcqd_diagnostics_by_query["query"]["validation_rejections"]},
                         {"invalid_hint_reference", "unsupported_entity_binding"})

    def test_top10_export_preserves_deep_candidates_and_gold_ranks(self):
        docs = [f"title\npassage {i}" for i in range(200)]
        scores = np.linspace(1, 0, 200)
        solution = QuerySolution("q", docs, scores, retrieval_trace={"evidence": {"stage": 3}})
        row = detailed_result(solution, {"id": "4hop_x"}, 5, [docs[0], docs[8], docs[100]], 4)
        self.assertEqual(len(row["docs"]), 10)
        self.assertEqual(len(row["candidate_docs"]), 200)
        self.assertEqual(row["retrieval_metrics"]["Recall@5"], 1 / 3)
        self.assertEqual([g["rank"] for g in row["gold_document_ranks"]], [1, 9, 101])
        self.assertEqual(row["benchmark_hops"], 4)
        self.assertEqual(row["candidate_docs"][:10], row["docs"])
        self.assertEqual(len(solution.docs), 200)
        json.dumps(row)


if __name__ == "__main__":
    unittest.main()
