"""CPU checks for benchmark-hop report grouping and answer alignment."""

import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from eval_utils import get_benchmark_hops  # noqa: E402
from stratified_eval import evaluate_stratified, stratify_by_hops  # noqa: E402
from pathcondrag.utils.misc_utils import QuerySolution  # noqa: E402


class StratifiedEvaluationTests(unittest.TestCase):
    def test_benchmark_labels_drive_retrieval_and_qa_groups(self):
        # All four query strings look single-hop to the legacy heuristic.
        queries = ["Name A?", "Name B?", "Name C?", "Name D?"]
        labels = [4, 2, 3, 1]
        docs = [[f"doc-{i}"] for i in range(4)]
        answers = [[f"answer-{i}"] for i in range(4)]
        results = [QuerySolution(question=q, docs=d) for q, d in zip(queries, docs)]
        results[0].docs = ["wrong-doc"]
        provenance = {"kind": "per_question_oracle", "oracle": True}

        with patch("stratified_eval.estimate_query_hops_simple", side_effect=AssertionError):
            report = evaluate_stratified(
                queries, results, docs, answers, global_config=None,
                predicted_answers=["wrong-answer", "answer-1", "answer-2", "answer-3"],
                k_list=[1], query_hops=labels, hop_provenance=provenance,
            )

        self.assertEqual(report["overall"]["retrieval"]["Recall@1"], 0.75)
        self.assertEqual(report["overall"]["qa"]["ExactMatch"], 0.75)
        for group in ("single_hop", "two_hop", "three_hop", "four_hop"):
            self.assertEqual(report[group]["count"], 1)
        self.assertEqual(report["four_hop"]["retrieval"]["Recall@1"], 0.0)
        self.assertEqual(report["four_hop"]["qa"]["ExactMatch"], 0.0)
        for group in ("single_hop", "two_hop", "three_hop"):
            self.assertEqual(report[group]["retrieval"]["Recall@1"], 1.0)
            self.assertEqual(report[group]["qa"]["ExactMatch"], 1.0)
        self.assertEqual(report["stratification"]["hop_provenance"], provenance)
        self.assertEqual(report["stratification"]["hop_distribution"],
                         {"4": 1, "2": 1, "3": 1, "1": 1})
        self.assertNotIn("multi_hop", report)

    def test_dataset_prior_one_overrides_multihop_keywords(self):
        queries = ["Who and which former leader came before the latter?"]
        docs = [["doc"]]
        results = [QuerySolution(question=queries[0], docs=docs[0])]
        for dataset in ("nq", "popqa"):
            labels = get_benchmark_hops([{}], dataset)
            report = evaluate_stratified(
                queries, results, docs, [[]], global_config=None, k_list=[1],
                query_hops=labels, hop_provenance={"kind": "dataset_level_prior", "value": 1},
            )
            self.assertEqual(report["single_hop"]["count"], 1)
            self.assertEqual(report["two_hop"]["count"], 0)
            self.assertEqual(report["three_hop"]["count"], 0)
            self.assertEqual(report["four_hop"]["count"], 0)

    def test_estimated_mode_preserves_legacy_groups(self):
        queries = ["Name A?", "Who and which leader?",
                   "Who and which former leader came before the latter?"]
        results = [QuerySolution(question=q, docs=["doc"]) for q in queries]
        report = evaluate_stratified(
            queries, results, [["doc"]] * 3, [[]] * 3,
            global_config=None, k_list=[1],
        )
        self.assertEqual(set(report), {"overall", "single_hop", "two_hop", "multi_hop"})
        self.assertEqual([report[g]["count"] for g in ("single_hop", "two_hop", "multi_hop")],
                         [1, 1, 1])

    def test_label_validation_rejects_misalignment_and_invalid_hops(self):
        for labels in ([], [0], [5], [2.0], [True]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                stratify_by_hops(["q"], [QuerySolution("q", [])], [[]], [[]], query_hops=labels)

    def test_local_musique_reports_all_three_oracle_groups(self):
        dataset_path = ROOT.parent / "datasets" / "musique.json"
        if not dataset_path.exists():
            self.skipTest("Local MuSiQue dataset is not present")
        samples = json.loads(dataset_path.read_text(encoding="utf-8"))
        queries = [sample["question"] for sample in samples]
        labels = get_benchmark_hops(samples, "musique")
        grouped = stratify_by_hops(
            queries, [QuerySolution(q, []) for q in queries],
            [[] for _ in queries], [[] for _ in queries], query_hops=labels,
        )
        counts = {group: len(data["queries"]) for group, data in grouped.items()}
        self.assertEqual(counts, {"single_hop": 0, "two_hop": 518,
                                  "three_hop": 316, "four_hop": 166})
        self.assertEqual(Counter(labels), Counter({2: 518, 3: 316, 4: 166}))


if __name__ == "__main__":
    unittest.main()
