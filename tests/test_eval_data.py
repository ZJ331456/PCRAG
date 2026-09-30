"""Fast checks for benchmark-hop labels and PopQA answer aliases."""

import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_utils import get_benchmark_hops, get_gold_answers  # noqa: E402


class EvalDataTests(unittest.TestCase):
    def test_musique_hops_preserve_order_and_reject_missing_labels(self):
        samples = [
            {"question_decomposition": [{"question": "one"}] * 4},
            {"question_decomposition": [{"question": "two"}] * 2},
            {"question_decomposition": [{"question": "three"}] * 3},
        ]
        self.assertEqual(get_benchmark_hops(samples, "musique"), [4, 2, 3])
        samples[1]["question_decomposition"] = []
        with self.assertRaisesRegex(ValueError, "sample index 1"):
            get_benchmark_hops(samples, "musique")

    def test_dataset_priors_do_not_use_support_count(self):
        samples = [{"supporting_facts": [1, 2, 3, 4]}, {}]
        self.assertEqual(get_benchmark_hops(samples, "hotpotqa"), [2, 2])
        self.assertEqual(get_benchmark_hops(samples, "2wikimultihopqa"), [2, 2])
        self.assertEqual(get_benchmark_hops(samples, "nq"), [1, 1])
        self.assertEqual(get_benchmark_hops(samples, "popqa"), [1, 1])

    def test_popqa_json_encoded_aliases_are_words_not_characters(self):
        sample = {
            "obj": "politician",
            "possible_answers": '["politician", "political leader"]',
            "o_aliases": '["political figure"]',
        }
        answers = set(get_gold_answers([sample])[0])
        self.assertEqual(answers, {"politician", "political leader", "political figure"})

    def test_local_musique_labels_are_complete_when_available(self):
        data_path = ROOT.parent / "datasets" / "musique.json"
        if not data_path.exists():
            self.skipTest("Local MuSiQue dataset is not present")
        samples = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(Counter(get_benchmark_hops(samples, "musique")),
                         Counter({2: 518, 3: 316, 4: 166}))


if __name__ == "__main__":
    unittest.main()
