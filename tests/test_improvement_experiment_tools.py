"""Integrity gates for shared-index, seven-case retrieval experiments."""

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import improvement_experiments as helpers


class ImprovementExperimentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "reports with spaces"
        self.source = self.root / "hippo"
        self.model = self.source / helpers.MODEL_DIR
        self.model.mkdir(parents=True)
        self.corpus = [{"title": f"Document {i}", "text": f"Text {i}"} for i in range(200)]
        self.data = []
        for i, hop in enumerate((2, 3, 4)):
            self.data.append({
                "id": f"{hop}hop__{i}", "question": f"Question {i}?",
                "question_decomposition": [{"question": f"Hop {j}?"} for j in range(hop)],
                "paragraphs": [{**p, "is_supporting": True} for p in self.corpus[i * 4:i * 4 + hop]],
            })
        self.data_path, self.corpus_path = self.root / "data.json", self.root / "corpus.json"
        self.write(self.data_path, self.data)
        self.write(self.corpus_path, self.corpus)
        self.write(self.model / "index_manifest.json", {
            "embedding": {"model_name": helpers.EMBEDDING_MODEL, "provider": "transformers"},
            "openie": {"identity": {"model_name": "qwen3-8b"}},
        })
        self.write(self.model / "chunk_metadata.json", {
            "chunk-" + hashlib.md5(helpers.passage_text(p).encode()).hexdigest(): {} for p in self.corpus
        })
        for asset in helpers.ASSETS:
            path = self.model / asset
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(asset.encode())
        cache = self.source / "llm_cache"
        cache.mkdir()
        (cache / "qwen.sqlite").write_bytes(b"initial-cache")
        (cache / "qwen.sqlite.00.lock").write_bytes(b"")
        self.log = self.root / "vllm.log"
        self.log.write_text('POST /v1/chat/completions HTTP/1.1" 200\n')
        self.prepare_args = argparse.Namespace(
            out_root=str(self.out), source_index=str(self.source), data_path=str(self.data_path),
            corpus_path=str(self.corpus_path), sample_size=0, sample_seed=42,
            sample_indices_file=None, cases="pathcondrag_original exp1_correctness",
        )
        helpers.prepare(self.prepare_args)
        self.manifest = helpers.manifest_at(self.out)

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def result(self, stage=0):
        docs = [helpers.passage_text(p) for p in self.corpus]
        rows = []
        totals = {f"Recall@{k}": 0.0 for k in (1, 2, 5, 10, 20, 200)}
        for i, sample in enumerate(self.data):
            gold = helpers.gold_docs(sample)
            metrics = {f"Recall@{k}": helpers.recall(gold, docs[:k]) for k in (1, 2, 5, 10, 20, 200)}
            for key, value in metrics.items():
                totals[key] += value / len(self.data)
            rows.append({
                "question": sample["question"], "query_index": i, "sample_id": sample["id"],
                "benchmark_hops": len(sample["question_decomposition"]), "docs": docs[:10],
                "doc_scores": [1.0] * 10, "candidate_docs": docs, "candidate_doc_scores": [1.0] * 200,
                "gold_docs": sorted(gold), "retrieval_metrics": metrics,
                "gold_document_ranks": [{"doc": doc, "rank": docs.index(doc) + 1} for doc in sorted(gold)],
                "all_gold_in_top5": gold <= set(docs[:5]), "all_gold_in_top10": gold <= set(docs[:10]),
                "retrieval_trace": {},
            })
        return {
            "selected_indices": [0, 1, 2], "sample_size_effective": 3,
            "result_top_k": 10, "candidate_output_top_k": 200,
            "runtime_config": {
                **self.manifest["runtime"], "improvement_stage": stage,
            },
            "hop_source": "benchmark", "hop_distribution": {"2": 1, "3": 1, "4": 1},
            "results": rows, "retrieval_metrics": totals,
            "llm_request_stats": {"failures": 0, "http_attempts": 1, "max_in_flight": 8},
            "retrieval_seconds": 2.0,
        }

    def init_case(self, name):
        args = argparse.Namespace(out_root=str(self.out), name=name)
        helpers.initialize_case(args)
        return self.out / "cases" / name

    def report_case(self, name):
        helpers.report(argparse.Namespace(
            out_root=str(self.out), name=name, elapsed=3, vllm_log=str(self.log),
            log_start=0, log_end=self.log.stat().st_size,
        ))

    def test_prepare_validates_all_hops_even_for_two_question_smoke(self):
        self.data[2]["question_decomposition"] = []
        self.write(self.data_path, self.data)
        self.prepare_args.sample_size = 2
        with self.assertRaisesRegex(ValueError, "lacks valid"):
            helpers.prepare(self.prepare_args)

    def test_prepare_rejects_mismatched_corpus_and_changed_resume_manifest(self):
        metadata = helpers.read_json(self.model / "chunk_metadata.json")
        metadata.pop(next(iter(metadata)))
        self.write(self.model / "chunk_metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "exactly the declared corpus"):
            helpers.prepare(self.prepare_args)
        self.write(self.model / "chunk_metadata.json", {
            "chunk-" + hashlib.md5(helpers.passage_text(p).encode()).hexdigest(): {} for p in self.corpus
        })
        self.prepare_args.sample_seed = 43
        with self.assertRaisesRegex(ValueError, "existing manifest differs"):
            helpers.prepare(self.prepare_args)

    def test_invalid_sample_indices_fail_without_silent_resampling(self):
        path = self.root / "indices.json"
        for indices in ([0, 0], [False, 1], [0, 999]):
            self.write(path, indices)
            with self.assertRaises(ValueError):
                helpers.load_indices(path, 3)

    def test_cases_have_independent_caches_from_one_snapshot(self):
        first = self.init_case("pathcondrag_original")
        (first / "index" / "llm_cache" / "qwen.sqlite").write_bytes(b"case-one-cache")
        second = self.init_case("exp1_correctness")
        self.assertEqual((second / "index" / "llm_cache" / "qwen.sqlite").read_bytes(), b"initial-cache")
        self.assertEqual((self.source / "llm_cache" / "qwen.sqlite").read_bytes(), b"initial-cache")
        self.assertFalse((second / "index" / "llm_cache" / "qwen.sqlite.00.lock").exists())

    def test_validated_marker_and_summary_include_true_hop_strata(self):
        for name, stage in (("pathcondrag_original", 0), ("exp1_correctness", 1)):
            case = self.init_case(name)
            self.write(case / "result.json", self.result(stage))
            self.report_case(name)
            self.assertEqual(helpers.ready(argparse.Namespace(out_root=str(self.out), name=name)), 0)
        helpers.summary(argparse.Namespace(out_root=str(self.out)))
        comparison = helpers.read_json(self.out / "comparison.json")
        report = comparison["cases"]["pathcondrag_original"]
        self.assertEqual(set(report["stratified"]), {"2", "3", "4"})
        self.assertAlmostEqual(report["all_gold_top5"], 1 / 3)
        self.assertTrue((self.out / "completed.ok").is_file())

    def test_wrong_ranking_runtime_or_metrics_cannot_be_validated(self):
        for change in ("indices", "budget", "batch", "stage", "gold", "metric", "prefix", "nan"):
            result = copy.deepcopy(self.result())
            if change == "indices": result["selected_indices"] = [2, 1, 0]
            elif change == "budget": result["results"][0]["docs"] = result["results"][0]["docs"][:5]
            elif change == "batch": result["runtime_config"]["embedding_batch_size"] = 8
            elif change == "stage": result["runtime_config"]["improvement_stage"] = 2
            elif change == "gold": result["results"][0]["gold_docs"] = []
            elif change == "metric": result["retrieval_metrics"]["Recall@5"] = 0.999
            elif change == "prefix": result["results"][0]["candidate_docs"] = list(reversed(result["results"][0]["candidate_docs"]))
            elif change == "nan": result["results"][0]["doc_scores"][0] = float("nan")
            with self.subTest(change=change), self.assertRaises(ValueError):
                helpers.validate_result(result, self.manifest, "pathcondrag_original", self.data,
                                        set(helpers.passage_text(p) for p in self.corpus))

    def test_non200_and_llm_failures_do_not_get_ready_markers(self):
        case = self.init_case("pathcondrag_original")
        result = self.result()
        self.write(case / "result.json", result)
        self.log.write_text('POST /v1/chat/completions HTTP/1.1" 503\n')
        with self.assertRaisesRegex(ValueError, "non-200"):
            self.report_case("pathcondrag_original")
        self.assertFalse((case / "validated.ok").exists())
        self.log.write_text('POST /v1/chat/completions HTTP/1.1" 200\n')
        result["llm_request_stats"]["failures"] = 1
        self.write(case / "result.json", result)
        with self.assertRaisesRegex(ValueError, "request failures"):
            self.report_case("pathcondrag_original")
        self.assertFalse((case / "validated.ok").exists())

    def test_graph_mutation_and_post_validation_edits_are_detected(self):
        case = self.init_case("pathcondrag_original")
        self.write(case / "result.json", self.result())
        graph = case / "index" / helpers.MODEL_DIR / "graph.pickle"
        graph.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "graph/embeddings changed"):
            self.report_case("pathcondrag_original")
        graph.write_bytes((self.model / "graph.pickle").read_bytes())
        self.report_case("pathcondrag_original")
        result = self.result()
        result["retrieval_seconds"] = 999
        self.write(case / "result.json", result)
        with self.assertRaisesRegex(ValueError, "validated files changed"):
            helpers.ready(argparse.Namespace(out_root=str(self.out), name="pathcondrag_original"))

    def test_incomplete_existing_case_and_missing_summary_case_stop(self):
        self.init_case("pathcondrag_original")
        with self.assertRaisesRegex(ValueError, "incomplete case exists"):
            self.init_case("pathcondrag_original")
        with self.assertRaisesRegex(ValueError, "case is incomplete"):
            helpers.summary(argparse.Namespace(out_root=str(self.out)))


if __name__ == "__main__":
    unittest.main()
