"""Retrieval protocol checks against native Hotpot/2Wiki/MuSiQue schemas."""

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_utils import get_benchmark_hops, get_gold_docs
from utils import improvement_experiments as protocol


class MultiDatasetProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.corpus = [{"title": f"Doc {i}", "text": f"Sentence {i}. Next {i}."}
                       for i in range(200)]

    @staticmethod
    def write(path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def samples(self, name):
        rows = []
        for index, count in enumerate((2, 4)):
            passages = self.corpus[index * 4:index * 4 + count]
            if name == "musique":
                sample = {"id": f"{count}hop__{index}", "question": f"Question {index}?",
                          "question_decomposition": [{"question": f"Step {step}?"}
                                                     for step in range(count)],
                          "paragraphs": [{**item, "is_supporting": True} for item in passages]}
            else:
                if name == "hotpotqa":
                    passages = passages[:2]
                sample = {"_id": f"sample-{index}", "question": f"Question {index}?",
                          "supporting_facts": [[item["title"], 0] for item in passages],
                          "context": [[item["title"], [item["text"]]] for item in passages]}
                if name == "2wikimultihopqa":
                    sample.update(type="bridge_comparison" if count == 4 else "compositional",
                                  evidences=[["s", "r", "o"] for _ in passages])
            rows.append(sample)
        return rows

    def prepare(self, name):
        data = self.samples(name)
        data_path = self.root / "datasets" / f"{name}.json"
        corpus_path = self.root / "datasets" / f"{name}_corpus.json"
        self.write(data_path, data)
        self.write(corpus_path, self.corpus)
        out = self.root / "results"
        args = argparse.Namespace(out_root=str(out / "metadata" / name),
                                  source_index=str(out / "shared_indexes" / name),
                                  shared_output_root=str(out), dataset=name,
                                  data_path=str(data_path), corpus_path=str(corpus_path),
                                  build_shared_index=True, sample_size=2, sample_seed=42,
                                  sample_indices_file=None,
                                  cases="hipporag2 pathcondrag_original exp4_dependency_binding")
        protocol.prepare(args)
        return args, data, protocol.manifest_at(args.out_root)

    def result(self, manifest, data, name):
        candidates = [protocol.passage_text(item) for item in self.corpus]
        rows = []
        totals = Counter()
        for index in manifest["selected_indices"]:
            gold = protocol.gold_docs(data[index], manifest["dataset"])
            metrics = {f"Recall@{k}": protocol.recall(gold, candidates[:k])
                       for k in (1, 2, 5, 10, 20, 200)}
            for key, value in metrics.items():
                totals[key] += value / len(data)
            rows.append({"sample_id": protocol.sample_identity(data[index], index),
                         "query_index": index, "question": data[index]["question"],
                         "benchmark_hops": (None if name == "hipporag2" and manifest["dataset"] != "musique"
                                            else manifest["benchmark_hops"][len(rows)]),
                         "docs": candidates[:10], "doc_scores": [1.0] * 10,
                         "candidate_docs": candidates, "candidate_doc_scores": [1.0] * len(candidates),
                         "gold_docs": sorted(gold), "retrieval_metrics": metrics,
                         "gold_document_ranks": [{"doc": doc, "rank": candidates.index(doc) + 1}
                                                 for doc in sorted(gold)],
                         "all_gold_in_top5": gold <= set(candidates[:5]),
                         "all_gold_in_top10": gold <= set(candidates[:10]),
                         "retrieval_trace": {}})
        return {"dataset": manifest["dataset"], "eval_mode": "retrieve", "qa_metrics": {},
                "n_docs": len(candidates), "indexed_docs": len(candidates),
                "selected_indices": manifest["selected_indices"], "sample_size_effective": len(rows),
                "result_top_k": 10, "candidate_output_top_k": 200,
                "runtime_config": {**manifest["runtime"], "improvement_stage": protocol.CASES[name]},
                "hop_source": "benchmark", "hop_distribution": manifest["hop_distribution"],
                "results": rows, "retrieval_metrics": dict(totals)}

    def test_native_schemas_share_gold_and_hop_readers(self):
        for name in protocol.DATASETS:
            with self.subTest(dataset=name):
                args, data, manifest = self.prepare(name)
                loaded, docs, hops = protocol.validated_dataset(args.data_path, args.corpus_path)
                self.assertEqual(loaded, data)
                self.assertEqual(hops, get_benchmark_hops(data, name))
                self.assertEqual(manifest["benchmark_hops"], hops)
                self.assertEqual(manifest["selected_sample_ids"],
                                 [protocol.sample_identity(row, i) for i, row in enumerate(data)])
                self.assertEqual([protocol.gold_docs(row) for row in data],
                                 [set(docs) for docs in get_gold_docs(data, name)])
                self.assertEqual(len(docs), 200)
                self.assertEqual(manifest["benchmark_hop_policy"]["scope"],
                                 "per_question_decomposition" if name == "musique" else "dataset_prior")
                protocol.prepare(args)

    def test_sentence_join_rules_are_not_interchangeable(self):
        sample = {"supporting_facts": [["Title", 0]], "context": [["Title", ["A.", " B."]]]}
        self.assertEqual(protocol.gold_docs(sample, "hotpotqa"), {"Title\nA. B."})
        self.assertEqual(protocol.gold_docs(sample, "2wikimultihopqa"), {"Title\nA.  B."})

    def test_all_three_case_exports_validate_with_underscore_id(self):
        for dataset in ("hotpotqa", "2wikimultihopqa"):
            args, data, manifest = self.prepare(dataset)
            docs = {protocol.passage_text(item) for item in self.corpus}
            for name in manifest["cases"]:
                with self.subTest(dataset=dataset, case=name):
                    result = self.result(manifest, data, name)
                    measurements, metrics = protocol.validate_result(result, manifest, name, data, docs)
                    self.assertEqual(len(measurements), 2)
                    self.assertEqual(metrics, result["retrieval_metrics"])
                    result["results"][0]["sample_id"] = "wrong-sample"
                    with self.assertRaisesRegex(ValueError, "sample identity"):
                        protocol.validate_result(result, manifest, name, data, docs)

    def test_tiny_corpus_exports_available_candidates_without_padding(self):
        self.corpus = self.corpus[:20]
        _, data, manifest = self.prepare("hotpotqa")
        docs = {protocol.passage_text(item) for item in self.corpus}
        for name in manifest["cases"]:
            with self.subTest(case=name):
                result = self.result(manifest, data, name)
                measurements, metrics = protocol.validate_result(result, manifest, name, data, docs)
                self.assertEqual(len(measurements), 2)
                self.assertEqual(metrics["Recall@200"], 1.0)
                self.assertEqual(len(result["results"][0]["docs"]), 10)
                self.assertEqual(len(result["results"][0]["candidate_docs"]), 20)
                result["results"][0]["candidate_doc_scores"].pop()
                with self.assertRaisesRegex(ValueError, "invalid candidate_doc_scores"):
                    protocol.validate_result(result, manifest, name, data, docs)

    def test_missing_support_and_duplicate_identity_are_rejected(self):
        args, data, _ = self.prepare("hotpotqa")
        data[0]["context"] = data[0]["context"][:1]
        self.write(args.data_path, data)
        with self.assertRaisesRegex(ValueError, "supporting titles absent"):
            protocol.validated_dataset(args.data_path, args.corpus_path)
        data = self.samples("hotpotqa")
        data[1]["_id"] = data[0]["_id"]
        self.write(args.data_path, data)
        with self.assertRaisesRegex(ValueError, "duplicate sample ID"):
            protocol.validated_dataset(args.data_path, args.corpus_path)

    def test_qa_or_partial_corpus_cannot_get_retrieval_validation(self):
        _, data, manifest = self.prepare("hotpotqa")
        docs = {protocol.passage_text(item) for item in self.corpus}
        for key, value in (("eval_mode", "rag_qa"), ("qa_metrics", {"f1": 0.5}),
                           ("n_docs", 199), ("dataset", "musique")):
            with self.subTest(key=key):
                result = self.result(manifest, data, "hipporag2")
                result[key] = value
                with self.assertRaises(ValueError):
                    protocol.validate_result(result, manifest, "hipporag2", data, docs)

    def test_shared_dataset_index_path_cannot_escape_declared_root(self):
        args, _, _ = self.prepare("hotpotqa")
        args.source_index = str(self.root / "unrelated-index")
        with self.assertRaisesRegex(ValueError, "declared shared index"):
            protocol.prepare(args)
        args.source_index = str(Path(args.shared_output_root) / "shared_indexes" / "hotpotqa")
        args.out_root = str(self.root / "unrelated-metadata")
        with self.assertRaisesRegex(ValueError, "metadata must be"):
            protocol.prepare(args)

    def fake_build(self, args, manifest):
        source = Path(args.source_index)
        model = source / manifest["model_dir"]
        self.write(model / "index_manifest.json", {
            "embedding": {"model_name": protocol.EMBEDDING_MODEL, "provider": "transformers"},
            "openie": {"identity": {"model_name": "qwen3-8b"}}})
        self.write(model / "chunk_metadata.json", {
            "chunk-" + hashlib.md5(protocol.passage_text(item).encode()).hexdigest(): {}
            for item in self.corpus})
        for asset in protocol.ASSETS:
            path = model / asset
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(asset.encode())
        cache = source / "llm_cache"
        cache.mkdir()
        (cache / "test.sqlite").write_bytes(b"fixed-cache")
        rows = [{"idx": "chunk-" + hashlib.md5(protocol.passage_text(item).encode()).hexdigest(),
                 "passage": protocol.passage_text(item), "extracted_entities": [],
                 "extracted_triples": [], "openie_metadata": {
                     "ner": {"finish_reason": "stop"}, "triples": {"finish_reason": "stop"}}}
                for item in self.corpus]
        rows[0]["openie_metadata"].update(
            triples={"finish_reason": "length", "quality_status": "failed"},
            publication={"openie_strict": False, "skipped_from_graph": True})
        openie_path = source / "openie_results_ner_qwen3-8b.json"
        self.write(openie_path, {"docs": rows})
        result = {"eval_mode": "index_only", "index_build_complete": True,
                  "indexed_docs": 200, "openie_document_count": 200, "openie_failure_count": 1,
                  "runtime_config": {**manifest["runtime"], "openie_strict": False,
                                     "force_index_from_scratch": True,
                                     "force_openie_from_scratch": True, "openie_mode": "online"},
                  "llm_request_stats": {"failures": 1, "max_in_flight": 8, "http_attempts": 400}}
        self.write(Path(args.out_root) / "index_build_result.json", result)
        log = self.root / "vllm.log"
        log.write_text('POST /v1/chat/completions HTTP/1.1" 200\n', encoding="utf-8")
        freeze = argparse.Namespace(out_root=args.out_root, openie_strict=False,
                                    elapsed=3, vllm_log=str(log), log_start=0, log_end=log.stat().st_size)
        return freeze, rows, openie_path

    def test_tolerant_build_freezes_explicit_exclusion_and_reusable_shared_index(self):
        args, _, manifest = self.prepare("2wikimultihopqa")
        freeze, _, _ = self.fake_build(args, manifest)
        protocol.freeze_index(freeze)
        self.assertEqual(protocol.index_ready(args), 0)
        report = protocol.read_json(Path(args.out_root) / "index_build_report.json")
        self.assertFalse(report["openie_strict"])
        self.assertEqual(report["openie_failure_count"], 1)
        protocol.prepare(args)
        protocol.initialize_case(argparse.Namespace(out_root=args.out_root, name="hipporag2"))

    def test_tolerant_build_cannot_publish_failed_graph_facts(self):
        args, _, manifest = self.prepare("hotpotqa")
        freeze, rows, path = self.fake_build(args, manifest)
        rows[0]["extracted_triples"] = [["unverified", "relation", "fact"]]
        self.write(path, {"docs": rows})
        with self.assertRaisesRegex(ValueError, "retained graph facts"):
            protocol.freeze_index(freeze)
        self.assertEqual(protocol.manifest_at(args.out_root)["index_build_status"], "pending")
        self.assertFalse((Path(args.out_root) / "initial_llm_cache").exists())


if __name__ == "__main__":
    unittest.main()
