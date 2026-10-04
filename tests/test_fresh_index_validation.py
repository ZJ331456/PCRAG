"""CPU artifact checks for fresh shared-index publication gates."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import igraph as ig
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from utils.fresh_index_validation import validate_fresh_index, unicode_normalize
from pathcondrag.index.openie_source_evidence import (
    FLAGS, VERIFIER_VERSION, _base_audit, _evaluate, _quote_offsets,
)


def key(content, namespace):
    return namespace + "-" + hashlib.md5(content.encode()).hexdigest()


class FreshIndexValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.index = self.out / "shared_hipporag2_index"
        self.model = self.index / "qwen3-8b__root_models_Qwen3-Embedding-8B"
        self.model.mkdir(parents=True)
        self.passage = "Café\nCafé is located in Paris."
        self.rows = [{"idx": key(self.passage, "chunk"), "passage": self.passage,
                      "extracted_entities": ["Café", "Paris"],
                      "extracted_triples": [["Café", "located in", "Paris"]],
                      "openie_metadata": {"ner": {"finish_reason": "stop"},
                                          "triples": {"finish_reason": "stop", "quality_status": "success"}}}]
        self.profile = {"extractor": "pathcondrag.information_extraction.source_verified_openie.SourceVerifiedOpenIE",
                        "fresh_recovery_semantic_verifier": "source-verifier-test-v1"}
        self.provenance = {"quality_profile": self.profile}
        self.write_fixture()

    def json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_fixture(self):
        state = {"docs": self.rows, "provenance": self.provenance}
        self.json(self.model / "openie_state.json", state)
        self.json(self.index / "openie_results_ner_qwen3-8b.json", state)
        self.json(self.model / "index_manifest.json",
                  {"text_normalization": "unicode_alnum_casefold_v1", "openie": self.provenance})
        self.json(self.model / "chunk_metadata.json", {row["idx"]: {} for row in self.rows})
        needed = {"chunk": {row["passage"] for row in self.rows}, "entity": set(), "fact": set()}
        for row in self.rows:
            for triple in row["extracted_triples"]:
                normalized = tuple(unicode_normalize(field) for field in triple)
                needed["entity"].update((normalized[0], normalized[2]))
                needed["fact"].add(str(normalized))
        for namespace, texts in needed.items():
            path = self.model / f"{namespace}_embeddings/vdb_{namespace}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"hash_id": key(text, namespace), "content": text,
                           "embedding": [1.0] + [0.0] * 4095} for text in sorted(texts)],
                         columns=["hash_id", "content", "embedding"]).to_parquet(path, index=False)
        graph = ig.Graph()
        graph["hipporag_edge_schema"] = 1
        graph.add_vertices([key(text, "chunk") for text in sorted(needed["chunk"])]
                           + [key(text, "entity") for text in sorted(needed["entity"])])
        for row in self.rows:
            chunk, entities = row["idx"], set()
            for triple in row["extracted_triples"]:
                normalized = [unicode_normalize(field) for field in triple]
                left, right = key(normalized[0], "entity"), key(normalized[2], "entity")
                entities.update((left, right))
                first, second = sorted((left, right))
                graph.add_edge(first, second, weight=1.0, edge_kind="fact",
                               fact_source_counts={chunk: 1}, synonym_score=0.0, passage_source=None,
                               source_key=first, target_key=second)
            for entity in entities:
                first, second = sorted((chunk, entity))
                graph.add_edge(first, second, weight=1.0, edge_kind="passage", fact_source_counts={},
                               synonym_score=0.0, passage_source=chunk, source_key=first, target_key=second)
        graph.write_pickle(str(self.model / "graph.pickle"))

    def mark_recovered(self):
        triples = self.rows[0]["extracted_triples"]
        audit = {"complete": True, "error": None, "n_unverified": 0,
                 "n_input": len(triples), "n_accepted": len(triples), "n_rejected": 0,
                 "supported": [True] * len(triples),
                 "checks": [{"index": i, "triple": triple, "supported": True}
                            for i, triple in enumerate(triples)],
                 "contract_version": "source-verifier-test-v1"}
        self.rows[0]["openie_metadata"]["triples"].update(
            source_verified_schema="pathcondrag_fresh_source_verified_openie_v1",
            semantic_verified=True, complete=True, requires_semantic_verification=False,
            quality_recovered=True, semantic_verifier_contract="source-verifier-test-v1",
            semantic_verification_history=[audit])
        return audit

    def mark_strict(self, *, recovered=False):
        self.profile.update(name='pathcondrag_openie_quality_v3', schema_version=3,
                            semantic_scope='all_final_relations',
                            fresh_recovery_semantic_verifier=VERIFIER_VERSION)
        row = self.rows[0]
        passage, triples = row['passage'], row['extracted_triples']
        quote = passage.split('\n', 1)[-1]
        checks = []
        for triple in triples:
            verdict = {'supported': True, 'quote': quote, 'reason': 'Explicit original-source assertion.',
                       'subject_roles': [{'source_subject': triple[0], 'mention': triple[0],
                                          'relation_quote': quote, 'relation_supported': True}],
                       **{flag: True for flag in FLAGS}}
            checks.append(_evaluate(passage, triple, verdict))
        audit = _base_audit(triples, checks, [], complete=True)
        source_fields = {'source_has_supported_relations': bool(triples),
                         'source_no_supported_relations': not triples,
                         'source_evidence_quote': quote,
                         'source_evidence_quote_offsets': _quote_offsets(passage, quote),
                         'source_reason': 'Original source relation audit.', 'finish_reason': 'stop'}
        batch = {**copy.deepcopy(audit), **source_fields}
        audit.update(source_fields)
        audit['batches'] = [batch]
        row['openie_metadata']['triples'].update(
            source_verified_schema='pathcondrag_fresh_source_verified_openie_v2',
            semantic_verified=True, complete=True, requires_semantic_verification=False,
            quality_recovered=recovered, semantic_verifier_contract=VERIFIER_VERSION,
            semantic_verification_history=[audit], source_no_supported_relations=not triples)
        return audit

    def validate(self):
        return validate_fresh_index(self.out, self.index, {row["passage"] for row in self.rows})

    def test_unicode_facts_vectors_and_actual_graph_pass(self):
        report = self.validate()
        self.assertEqual(unicode_normalize(" CAFÉ—北京! "), "café 北京")
        self.assertEqual(report["vectors"]["entity"]["count"], 2)
        self.assertTrue(report["edge_validation"]["fact_provenance_exact"])
        self.assertFalse(report["shared_knn_execution"]["observed"])
        self.assertTrue((self.out / "fresh_index_validation.json").is_file())

    def test_corrupted_state_top_and_provenance_are_rejected(self):
        top = copy.deepcopy({"docs": self.rows, "provenance": self.provenance})
        top["docs"][0]["extracted_triples"][0][2] = "Berlin"
        self.json(self.index / "openie_results_ner_qwen3-8b.json", top)
        with self.assertRaisesRegex(ValueError, "artifacts differ"):
            self.validate()

    def test_incomplete_stage_and_invalid_triples_are_rejected(self):
        self.rows[0]["openie_metadata"]["ner"]["openie_skipped"] = True
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Incomplete fresh ner"):
            self.validate()
        self.rows[0]["openie_metadata"]["ner"].pop("openie_skipped")
        self.rows[0]["extracted_triples"].append(["Café", "bad", ""])
        state = {"docs": self.rows, "provenance": self.provenance}
        self.json(self.model / "openie_state.json", state)
        self.json(self.index / "openie_results_ner_qwen3-8b.json", state)
        with self.assertRaisesRegex(ValueError, "Structurally invalid"):
            self.validate()

    def test_recovered_facts_must_match_final_source_verdict(self):
        audit = self.mark_recovered()
        self.write_fixture()
        report = self.validate()
        self.assertEqual(report["recovered_chunks"], 1)
        self.assertEqual(report["accepted_relations_final"], 1)
        audit["checks"][0]["triple"] = ["Café", "located in", "Berlin"]
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "differ from accepted"):
            self.validate()

    def test_empty_facts_require_explicit_legitimate_empty_audit(self):
        self.rows[0]["extracted_triples"] = []
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Empty extraction lacks"):
            self.validate()
        self.mark_recovered()
        self.rows[0]["openie_metadata"]["triples"]["source_no_supported_relations"] = True
        self.write_fixture()
        report = self.validate()
        self.assertEqual(report["legitimate_empty_chunks"], 1)
        self.assertEqual(report["graph_nodes"], 1)

    def test_stale_vector_contents_and_bad_dimensions_are_rejected(self):
        path = self.model / "entity_embeddings/vdb_entity.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "content"] = "stale"
        frame.loc[0, "hash_id"] = key("stale", "entity")
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "contents differ"):
            self.validate()
        self.write_fixture()
        frame = pd.read_parquet(path)
        frame.at[0, "embedding"] = [1.0]
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.validate()

    def test_graph_stale_fact_source_is_rejected(self):
        graph = ig.Graph.Read_Pickle(str(self.model / "graph.pickle"))
        graph.es[0]["fact_source_counts"] = {"chunk-stale": 1}
        graph.write_pickle(str(self.model / "graph.pickle"))
        with self.assertRaisesRegex(ValueError, "source counts do not match"):
            self.validate()

    def test_knn_profile_reports_only_actual_completed_trace(self):
        self.json(self.out / "index_build_result.json", {"runtime_config": {"synonymy_edge_query_batch_size": 1000}})
        self.assertFalse(self.validate()["shared_knn_execution"]["observed"])
        trace = {"device": "cuda", "dtype": "float32", "allow_tf32": False,
                 "query_batch_size": 1000, "key_batch_size": 16384,
                 "native_cosine_topk_and_threshold_unchanged": True, "complete": True}
        self.json(self.model / "shared_build_execution.json", trace)
        self.assertTrue(self.validate()["shared_knn_execution"]["observed"])
        trace["complete"] = False
        self.json(self.model / "shared_build_execution.json", trace)
        with self.assertRaisesRegex(ValueError, "KNN execution profile"):
            self.validate()

    def test_strict_normal_extraction_cannot_bypass_source_audit(self):
        self.profile.update(semantic_scope='all_final_relations',
                            fresh_recovery_semantic_verifier=VERIFIER_VERSION)
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'Normal initial extraction bypassed'):
            self.validate()

    def test_strict_ordinary_evidence_audit_passes_without_marking_recovery(self):
        self.mark_strict()
        self.write_fixture()
        report = self.validate()
        self.assertEqual(report['audited_chunks'], 1)
        self.assertEqual(report['recovered_chunks'], 0)
        self.assertEqual(report['accepted_relations_final'], 1)

    def test_strict_accepted_fact_requires_exact_quote_and_offsets(self):
        audit = self.mark_strict()
        check = audit['checks'][0]
        quote = check.pop('quote')
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'no exact source evidence'):
            self.validate()
        check['quote'] = quote
        check['quote_offsets'][0]['start'] += 1
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'offsets mismatch'):
            self.validate()

    def test_strict_positive_role_evidence_and_subject_coverage_are_recomputed(self):
        audit = self.mark_strict()
        audit['checks'][0]['subject_roles'][0]['mention'] = 'Paris'
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'source roles or coverage differ'):
            self.validate()

    def test_guided_quote_ids_preserve_resolved_exact_source_validation(self):
        audit = self.mark_strict()
        for item in [audit, *audit['batches']]:
            item['source_evidence_quote_id'] = 'span-1'
            item['checks'][0]['quote_id'] = 'span-1'
            item['checks'][0]['subject_roles'][0]['relation_quote_id'] = 'span-1'
        self.write_fixture()
        self.assertEqual(self.validate()['accepted_relations_final'], 1)
        audit['checks'][0]['quote_offsets'][0]['start'] += 1
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'offsets mismatch'):
            self.validate()

    def test_strict_empty_requires_independent_whole_source_verdict(self):
        passage = 'A heading without a factual assertion'
        self.rows[0].update(idx=key(passage, 'chunk'), passage=passage,
                            extracted_entities=[], extracted_triples=[])
        audit = self.mark_strict()
        self.write_fixture()
        self.assertEqual(self.validate()['legitimate_empty_chunks'], 1)
        audit.pop('batches')
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'no independent evidence batches'):
            self.validate()

    def test_strict_atomic_empty_focus_requires_matching_independent_audit(self):
        self.mark_strict(recovered=True)
        metadata = self.rows[0]['openie_metadata']['triples']
        metadata['unverified_empty_focuses'] = [[0, 4]]
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'focuses lack independent audits'):
            self.validate()
        focus_audit = _base_audit([], [], [], complete=True)
        focus_audit.update(
            source_scope='whole_source_with_original_focus',
            source_relation_verdict_scope='original_focus_only',
            source_focus={'start': 0, 'end': 4, 'text': self.passage[:4]},
            source_has_supported_relations=False, source_no_supported_relations=True,
            source_evidence_quote='Café', source_evidence_quote_offsets=_quote_offsets(self.passage, 'Café'),
            source_reason='The original focus contains only the heading name.', finish_reason='stop')
        metadata['atomic_empty_focus_audits'] = [focus_audit]
        self.write_fixture()
        self.assertEqual(self.validate()['audited_chunks'], 1)
        focus_audit['source_focus']['end'] = 5
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'no complete independent no-relation verdict'):
            self.validate()

    def test_nonempty_focus_cannot_claim_deterministic_whitespace(self):
        self.mark_strict()
        metadata = self.rows[0]['openie_metadata']['triples']
        metadata['unverified_empty_focuses'] = [[0, 4]]
        focus_audit = _base_audit([], [], [], complete=True)
        focus_audit.update(
            source_scope='whole_source_with_original_focus',
            source_relation_verdict_scope='original_focus_only',
            source_focus={'start': 0, 'end': 4, 'text': self.passage[:4]},
            source_has_supported_relations=False, source_no_supported_relations=True,
            deterministic_empty_focus=True, source_reason='Pretended whitespace.', finish_reason='stop')
        metadata['atomic_empty_focus_audits'] = [focus_audit]
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'falsely marked as deterministic whitespace'):
            self.validate()


if __name__ == "__main__":
    unittest.main()
