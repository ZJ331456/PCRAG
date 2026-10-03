"""CPU checks for source-bound repair checkpoints and index compatibility."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
HIPPO_ROOT = Path('/root/baseline/HippoRAG')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HIPPO_ROOT / 'src'))

spec = importlib.util.spec_from_file_location(
    'pathcondrag_openie_repair_driver', ROOT / 'scripts/utils/openie_repair.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

from hipporag.HippoRAG import HippoRAG  # noqa: E402
from hipporag.utils.misc_utils import text_processing as baseline_normalize  # noqa: E402


def source_row(triples=None):
    passage = 'Station\nStation opened in 1910 and is below grade.'
    return {
        'idx': repair.md5_id(passage, 'chunk-'), 'passage': passage,
        'extracted_entities': ['Station', '1910'],
        'extracted_triples': triples if triples is not None else [
            ['Station', 'opened in', '1910'], ['Station', 'is below grade', '']],
        'openie_metadata': {'ner': {'finish_reason': 'stop'},
                            'triples': {'finish_reason': 'stop'}},
    }


class RepairDriverTests(unittest.TestCase):
    def test_unicode_normalization_matches_source_baseline(self):
        for value in ('静岡県', 'κλῆς', 'Straße', 'İSTANBUL',
                      '  A.C.\tMilan\n1999  ', 'New---York', '—\t'):
            with self.subTest(value=value):
                self.assertEqual(repair.normalize(value), baseline_normalize(value))
        self.assertEqual(repair.normalize('静岡県'), '静岡県')
        self.assertEqual(repair.normalize('Straße'), 'strasse')

    def test_split_rejects_empty_fields_without_losing_unicode_relations(self):
        row = source_row([
            ['静岡県', 'located in', 'Japan'], ['Station', 'opened in', '1910'],
            ['Station', 'is below grade', ''], ['Station', 'is', '---'],
            {'subject': 'Station', 'relation': 'is', 'object': 'below grade'},
            ['Station', 'opened', '1910', 'October'],
        ])
        before = copy.deepcopy(row)
        valid, invalid, defective = repair.split_row(row)
        self.assertEqual(valid, row['extracted_triples'][:2])
        self.assertEqual(len(invalid), 4)
        self.assertTrue(defective)
        self.assertEqual(row, before)

    def test_failed_metadata_remains_a_repair_target_with_valid_records(self):
        row = source_row([['Station', 'opened in', '1910']])
        self.assertFalse(repair.split_row(row)[2])
        row['openie_metadata']['triples']['openie_skipped'] = True
        self.assertTrue(repair.split_row(row)[2])

    def test_required_sets_use_source_normalized_tuple_encoding(self):
        rows = [source_row([
            ['静岡県', 'Located in', 'Japan'],
            ['静岡県', 'located  in', 'JAPAN'],
            ['A.C. Milan', 'won', 'Serie A'],
        ])]
        entities, facts = repair.required_contents(rows)
        self.assertEqual(entities, {'静岡県', 'japan', 'a c milan', 'serie a'})
        self.assertEqual(facts, {"('静岡県', 'located in', 'japan')",
                                "('a c milan', 'won', 'serie a')"})
        rows[0]['extracted_triples'].append(['Station', 'is', ''])
        with self.assertRaisesRegex(ValueError, 'Invalid triples'):
            repair.required_contents(rows)

    def test_checkpoint_is_bound_to_the_original_row_including_metadata(self):
        original = source_row()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            checkpoint = out / 'repairs' / (original['idx'] + '.json')
            repair.write_json(checkpoint, {
                'repair_version': repair.REPAIR_VERSION,
                'original_sha256': repair.row_hash(original), 'complete': True,
                'row': source_row([['Station', 'opened in', '1910']]),
                'summary': {'idx': original['idx'], 'complete': True},
            })
            rows, pending, summaries = repair.recovered_rows({'docs': [original]}, out)
            self.assertEqual(len(rows), 1)
            self.assertEqual(pending, [])
            self.assertTrue(summaries[0]['complete'])
            changed = copy.deepcopy(original)
            changed['openie_metadata']['triples']['new_attempt'] = 1
            with self.assertRaisesRegex(ValueError, 'Stale repair checkpoint'):
                repair.recovered_rows({'docs': [changed]}, out)

    def test_publication_rejects_partial_cache_even_when_all_triples_are_valid(self):
        for stage in ('ner', 'triples'):
            for failure in ({'quality_status': 'partial'}, {'error': 'parse failure'},
                            {'openie_skipped': True}):
                row = source_row([['Station', 'opened in', '1910']])
                row['openie_metadata'][stage].update(failure)
                with self.subTest(stage=stage, failure=failure), self.assertRaisesRegex(
                        ValueError, 'Incomplete OpenIE'):
                    repair.required_contents([row])
        row = source_row([['Station', 'opened in', '1910']])
        row['openie_quality_repair'] = {'complete': False}
        with self.assertRaisesRegex(ValueError, 'Incomplete repair'):
            repair.required_contents([row])

    def test_source_output_overlap_is_rejected_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'PathCondRAG'
            source = project / 'outputs' / 'source'
            source.mkdir(parents=True)
            args = SimpleNamespace(source_index=str(source), hippo_root=str(HIPPO_ROOT))
            layout = (source, 'model', {}, {'docs': []})
            with patch.object(repair, 'ROOT', project), patch.object(
                    repair, 'source_layout', return_value=layout):
                for out in (source, source / 'nested', source.parent):
                    args.out_root = str(out)
                    with self.subTest(out=out), self.assertRaisesRegex(
                            ValueError, 'separate from the frozen source'):
                        repair.initialize(args)
                args.out_root = str(Path(directory) / 'elsewhere')
                with self.assertRaisesRegex(ValueError, 'under PathCondRAG/outputs'):
                    repair.initialize(args)
            self.assertFalse((source / 'source_snapshot.json').exists())
            self.assertFalse((source / 'nested').exists())

    def test_repair_keeps_healthy_relations_and_does_not_mutate_source(self):
        original = source_row()
        before = copy.deepcopy(original)
        extractor = SimpleNamespace(triple_extraction=Mock(return_value=SimpleNamespace(
            triples=[['Station', 'is', 'below grade']],
            response='{"triples":[["Station","is","below grade"]]}',
            metadata={'quality_status': 'success', 'finish_reason': 'stop',
                      'support_quotes': ['Station opened in 1910 and is below grade.'],
                      'support_quote_matches': [True], 'openie_attempt_count': 1})))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            summary = repair.repair_one(extractor, original, out)
            saved = repair.read_json(out / 'repairs' / (original['idx'] + '.json'))
        self.assertTrue(summary['complete'])
        self.assertEqual(saved['row']['extracted_triples'], [
            ['Station', 'opened in', '1910'], ['Station', 'is', 'below grade']])
        self.assertEqual(original, before)
        context = extractor.triple_extraction.call_args.kwargs['repair_context']
        self.assertIn('is below grade', context)
        self.assertIn('Preserve negatives, dates', context)

    def test_an_empty_recovery_does_not_complete_a_failed_relation_rich_chunk(self):
        original = source_row([])
        original['openie_metadata']['triples']['openie_skipped'] = True
        extractor = SimpleNamespace(triple_extraction=Mock(return_value=SimpleNamespace(
            triples=[], response='{"triples":[]}', metadata={
                'quality_status': 'empty_valid', 'finish_reason': 'stop',
                'support_quotes': [], 'support_quote_matches': []})))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            summary = repair.repair_one(extractor, original, out)
            self.assertFalse(summary['complete'])
            _, pending, _ = repair.recovered_rows({'docs': [original]}, out)
            self.assertEqual(len(pending), 1)

    def test_invalid_recovery_remains_incomplete_and_preserves_healthy_records(self):
        original = source_row()
        extractor = SimpleNamespace(triple_extraction=Mock(return_value=SimpleNamespace(
            triples=[['Station', 'is', '']], response='bad',
            metadata={'quality_status': 'partial', 'finish_reason': 'stop'})))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            summary = repair.repair_one(extractor, original, out)
            checkpoint = repair.read_json(out / 'repairs' / (original['idx'] + '.json'))
        self.assertFalse(summary['complete'])
        self.assertEqual(checkpoint['row']['extracted_triples'],
                         [['Station', 'opened in', '1910']])

    def test_baseline_accepts_and_retains_a_truthful_repair_provenance_overlay(self):
        identity = {'prompt_schema': 'hipporag_openie_v1'}
        producer = {'class': 'hipporag.llm.openai_gpt.CacheOpenAI', 'mode': 'online'}
        provenance = {'identity': identity, 'producer': producer,
                      'quality_repair': {'schema': repair.REPAIR_VERSION}}
        rag = HippoRAG.__new__(HippoRAG)
        rag.global_config = SimpleNamespace(force_openie_from_scratch=False)
        rag._openie_state_identity = lambda: identity
        rag._current_openie_provenance = lambda: {'identity': identity, 'producer': producer}
        with tempfile.TemporaryDirectory() as directory:
            rag.openie_state_path = str(Path(directory) / 'openie_state.json')
            rag.openie_results_path = str(Path(directory) / 'export.json')
            repair.write_json(rag.openie_state_path, {
                'docs': [source_row()], 'provenance': provenance})
            self.assertEqual(rag._openie_provenance_for_manifest(), provenance)
        changed = copy.deepcopy(provenance)
        changed['producer']['class'] = 'pathcondrag.llm.openai_gpt.CacheOpenAI'
        with self.assertRaisesRegex(RuntimeError, 'producer changed'):
            rag._validate_openie_provenance(changed, 'fixture')


if __name__ == '__main__':
    unittest.main()
