"""CPU integration checks for the separate NER and OpenIE stages."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.index.ner.openai import OpenAINERMixin
from pathcondrag.index.ner.source_verified import SourceVerifiedNERMixin
from pathcondrag.index.openie.offline import OfflineOpenIE
from pathcondrag.index.openie.openie_openai import OpenIE
from pathcondrag.index.openie.openie_source_evidence import FLAGS
from pathcondrag.index.openie.source_verified_openie import SourceVerifiedOpenIE
from pathcondrag.index.openie_checkpoint import OpenIECheckpoint
from pathcondrag.index.shared_index_builder import SharedQualityOpenIE
from pathcondrag.prompts import PromptTemplateManager


SOURCE = 'Ada trained in theater.'
TRIPLE = ['Ada', 'trained in', 'theater']


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def count_prompt_tokens(self, messages):
        return 100

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.replies), {'finish_reason': 'stop'}, False


class ExtractionLayoutTests(unittest.TestCase):
    def test_single_chunk_runs_ner_triples_evidence_and_checkpoint_resume(self):
        verdict = {
            'supported': True, 'quote_id': 0,
            'subject_roles': [{'source_subject': 'Ada', 'mention': 'Ada',
                               'relation_quote_id': 0, 'relation_supported': True}],
            'reason': 'The source explicitly states this relationship.',
            **{flag: True for flag in FLAGS},
        }
        llm = ScriptedLLM([
            json.dumps({'named_entities': ['Ada']}),
            json.dumps({'triples': [TRIPLE]}),
            json.dumps({'source_has_supported_relations': True,
                        'source_evidence_quote_id': 0,
                        'source_reason': 'The source contains an explicit relation.',
                        'verdicts': [verdict]}),
        ])
        chunks = {'chunk': {'content': SOURCE}}
        extractor = SharedQualityOpenIE(llm, max_workers=8)
        self.assertIs(OpenIE.ner, OpenAINERMixin.ner)
        self.assertIs(SourceVerifiedOpenIE.ner, SourceVerifiedNERMixin.ner)
        with tempfile.TemporaryDirectory() as directory:
            progress = OpenIECheckpoint(Path(directory) / 'progress.sqlite', {'case': 'split_stages'})
            try:
                extractor.checkpoint = progress.save
                ner, triples = extractor.batch_openie(chunks)
                self.assertEqual(ner['chunk'].unique_entities, ['Ada'])
                self.assertEqual(triples['chunk'].triples, [TRIPLE])
                self.assertTrue(extractor.is_verified_complete(triples['chunk']))
                self.assertEqual([call['max_completion_tokens'] for call in llm.calls],
                                 [512, 2048, 2048])
                for call in (llm.calls[0], llm.calls[2]):
                    self.assertFalse(call['extra_body']['chat_template_kwargs']['enable_thinking'])
                extractor.initial_rows = progress.overlay([], chunks)
                resumed_ner, resumed_triples = extractor.batch_openie(chunks)
                self.assertEqual(resumed_ner['chunk'].unique_entities, ['Ada'])
                self.assertEqual(resumed_triples['chunk'].triples, [TRIPLE])
                self.assertEqual(len(llm.calls), 3)
            finally:
                extractor.checkpoint = None
                progress.close()

    def test_strict_ner_invalid_responses_preserve_512_budget_and_failure(self):
        llm = ScriptedLLM(['{"named_entities":"Ada"}'] * 3)
        result = SharedQualityOpenIE(llm).ner('chunk', SOURCE)
        self.assertEqual(result.unique_entities, [])
        self.assertFalse(result.metadata['complete'])
        self.assertEqual(result.metadata['quality_status'], 'failed')
        self.assertEqual([call['max_completion_tokens'] for call in llm.calls], [512] * 3)

    def offline_extractor(self, triple_response):
        class BatchLLM:
            def __init__(self):
                self.calls = []

            def batch_infer(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                response = (json.dumps({'named_entities': ['Ada', 'Ada']})
                            if kwargs['json_template'] == 'ner' else triple_response)
                return [response], {}

        extractor = OfflineOpenIE.__new__(OfflineOpenIE)
        extractor.llm_model = BatchLLM()
        extractor.prompt_template_manager = PromptTemplateManager()
        return extractor

    def test_offline_stages_keep_original_payload_and_budgets(self):
        extractor = self.offline_extractor(json.dumps({'triples': [TRIPLE]}))
        ner, triples = extractor.batch_openie({'chunk': {'content': SOURCE}})
        self.assertEqual(ner['chunk'].unique_entities, ['Ada', 'Ada'])
        self.assertEqual(triples['chunk'].triples, [TRIPLE])
        self.assertEqual([call[1]['max_tokens'] for call in extractor.llm_model.calls], [512, 2048])
        self.assertEqual(ner['chunk'].metadata, {})
        self.assertEqual(triples['chunk'].metadata, {})
        self.assertIn(json.dumps({'named_entities': ['Ada', 'Ada']}),
                      extractor.llm_model.calls[1][0][0][-1]['content'])

    def test_offline_invalid_triples_do_not_raise_unbound_error(self):
        extractor = self.offline_extractor('invalid JSON')
        _, triples = extractor.batch_openie({'chunk': {'content': SOURCE}})
        self.assertEqual(triples['chunk'].triples, [])


if __name__ == '__main__':
    unittest.main()
