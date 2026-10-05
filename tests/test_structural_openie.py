"""Ordinary-cost extraction keeps fixed budgets and truthful validation scope."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.index.openie.structural_openie import StructuralOpenIE, STRUCTURAL_VERSION, RECOVERY_BUDGET
from pathcondrag.index.openie_build_queue import run_openie_queue
from pathcondrag.utils.misc_utils import TripleRawOutput


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.llm_config = SimpleNamespace(generate_params={'seed': 42})

    def infer(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        response, finish = next(self.replies)
        return response, {'finish_reason': finish}, False


class StructuralOpenIETests(unittest.TestCase):
    def test_normal_queue_uses_one_ner_and_one_triple_request_without_audits(self):
        triples = [[f'Alpha{index}', 'is', 'Beta'] for index in range(20)]
        llm = ScriptedLLM([('{"named_entities":["Alpha","Beta"]}', 'stop'),
                           (json.dumps({'triples': triples}), 'stop')])
        extractor = StructuralOpenIE(llm, max_workers=1, respect_env_workers=False)
        with patch('pathcondrag.index.openie.openie_source_evidence.verify_source_relations',
                   side_effect=AssertionError('semantic audit must not run')), \
                patch('pathcondrag.index.openie.openie_compact_recovery.compact_recovery',
                      side_effect=AssertionError('compact must not run')), \
                patch('pathcondrag.index.openie.openie_atomic_recovery.atomic_recovery',
                      side_effect=AssertionError('atomic must not run')):
            ner, outputs = run_openie_queue(extractor, {'chunk': {'content': 'Alpha is Beta.'}})
        result = outputs['chunk']
        self.assertEqual([call['max_completion_tokens'] for call in llm.calls], [512, 2048])
        self.assertEqual(result.triples, triples)
        self.assertTrue(extractor.is_verified_complete(result))
        self.assertFalse(result.metadata['semantic_verified'])
        self.assertEqual(result.metadata['validation_scope'], 'structural')
        self.assertEqual(result.metadata['structural_infer_calls'], 1)
        self.assertTrue(all(call['extra_body']['chat_template_kwargs']['enable_thinking'] is False
                            for call in llm.calls))

    def test_shape_and_length_failures_share_three_call_budget_then_success(self):
        llm = ScriptedLLM([('{"triples":[["Alpha","is","Beta","extra"]]}', 'stop'),
                           ('{"triples":[["Alpha","is","Beta"]]}', 'length'),
                           ('{"triples":[["Alpha","is","Beta"]]}', 'stop')])
        extractor = StructuralOpenIE(llm)
        result = extractor.triple_extraction('chunk', 'Alpha is Beta.', ['Alpha', 'Beta'])
        self.assertTrue(extractor.is_verified_complete(result))
        self.assertEqual(result.metadata['structural_infer_calls'], RECOVERY_BUDGET)
        self.assertEqual(len(llm.calls), RECOVERY_BUDGET)
        self.assertTrue(all(call['max_completion_tokens'] == 2048 for call in llm.calls))
        self.assertNotEqual(llm.calls[0]['messages'], llm.calls[1]['messages'])
        self.assertEqual(result.metadata['structural_attempts'][0]['invalid_triple_count'], 1)

    def test_repeated_empty_results_fail_once_budget_used_and_queue_does_not_retry_llm(self):
        llm = ScriptedLLM([('{"named_entities":["Alpha"]}', 'stop')]
                          + [('{"triples":[]}', 'stop')] * RECOVERY_BUDGET)
        extractor = StructuralOpenIE(llm, max_workers=1, respect_env_workers=False)
        _, triples = run_openie_queue(extractor, {'chunk': {'content': 'Alpha is Beta.'}})
        result = triples['chunk']
        self.assertFalse(result.metadata['complete'])
        self.assertFalse(result.metadata['semantic_verified'])
        self.assertFalse(result.metadata['source_no_supported_relations'])
        self.assertEqual(result.metadata['structural_infer_calls'], RECOVERY_BUDGET)
        self.assertEqual(len(llm.calls), 1 + RECOVERY_BUDGET)
        self.assertEqual(result.triples, [])
        self.assertFalse(extractor.is_verified_complete(result))

    def test_saved_attempts_reduce_remaining_budget_without_counting_ner(self):
        llm = ScriptedLLM([('{"triples":[["Alpha","is","Beta"]]}', 'stop')])
        extractor = StructuralOpenIE(llm)
        previous = TripleRawOutput('chunk', 'bad response', [],
                                   {'structural_infer_calls': 2, 'ner_attempts': [{}] * 64,
                                    'complete': False, 'error': 'bad JSON'})
        result = extractor.recover_pending_triples('chunk', 'Alpha is Beta.', ['Alpha'], previous, 2)
        self.assertTrue(extractor.is_verified_complete(result))
        self.assertEqual(result.metadata['structural_infer_calls'], 3)
        self.assertEqual(len(llm.calls), 1)
        exhausted = TripleRawOutput('chunk', 'truncated', [], {'openie_attempt_count': 3,
                                                              'finish_reason': 'length'})
        result = extractor.recover_pending_triples('chunk', 'Alpha is Beta.', ['Alpha'], exhausted, 3)
        self.assertFalse(result.metadata['complete'])
        self.assertEqual(len(llm.calls), 1)

    def test_cached_initial_result_revalidated_locally_without_semantic_claim(self):
        llm = ScriptedLLM([])
        extractor = StructuralOpenIE(llm)
        previous = TripleRawOutput('chunk', '{"triples":[["Alpha","is","Beta"]]}',
                                   [['Alpha', 'is', 'Beta']],
                                   {'finish_reason': 'stop', 'openie_attempt_count': 1,
                                    'semantic_verified': True, 'source_verified_schema': 'old-schema',
                                    'semantic_verification_history': [{'complete': True}],
                                    'semantic_verifier_contract': 'old-verifier'})
        result = extractor.audit_existing_triples('chunk', 'Alpha is Beta.', ['Alpha'], previous)
        self.assertTrue(extractor.is_verified_complete(result))
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(result.metadata['structural_schema'], STRUCTURAL_VERSION)
        self.assertFalse(result.metadata['semantic_verified'])
        self.assertNotIn('source_verified_schema', result.metadata)
        self.assertNotIn('semantic_verification_history', result.metadata)
        self.assertNotIn('semantic_verifier_contract', result.metadata)
        self.assertEqual(result.metadata['historical_semantic_metadata']['source_verified_schema'], 'old-schema')

    def test_only_whitespace_gets_deterministic_empty_completion(self):
        extractor = StructuralOpenIE(ScriptedLLM([]))
        result = extractor.triple_extraction('chunk', ' \n\t ', [])
        self.assertTrue(extractor.is_verified_complete(result))
        self.assertFalse(result.metadata['semantic_verified'])
        self.assertTrue(result.metadata['deterministically_empty_source'])
        self.assertEqual(result.metadata['structural_infer_calls'], 0)

    def test_checker_rejects_fake_semantic_flags_and_wrong_fields(self):
        llm = ScriptedLLM([('{"triples":[["Alpha","is","Beta"]]}', 'stop')])
        extractor = StructuralOpenIE(llm)
        result = extractor.triple_extraction('chunk', 'Alpha is Beta.', ['Alpha'])
        result.metadata['semantic_verified'] = True
        self.assertFalse(extractor.is_verified_complete(result))
        result.metadata['semantic_verified'] = False
        result.triples = [['Alpha', 'is', '']]
        self.assertFalse(extractor.is_verified_complete(result))


if __name__ == '__main__':
    unittest.main()
