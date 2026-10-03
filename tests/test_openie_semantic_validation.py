"""No-model checks for semantic filtering and bounded source-only verification."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from utils.openie_semantic_validation import SemanticVerificationError, verify_repaired_triples  # noqa: E402


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            return reply
        return reply, {'finish_reason': 'stop'}, False


class SemanticVerificationTests(unittest.TestCase):
    def setUp(self):
        self.passage = ('Actor Iris had classical theater training and appeared in Moon Harbor. '
                        'Record A originally appeared on Label B. Its tracks were reissued as part of "".')
        self.triples = [
            ['Actor Iris', 'had classical theater training', 'Moon Harbor'],
            ['Actor Iris', 'trained in', 'classical theater'],
            ['Record A', 'tracks reissued as part of', 'Label B'],
        ]

    def test_jointly_unsupported_cooccurring_arguments_are_filtered_without_rewriting(self):
        original = copy.deepcopy(self.triples)
        metadata = {'finish_reason': 'stop', 'response_id': 'verification'}
        llm = FakeLLM([('{"supported":[false,true,false]}', metadata, True)])
        accepted, audit = verify_repaired_triples(llm, self.passage, self.triples)
        self.assertEqual(accepted, [self.triples[1]])
        self.assertEqual(self.triples, original)
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['rejected_indices'], [0, 2])
        self.assertEqual(audit['checks'][0]['triple'], self.triples[0])
        self.assertEqual(audit['raw_response'], '{"supported":[false,true,false]}')
        self.assertEqual(audit['llm_metadata'], metadata)
        self.assertTrue(audit['cache_hit'])

    def test_all_false_is_a_complete_rejection_not_a_transport_failure(self):
        accepted, audit = verify_repaired_triples(
            FakeLLM(['{"supported":[false,false,false]}']), self.passage, self.triples)
        self.assertEqual(accepted, [])
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['n_rejected'], 3)
        self.assertEqual(audit['n_unverified'], 0)

    def test_empty_candidates_need_no_llm_request(self):
        llm = FakeLLM([])
        accepted, audit = verify_repaired_triples(llm, self.passage, [])
        self.assertEqual(accepted, [])
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['attempt_count'], 0)
        self.assertEqual(llm.calls, [])

    def test_wrong_length_and_numeric_flags_receive_specific_feedback(self):
        llm = FakeLLM(['{"supported":[true]}', '{"supported":[0,1,0]}',
                       '{"supported":[false,true,false]}'])
        accepted, audit = verify_repaired_triples(llm, self.passage, self.triples)
        self.assertEqual(accepted, [self.triples[1]])
        self.assertEqual(audit['shape_repair_count'], 2)
        self.assertIn('exactly 3', llm.calls[1]['messages'][-1]['content'])
        self.assertIn('not JSON booleans', llm.calls[2]['messages'][-1]['content'])
        for call in llm.calls:
            self.assertEqual(call['max_completion_tokens'], 2048)
            self.assertEqual(call['temperature'], 0.0)
            self.assertFalse(call['extra_body']['chat_template_kwargs']['enable_thinking'])
            schema = call['extra_body']['guided_json']['properties']['supported']
            self.assertEqual(schema['items'], {'type': 'boolean'})
            self.assertEqual((schema['minItems'], schema['maxItems']), (3, 3))

    def test_persistent_shape_failure_is_unverified_not_a_false_semantic_judgment(self):
        llm = FakeLLM(['not json'] * 3)
        with self.assertRaises(SemanticVerificationError) as caught:
            verify_repaired_triples(llm, self.passage, self.triples)
        audit = caught.exception.audit_metadata
        self.assertFalse(audit['complete'])
        self.assertIsNone(audit['supported'])
        self.assertIsNone(audit['n_rejected'])
        self.assertEqual(audit['n_unverified'], 3)
        self.assertTrue(all(check['supported'] is None for check in audit['checks']))
        self.assertEqual(len(llm.calls), 3)

    def test_parseable_but_truncated_verdict_is_not_accepted(self):
        llm = FakeLLM([('{"supported":[true,true,true]}', {'finish_reason': 'length'}, False),
                       '{"supported":[false,true,false]}'])
        accepted, audit = verify_repaired_triples(llm, self.passage, self.triples)
        self.assertEqual(accepted, [self.triples[1]])
        self.assertIn('finish_reason', audit['attempts'][0]['validation_error'])
        self.assertEqual(len(llm.calls), 2)

    def test_extra_keys_and_string_flags_cannot_bypass_boolean_schema(self):
        llm = FakeLLM(['{"supported":[true,true,true],"reason":"cooccur"}',
                       '{"supported":["true","false","true"]}',
                       '{"supported":[false,false,false]}'])
        accepted, audit = verify_repaired_triples(llm, self.passage, self.triples)
        self.assertEqual(accepted, [])
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['attempt_count'], 3)

    def test_client_transport_failure_is_recorded_without_multiplying_http_retry_loops(self):
        llm = FakeLLM([TimeoutError('local endpoint unavailable')])
        with self.assertRaises(SemanticVerificationError) as caught:
            verify_repaired_triples(llm, self.passage, self.triples)
        audit = caught.exception.audit_metadata
        self.assertFalse(audit['complete'])
        self.assertIn('TimeoutError', audit['error'])
        self.assertEqual(len(llm.calls), 1)

    def test_whole_source_is_sent_once_and_long_error_feedback_is_bounded(self):
        long_output = 'x' * 10000
        llm = FakeLLM([long_output, '{"supported":[false,true,false]}'])
        _, audit = verify_repaired_triples(llm, self.passage, self.triples)
        original_request = json.loads(llm.calls[0]['messages'][1]['content'])
        self.assertEqual(original_request['SOURCE'], self.passage)
        self.assertEqual(original_request['candidate_triples'], self.triples)
        self.assertEqual(audit['attempts'][0]['raw_response'], long_output)
        self.assertLessEqual(len(llm.calls[1]['messages'][-2]['content']), 1200)
        self.assertEqual(llm.calls[1]['messages'][1], llm.calls[0]['messages'][1])

    def test_prompt_requires_joint_entailment_missing_arguments_and_correct_table_roles(self):
        llm = FakeLLM(['{"supported":[false,true,false]}'])
        verify_repaired_triples(llm, self.passage, self.triples)
        system = llm.calls[0]['messages'][0]['content']
        for rule in ('ONLY the complete supplied SOURCE', 'co-occurring', 'empty quotation',
                     'missing object', 'time/location qualifiers', 'column header',
                     'benchmark answers', 'classical theater', 'Label B'):
            self.assertIn(rule, system)

    def test_original_unicode_and_spacing_are_preserved(self):
        triple = ['  静岡県 ', ' located in ', ' Japan  ']
        accepted, _ = verify_repaired_triples(
            FakeLLM(['{"supported":[true]}']), '静岡県 is located in Japan.', [triple])
        self.assertEqual(accepted, [triple])
        self.assertIsNot(accepted[0], triple)

    def test_invalid_candidates_fail_before_any_model_request(self):
        llm = FakeLLM([])
        for triples in (None, [['A', 'r', '']], [['A', 'r', 'B', 'C']], ['abc']):
            with self.subTest(triples=triples), self.assertRaises(ValueError):
                verify_repaired_triples(llm, self.passage, triples)
        self.assertEqual(llm.calls, [])

    def test_large_batches_keep_order_and_preserve_each_original_api_response(self):
        triples = [['Subject', f'relation {index}', 'Object'] for index in range(65)]
        responses = [json.dumps({'supported': [True] * 30}),
                     json.dumps({'supported': [False] * 30}),
                     json.dumps({'supported': [True] * 5})]
        llm = FakeLLM(responses)
        accepted, audit = verify_repaired_triples(llm, 'Subject has the listed relationships.', triples)
        self.assertEqual(accepted, triples[:30] + triples[60:])
        self.assertEqual(audit['batch_count'], 3)
        self.assertEqual(audit['shape_repair_count'], 0)
        self.assertEqual(audit['raw_responses'], responses)
        self.assertTrue(audit['response_is_aggregate'])
        self.assertEqual(json.loads(audit['response'])['supported'], audit['supported'])
        self.assertTrue(all(len(json.loads(call['messages'][1]['content'])['candidate_triples']) <= 30
                            for call in llm.calls))

    def test_cpu_token_count_shrinks_batches_without_truncating_source(self):
        triples = [['Subject', f'relation {index}', 'Object'] for index in range(4)]
        llm = FakeLLM(['{"supported":[true,true]}', '{"supported":[false,true]}'])
        llm.count_prompt_tokens = lambda messages: (
            4000 + 500 * len(json.loads(messages[1]['content'])['candidate_triples']))
        accepted, audit = verify_repaired_triples(llm, self.passage, triples)
        self.assertEqual(accepted, triples[:2] + triples[3:])
        self.assertEqual(audit['batch_count'], 2)
        self.assertTrue(audit['exact_prompt_token_accounting'])
        self.assertTrue(all(record['prompt_token_count'] + 2048 <= 8192 for record in audit['attempts']))
        self.assertTrue(all(json.loads(call['messages'][1]['content'])['SOURCE'] == self.passage
                            for call in llm.calls))

    def test_oversize_whole_source_blocks_requests_instead_of_cutting_evidence(self):
        llm = FakeLLM([])
        llm.count_prompt_tokens = lambda messages: 7000
        with self.assertRaisesRegex(SemanticVerificationError, 'cannot fit'):
            verify_repaired_triples(llm, self.passage, self.triples)
        self.assertEqual(llm.calls, [])


if __name__ == '__main__':
    unittest.main()
