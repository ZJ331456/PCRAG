"""CPU-only checks for the bounded compact fallback and context preflight."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from utils.openie_compact_recovery import _child_windows, _windows, compact_recovery


class FakeLLM:
    def __init__(self, responses, token_count=100):
        self.responses = iter(responses)
        self.token_count = token_count
        self.calls = []

    def count_prompt_tokens(self, messages):
        return self.token_count

    def infer(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, tuple):
            return response[0], {'finish_reason': response[1]}, False
        return response, {'finish_reason': 'stop'}, False


OK = '{"triples":[["A","located in","B"]],"status":"success"}'


class CompactRecoveryTests(unittest.TestCase):
    def test_whole_call_is_flat_and_requires_later_semantic_verification(self):
        llm = FakeLLM([OK])
        output = compact_recovery(llm, 'original-id', 'A is located in B.', ['A', 'B'], 'Repair source.')
        self.assertEqual(output.chunk_id, 'original-id')
        self.assertEqual(output.triples, [['A', 'located in', 'B']])
        self.assertTrue(output.metadata['requires_semantic_verification'])
        self.assertFalse(output.metadata['semantic_verified'])
        self.assertEqual(len(llm.calls), 1)
        settings = llm.calls[0]
        self.assertEqual(settings['max_completion_tokens'], 2048)
        self.assertEqual(settings['temperature'], 0.0)
        self.assertFalse(settings['extra_body']['chat_template_kwargs']['enable_thinking'])
        self.assertTrue(settings['extra_body']['guided_grammar'])
        self.assertNotIn('guided_json', settings['extra_body'])
        self.assertNotIn('support_quotes', settings['extra_body']['guided_grammar'])

    def test_length_immediately_uses_fully_covering_windows(self):
        passage = 'Table title\n' + 'row item amount ' * 90
        windows = _windows(passage)
        llm = FakeLLM([('truncated', 'length')] + [OK] * len(windows))
        output = compact_recovery(llm, 'id', passage, [], 'Recover source facts.')
        self.assertTrue(output.metadata['window_recovery_complete'])
        self.assertEqual(len(llm.calls), 1 + len(windows))
        self.assertEqual(output.metadata['window_recovery'][-1]['source_end'], len(passage))
        self.assertTrue(all(end - start <= 450 for start, end, _ in windows))

    def test_source_over_eight_windows_stays_failed_without_dropping_tail(self):
        llm = FakeLLM([('truncated', 'length')])
        output = compact_recovery(llm, 'id', 'word ' * 1200, [], 'Recover source facts.')
        self.assertEqual(output.metadata['quality_status'], 'failed')
        self.assertFalse(output.metadata['window_coverage_complete'])
        self.assertEqual(len(llm.calls), 1)

    def test_window_failure_is_partial_and_never_complete(self):
        passage = 'row item amount ' * 40
        windows = _windows(passage)
        self.assertEqual(len(windows), 2)
        children = _child_windows(passage, windows[1][0], windows[1][1])
        llm = FakeLLM(['not json', OK, 'not json', 'not json']
                      + ['not json'] * (2 * len(children)))
        output = compact_recovery(llm, 'id', passage, [], 'Recover source facts.')
        self.assertEqual(output.metadata['quality_status'], 'partial')
        self.assertFalse(output.metadata['complete'])
        self.assertTrue(output.metadata['openie_skipped'])
        self.assertEqual(len(llm.calls), 4 + 2 * len(children))
        self.assertFalse(output.metadata['window_recovery'][1]['metadata']
                         ['child_window_recovery_complete'])

    def test_over_context_is_blocked_before_any_http_request(self):
        llm = FakeLLM([], token_count=7000)
        output = compact_recovery(llm, 'id', 'A is located in B.', [], 'Recover source facts.')
        self.assertEqual(output.metadata['quality_status'], 'failed')
        self.assertEqual(llm.calls, [])
        self.assertIn('exceeds 8192', output.metadata['window_recovery'][0]['metadata']
                      ['parent_attempt_metadata']['openie_skip_reason'])

    def test_failed_parent_recovers_with_children_and_preserves_attempts(self):
        passage = 'Table title\n' + 'row item amount ' * 40
        windows = _windows(passage)
        children = _child_windows(passage, windows[0][0], windows[0][1])
        llm = FakeLLM([('whole truncated', 'length'), ('parent truncated', 'length'),
                       ('parent truncated again', 'length')]
                      + [OK] * len(children) + [OK] * (len(windows) - 1))
        output = compact_recovery(llm, 'id', passage, [], 'Recover source facts.')
        self.assertTrue(output.metadata['complete'])
        parent = output.metadata['window_recovery'][0]['metadata']
        self.assertEqual(parent['finish_reason'], 'stop')
        self.assertTrue(parent['aggregated_from_complete_children'])
        self.assertEqual(parent['parent_attempt_metadata']['finish_reason'], 'length')
        self.assertEqual(len(parent['parent_attempt_metadata']['attempts']), 2)
        self.assertEqual(parent['openie_attempt_count'], 2 + len(children))
        self.assertEqual(output.metadata['window_recovery_attempt_count'],
                         2 + len(children) + len(windows) - 1)
        self.assertEqual(len(llm.calls), 3 + len(children) + len(windows) - 1)
        self.assertNotIn('child_window_recovery',
                         output.metadata['window_recovery'][1]['metadata'])
        recovered_response = json.loads(json.loads(output.response)['window_responses'][0])
        self.assertEqual(recovered_response['parent_response'], 'parent truncated again')
        self.assertEqual(recovered_response['child_responses'], [OK] * len(children))

    def test_child_windows_retain_absolute_offsets_and_complete_span(self):
        passage = 'Title\n' + 'row item amount ' * 40
        for start, end, _ in _windows(passage):
            children = _child_windows(passage, start, end)
            self.assertGreater(len(children), 0)
            self.assertLessEqual(len(children), 3)
            self.assertEqual(children[0][0], start)
            self.assertEqual(children[-1][1], end)
            previous_end = start
            for child_start, child_end, text in children:
                self.assertLessEqual(child_start, previous_end)
                self.assertGreater(child_end, child_start)
                self.assertLessEqual(child_end - child_start, 225)
                self.assertEqual(text, 'Title\n' + passage[child_start:child_end])
                previous_end = child_end

    def test_one_child_failure_never_marks_parent_complete(self):
        passage = 'row item amount ' * 20
        windows = _windows(passage)
        self.assertEqual(len(windows), 1)
        children = _child_windows(passage, windows[0][0], windows[0][1])
        llm = FakeLLM([('whole truncated', 'length'), ('parent truncated', 'length'),
                       ('parent truncated again', 'length'), OK]
                      + [('child truncated', 'length')] * (2 * (len(children) - 1)))
        output = compact_recovery(llm, 'id', passage, [], 'Recover source facts.')
        self.assertFalse(output.metadata['complete'])
        parent = output.metadata['window_recovery'][0]['metadata']
        self.assertFalse(parent['complete'])
        self.assertEqual(parent['finish_reason'], 'length')
        self.assertNotIn('aggregated_from_complete_children', parent)
        self.assertEqual(output.metadata['quality_status'], 'partial')

    def test_explicit_no_supported_relations_is_valid_empty(self):
        llm = FakeLLM(['{"triples":[],"status":"no_supported_relations"}'])
        output = compact_recovery(llm, 'id', 'An unspecified name was omitted.', [], 'Recover missing name.')
        self.assertEqual(output.metadata['quality_status'], 'empty_valid')
        self.assertTrue(output.metadata['complete'])
        self.assertEqual(output.triples, [])

    def test_short_child_quality_failure_receives_source_grounded_feedback(self):
        passage = 'Title\nAda and Bo are fictional characters in Show C.'
        bad = '{"triples":[["Ada","and","Bo"]],"status":"success"}'
        llm = FakeLLM([('whole truncated', 'length'), ('parent truncated', 'length'),
                       ('parent truncated again', 'length'), bad, OK])
        output = compact_recovery(llm, 'id', passage, [], 'Recover source facts.')
        self.assertTrue(output.metadata['complete'])
        child = output.metadata['window_recovery'][0]['metadata']['child_window_recovery'][0]['metadata']
        self.assertEqual(child['openie_attempt_count'], 2)
        self.assertIn('coordinating word', child['attempts'][0]['validation_error'])
        self.assertIn('separately extract', llm.calls[-1]['messages'][-1]['content'])
        self.assertEqual(llm.calls[-1]['max_completion_tokens'], 2048)
        self.assertFalse(llm.calls[-1]['extra_body']['chat_template_kwargs']['enable_thinking'])


if __name__ == '__main__':
    unittest.main()
