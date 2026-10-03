"""CPU-only checks for the bounded compact fallback and context preflight."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from utils.openie_compact_recovery import _windows, compact_recovery


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
        self.assertNotIn('support_quotes', settings['extra_body']['guided_json']['properties'])

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
        llm = FakeLLM(['not json', OK, 'not json', 'not json'])
        output = compact_recovery(llm, 'id', passage, [], 'Recover source facts.')
        self.assertEqual(output.metadata['quality_status'], 'partial')
        self.assertFalse(output.metadata['complete'])
        self.assertTrue(output.metadata['openie_skipped'])
        self.assertEqual(len(llm.calls), 4)

    def test_over_context_is_blocked_before_any_http_request(self):
        llm = FakeLLM([], token_count=7000)
        output = compact_recovery(llm, 'id', 'A is located in B.', [], 'Recover source facts.')
        self.assertEqual(output.metadata['quality_status'], 'failed')
        self.assertEqual(llm.calls, [])
        self.assertIn('exceeds 8192', output.metadata['window_recovery'][0]['metadata']['openie_skip_reason'])

    def test_explicit_no_supported_relations_is_valid_empty(self):
        llm = FakeLLM(['{"triples":[],"status":"no_supported_relations"}'])
        output = compact_recovery(llm, 'id', 'An unspecified name was omitted.', [], 'Recover missing name.')
        self.assertEqual(output.metadata['quality_status'], 'empty_valid')
        self.assertTrue(output.metadata['complete'])
        self.assertEqual(output.triples, [])


if __name__ == '__main__':
    unittest.main()
