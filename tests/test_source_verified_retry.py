"""Regression checks for an incomplete first compact pass and bounded retry."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.index.openie.openie_openai import OpenIE
from pathcondrag.index.openie import source_verified_openie as module
from pathcondrag.utils.misc_utils import TripleRawOutput
from pathcondrag.index.openie.openie_semantic_validation import verify_repaired_triples as boolean_verifier


class FakeLLM:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return next(self.responses), {'finish_reason': 'stop'}, False


def extraction(triples, **metadata):
    return TripleRawOutput('chunk', json.dumps({'triples': triples}), triples,
                           {'finish_reason': 'stop', **metadata})


class IncompleteCompactRetryTests(unittest.TestCase):
    def call(self, recovery, responses=()):
        llm = FakeLLM(responses)
        initial = extraction([], quality_status='failed', openie_skipped=True)
        with patch.object(OpenIE, 'triple_extraction', return_value=initial), \
                patch.object(module, 'verify_repaired_triples',
                             side_effect=lambda llm, passage, values, **kwargs:
                             boolean_verifier(llm, passage, values)), \
                patch('pathcondrag.index.openie.openie_atomic_recovery.atomic_recovery',
                      return_value=extraction([], complete=False, openie_skipped=True)):
            with patch.object(module, 'compact_recovery', side_effect=recovery) as fallback:
                output = module.SourceVerifiedOpenIE(llm).triple_extraction(
                    'chunk', 'Ada and Bo are characters in Show C.', ['Ada', 'Bo', 'Show C'])
        return output, fallback, llm

    def test_incomplete_pass_retries_and_filters_against_whole_source(self):
        supported = ['Ada', 'is a character in', 'Show C']
        unsupported = ['Ada', 'is parent of', 'Bo']
        failed = extraction([unsupported], complete=False, openie_skipped=True,
                            attempts=[{'validation_error': 'coordinating word as predicate'}])
        recovered = extraction([supported, unsupported], complete=True)
        output, fallback, llm = self.call([failed, recovered], ['{"supported":[true,false]}'])
        self.assertEqual(fallback.call_count, 2)
        self.assertEqual(output.triples, [supported])
        self.assertTrue(output.metadata['semantic_verified'])
        self.assertEqual(len(output.metadata['fresh_extraction_history']), 3)
        self.assertIn('coordinating word', fallback.call_args.args[4])
        self.assertEqual(llm.calls[0]['max_completion_tokens'], 2048)
        self.assertFalse(llm.calls[0]['extra_body']['chat_template_kwargs']['enable_thinking'])

    def test_second_incomplete_pass_uses_atomic_then_stops_without_publishing_partial_candidates(self):
        partial = extraction([['Ada', 'is parent of', 'Bo']], complete=False,
                             openie_skipped=True, quality_status='partial')
        output, fallback, llm = self.call([partial, partial])
        self.assertEqual(fallback.call_count, 2)
        self.assertEqual(output.triples, [])
        self.assertTrue(output.metadata['openie_skipped'])
        self.assertFalse(output.metadata['complete'])
        self.assertEqual(len(output.metadata['fresh_extraction_history']), 4)
        self.assertEqual(output.metadata['fresh_extraction_history'][-1]['stage'], 'atomic')
        self.assertEqual(llm.calls, [])

    def test_first_complete_pass_does_not_add_a_second_request(self):
        supported = ['Ada', 'is a character in', 'Show C']
        output, fallback, _ = self.call([extraction([supported], complete=True)],
                                        ['{"supported":[true]}'])
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(output.triples, [supported])

    def test_feedback_collects_failed_child_diagnostics(self):
        metadata = {'window_recovery': [{'metadata': {'child_window_recovery': [
            {'metadata': {'attempts': [{'validation_error': 'child uses and as predicate'}]}}
        ]}}]}
        feedback = module.SourceVerifiedOpenIE._incomplete_recovery_feedback(metadata)
        self.assertIn('child uses and as predicate', feedback)
        self.assertIn('not factual evidence', feedback)


if __name__ == '__main__':
    unittest.main()
