"""Bounded NER fallback covers unchanged source and replays complete leaves."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.index.openie.source_verified_openie import SourceVerifiedOpenIE
from pathcondrag.index.ner import source_verified as ner_module
from pathcondrag.utils.misc_utils import NerRawOutput


class NERSpanRecoveryTests(unittest.TestCase):
    source = 'Title\n' + ' '.join(f'City{index}' for index in range(32)) + '\nTail'

    def previous(self):
        return NerRawOutput('chunk', '', [], {'quality_status': 'failed', 'complete': False,
                                               'ner_attempts': [{'attempt': 1}]})

    def test_failed_long_unit_is_split_and_whole_original_source_covered(self):
        requests = []

        def infer(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            source = kwargs['messages'][3]['content']
            if len(source) > 140:
                return '{"named_entities":[', {'finish_reason': 'length'}, False
            names = [word for word in source.split() if word.startswith('City')]
            return json.dumps({'named_entities': names}), {'finish_reason': 'stop'}, False

        extractor = SourceVerifiedOpenIE(SimpleNamespace(infer=infer))
        result = extractor.recover_pending_ner('chunk', self.source, self.previous(), 1)
        self.assertTrue(result.metadata['complete'])
        self.assertEqual(result.unique_entities, [f'City{index}' for index in range(32)])
        leaves = result.metadata['ner_units']
        self.assertEqual(''.join(unit['source_text'] for unit in leaves), self.source)
        self.assertEqual([(unit['source_start'], unit['source_end']) for unit in leaves],
                         [(sum(len(leaf['source_text']) for leaf in leaves[:index]),
                           sum(len(leaf['source_text']) for leaf in leaves[:index + 1]))
                          for index in range(len(leaves))])
        self.assertTrue(any(unit['status'] == 'split_parent'
                            for unit in result.metadata['ner_recovery_history_failedunits']))
        self.assertTrue(all(call['max_completion_tokens'] == 512 for call in requests))
        self.assertTrue(all(call['extra_body']['chat_template_kwargs']['enable_thinking'] is False
                            for call in requests))
        first_call_count = len(requests)
        replay = extractor.recover_pending_ner('chunk', self.source, result, 2)
        self.assertTrue(replay.metadata['complete'])
        self.assertEqual(replay.unique_entities, result.unique_entities)
        self.assertEqual(len(requests), first_call_count)

    def test_budget_exhaustion_cannot_mark_unprocessed_tail_complete(self):
        calls = []

        def attempts(key, source, context):
            calls.append(source)
            return NerRawOutput(key, '{}', ['Only title'], {'complete': True, 'finish_reason': 'stop',
                                                           'ner_attempts': [{'attempt': 1}]})

        extractor = SourceVerifiedOpenIE(SimpleNamespace())
        extractor._ner_attempts = attempts
        with patch.object(ner_module, 'MAX_NER_RECOVERY_CALLS', 4):
            result = extractor.recover_pending_ner('chunk', self.source, self.previous(), 1)
        self.assertEqual(calls, ['Title\n'])
        self.assertFalse(result.metadata['complete'])
        self.assertFalse(result.metadata['ner_source_coverage_complete'])
        self.assertEqual(result.unique_entities, [])
        self.assertTrue(result.metadata['ner_pending_spans'])
        self.assertLessEqual(result.metadata['ner_recovery_calls'], 4)
        self.assertEqual(result.metadata['ner_units'][0]['source_text'], 'Title\n')

    def test_resume_reuses_completed_leaf_and_only_processes_unfinished_spans(self):
        calls = []
        fail = {'remaining': True}

        def attempts(key, source, context):
            calls.append(source)
            if source == 'Tail' and fail['remaining']:
                return NerRawOutput(key, '{bad', [], {'complete': False, 'error': 'bad JSON',
                                                     'ner_attempts': [{'attempt': 1}]})
            return NerRawOutput(key, '{}', [source.strip()], {'complete': True, 'finish_reason': 'stop',
                                                            'ner_attempts': [{'attempt': 1}]})

        extractor = SourceVerifiedOpenIE(SimpleNamespace())
        extractor._ner_attempts = attempts
        first = extractor.recover_pending_ner('chunk', self.source, self.previous(), 1)
        self.assertFalse(first.metadata['complete'])
        first_success_count = len(first.metadata['ner_units'])
        self.assertEqual(first_success_count, 2)
        calls.clear()
        fail['remaining'] = False
        resumed = extractor.recover_pending_ner('chunk', self.source, first, 2)
        self.assertEqual(calls, ['Tail'])
        self.assertTrue(resumed.metadata['complete'])
        self.assertEqual(''.join(unit['source_text'] for unit in resumed.metadata['ner_units']), self.source)


if __name__ == '__main__':
    unittest.main()
