"""CPU-only checks for resumable strict OpenIE pending work."""

import copy
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.utils.misc_utils import NerRawOutput, TripleRawOutput
from pathcondrag.index.openie_build_queue import run_openie_queue


FACT = ['Iris', 'trained in', 'classical theater']


def ner(key, error=None):
    return NerRawOutput(key, '{}', [] if error else ['Iris'],
                        {'finish_reason': 'stop', **({'error': error} if error else {})})


def triples(key, success=True, contract='new', values=None, **metadata):
    values = [FACT] if values is None else values
    return TripleRawOutput(key, '{}', copy.deepcopy(values), {
        'finish_reason': 'stop', 'complete': success, 'semantic_verified': success,
        'quality_status': 'success' if success else 'failed',
        'source_verified_schema': contract,
        **({'error': 'invalid response', 'openie_skipped': True} if not success else {}),
        **metadata})


def row(key, triple_result=None, ner_result=None, passage=None):
    triple_result, ner_result = triple_result or triples(key), ner_result or ner(key)
    return {'idx': key, 'passage': passage or key, 'extracted_entities': ner_result.unique_entities,
            'extracted_triples': triple_result.triples,
            'openie_metadata': {'ner': ner_result.metadata, 'triples': triple_result.metadata},
            'openie_responses': {'ner': ner_result.response, 'triples': triple_result.response}}


class Extractor:
    def __init__(self, failures=None, ner_failures=None, throw=None, slow=False):
        self.calls = []
        self.failures = failures or {}
        self.ner_failures = ner_failures or {}
        self.throw = throw or {}
        self.slow = slow
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def worker_limits(self):
        return 64, 64

    def is_verified_complete(self, output):
        return (output.metadata.get('source_verified_schema') == 'new'
                and output.metadata.get('complete') is True
                and output.metadata.get('semantic_verified') is True)

    def _call(self, stage, key, round_number=1):
        with self.lock:
            self.calls.append((stage, key, round_number))
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.slow:
                time.sleep(.004)
            failure = self.throw.get((stage, key, round_number))
            if failure:
                raise failure
            if stage in ('ner', 'recover_ner'):
                return ner(key, 'NER transport failed' if round_number <= self.ner_failures.get(key, 0)
                           else None)
            return triples(key, success=round_number > self.failures.get(key, 0))
        finally:
            with self.lock:
                self.active -= 1

    def ner(self, key, passage):
        return self._call('ner', key)

    def recover_pending_ner(self, key, passage, previous, round_number):
        return self._call('recover_ner', key, round_number)

    def triple_extraction(self, key, passage, entities):
        return self._call('triples', key)

    def audit_existing_triples(self, key, passage, entities, prior):
        return self._call('audit', key)

    def recover_pending_triples(self, key, passage, entities, previous, round_number):
        return self._call('recover_triples', key, round_number)


class BuildQueueTests(unittest.TestCase):
    def chunks(self, *keys):
        return {key: {'content': key} for key in keys}

    def test_restart_uses_prior_recovery_progress_without_starting_normal_extraction(self):
        initial = row('chunk', triples('chunk', success=False,
            fresh_extraction_history=[{'stage': 'compact', 'triples': [FACT]}]))
        extractor = Extractor()
        _, outputs = run_openie_queue(extractor, self.chunks('chunk'), [initial])
        self.assertEqual(extractor.calls, [('recover_triples', 'chunk', 1)])
        self.assertTrue(outputs['chunk'].metadata['complete'])

    def test_success_keeps_source_order_and_checkpoints_each_stage_on_main_thread(self):
        extractor, saved, threads = Extractor(), [], []

        def checkpoint(event):
            threads.append(threading.get_ident())
            saved.append(event)

        ner_outputs, triple_outputs = run_openie_queue(extractor, self.chunks('b', 'a'), checkpoint=checkpoint)
        self.assertEqual(list(ner_outputs), ['b', 'a'])
        self.assertEqual(list(triple_outputs), ['b', 'a'])
        self.assertEqual(threads, [threading.get_ident()] * 4)
        self.assertEqual(sum(event['openie_metadata']['triples']['complete'] is False for event in saved), 2)
        self.assertEqual(sum(event['openie_metadata']['triples']['complete'] is True for event in saved), 2)
        self.assertTrue(all(set(event) == {'idx', 'passage', 'extracted_entities', 'extracted_triples',
                                          'openie_metadata', 'openie_responses'} for event in saved))
        self.assertEqual(len(extractor.calls), 4)

    def test_only_failed_triples_are_retried_and_prior_errors_remain(self):
        extractor = Extractor(failures={'bad': 2})
        _, outputs = run_openie_queue(extractor, self.chunks('good', 'bad'))
        self.assertTrue(outputs['bad'].metadata['complete'])
        self.assertEqual([call for call in extractor.calls if call[1] == 'good'],
                         [('ner', 'good', 1), ('triples', 'good', 1)])
        self.assertEqual([call for call in extractor.calls if call[0] == 'recover_triples'],
                         [('recover_triples', 'bad', 2), ('recover_triples', 'bad', 3)])
        history = outputs['bad'].metadata['build_queue_history']
        self.assertEqual([entry['round'] for entry in history], [1, 2, 3])
        self.assertEqual(history[0]['metadata']['error'], 'invalid response')
        self.assertNotIn('error', outputs['bad'].metadata)

    def test_old_success_is_audited_and_new_complete_contract_is_skipped(self):
        extractor, saved = Extractor(), []
        initial = [row('old', triples('old', contract='old')), row('new')]
        originals = copy.deepcopy(initial)
        _, outputs = run_openie_queue(extractor, self.chunks('old', 'new'), initial, saved.append)
        self.assertEqual(extractor.calls, [('audit', 'old', 1)])
        self.assertEqual(initial, originals)
        self.assertEqual([event['idx'] for event in saved], ['old'])
        self.assertEqual(outputs['old'].metadata['source_verified_schema'], 'new')
        self.assertEqual(outputs['old'].metadata['build_queue_history'][0]['metadata']
                         ['source_verified_schema'], 'old')

    def test_failed_ner_recovers_before_first_triple_request(self):
        extractor, saved = Extractor(ner_failures={'chunk': 1}), []
        _, outputs = run_openie_queue(extractor, self.chunks('chunk'), checkpoint=saved.append)
        self.assertTrue(outputs['chunk'].metadata['complete'])
        self.assertEqual(extractor.calls,
                         [('ner', 'chunk', 1), ('recover_ner', 'chunk', 2), ('triples', 'chunk', 1)])
        self.assertEqual(len(saved), 3)
        self.assertTrue(saved[0]['openie_metadata']['triples']['openie_skipped'])

    def test_transport_exception_is_checkpointed_and_other_chunks_complete(self):
        extractor = Extractor(throw={('triples', 'bad', 1): TimeoutError('server timed out')})
        saved = []
        _, outputs = run_openie_queue(extractor, self.chunks('bad', 'good'), checkpoint=saved.append)
        self.assertTrue(all(output.metadata['complete'] for output in outputs.values()))
        failed = [event for event in saved if event['openie_metadata']['triples'].get('build_queue_exception')]
        self.assertEqual(len(failed), 1)
        self.assertIn('TimeoutError: server timed out', failed[0]['openie_metadata']['triples']['error'])
        self.assertIn('TimeoutError', outputs['bad'].metadata['build_queue_history'][0]['metadata']['error'])

    def test_resume_only_retries_failed_ner_and_keeps_completed_triples(self):
        extractor = Extractor()
        initial = [row('chunk', ner_result=ner('chunk', 'old NER transport failure'))]
        _, outputs = run_openie_queue(extractor, self.chunks('chunk'), initial)
        self.assertEqual(extractor.calls, [('ner', 'chunk', 1)])
        self.assertEqual(outputs['chunk'].metadata['source_verified_schema'], 'new')
        self.assertTrue(outputs['chunk'].metadata['complete'])

    def test_three_round_limit_does_not_approve_unresolved_rows(self):
        extractor = Extractor(failures={'bad': 99}, ner_failures={'ner_bad': 99})
        ner_outputs, outputs = run_openie_queue(extractor, self.chunks('bad', 'ner_bad', 'good'))
        self.assertFalse(outputs['bad'].metadata['complete'])
        self.assertTrue(outputs['bad'].metadata['openie_skipped'])
        self.assertEqual(outputs['bad'].metadata['build_queue_round'], 3)
        self.assertFalse(outputs['ner_bad'].metadata['complete'])
        self.assertIn('error', ner_outputs['ner_bad'].metadata)
        self.assertTrue(outputs['good'].metadata['complete'])
        self.assertEqual(len([call for call in extractor.calls if call[1] == 'bad']), 4)
        self.assertFalse(any(call[0] in ('triples', 'recover_triples') and call[1] == 'ner_bad'
                             for call in extractor.calls))

    def test_unapproved_empty_result_and_invalid_relation_remain_pending(self):
        class InvalidExtractor(Extractor):
            def triple_extraction(self, key, passage, entities):
                return triples(key, values=[] if key == 'empty' else [['Iris', 'and', 'Lena']])

            def recover_pending_triples(self, key, passage, entities, previous, round_number):
                return self.triple_extraction(key, passage, entities)

        _, outputs = run_openie_queue(InvalidExtractor(), self.chunks('empty', 'invalid'))
        self.assertTrue(all(output.metadata['complete'] is False for output in outputs.values()))

    def test_legitimate_audited_empty_is_allowed(self):
        extractor = Extractor()
        initial = [row('heading', triples('heading', values=[], source_no_supported_relations=True))]
        _, outputs = run_openie_queue(extractor, self.chunks('heading'), initial)
        self.assertEqual(extractor.calls, [])
        self.assertEqual(outputs['heading'].triples, [])

    def test_resume_rejects_changed_source_and_duplicate_rows_before_requests(self):
        for initial in ([row('chunk', passage='different')], [row('chunk'), row('chunk')]):
            with self.subTest(initial=initial):
                extractor = Extractor()
                with self.assertRaises(ValueError):
                    run_openie_queue(extractor, self.chunks('chunk'), initial)
                self.assertEqual(extractor.calls, [])

    def test_checkpoint_io_failure_propagates(self):
        def failing_checkpoint(event):
            raise OSError('disk full')

        with self.assertRaisesRegex(OSError, 'disk full'):
            run_openie_queue(Extractor(), self.chunks('chunk'), checkpoint=failing_checkpoint)

    def test_requested_worker_limit_is_clamped_to_eight(self):
        extractor = Extractor(slow=True)
        keys = tuple(f'chunk-{index}' for index in range(24))
        run_openie_queue(extractor, self.chunks(*keys))
        self.assertLessEqual(extractor.maximum_active, 8)
        self.assertGreater(extractor.maximum_active, 1)

    def test_queue_uses_named_tqdm_bars_for_ner_and_triples(self):
        class RecordingProgress:
            calls = []

            def __init__(self, iterable, **kwargs):
                RecordingProgress.calls.append(kwargs)
                self.iterable = iterable

            def __iter__(self):
                return iter(self.iterable)

            def set_postfix(self, **kwargs):
                return None

        RecordingProgress.calls = []
        with patch('pathcondrag.index.openie_build_queue.tqdm', RecordingProgress):
            run_openie_queue(Extractor(), self.chunks('a', 'b'))
        self.assertEqual([call.get('desc') for call in RecordingProgress.calls],
                         ['NER', 'Extracting triples'])
        self.assertEqual([call.get('total') for call in RecordingProgress.calls], [2, 2])


if __name__ == '__main__':
    unittest.main()
