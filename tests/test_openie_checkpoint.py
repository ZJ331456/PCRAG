"""Check persisted stage progress, replay and source/configuration integrity."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathcondrag.utils.openie_checkpoint import OpenIECheckpoint


class OpenIECheckpointTests(unittest.TestCase):
    def test_stage_commit_survives_reopen_and_replaces_only_matching_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'progress.sqlite'
            progress = OpenIECheckpoint(path, {'quality': 'strict', 'corpus': 'same'})
            row = {'idx': 'chunk-a', 'passage': 'Original source.', 'openie_metadata': {
                'ner': {'finish_reason': 'stop'}, 'triples': {'complete': False}}}
            progress.save(row)
            progress.close()
            reloaded = OpenIECheckpoint(path, {'corpus': 'same', 'quality': 'strict'})
            restored = reloaded.overlay([], {'chunk-a': {'content': 'Original source.'}})
            self.assertEqual(restored, [row])
            row['openie_metadata']['triples']['complete'] = True
            reloaded.save(row)
            self.assertTrue(reloaded.overlay([], {'chunk-a': {'content': 'Original source.'}})
                            [0]['openie_metadata']['triples']['complete'])
            reloaded.close()

    def test_other_contract_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'progress.sqlite'
            progress = OpenIECheckpoint(path, {'quality': 'v2'})
            progress.close()
            with self.assertRaisesRegex(RuntimeError, 'another corpus/producer/quality'):
                OpenIECheckpoint(path, {'quality': 'v3'})

    def test_changed_source_and_extraneous_progress_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            progress = OpenIECheckpoint(Path(temporary) / 'progress.sqlite', {})
            progress.save({'idx': 'chunk-a', 'passage': 'Original source.'})
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                progress.overlay([], {'chunk-a': {'content': 'Different source.'}})
            with self.assertRaisesRegex(RuntimeError, 'out-of-corpus'):
                progress.overlay([], {'chunk-b': {'content': 'Original source.'}})
            progress.close()

    def test_failed_stage_persisted_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            progress = OpenIECheckpoint(Path(temporary) / 'progress.sqlite', {})
            progress.save({'idx': 'a', 'passage': 'Source', 'extracted_triples': [],
                           'openie_metadata': {'triples': {'complete': False,
                                                        'quality_status': 'pending'}}})
            restored = progress.overlay([], {'a': {'content': 'Source'}})[0]
            self.assertEqual(restored['openie_metadata']['triples']['quality_status'], 'pending')
            progress.close()


if __name__ == '__main__':
    unittest.main()
