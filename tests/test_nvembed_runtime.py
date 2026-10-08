"""CPU checks for the OOM retry boundary and unchanged embedding semantics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'src/pathcondrag/embedding_model/nvembed_runtime.py'
SPEC = importlib.util.spec_from_file_location('nvembed_runtime_test_target', SOURCE)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class FakeEncoder:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.calls = []
        self.embedding_model = SimpleNamespace(config=SimpleNamespace(use_cache=True))

    def encode(self, prompts, **params):
        self.calls.append((list(prompts), dict(params)))
        if len(prompts) > self.capacity:
            raise torch.cuda.OutOfMemoryError('simulated oversized original batch')
        return torch.tensor([[float(text), float(text) + 1] for text in prompts])


def holder(capacity=100, norm=False):
    return SimpleNamespace(
        embedding_model=FakeEncoder(capacity), embedding_dim=2,
        embedding_config=SimpleNamespace(
            norm=norm, encode_params={'instruction': '', 'max_length': 2048,
                                      'batch_size': 4, 'num_workers': 32}))


class NVEmbedRuntimeTests(unittest.TestCase):
    def test_retry_preserves_all_original_rows_order_and_parameters(self):
        model = holder(capacity=2)
        with patch.object(runtime.torch.cuda, 'empty_cache') as empty_cache:
            actual = runtime.memory_bounded_batch_encode(
                model, ['1', '2', '3', '4', '5'], instruction='Find supporting passages')
        np.testing.assert_array_equal(actual, [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.assertEqual(model._nvembed_execution_stats['cuda_oom_splits'], 1)
        self.assertEqual(model._nvembed_execution_stats['encoded_texts'], 5)
        empty_cache.assert_called_once()
        self.assertFalse(model.embedding_model.embedding_model.config.use_cache)
        for _, params in model.embedding_model.calls:
            self.assertEqual(params['instruction'], 'Instruct: Find supporting passages\nQuery: ')
            self.assertEqual(params['max_length'], 2048)
        self.assertEqual(model.embedding_config.encode_params['instruction'], '')

    def test_single_passage_oom_is_a_real_failure_not_empty_or_truncated_output(self):
        model = holder(capacity=0)
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            runtime.memory_bounded_batch_encode(model, ['1'])
        self.assertEqual(len(model.embedding_model.calls), 1)
        self.assertEqual(model.embedding_model.calls[0][1]['max_length'], 2048)

    def test_normalized_output_and_empty_instruction_match_native_contract(self):
        model = holder(norm=True)
        actual = runtime.memory_bounded_batch_encode(model, '3', instruction='')
        np.testing.assert_array_equal(actual, np.array([[0.6, 0.8]], dtype=np.float32))
        self.assertEqual(model.embedding_model.calls[0][1]['instruction'], '')

    def test_read_only_baseline_patch_is_restored_even_when_execution_fails(self):
        class BaselineStub:
            def batch_encode(self, texts):
                return texts
        original = BaselineStub.batch_encode
        with patch.object(runtime.importlib, 'import_module', return_value=SimpleNamespace(
                NVEmbedV2EmbeddingModel=BaselineStub)):
            with self.assertRaisesRegex(RuntimeError, 'execution failed'):
                with runtime.baseline_nvembed_runtime():
                    self.assertIs(BaselineStub.batch_encode, runtime.memory_bounded_batch_encode)
                    raise RuntimeError('execution failed')
        self.assertIs(BaselineStub.batch_encode, original)


if __name__ == '__main__':
    unittest.main()
