"""Memory-bounded NV-Embed inference shared by both comparison methods.

Only execution/storage changes: keep each encoded batch on CPU, disable the
unused generation KV cache, and split a batch if CUDA runs out of memory.
Text, token limit, instructions, dtype and normalization retain native values.
"""

from contextlib import contextmanager
from copy import deepcopy
import gc
import importlib
import logging

import numpy as np
import torch
from tqdm import tqdm


LOG = logging.getLogger(__name__)
RUNTIME_VERSION = 'nvembed_cpu_batch_storage_oom_split_v1'


def _encode_cpu(model, texts, params, stats):
    """Return a complete batch, or retry smaller batches after releasing OOM."""
    failed = False
    try:
        output = model.encode(prompts=texts, **params)
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu()
        else:
            output = torch.as_tensor(np.asarray(output))
        stats['successful_microbatches'] += 1
        stats['smallest_successful_microbatch'] = min(
            stats['smallest_successful_microbatch'], len(texts))
        return output
    except torch.cuda.OutOfMemoryError:
        if len(texts) == 1:
            # A single original passage remains a real failure. Do not shorten
            # its token limit or substitute an empty vector to hide the error.
            raise
        failed = True
        stats['cuda_oom_splits'] += 1
    # Leave the exception scope before collecting its failed forward frames.
    if failed:
        gc.collect()
        torch.cuda.empty_cache()
        middle = len(texts) // 2
        LOG.warning('NV embedding CUDA OOM: retrying the same %d texts as %d+%d; '
                    'max_length=%s and instruction unchanged',
                    len(texts), middle, len(texts) - middle, params.get('max_length'))
        left = _encode_cpu(model, texts[:middle], params, stats)
        right = _encode_cpu(model, texts[middle:], params, stats)
        return torch.cat((left, right), dim=0)


def memory_bounded_batch_encode(self, texts, **kwargs):
    """Native batch_encode contract without retaining whole-corpus GPU tensors."""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    params = deepcopy(self.embedding_config.encode_params)
    params.update(kwargs)
    if 'instruction' in kwargs and kwargs['instruction'] != '':
        params['instruction'] = f"Instruct: {kwargs['instruction']}\nQuery: "
    batch_size = int(params.pop('batch_size', 16))
    if batch_size < 1:
        raise ValueError('NV embedding batch_size must be at least 1.')
    if not texts:
        return np.empty((0, self.embedding_dim), dtype=np.float32)

    model = self.embedding_model
    # NV calls its bidirectional text encoder once, without past_key_values.
    # Generating and returning those caches wastes memory and is never read by
    # the latent pooling layer. This leaves attention/token representations
    # unchanged and does not change LLM generation/cache settings.
    encoder = getattr(model, 'embedding_model', None)
    encoder_config = getattr(encoder, 'config', None)
    if encoder_config is not None:
        encoder_config.use_cache = False
    stats = getattr(self, '_nvembed_execution_stats', None)
    if stats is None:
        stats = self._nvembed_execution_stats = {
            'version': RUNTIME_VERSION,
            'output_batch_device': 'cpu',
            'encoder_use_cache': False,
            'cuda_oom_splits': 0,
            'successful_microbatches': 0,
            'smallest_successful_microbatch': batch_size,
            'encoded_texts': 0,
        }
        LOG.info('NV embedding runtime=%s; batch=%d; max_length=%s; '
                 'CPU batch storage and unused encoder KV cache disabled',
                 RUNTIME_VERSION, batch_size, params.get('max_length'))
    chunks = []
    with torch.no_grad(), tqdm(total=len(texts), desc='Batch Encoding',
                               disable=len(texts) <= batch_size) as progress:
        for start in range(0, len(texts), batch_size):
            current = texts[start:start + batch_size]
            chunks.append(_encode_cpu(model, current, params, stats))
            stats['encoded_texts'] += len(current)
            progress.update(len(current))
    results = torch.cat(chunks, dim=0).numpy()
    if self.embedding_config.norm:
        results = (results.T / np.linalg.norm(results, axis=1)).T
    return results


@contextmanager
def baseline_nvembed_runtime():
    """Patch only this process's NV encoder; never edit the baseline checkout."""
    module = importlib.import_module('hipporag.embedding_model.NVEmbedV2')
    model_class = module.NVEmbedV2EmbeddingModel
    original = model_class.batch_encode
    model_class.batch_encode = memory_bounded_batch_encode
    try:
        yield
    finally:
        model_class.batch_encode = original
