from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction


class TransformersEmbeddingModel(BaseEmbeddingModel):
    """
    In-process SentenceTransformer backend (Qwen3-Embedding etc.).

    Select via embedding_model_name starting with ``Transformers/``, e.g.:
      Transformers//root/models/Qwen3-Embedding-8B
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        raw = embedding_model_name or self.embedding_model_name
        self.model_id = raw.removeprefix("Transformers/").removeprefix("transformers/")
        self.embedding_type = "float"
        self.batch_size = int(getattr(self.global_config, "embedding_batch_size", 2) or 2)
        self.normalize = bool(getattr(self.global_config, "embedding_return_as_normalized", True))
        max_seq = int(getattr(self.global_config, "embedding_max_seq_len", 2048) or 2048)

        dtype_name = str(getattr(self.global_config, "embedding_model_dtype", "auto") or "auto")
        if dtype_name in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16
        elif dtype_name in ("float16", "fp16"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            self.model_id,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch_dtype},
        )
        self.model.max_seq_length = max_seq

        self.search_query_instr = set(
            [
                get_query_instruction("query_to_fact"),
                get_query_instruction("query_to_passage"),
            ]
        )
        self._is_qwen3_embedding = "qwen3-embedding" in self.model_id.lower()

    def _encode(self, texts: List[str], instruction: str = "") -> np.ndarray:
        encode_kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize,
            "show_progress_bar": False,
        }
        # Qwen3-Embedding is instruction-aware: inject Hippo's task prompt
        # (query_to_fact / query_to_passage / ...), do NOT use the fixed
        # prompt_name="query" web-search template.
        if instruction:
            if self._is_qwen3_embedding:
                encode_kwargs["prompt"] = f"Instruct: {instruction}\nQuery:"
            else:
                texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
        return np.asarray(self.model.encode(texts, **encode_kwargs))

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        instruction = kwargs.get("instruction", "") or ""
        if len(texts) <= self.batch_size:
            return self._encode(texts, instruction=instruction)

        results = []
        batch_indexes = list(range(0, len(texts), self.batch_size))
        for i in tqdm(batch_indexes, desc="Batch Encoding"):
            results.append(self._encode(texts[i : i + self.batch_size], instruction=instruction))
        return np.concatenate(results, axis=0)
