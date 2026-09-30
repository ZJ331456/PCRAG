import os
import re
import time
from copy import deepcopy
from typing import List, Optional

import httpx
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from openai import OpenAI
from openai import AzureOpenAI

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)


def _safe_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return value if value > 0 else default


def _safe_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    return value if value > 0 else default


def _extract_context_limit(error: Exception) -> Optional[int]:
    message = str(error)
    patterns = [
        r"context length is only\s+(\d+)",
        r"maximum input length of\s+(\d+)",
        r"maximum input length is\s+(\d+)",
        r"maximum context length is\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()
        self.embedding_max_input_tokens = _safe_env_int("HIPPO_EMBEDDING_MAX_INPUT_TOKENS", 0)
        self._token_encoder = None
        self._token_encoder_loaded = False
        self._truncation_warning_count = 0

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        connect_timeout = _safe_env_float("HIPPO_HTTP_CONNECT_TIMEOUT_SEC", 20.0)
        read_timeout = _safe_env_float("HIPPO_HTTP_READ_TIMEOUT_SEC", 300.0)
        write_timeout = _safe_env_float("HIPPO_HTTP_WRITE_TIMEOUT_SEC", read_timeout)
        pool_timeout = _safe_env_float("HIPPO_HTTP_POOL_TIMEOUT_SEC", 30.0)
        max_retries = _safe_env_int("HIPPO_OPENAI_MAX_RETRIES", 2)
        max_connections = _safe_env_int("HIPPO_HTTP_MAX_CONNECTIONS", 500)
        max_keepalive = min(_safe_env_int("HIPPO_HTTP_MAX_KEEPALIVE", 100), max_connections)
        http_client = httpx.Client(
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive),
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            ),
        )

        if self.global_config.azure_embedding_endpoint is None:
            self.client = OpenAI(
                base_url=self.global_config.embedding_base_url,
                http_client=http_client,
                max_retries=max_retries,
            )
        else:
            self.client = AzureOpenAI(
                api_version=self.global_config.azure_embedding_endpoint.split('api-version=')[1],
                azure_endpoint=self.global_config.azure_embedding_endpoint,
                http_client=http_client,
                max_retries=max_retries,
            )


    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                # "torch_dtype": "auto",
                'device_map': "auto",  # added this line to use multiple GPUs
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def _get_token_encoder(self):
        if self._token_encoder_loaded:
            return self._token_encoder
        self._token_encoder_loaded = True
        encoding_name = os.environ.get("HIPPO_EMBEDDING_TIKTOKEN_ENCODING", "cl100k_base")
        try:
            import tiktoken
            self._token_encoder = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            self._token_encoder = None
            logger.warning(
                f"Could not load tiktoken encoding '{encoding_name}' for embedding truncation; "
                f"falling back to character truncation: {e}"
            )
        return self._token_encoder

    def _clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        return text if text != "" else " "

    def _truncate_texts(self, texts: List[str], token_limit: int) -> List[str]:
        if token_limit <= 0:
            return texts

        encoder = self._get_token_encoder()
        truncated = 0
        if encoder is not None:
            output = []
            for text in texts:
                tokens = encoder.encode(text)
                if len(tokens) > token_limit:
                    text = encoder.decode(tokens[:token_limit])
                    truncated += 1
                output.append(text)
        else:
            char_limit = _safe_env_int("HIPPO_EMBEDDING_MAX_INPUT_CHARS", max(1, token_limit * 3))
            output = []
            for text in texts:
                if len(text) > char_limit:
                    text = text[:char_limit]
                    truncated += 1
                output.append(text)

        if truncated and self._truncation_warning_count < 3:
            logger.warning(f"Truncated {truncated} embedding input(s) to <= {token_limit} tokens")
            self._truncation_warning_count += 1
        return output

    def _create_embeddings(self, texts: List[str]):
        response = self.client.embeddings.create(input=texts, model=self.embedding_model_name)
        return np.array([v.embedding for v in response.data])

    def encode(self, texts: List[str]):
        texts = [self._clean_text(t) for t in texts]
        request_texts = self._truncate_texts(texts, self.embedding_max_input_tokens)
        try:
            return self._create_embeddings(request_texts)
        except Exception as e:
            context_limit = _extract_context_limit(e)
            if context_limit is None:
                raise

            retry_limits = []
            if self.embedding_max_input_tokens > 0:
                retry_limits.extend([
                    int(self.embedding_max_input_tokens * 0.9),
                    int(self.embedding_max_input_tokens * 0.75),
                ])
            retry_limits.extend([
                context_limit - 32,
                int(context_limit * 0.85),
                int(context_limit * 0.70),
            ])

            last_error = e
            seen = set()
            for limit in retry_limits:
                limit = max(1, int(limit))
                if limit in seen:
                    continue
                seen.add(limit)
                try:
                    logger.warning(
                        f"Retrying embedding request with inputs truncated to <= {limit} tokens "
                        f"after context-length error: {e}"
                    )
                    return self._create_embeddings(self._truncate_texts(texts, limit))
                except Exception as retry_error:
                    last_error = retry_error
            raise last_error

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
            # del params["instruction"]

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")

        batch_size = params.pop("batch_size", 16)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            max_attempts = _safe_env_int("HIPPO_EMBED_BATCH_RETRIES", 3)
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                last_error = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        results.append(self.encode(batch))
                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < max_attempts:
                            logger.warning(f"Embedding batch {i // batch_size + 1} failed (attempt {attempt}/{max_attempts}): {e}")
                            time.sleep(min(2, attempt))
                if last_error is not None:
                    raise RuntimeError(f"Embedding batch failed after {max_attempts} attempts: {last_error}") from last_error
                pbar.update(batch_size)
            pbar.close()
            results = np.concatenate(results)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
