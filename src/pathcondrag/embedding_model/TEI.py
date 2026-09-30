"""
Hugging Face Text Embeddings Inference (TEI) client for HippoRAG / PCRAG.

Select this backend by setting embedding_model_name with a ``TEI/`` prefix, e.g.:
    embedding_model_name="TEI/e5-base-v2"
    embedding_base_url="http://127.0.0.1:8036/v1"   # OpenAI-compatible
    # or
    embedding_base_url="http://127.0.0.1:8036"      # native TEI /embed

Env overrides:
    HIPPO_TEI_API_MODE=openai|native   (default: auto from base_url)
    HIPPO_EMBEDDING_MAX_INPUT_TOKENS
    HIPPO_EMBED_BATCH_RETRIES
"""

from __future__ import annotations

import os
import re
import time
from copy import deepcopy
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
import numpy as np
import torch
from openai import OpenAI
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

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


def _strip_tei_prefix(embedding_model_name: str) -> str:
    name = embedding_model_name.strip()
    for prefix in ("TEI/", "tei/", "TEI:", "tei:"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _normalize_base_url(url: Optional[str]) -> str:
    if not url:
        return "http://127.0.0.1:8036"
    return url.rstrip("/")


def _detect_api_mode(base_url: str) -> str:
    forced = os.environ.get("HIPPO_TEI_API_MODE", "").strip().lower()
    if forced in {"openai", "native"}:
        return forced
    path = urlparse(base_url).path.rstrip("/")
    if path.endswith("/v1") or path.endswith("/embeddings"):
        return "openai"
    return "native"


class TEIEmbeddingModel(BaseEmbeddingModel):
    """HTTP client for TEI (/v1/embeddings or native /embed)."""

    def __init__(
        self,
        global_config: Optional[BaseConfig] = None,
        embedding_model_name: Optional[str] = None,
    ) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name

        self.served_model_name = _strip_tei_prefix(self.embedding_model_name)
        self.base_url = _normalize_base_url(self.global_config.embedding_base_url)
        self.api_mode = _detect_api_mode(self.base_url)

        self._init_embedding_config()
        self.embedding_max_input_tokens = _safe_env_int("HIPPO_EMBEDDING_MAX_INPUT_TOKENS", 0)
        self._token_encoder = None
        self._token_encoder_loaded = False
        self._truncation_warning_count = 0

        connect_timeout = _safe_env_float("HIPPO_HTTP_CONNECT_TIMEOUT_SEC", 20.0)
        read_timeout = _safe_env_float("HIPPO_HTTP_READ_TIMEOUT_SEC", 300.0)
        write_timeout = _safe_env_float("HIPPO_HTTP_WRITE_TIMEOUT_SEC", read_timeout)
        pool_timeout = _safe_env_float("HIPPO_HTTP_POOL_TIMEOUT_SEC", 30.0)
        max_retries = _safe_env_int("HIPPO_OPENAI_MAX_RETRIES", 2)
        max_connections = _safe_env_int("HIPPO_HTTP_MAX_CONNECTIONS", 500)
        max_keepalive = min(_safe_env_int("HIPPO_HTTP_MAX_KEEPALIVE", 100), max_connections)

        self.http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
            ),
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            ),
        )

        self.openai_client = None
        if self.api_mode == "openai":
            openai_base = self.base_url
            if not openai_base.endswith("/v1"):
                openai_base = f"{openai_base}/v1"
            self.openai_client = OpenAI(
                base_url=openai_base,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                http_client=self.http_client,
                max_retries=max_retries,
            )

        logger.info(
            f"TEIEmbeddingModel ready: served={self.served_model_name} "
            f"base_url={self.base_url} api_mode={self.api_mode}"
        )

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.served_model_name,
                "trust_remote_code": True,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)

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
                f"Could not load tiktoken encoding '{encoding_name}' for TEI truncation; "
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
            logger.warning(f"Truncated {truncated} TEI embedding input(s) to <= {token_limit} tokens")
            self._truncation_warning_count += 1
        return output

    def _create_embeddings_openai(self, texts: List[str]) -> np.ndarray:
        assert self.openai_client is not None
        response = self.openai_client.embeddings.create(
            input=texts,
            model=self.served_model_name,
        )
        return np.array([v.embedding for v in response.data], dtype=np.float32)

    def _create_embeddings_native(self, texts: List[str]) -> np.ndarray:
        # Native TEI: POST /embed  {"inputs": [...]}
        url = self.base_url
        if url.endswith("/v1"):
            url = url[: -len("/v1")]
        embed_url = urljoin(url + "/", "embed")
        resp = self.http_client.post(embed_url, json={"inputs": texts})
        resp.raise_for_status()
        data = resp.json()
        # TEI returns a list of vectors, or {"embeddings": [...]} in some versions.
        if isinstance(data, dict) and "embeddings" in data:
            vectors = data["embeddings"]
        else:
            vectors = data
        return np.array(vectors, dtype=np.float32)

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.api_mode == "openai":
            return self._create_embeddings_openai(texts)
        return self._create_embeddings_native(texts)

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
                retry_limits.extend(
                    [
                        int(self.embedding_max_input_tokens * 0.9),
                        int(self.embedding_max_input_tokens * 0.75),
                    ]
                )
            retry_limits.extend(
                [
                    context_limit - 32,
                    int(context_limit * 0.85),
                    int(context_limit * 0.70),
                ]
            )

            last_error = e
            seen = set()
            for limit in retry_limits:
                limit = max(1, int(limit))
                if limit in seen:
                    continue
                seen.add(limit)
                try:
                    logger.warning(
                        f"Retrying TEI embedding with inputs truncated to <= {limit} tokens "
                        f"after context-length error: {e}"
                    )
                    return self._create_embeddings(self._truncate_texts(texts, limit))
                except Exception as retry_error:
                    last_error = retry_error
            raise last_error

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        instruction = params.get("instruction", "")
        if kwargs.get("instruction"):
            instruction = kwargs["instruction"]
        if instruction:
            # Keep parity with OpenAI adapter instruction formatting used by HippoRAG.
            prefix = f"Instruct: {instruction}\nQuery: "
            texts = [prefix + t for t in texts]

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")

        batch_size = params.pop("batch_size", 16)
        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="TEI Batch Encoding")
            results = []
            max_attempts = _safe_env_int("HIPPO_EMBED_BATCH_RETRIES", 3)
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                last_error = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        results.append(self.encode(batch))
                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < max_attempts:
                            logger.warning(
                                f"TEI embedding batch {i // batch_size + 1} failed "
                                f"(attempt {attempt}/{max_attempts}): {e}"
                            )
                            time.sleep(min(2, attempt))
                if last_error is not None:
                    raise RuntimeError(
                        f"TEI embedding batch failed after {max_attempts} attempts: {last_error}"
                    ) from last_error
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(results)

        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T
        return results
