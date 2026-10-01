import functools
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

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


# This bound is shared by every CacheOpenAI instance in this process. Waiting and
# retry backoff happen outside the semaphore, so only active HTTP calls occupy it.
LLM_MAX_IN_FLIGHT = _safe_env_int("PATHCONDRAG_LLM_MAX_IN_FLIGHT", 8)
_LLM_HTTP_SEMAPHORE = threading.BoundedSemaphore(LLM_MAX_IN_FLIGHT)


def _effective_generation_params(self, kwargs):
    """Build the same generation parameters for both the request and cache key."""
    params = deepcopy(self.llm_config.generate_params)
    params.update(deepcopy(kwargs))
    params.pop("messages", None)
    model_name = str(params.get("model") or "").lower()
    configured_name = str(getattr(self, "llm_name", "") or "").lower()
    if "qwen3" in model_name or "qwen3" in configured_name:
        # A caller override must not accidentally switch Qwen3 back to thinking.
        extra_body = dict(params.get("extra_body") or {})
        chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_kwargs
        params["extra_body"] = extra_body
    return params


def _is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        return status in (408, 409, 429) or status >= 500
    return False


def _retry_after_seconds(exc: Exception):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(header)
        if value is None:
            continue
        try:
            return max(0.0, float(value) * scale)
        except (TypeError, ValueError):
            if header == "retry-after":
                try:
                    when = parsedate_to_datetime(value)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
    return None


def _increment_stat(self, name: str):
    lock = getattr(self, "_llm_stats_lock", None)
    if lock is not None:
        with lock:
            setattr(self, name, getattr(self, name, 0) + 1)


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # Include every effective generation setting, including Qwen3 thinking mode,
        # rather than just temperature and length. The version prevents collisions
        # with old keys that did not record the chat template settings.
        generation_params = _effective_generation_params(self, kwargs)
        key_data = {
            "cache_key_version": 2,
            "messages": messages,
            "generation": generation_params,
            "base_url": getattr(self, "llm_base_url", None),
        }
        key_hash = hashlib.sha256(
            json.dumps(key_data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        # Existing non-Qwen caches remain usable. Qwen3 legacy entries cannot
        # establish whether thinking was enabled, so never read them.
        legacy_hashes = []
        model_name = str(generation_params.get("model") or "").lower()
        configured_name = str(getattr(self, "llm_name", "") or "").lower()
        old_key_fields = {"model", "seed", "temperature", "max_completion_tokens",
                          "max_tokens", "max_new_tokens", "n"}
        legacy_compatible = (set(generation_params).issubset(old_key_fields)
                             and generation_params.get("n", 1) == 1)
        if "qwen3" not in model_name and "qwen3" not in configured_name and legacy_compatible:
            gen_params = self.llm_config.generate_params
            model = kwargs.get("model", gen_params.get("model"))
            seed = kwargs.get("seed", gen_params.get("seed"))
            temperature = kwargs.get("temperature", gen_params.get("temperature"))
            max_tokens = (
                kwargs.get("max_completion_tokens")
                or kwargs.get("max_tokens")
                or kwargs.get("max_new_tokens")
                or gen_params.get("max_completion_tokens")
                or gen_params.get("max_tokens")
                or gen_params.get("max_new_tokens")
            )
            for legacy_key_data in (
                {"messages": messages, "model": model, "seed": seed,
                 "temperature": temperature, "max_tokens": max_tokens},
                {"messages": messages, "model": model, "seed": seed,
                 "temperature": temperature},
            ):
                legacy_hashes.append(hashlib.sha256(
                    json.dumps(legacy_key_data, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest())

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            if row is None:
                for legacy_idx, legacy_hash in enumerate(legacy_hashes):
                    c.execute("SELECT message, metadata FROM cache WHERE key = ?", (legacy_hash,))
                    row = c.fetchone()
                    if row is not None and legacy_idx == 1:
                        # The oldest key omitted the token budget. Its truncated
                        # response must not satisfy a larger-budget retry.
                        _msg, _meta_str = row
                        if json.loads(_meta_str).get("finish_reason") == "length":
                            row = None
                    if row is not None:
                        break
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                _increment_stat(self, "llm_cache_hit_count")
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        _increment_stat(self, "llm_cache_miss_count")
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_attempts = max(1, int(getattr(self, "max_retries", 5)))
        for attempt in range(1, max_attempts + 1):
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:
                if not _is_retryable_llm_error(exc) or attempt == max_attempts:
                    _increment_stat(self, "llm_failure_count")
                    logger.error("LLM request failed after %d attempt(s): %s", attempt, exc)
                    raise
                _increment_stat(self, "llm_retry_count")
                backoff = min(16.0, 2.0 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                retry_after = _retry_after_seconds(exc)
                delay = max(backoff, retry_after or 0.0)
                logger.warning(
                    "Retryable LLM request error (attempt %d/%d, sleep %.2fs): %s",
                    attempt, max_attempts, delay, exc,
                )
                time.sleep(delay)
    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        self._llm_stats_lock = threading.Lock()
        self.llm_cache_hit_count = 0
        self.llm_cache_miss_count = 0
        self.llm_http_attempt_count = 0
        self.llm_retry_count = 0
        self.llm_failure_count = 0

        self._init_llm_config()
        if high_throughput:
            max_connections = _safe_env_int("HIPPO_HTTP_MAX_CONNECTIONS", 500)
            max_keepalive = min(_safe_env_int("HIPPO_HTTP_MAX_KEEPALIVE", 100), max_connections)
            connect_timeout = _safe_env_float("HIPPO_HTTP_CONNECT_TIMEOUT_SEC", 20.0)
            read_timeout = _safe_env_float("HIPPO_HTTP_READ_TIMEOUT_SEC", 300.0)
            write_timeout = _safe_env_float("HIPPO_HTTP_WRITE_TIMEOUT_SEC", read_timeout)
            pool_timeout = _safe_env_float("HIPPO_HTTP_POOL_TIMEOUT_SEC", 30.0)
            limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
            client = httpx.Client(
                limits=limits,
                timeout=httpx.Timeout(
                    connect=connect_timeout,
                    read=read_timeout,
                    write=write_timeout,
                    pool=pool_timeout,
                ),
            )
        else:
            client = None

        # max_retries is the total attempt budget for our one retry layer.
        self.max_retries = kwargs.get("max_retries", _safe_env_int("HIPPO_OPENAI_MAX_RETRIES", 2))

        if self.global_config.azure_endpoint is None:
            self.openai_client = OpenAI(base_url=self.llm_base_url, http_client=client, max_retries=0)
        else:
            self.openai_client = AzureOpenAI(api_version=self.global_config.azure_endpoint.split('api-version=')[1],
                                             azure_endpoint=self.global_config.azure_endpoint, max_retries=0)

    def get_request_stats(self):
        """Return a consistent snapshot for fail-fast checks and benchmarking."""
        with self._llm_stats_lock:
            return {
                "cache_hits": self.llm_cache_hit_count,
                "cache_misses": self.llm_cache_miss_count,
                "http_attempts": self.llm_http_attempt_count,
                "retries": self.llm_retry_count,
                "failures": self.llm_failure_count,
                "max_in_flight": LLM_MAX_IN_FLIGHT,
            }

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url
        generate_params = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }
        # Qwen3 defaults to thinking mode; disable for OpenIE/QA JSON reliability & latency.
        if "qwen3" in (self.global_config.llm_name or "").lower():
            extra_body = dict(generate_params.get("extra_body") or {})
            chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = chat_kwargs
            generate_params["extra_body"] = extra_body
        config_dict['generate_params'] = generate_params

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = _effective_generation_params(self, kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')

        with _LLM_HTTP_SEMAPHORE:
            _increment_stat(self, "llm_http_attempt_count")
            response = self.openai_client.chat.completions.create(**params)

        message_obj = response.choices[0].message
        response_message = ""

        # OpenAI-compatible content field.
        raw_content = getattr(message_obj, "content", None)
        if isinstance(raw_content, str):
            response_message = raw_content
        elif isinstance(raw_content, list):
            # Some OpenAI-compatible providers return segmented content.
            parts: List[str] = []
            for part in raw_content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text_val = part.get("text", None)
                    if isinstance(text_val, str):
                        parts.append(text_val)
                    elif part.get("type") == "text":
                        content_val = part.get("content", None)
                        if isinstance(content_val, str):
                            parts.append(content_val)
                else:
                    text_attr = getattr(part, "text", None)
                    if isinstance(text_attr, str):
                        parts.append(text_attr)
            response_message = "".join(parts).strip()

        # Some models (e.g. reasoning parser backends) may place text in reasoning fields.
        if not isinstance(response_message, str) or len(response_message.strip()) == 0:
            reasoning_val = getattr(message_obj, "reasoning_content", None)
            if reasoning_val is None:
                reasoning_val = getattr(message_obj, "reasoning", None)
            if isinstance(reasoning_val, str):
                response_message = reasoning_val

        if not isinstance(response_message, str):
            try:
                response_message = str(response_message)
            except Exception:
                response_message = ""
        response_message = response_message if response_message is not None else ""
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata
