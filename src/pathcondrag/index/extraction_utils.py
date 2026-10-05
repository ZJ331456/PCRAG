"""Shared parsing, worker configuration and retry settings for extraction."""

import json
import os
from typing import List

from ..llm.openai_gpt import CacheOpenAI


_LENGTH_RETRY_FREQUENCY_PENALTIES = (0.2, 0.5)



_DEFAULT_QUALITY_MAX_RETRIES = 5



def _safe_env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        val = int(raw)
        if val <= 0:
            return default
        return val
    except Exception:
        return default



def _extract_json_list_field(response: str, field_name: str) -> List:
    decoder = json.JSONDecoder()
    for start_index, character in enumerate(response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response[start_index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or field_name not in payload:
            continue
        value = payload[field_name]
        if not isinstance(value, list):
            raise ValueError(f"OpenIE response field {field_name!r} must be a list.")
        return value
    raise ValueError(f"OpenIE response does not contain a valid JSON object with {field_name!r}.")



def _extract_ner_from_response(real_response):
    return _extract_json_list_field(real_response, "named_entities")



def _base_generate_seed(llm_model: CacheOpenAI) -> int:
    config = getattr(llm_model, "llm_config", None)
    params = getattr(config, "generate_params", {}) or {}
    base_seed = params.get("seed")
    if base_seed is None:
        base_seed = 0
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError(f"OpenIE retry requires an integer or null seed, got {base_seed!r}")
    return base_seed



def _length_retry_seed(llm_model: CacheOpenAI) -> int:
    """Change the cache key once without changing the decoding settings."""
    return _base_generate_seed(llm_model) + 1



def _resolve_quality_max_retries(explicit):
    if explicit is not None:
        if explicit < 0:
            raise ValueError("quality_max_retries cannot be negative.")
        return explicit
    raw = os.environ.get("HIPPO_OPENIE_QUALITY_MAX_RETRIES", "").strip()
    if raw:
        value = int(raw)
        if value < 0:
            raise ValueError("HIPPO_OPENIE_QUALITY_MAX_RETRIES cannot be negative.")
        return value
    return _DEFAULT_QUALITY_MAX_RETRIES
