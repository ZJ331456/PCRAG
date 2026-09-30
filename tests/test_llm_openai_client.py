"""Small offline checks for the PathCondRAG OpenAI-compatible LLM client."""

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathcondrag.llm import openai_gpt as client_module  # noqa: E402


class FakeLLM:
    def __init__(self, cache_file_name, llm_name="qwen3-8b"):
        self.cache_file_name = cache_file_name
        self.llm_name = llm_name
        self.llm_base_url = "http://localhost:8035/v1"
        self.llm_config = SimpleNamespace(generate_params={
            "model": self.llm_name,
            "temperature": 0,
            "max_completion_tokens": 2048,
            "n": 1,
        })
        if "qwen3" in self.llm_name:
            self.llm_config.generate_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        self.calls = 0

    @client_module.cache_response
    def infer(self, messages, **kwargs):
        self.calls += 1
        return f"response {self.calls}", {"finish_reason": "stop"}


def status_error(status, headers=None):
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "http://localhost:8035/v1/chat/completions"),
        headers=headers,
    )
    return openai.APIStatusError("temporary failure", response=response, body=None)


class CacheKeyTests(unittest.TestCase):
    def test_qwen3_ignores_old_cache_without_thinking_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeLLM(os.path.join(tmp, "llm.sqlite"))
            messages = [{"role": "user", "content": "question"}]
            old_data = {"messages": messages, "model": "qwen3-8b", "seed": None,
                        "temperature": 0, "max_tokens": 2048}
            old_key = hashlib.sha256(json.dumps(old_data, sort_keys=True).encode()).hexdigest()
            with sqlite3.connect(model.cache_file_name) as conn:
                conn.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, message TEXT, metadata TEXT)")
                conn.execute("INSERT INTO cache VALUES (?, ?, ?)",
                             (old_key, "old thinking response", '{"finish_reason":"stop"}'))

            first = model.infer(messages)
            second = model.infer(messages)
            self.assertEqual(first[0], "response 1")
            self.assertFalse(first[2])
            self.assertEqual(second[0], "response 1")
            self.assertTrue(second[2])
            self.assertEqual(model.calls, 1)

    def test_all_generation_settings_participate_in_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeLLM(os.path.join(tmp, "llm.sqlite"))
            messages = [{"role": "user", "content": "question"}]
            self.assertFalse(model.infer(messages, top_p=0.9)[2])
            self.assertTrue(model.infer(messages, top_p=0.9)[2])
            self.assertFalse(model.infer(messages, top_p=0.8)[2])
            self.assertFalse(model.infer(messages, max_completion_tokens=512)[2])
            self.assertFalse(model.infer(messages, stop=["END"])[2])
            self.assertEqual(model.calls, 4)

    def test_non_qwen_old_cache_is_used_only_for_compatible_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeLLM(os.path.join(tmp, "llm.sqlite"), llm_name="older-model")
            messages = [{"role": "user", "content": "question"}]
            old_data = {"messages": messages, "model": "older-model", "seed": None,
                        "temperature": 0, "max_tokens": 2048}
            old_key = hashlib.sha256(json.dumps(old_data, sort_keys=True).encode()).hexdigest()
            with sqlite3.connect(model.cache_file_name) as conn:
                conn.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, message TEXT, metadata TEXT)")
                conn.execute("INSERT INTO cache VALUES (?, ?, ?)",
                             (old_key, "old response", '{"finish_reason":"stop"}'))

            self.assertEqual(model.infer(messages)[0], "old response")
            self.assertEqual(model.calls, 0)
            self.assertEqual(model.infer(messages, top_p=0.8)[0], "response 1")
            self.assertEqual(model.calls, 1)

    def test_qwen3_thinking_is_forced_off_even_with_caller_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeLLM(os.path.join(tmp, "llm.sqlite"))
            params = client_module._effective_generation_params(model, {
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
            })
            self.assertIs(params["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)


class RetryTests(unittest.TestCase):
    def test_only_transient_errors_retry_with_single_attempt_budget(self):
        calls = []

        @client_module.dynamic_retry_decorator
        def request(self):
            calls.append(1)
            if len(calls) < 3:
                raise status_error(429, {"retry-after": "0.01"})
            return "ok"

        model = SimpleNamespace(max_retries=3)
        with patch.object(client_module.time, "sleep") as sleep:
            with patch.object(client_module.random, "uniform", return_value=1.0):
                self.assertEqual(request(model), "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleep.call_count, 2)

        calls.clear()

        @client_module.dynamic_retry_decorator
        def bad_request(self):
            calls.append(1)
            raise status_error(400)

        with patch.object(client_module.time, "sleep") as sleep:
            with self.assertRaises(openai.APIStatusError):
                bad_request(model)
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()


class ConcurrencyTests(unittest.TestCase):
    def test_http_limit_is_shared_across_calls_and_thinking_stays_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = threading.Lock()
            active = 0
            high_water = 0
            seen_thinking = []

            def create(**params):
                nonlocal active, high_water
                with gate:
                    active += 1
                    high_water = max(high_water, active)
                    seen_thinking.append(params["extra_body"]["chat_template_kwargs"]["enable_thinking"])
                time.sleep(0.03)
                with gate:
                    active -= 1
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
                )

            model = SimpleNamespace(
                cache_file_name=os.path.join(tmp, "llm.sqlite"),
                llm_name="qwen3-8b", llm_base_url="http://localhost:8035/v1",
                llm_config=SimpleNamespace(generate_params={
                    "model": "qwen3-8b", "temperature": 0,
                    "max_completion_tokens": 2048,
                }),
                max_retries=1,
                openai_client=SimpleNamespace(chat=SimpleNamespace(
                    completions=SimpleNamespace(create=create))),
                _llm_stats_lock=threading.Lock(), llm_http_attempt_count=0,
            )
            with patch.object(client_module, "_LLM_HTTP_SEMAPHORE", threading.BoundedSemaphore(2)):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(
                        client_module.CacheOpenAI.infer, model,
                        [{"role": "user", "content": f"question {i}"}],
                    ) for i in range(4)]
                    results = [future.result() for future in futures]

            self.assertEqual(high_water, 2)
            self.assertEqual(model.llm_http_attempt_count, 4)
            self.assertTrue(all(value is False for value in seen_thinking))
            self.assertEqual([result[0] for result in results], ["ok"] * 4)


if __name__ == "__main__":
    unittest.main()
