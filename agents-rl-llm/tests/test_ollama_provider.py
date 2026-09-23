"""Unit tests for `agents-rl-llm/llm-rag/ollama_provider.py` (Step 5).

Run from the repository root, e.g.:
    python3 -m unittest agents-rl-llm.tests.test_ollama_provider -v
or, without relying on package discovery:
    python3 -m unittest discover -s agents-rl-llm/tests -p "test_ollama_provider.py" -v

Like `test_task_decomposition.py`, this file loads the module under test
directly from its file path (via `importlib.util`), since `agents-rl-llm/`
and `llm-rag/` are hyphenated directories and cannot be reached via a
normal dotted import.

None of these tests touch the network or require a real Ollama server:
every test constructs `OllamaTaskDecompositionProvider` with an injected
fake `transport` callable that returns canned bytes (or raises, to
simulate failures) instead of calling `urllib.request.urlopen`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import unittest
import urllib.error
from typing import Any, Dict, List, Optional
from unittest import mock

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LLM_RAG_DIR = os.path.join(_THIS_DIR, "..", "llm-rag")
_TASK_DECOMPOSITION_PATH = os.path.join(_LLM_RAG_DIR, "task_decomposition.py")
_OLLAMA_PROVIDER_PATH = os.path.join(_LLM_RAG_DIR, "ollama_provider.py")


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load task_decomposition first (ollama_provider imports it as a plain
# top-level module once its own directory is on sys.path).
_task_decomposition = _load_module("task_decomposition", _TASK_DECOMPOSITION_PATH)
_ollama_provider = _load_module("ollama_provider", _OLLAMA_PROVIDER_PATH)

RuleBasedProvider = _task_decomposition.RuleBasedProvider
decompose_task_structured = _task_decomposition.decompose_task_structured
OllamaTaskDecompositionProvider = _ollama_provider.OllamaTaskDecompositionProvider

REQUIRED_FIELDS = {"task_id", "task", "priority", "dependencies", "disaster_type"}

_SAMPLE_MISSION = "Flood waters are rising near the levee, rescue trapped residents."

_SAMPLE_TASK_LIST = [
    {
        "task_id": "llm_assess",
        "task": "Assess flood impact from live sensor feeds",
        "priority": "critical",
        "dependencies": [],
        "disaster_type": "flood",
    },
    {
        "task_id": "llm_rescue",
        "task": "Dispatch boats to rescue trapped residents",
        "priority": "critical",
        "dependencies": ["llm_assess"],
        "disaster_type": "flood",
    },
]


def _ollama_response_bytes(response_payload: Any, extra_outer: Optional[Dict[str, Any]] = None) -> bytes:
    """Build the raw bytes Ollama's /api/generate would return: a JSON
    object whose "response" field is a *string* containing the model's
    (JSON) output -- matching Ollama's real wire format."""
    outer: Dict[str, Any] = {
        "model": "llama3.2",
        "done": True,
        "response": json.dumps(response_payload),
    }
    if extra_outer:
        outer.update(extra_outer)
    return json.dumps(outer).encode("utf-8")


class RecordingTransport:
    """Fake transport that records every call and returns a fixed body
    (or raises a fixed exception), so tests never touch the network."""

    def __init__(self, body: Optional[bytes] = None, exc: Optional[BaseException] = None):
        self.body = body
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url: str, data: bytes, headers: Dict[str, str], timeout: float) -> bytes:
        self.calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        assert self.body is not None
        return self.body


class SuccessfulResponseTests(unittest.TestCase):
    """Successful Ollama response + JSON parsing."""

    def test_json_array_response_is_parsed_and_returned(self):
        transport = RecordingTransport(body=_ollama_response_bytes(_SAMPLE_TASK_LIST))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["task_id"], "llm_assess")
        self.assertEqual(tasks[1]["dependencies"], ["llm_assess"])
        for task in tasks:
            self.assertTrue(REQUIRED_FIELDS.issubset(task.keys()))
        # Exactly one HTTP call was made, and it hit /api/generate.
        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(transport.calls[0]["url"].endswith("/api/generate"))

    def test_tasks_wrapped_in_dict_response(self):
        payload = {"tasks": _SAMPLE_TASK_LIST}
        transport = RecordingTransport(body=_ollama_response_bytes(payload))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(len(tasks), 2)
        self.assertEqual([t["task_id"] for t in tasks], ["llm_assess", "llm_rescue"])

    def test_integrates_with_decompose_task_structured(self):
        transport = RecordingTransport(body=_ollama_response_bytes(_SAMPLE_TASK_LIST))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        structured = decompose_task_structured(_SAMPLE_MISSION, provider=provider)

        self.assertEqual(len(structured), 2)
        for task in structured:
            self.assertTrue(REQUIRED_FIELDS.issubset(task.keys()))


class RagContextTests(unittest.TestCase):
    def test_context_is_included_in_the_request_prompt(self):
        transport = RecordingTransport(body=_ollama_response_bytes(_SAMPLE_TASK_LIST))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        provider.decompose(_SAMPLE_MISSION, context="Levee is at 90% capacity near Sector 4.")

        sent_body = json.loads(transport.calls[0]["data"].decode("utf-8"))
        self.assertIn("Levee is at 90% capacity near Sector 4.", sent_body["prompt"])
        self.assertIn(_SAMPLE_MISSION, sent_body["prompt"])

    def test_no_context_still_sends_valid_request(self):
        transport = RecordingTransport(body=_ollama_response_bytes(_SAMPLE_TASK_LIST))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION, context=None)

        self.assertEqual(len(tasks), 2)
        sent_body = json.loads(transport.calls[0]["data"].decode("utf-8"))
        self.assertIn(_SAMPLE_MISSION, sent_body["prompt"])


class ConfigurationTests(unittest.TestCase):
    def test_custom_model_and_host_are_used(self):
        transport = RecordingTransport(body=_ollama_response_bytes(_SAMPLE_TASK_LIST))
        provider = OllamaTaskDecompositionProvider(
            host="http://custom-ollama-host:9999",
            model="mistral",
            transport=transport,
        )

        provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(transport.calls[0]["url"], "http://custom-ollama-host:9999/api/generate")
        sent_body = json.loads(transport.calls[0]["data"].decode("utf-8"))
        self.assertEqual(sent_body["model"], "mistral")

    def test_defaults_come_from_environment_variables(self):
        transport = RecordingTransport(body=_ollama_response_bytes(_SAMPLE_TASK_LIST))
        env = {
            "OLLAMA_HOST": "http://env-host:11111",
            "OLLAMA_MODEL": "env-model",
            "OLLAMA_TIMEOUT_SECONDS": "5",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            provider = OllamaTaskDecompositionProvider(transport=transport)
            provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(transport.calls[0]["url"], "http://env-host:11111/api/generate")
        self.assertEqual(transport.calls[0]["timeout"], 5.0)
        sent_body = json.loads(transport.calls[0]["data"].decode("utf-8"))
        self.assertEqual(sent_body["model"], "env-model")

    def test_falls_back_to_hardcoded_defaults_without_env_or_args(self):
        transport = RecordingTransport(body=_ollama_response_bytes(_SAMPLE_TASK_LIST))
        env_without_ollama_vars = {
            k: v for k, v in os.environ.items() if not k.startswith("OLLAMA_")
        }
        with mock.patch.dict(os.environ, env_without_ollama_vars, clear=True):
            provider = OllamaTaskDecompositionProvider(transport=transport)
            provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(transport.calls[0]["url"], "http://localhost:11434/api/generate")
        sent_body = json.loads(transport.calls[0]["data"].decode("utf-8"))
        self.assertEqual(sent_body["model"], "llama3.2")


class FallbackTests(unittest.TestCase):
    """Ollama failures of every kind must fall back to RuleBasedProvider's
    output for the same mission -- never raise, never return nothing."""

    def _expected_fallback(self, mission: str) -> List[Dict[str, Any]]:
        return RuleBasedProvider().decompose(mission)

    def test_timeout_falls_back_to_rule_based(self):
        transport = RecordingTransport(exc=socket.timeout("timed out"))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(tasks, self._expected_fallback(_SAMPLE_MISSION))

    def test_connection_error_falls_back_to_rule_based(self):
        transport = RecordingTransport(exc=urllib.error.URLError("connection refused"))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(tasks, self._expected_fallback(_SAMPLE_MISSION))

    def test_http_error_falls_back_to_rule_based(self):
        http_error = urllib.error.HTTPError(
            url="http://localhost:11434/api/generate", code=500, msg="Internal Error", hdrs=None, fp=None
        )
        transport = RecordingTransport(exc=http_error)
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(tasks, self._expected_fallback(_SAMPLE_MISSION))

    def test_invalid_json_body_falls_back_to_rule_based(self):
        transport = RecordingTransport(body=b"this is not json at all {{{")
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(tasks, self._expected_fallback(_SAMPLE_MISSION))

    def test_invalid_json_inside_response_field_falls_back(self):
        # Outer envelope is valid JSON, but the model's "response" text
        # (which should itself be JSON) is garbage.
        outer = {"model": "llama3.2", "done": True, "response": "not valid json {{"}
        transport = RecordingTransport(body=json.dumps(outer).encode("utf-8"))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(tasks, self._expected_fallback(_SAMPLE_MISSION))

    def test_unusable_shape_falls_back(self):
        # Valid JSON throughout, but neither a list nor {"tasks": [...]}.
        transport = RecordingTransport(body=_ollama_response_bytes({"unexpected": "shape"}))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(tasks, self._expected_fallback(_SAMPLE_MISSION))

    def test_malformed_tasks_fall_back_to_rule_based(self):
        # Valid JSON, valid list shape, but no entry has usable "task" text,
        # so normalize_tasks() drops everything and we must fall back.
        malformed = [
            {"task_id": "bad_1", "priority": "critical"},  # missing "task"
            {"task": "   ", "priority": "high"},  # blank task text
            "just a string, not an object",
        ]
        transport = RecordingTransport(body=_ollama_response_bytes(malformed))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(tasks, self._expected_fallback(_SAMPLE_MISSION))

    def test_partially_malformed_tasks_are_salvaged_not_dropped(self):
        # One good task, one unsalvageable task: normalize_tasks keeps the
        # good one, so this should NOT fall back to rule-based.
        mixed = [
            {"task_id": "ok_task", "task": "Deploy rescue boats", "priority": "critical"},
            {"priority": "high"},  # no task text -- dropped
        ]
        transport = RecordingTransport(body=_ollama_response_bytes(mixed))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task"], "Deploy rescue boats")

    def test_custom_fallback_provider_is_used(self):
        class StubFallback:
            def decompose(self, mission, context=None):
                return [
                    {
                        "task_id": "stub",
                        "task": "Stub fallback task",
                        "priority": "low",
                        "dependencies": [],
                        "disaster_type": "generic",
                    }
                ]

        transport = RecordingTransport(exc=RuntimeError("boom"))
        provider = OllamaTaskDecompositionProvider(transport=transport, fallback_provider=StubFallback())

        tasks = provider.decompose(_SAMPLE_MISSION)

        self.assertEqual(tasks[0]["task_id"], "stub")

    def test_decompose_never_raises_and_decompose_task_structured_stays_healthy(self):
        # End-to-end: even when Ollama is completely broken, the public
        # decompose_task_structured() API must still return a sane result.
        transport = RecordingTransport(exc=OSError("network unreachable"))
        provider = OllamaTaskDecompositionProvider(transport=transport)

        structured = decompose_task_structured(_SAMPLE_MISSION, provider=provider)

        self.assertTrue(len(structured) > 0)
        for task in structured:
            self.assertTrue(REQUIRED_FIELDS.issubset(task.keys()))


class NoNetworkDependencyTests(unittest.TestCase):
    """Guard against any test in this suite accidentally reaching the
    real network / a real Ollama server."""

    def test_real_urlopen_is_never_invoked_by_the_suite(self):
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted")):
            transport = RecordingTransport(body=_ollama_response_bytes(_SAMPLE_TASK_LIST))
            provider = OllamaTaskDecompositionProvider(transport=transport)
            tasks = provider.decompose(_SAMPLE_MISSION)
            self.assertEqual(len(tasks), 2)

    def test_default_transport_would_call_urlopen_but_is_not_exercised_here(self):
        # Sanity check that the *default* (real) transport is wired to
        # urllib, without ever actually invoking it.
        provider = OllamaTaskDecompositionProvider()
        self.assertIs(provider._transport, _ollama_provider._default_transport)


if __name__ == "__main__":
    unittest.main()
