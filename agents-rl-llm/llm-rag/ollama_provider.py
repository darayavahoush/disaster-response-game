"""Ollama-backed task decomposition provider (Step 5).

`OllamaTaskDecompositionProvider` implements the `TaskDecompositionProvider`
interface defined in `task_decomposition.py` by asking a local Ollama model
to decompose a mission into structured tasks, instead of using the
deterministic keyword pipeline. It is a drop-in replacement: callers use it
exactly like `RuleBasedProvider`, typically via
`decompose_task_structured(mission, provider=OllamaTaskDecompositionProvider())`.

Design notes
------------
* Standard library only (`urllib`), no new dependencies.
* The HTTP call is made through an injectable `transport` callable so unit
  tests never need a real Ollama server or network access.
* Any failure -- connection error, timeout, non-2xx response, invalid JSON,
  or a response that normalizes to zero usable tasks -- causes this
  provider to fall back to `RuleBasedProvider`, so `decompose()` never
  raises and never returns something `decompose_task_structured` can't use.

Import note
-----------
`agents-rl-llm/` and `llm-rag/` are hyphenated directory names, so they
cannot be reached via a normal dotted Python import (there is no
`agents_rl_llm` package on disk). This module sits next to
`task_decomposition.py`, so it adds its own directory to `sys.path` (if not
already present) and imports `task_decomposition` as a plain top-level
module -- the same file `task_decomposition.py` that
`tests/test_task_decomposition.py` loads directly by path. No new package
(e.g. `agents_rl_llm`) is created to make this work.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from task_decomposition import (  # noqa: E402 - see import note above
    RuleBasedProvider,
    TaskDecompositionProvider,
    normalize_tasks,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Defaults (overridable via environment variables or constructor args)
# --------------------------------------------------------------------------

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "llama3.2"
_DEFAULT_OLLAMA_TIMEOUT_SECONDS = 15.0

# Signature: (url, data, headers, timeout_seconds) -> raw response bytes.
# Swappable in tests so no real network/Ollama call is ever made.
Transport = Callable[[str, bytes, Dict[str, str], float], bytes]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return default


def _default_transport(url: str, data: bytes, headers: Dict[str, str], timeout: float) -> bytes:
    """Real Ollama call via stdlib `urllib`. Never used in unit tests --
    they inject a fake `transport` instead."""
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


class OllamaTaskDecompositionProvider(TaskDecompositionProvider):
    """Task decomposition backed by a local Ollama model, with a safe
    fallback to `RuleBasedProvider` on any failure."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        transport: Optional[Transport] = None,
        fallback_provider: Optional[TaskDecompositionProvider] = None,
    ) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST)).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
        self.timeout = (
            float(timeout) if timeout is not None else _env_float("OLLAMA_TIMEOUT_SECONDS", _DEFAULT_OLLAMA_TIMEOUT_SECONDS)
        )
        self._transport: Transport = transport or _default_transport
        self._fallback_provider: TaskDecompositionProvider = fallback_provider or RuleBasedProvider()

    # ----------------------------------------------------------------
    # Public interface (TaskDecompositionProvider)
    # ----------------------------------------------------------------

    def decompose(self, mission: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        mission_text = mission if isinstance(mission, str) else str(mission)

        try:
            raw_bytes = self._request_ollama(mission_text, context)
            raw_tasks = self._extract_tasks(raw_bytes)
        except Exception as exc:  # noqa: BLE001 - any transport/parse failure falls back
            logger.warning(
                "Ollama task decomposition request failed (%s); falling back to rule-based provider",
                exc,
            )
            return self._fallback_provider.decompose(mission_text, context=context)

        if raw_tasks is None:
            logger.warning(
                "Ollama response did not contain a usable task list; falling back to rule-based provider"
            )
            return self._fallback_provider.decompose(mission_text, context=context)

        normalized = normalize_tasks(raw_tasks)
        if not normalized:
            logger.warning(
                "Ollama response produced no valid tasks after normalization; "
                "falling back to rule-based provider"
            )
            return self._fallback_provider.decompose(mission_text, context=context)

        return normalized

    # ----------------------------------------------------------------
    # Ollama request/response handling
    # ----------------------------------------------------------------

    def _build_prompt(self, mission: str, context: Optional[str]) -> str:
        context_section = f"\nRelevant context:\n{context}\n" if context else ""
        return (
            "You are an emergency response planning assistant. Break the mission "
            "below into an ordered list of concrete disaster-response tasks.\n"
            f"Mission: {mission}\n"
            f"{context_section}"
            "Respond with ONLY valid JSON: either a JSON array of task objects, "
            'or an object of the form {"tasks": [...]}. Each task object must '
            "have the fields: task_id (string), task (string), priority (one of "
            '"critical", "high", "medium", "low"), dependencies (array of '
            'task_id strings), disaster_type (one of "flood", "earthquake", '
            '"cyclone", "generic"). Do not include any text outside the JSON.'
        )

    def _request_ollama(self, mission: str, context: Optional[str]) -> bytes:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(mission, context),
            "format": "json",
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        return self._transport(url, data, headers, self.timeout)

    def _extract_tasks(self, raw_bytes: bytes) -> Optional[Any]:
        """Parse the raw Ollama HTTP response body and pull out the task
        list. Returns None (rather than raising) when the shape is
        unusable, so callers can decide to fall back."""
        text = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
        outer = json.loads(text)

        # Ollama's /api/generate wraps the model's textual output in a
        # "response" string field. Some injected transports (in tests) may
        # also hand back the decoded task payload directly.
        if isinstance(outer, dict) and "response" in outer:
            inner = outer["response"]
            payload = json.loads(inner) if isinstance(inner, str) else inner
        else:
            payload = outer

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
            return payload["tasks"]
        return None
