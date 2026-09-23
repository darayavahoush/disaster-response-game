"""Agent Intelligence HTTP API.

Wires the existing, independently-developed pieces together behind the
three endpoints defined in `docs/api-contract.md`:

    GET  /health
    POST /decide           -> MAPPO policy inference (no training)
    POST /knowledge_query  -> RAG -> task decomposition -> optional Ollama
                               LLM -> deterministic (rule-based) fallback

This module does not implement RAG, task decomposition, the Ollama
integration, or MAPPO itself -- it only validates requests, calls into
`agents_rl_llm.llm_rag` / `agents_rl_llm.mappo_maddpg`, and shapes the
responses to match the documented contract.
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents_rl_llm.llm_rag.knowledge_base import EmergencyKnowledgeBase
from agents_rl_llm.llm_rag.ollama_provider import OllamaTaskDecompositionProvider
from agents_rl_llm.llm_rag.task_decomposition import RuleBasedProvider, decompose_task_structured
from agents_rl_llm.mappo_maddpg.env import DisasterResponseEnv

logger = logging.getLogger(__name__)

# MAPPOPolicy depends on torch. It's imported defensively so the service
# can still start (and /health can still report an honest status) in an
# environment where torch isn't installed -- /decide will respond with 503
# in that case rather than the process failing to boot. In the normal
# deployment environment (torch>=2.0.0, per requirements.txt) this import
# succeeds and behaves exactly as before.
try:
    from agents_rl_llm.mappo_maddpg.policy import MAPPOPolicy
except Exception as exc:  # pragma: no cover - exercised only when torch is absent
    MAPPOPolicy = None
    _POLICY_IMPORT_ERROR: Optional[str] = str(exc)
else:
    _POLICY_IMPORT_ERROR = None


# --------------------------------------------------------------------------
# Component wiring
# --------------------------------------------------------------------------

knowledge_base = EmergencyKnowledgeBase()
env = DisasterResponseEnv(agent_ids=["drone-1", "rescue_team-1", "vehicle-1"], num_victims=4)

_RULE_BASED_PROVIDER = RuleBasedProvider()


def _default_checkpoint_candidates() -> List[Path]:
    """Checkpoint paths produced by `agents-rl-llm/mappo-maddpg/train.py`.

    Preferred order: the best-reward checkpoint from training, then the
    most recent one. Both are written relative to the repo root.
    """
    root = Path(__file__).resolve().parent.parent
    ckpt_dir = root / "agents-rl-llm" / "mappo-maddpg" / "checkpoints"
    return [ckpt_dir / "mappo_best.pt", ckpt_dir / "mappo_latest.pt"]


def _resolve_checkpoint_path() -> Optional[str]:
    """`MAPPO_CHECKPOINT_PATH` env var wins if set; otherwise fall back to
    the default training output locations, first existing one wins."""
    env_path = os.getenv("MAPPO_CHECKPOINT_PATH")
    if env_path:
        return env_path
    for candidate in _default_checkpoint_candidates():
        if candidate.exists():
            return str(candidate)
    return None


REQUIRE_CHECKPOINT = os.getenv("MAPPO_REQUIRE_CHECKPOINT", "0") == "1"

policy = None
policy_error: Optional[str] = _POLICY_IMPORT_ERROR
checkpoint_path: Optional[str] = None
checkpoint_loaded = False

if MAPPOPolicy is not None:
    try:
        policy = MAPPOPolicy(
            num_agents=len(env.agent_ids),
            obs_dim=env.obs_dim,
            action_dim=len(env.action_space),
            action_names=env.action_space,
        )
        checkpoint_path = _resolve_checkpoint_path()
        if checkpoint_path:
            try:
                policy.load_checkpoint(checkpoint_path)
                checkpoint_loaded = True
            except Exception as ckpt_exc:  # noqa: BLE001
                policy_error = f"failed to load checkpoint '{checkpoint_path}': {ckpt_exc}"
                logger.warning(policy_error)
                if REQUIRE_CHECKPOINT:
                    policy = None
        elif REQUIRE_CHECKPOINT:
            policy_error = "MAPPO_REQUIRE_CHECKPOINT is set but no checkpoint was found"
            policy = None
    except Exception as exc:  # noqa: BLE001 - policy construction itself failed
        policy = None
        policy_error = f"failed to construct MAPPOPolicy: {exc}"
        logger.warning(policy_error)
else:
    logger.warning("MAPPO policy unavailable: %s", policy_error)


# --------------------------------------------------------------------------
# Small error type so both the HTTP handler and the in-process TestClient
# can surface the same (status_code, payload) shape for validation/runtime
# failures without duplicating the checks in two places.
# --------------------------------------------------------------------------


class ApiError(Exception):
    def __init__(self, status_code: int, message: str, **details: Any):
        super().__init__(message)
        self.status_code = status_code
        self.payload: Dict[str, Any] = {"error": message}
        if details:
            self.payload["details"] = details


def _read_json(req_body: bytes) -> Dict[str, Any]:
    """Parse the raw HTTP request body. Raises ApiError(400, ...) on
    anything that isn't valid JSON encoding a top-level object; an empty
    body is treated as `{}` (matching the previous, lenient behaviour)."""
    if not req_body:
        return {}
    try:
        parsed = json.loads(req_body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, f"malformed request: invalid JSON body ({exc})") from exc
    if not isinstance(parsed, dict):
        raise ApiError(400, "malformed request: request body must be a JSON object")
    return parsed


def _make_response(data: Dict[str, Any], status: int = 200):
    payload = json.dumps(data).encode("utf-8")
    return payload, status


# --------------------------------------------------------------------------
# /decide -- MAPPO policy inference only (no training / no PPO updates)
# --------------------------------------------------------------------------

_TYPE_INDEX = {"drone": 0.0, "rescue_team": 1.0, "vehicle": 2.0}
_NUMERIC_FIELDS = ("battery", "hazard_risk", "mission_priority", "victims_nearby", "supply", "x", "y")
_ACTIVE_STATUSES = {"idle", "scanning"}


def _validate_observations(observations: Any) -> List[Dict[str, Any]]:
    if observations is None:
        raise ApiError(400, "missing required field: observations")
    if not isinstance(observations, list):
        raise ApiError(400, "invalid request: 'observations' must be a list")

    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(observations):
        if not isinstance(item, dict):
            raise ApiError(400, f"invalid agent data at index {index}: expected an object", index=index)

        agent_id = item.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ApiError(
                400,
                f"invalid agent data at index {index}: 'agent_id' is required and must be a non-empty string",
                index=index,
            )

        obs = item.get("observation", {})
        if obs is None:
            obs = {}
        if not isinstance(obs, dict):
            raise ApiError(
                400,
                f"invalid observation for agent '{agent_id}' at index {index}: 'observation' must be an object",
                index=index,
                agent_id=agent_id,
            )

        for field in _NUMERIC_FIELDS:
            if field not in obs or obs[field] is None:
                continue
            try:
                float(obs[field])
            except (TypeError, ValueError):
                raise ApiError(
                    400,
                    f"invalid observation for agent '{agent_id}' at index {index}: "
                    f"field '{field}' must be numeric",
                    index=index,
                    agent_id=agent_id,
                    field=field,
                )

        validated.append({"agent_id": agent_id, "observation": obs})
    return validated


def _encode_observation_features(obs: Dict[str, Any]) -> List[float]:
    """Same 12-dim feature layout as `DisasterResponseEnv.encode_observation`,
    so features handed to the trained policy at inference time match what
    it was trained on."""
    battery = float(obs.get("battery", 0.5))
    hazard_risk = float(obs.get("hazard_risk", 0.1))
    mission_priority = float(obs.get("mission_priority", 0.5))
    victims_nearby = float(obs.get("victims_nearby", 0.0))
    supply = float(obs.get("supply", 0.5))
    status = 1.0 if str(obs.get("status", "idle")) in _ACTIVE_STATUSES else 0.0
    type_index = _TYPE_INDEX.get(str(obs.get("type", "drone")), 0.0)
    x = float(obs.get("x", 0.0))
    y = float(obs.get("y", 0.0))
    route_pressure = 0.5 + (0.5 if victims_nearby > 0 else 0.0)
    low_battery = 1.0 if battery < 0.3 else 0.0
    high_hazard = 1.0 if hazard_risk > 0.75 else 0.0
    return [
        battery, hazard_risk, mission_priority, victims_nearby, supply,
        status, type_index, x, y, route_pressure, low_battery, high_hazard,
    ]


def _handle_decide(data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    validated = _validate_observations(data.get("observations"))

    if not validated:
        return {"actions": []}, 200

    if policy is None:
        raise ApiError(503, f"MAPPO policy unavailable: {policy_error or 'not initialized'}")

    feature_batch = [_encode_observation_features(item["observation"]) for item in validated]

    try:
        action_names = policy.decide_batch(feature_batch)
    except Exception as exc:  # noqa: BLE001 - inference failure, not a client error
        raise ApiError(500, f"policy inference failed: {exc}") from exc

    actions: List[Dict[str, Any]] = []
    for item, action in zip(validated, action_names):
        obs = item["observation"]
        decision: Dict[str, Any] = {"agent_id": item["agent_id"], "action": action}
        if action in {"move", "deliver"}:
            decision["target"] = [
                int(float(obs.get("x", 0.0)) + 1),
                int(float(obs.get("y", 0.0)) + 1),
            ]
        actions.append(decision)

    return {"actions": actions}, 200


# --------------------------------------------------------------------------
# /knowledge_query -- RAG -> task decomposition -> optional Ollama LLM ->
# deterministic (rule-based) fallback
# --------------------------------------------------------------------------


def _task_provider():
    """LLM_PROVIDER=ollama opts into the Ollama-backed task-decomposition
    provider (which itself falls back to rule-based on any failure -- see
    ollama_provider.py). Any other value (default: unset/"local") uses the
    deterministic rule-based provider directly, so the default/test
    configuration never attempts to reach a network service."""
    if os.getenv("LLM_PROVIDER", "local") == "ollama":
        return OllamaTaskDecompositionProvider()
    return _RULE_BASED_PROVIDER


def _handle_knowledge_query(data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    raw_question = data.get("question", data.get("mission", ""))
    question = raw_question if isinstance(raw_question, str) else str(raw_question)
    question = question.strip()

    if not question:
        return {"answer": "No question supplied.", "sources": [], "suggested_task_decomposition": []}, 200

    # 1. RAG knowledge retrieval (also supplies grounding context for the
    #    task-decomposition step below).
    rag_result = knowledge_base.answer(question)
    context = rag_result["answer"]

    # 2. Task decomposition, 3. optional Ollama LLM, 4. deterministic
    #    fallback: all handled by decompose_task_structured() + the chosen
    #    provider (OllamaTaskDecompositionProvider already falls back to
    #    RuleBasedProvider on any failure).
    provider = _task_provider()
    structured_tasks = decompose_task_structured(question, provider=provider, context=context)
    suggested_task_decomposition: List[str] = [task["task"] for task in structured_tasks]

    return {
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
        "suggested_task_decomposition": suggested_task_decomposition,
    }, 200


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def _handle_health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "agent-intelligence",
        "policy": "mappo",
        "llm_provider": os.getenv("LLM_PROVIDER", "local"),
        # Additive diagnostics (existing consumers only check "ok"):
        "policy_ready": policy is not None,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": checkpoint_path,
    }


# --------------------------------------------------------------------------
# In-process test client (used by unit tests; mirrors the HTTP handler)
# --------------------------------------------------------------------------


class Response:
    def __init__(self, status_code: int, payload: Dict[str, Any]):
        self.status_code = status_code
        self._payload = payload

    def get_json(self):
        return self._payload


class TestClient:
    def __init__(self, app):
        self.app = app

    def get(self, path: str):
        if path == "/health":
            return Response(200, _handle_health())
        return Response(404, {"error": "not found"})

    def post(self, path: str, json: Dict[str, Any] | None = None):
        data = json if isinstance(json, dict) else {}
        try:
            if path == "/decide":
                payload, status = _handle_decide(data)
                return Response(status, payload)
            if path == "/knowledge_query":
                payload, status = _handle_knowledge_query(data)
                return Response(status, payload)
        except ApiError as exc:
            return Response(exc.status_code, exc.payload)
        return Response(404, {"error": "not found"})


class AgentIntelligenceApp:
    def test_client(self):
        return TestClient(self)


app = AgentIntelligenceApp()


# --------------------------------------------------------------------------
# Real HTTP server
# --------------------------------------------------------------------------


class AgentIntelligenceHandler(BaseHTTPRequestHandler):
    server_version = "AegisAgentIntelligence/1.0"

    def _write(self, body: bytes, status: int):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            body, status = _make_response(_handle_health())
            self._write(body, status)
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length else b""

        if self.path not in ("/decide", "/knowledge_query"):
            self.send_error(404, "Not found")
            return

        try:
            data = _read_json(raw_body)
            if self.path == "/decide":
                payload, status = _handle_decide(data)
            else:
                payload, status = _handle_knowledge_query(data)
        except ApiError as exc:
            body, status = _make_response(exc.payload, exc.status_code)
            self._write(body, status)
            return

        body, status = _make_response(payload, status)
        self._write(body, status)

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(host: str = "0.0.0.0", port: int = 8000):
    server = ThreadingHTTPServer((host, port), AgentIntelligenceHandler)
    print(f"Aegis agent intelligence running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server(host=os.getenv("AGENT_SERVICE_HOST", "0.0.0.0"), port=int(os.getenv("AGENT_SERVICE_PORT", "8000")))
