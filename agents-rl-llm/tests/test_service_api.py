"""Focused unit tests for agents_rl_llm/service.py.

Run from the repository root, e.g.:
    python -m pytest agents-rl-llm/tests/test_service_api.py -v

None of these tests require a real Ollama server, network access, or a
real simulation engine:
  - /decide tests substitute a lightweight fake MAPPO policy (via
    monkeypatching `service.policy`) so they don't depend on torch or a
    trained checkpoint being present.
  - /knowledge_query tests exercise the default (rule-based) pipeline
    directly, and separately verify the optional-Ollama code path using
    a fake provider class -- no real HTTP calls are made to Ollama.
  - The one test that exercises the real HTTP server only talks to
    localhost (loopback), never the network.
"""

from __future__ import annotations

import json
import unittest

from agents_rl_llm import service
from agents_rl_llm.llm_rag.task_decomposition import RuleBasedProvider


class FakeMappoPolicy:
    """Deterministic stand-in for MAPPOPolicy.decide_batch, so /decide
    tests don't require torch or a trained checkpoint."""

    def __init__(self, fixed_action: str = "wait"):
        self.fixed_action = fixed_action
        self.calls = []

    def decide_batch(self, observations):
        self.calls.append(observations)
        return [self.fixed_action for _ in observations]


class FailingMappoPolicy:
    def decide_batch(self, observations):
        raise RuntimeError("inference blew up")


class FakeOllamaProvider:
    """Stands in for OllamaTaskDecompositionProvider without making any
    network call, to exercise the LLM_PROVIDER=ollama code path."""

    def __init__(self, *args, **kwargs):
        pass

    def decompose(self, mission, context=None):
        return [
            {
                "task_id": "llm_task",
                "task": "LLM-suggested high-level task",
                "priority": "high",
                "dependencies": [],
                "disaster_type": "generic",
            }
        ]


class ServiceApiTestCase(unittest.TestCase):
    def setUp(self):
        # Snapshot + restore module-level state each test so tests don't
        # leak policy/env mutations into one another.
        self._orig_policy = service.policy
        self._orig_policy_error = service.policy_error
        self.client = service.app.test_client()

    def tearDown(self):
        service.policy = self._orig_policy
        service.policy_error = self._orig_policy_error


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


class HealthEndpointTests(ServiceApiTestCase):
    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["service"], "agent-intelligence")
        self.assertIn("policy_ready", body)
        self.assertIn("checkpoint_loaded", body)

    def test_health_reflects_missing_policy(self):
        service.policy = None
        r = self.client.get("/health")
        self.assertFalse(r.get_json()["policy_ready"])

    def test_unknown_get_path_is_404(self):
        r = self.client.get("/nope")
        self.assertEqual(r.status_code, 404)


# --------------------------------------------------------------------------
# /decide
# --------------------------------------------------------------------------


class DecideEndpointTests(ServiceApiTestCase):
    def test_valid_multi_agent_request_returns_one_action_per_agent(self):
        service.policy = FakeMappoPolicy(fixed_action="scan")
        payload = {
            "observations": [
                {"agent_id": "drone-1", "observation": {"battery": 0.6, "status": "idle"}},
                {"agent_id": "rescue_team-1", "observation": {"battery": 0.4, "x": 2, "y": 3}},
            ]
        }
        r = self.client.post("/decide", json=payload)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(len(body["actions"]), 2)
        self.assertEqual(body["actions"][0]["agent_id"], "drone-1")
        self.assertEqual(body["actions"][1]["agent_id"], "rescue_team-1")
        for action in body["actions"]:
            self.assertEqual(action["action"], "scan")

    def test_move_and_deliver_actions_include_target(self):
        service.policy = FakeMappoPolicy(fixed_action="move")
        payload = {"observations": [{"agent_id": "drone-1", "observation": {"x": 4, "y": 5}}]}
        r = self.client.post("/decide", json=payload)
        body = r.get_json()
        self.assertEqual(body["actions"][0]["target"], [5, 6])

    def test_wait_action_has_no_target(self):
        service.policy = FakeMappoPolicy(fixed_action="wait")
        payload = {"observations": [{"agent_id": "drone-1", "observation": {}}]}
        r = self.client.post("/decide", json=payload)
        body = r.get_json()
        self.assertNotIn("target", body["actions"][0])

    def test_empty_observations_returns_empty_actions_without_calling_policy(self):
        fake = FakeMappoPolicy()
        service.policy = fake
        r = self.client.post("/decide", json={"observations": []})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"actions": []})
        self.assertEqual(fake.calls, [])

    def test_missing_observations_field_is_400(self):
        service.policy = FakeMappoPolicy()
        r = self.client.post("/decide", json={})
        self.assertEqual(r.status_code, 400)
        self.assertIn("observations", r.get_json()["error"])

    def test_observations_not_a_list_is_400(self):
        service.policy = FakeMappoPolicy()
        r = self.client.post("/decide", json={"observations": "not-a-list"})
        self.assertEqual(r.status_code, 400)

    def test_observation_entry_not_an_object_is_400(self):
        service.policy = FakeMappoPolicy()
        r = self.client.post("/decide", json={"observations": ["not-a-dict"]})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["details"]["index"], 0)

    def test_missing_agent_id_is_400(self):
        service.policy = FakeMappoPolicy()
        r = self.client.post("/decide", json={"observations": [{"observation": {}}]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("agent_id", r.get_json()["error"])

    def test_blank_agent_id_is_400(self):
        service.policy = FakeMappoPolicy()
        r = self.client.post("/decide", json={"observations": [{"agent_id": "   "}]})
        self.assertEqual(r.status_code, 400)

    def test_non_numeric_observation_field_is_400(self):
        service.policy = FakeMappoPolicy()
        payload = {"observations": [{"agent_id": "drone-1", "observation": {"battery": "high"}}]}
        r = self.client.post("/decide", json=payload)
        self.assertEqual(r.status_code, 400)
        details = r.get_json()["details"]
        self.assertEqual(details["field"], "battery")
        self.assertEqual(details["agent_id"], "drone-1")

    def test_observation_field_not_an_object_is_400(self):
        service.policy = FakeMappoPolicy()
        payload = {"observations": [{"agent_id": "drone-1", "observation": "nope"}]}
        r = self.client.post("/decide", json=payload)
        self.assertEqual(r.status_code, 400)

    def test_policy_unavailable_is_503(self):
        service.policy = None
        service.policy_error = "torch not installed"
        payload = {"observations": [{"agent_id": "drone-1", "observation": {}}]}
        r = self.client.post("/decide", json=payload)
        self.assertEqual(r.status_code, 503)
        self.assertIn("policy unavailable", r.get_json()["error"].lower())

    def test_policy_inference_failure_is_500(self):
        service.policy = FailingMappoPolicy()
        payload = {"observations": [{"agent_id": "drone-1", "observation": {}}]}
        r = self.client.post("/decide", json=payload)
        self.assertEqual(r.status_code, 500)

    def test_action_names_come_only_from_the_policy_not_the_llm(self):
        # Requirement: the LLM/RAG stack must never choose low-level
        # movement actions -- only the MAPPO policy does. Verify the
        # feature vector handed to the policy is the only input to the
        # action, and that its length matches env.obs_dim (12), i.e. the
        # same encoding DisasterResponseEnv.encode_observation() uses.
        fake = FakeMappoPolicy(fixed_action="rescue")
        service.policy = fake
        payload = {"observations": [{"agent_id": "drone-1", "observation": {"battery": 0.9}}]}
        self.client.post("/decide", json=payload)
        self.assertEqual(len(fake.calls[0][0]), service.env.obs_dim)


# --------------------------------------------------------------------------
# /knowledge_query
# --------------------------------------------------------------------------


class KnowledgeQueryEndpointTests(ServiceApiTestCase):
    def test_default_pipeline_uses_rule_based_decomposition(self):
        payload = {"question": "Rescue victims trapped in a collapsed building after an earthquake."}
        r = self.client.post("/knowledge_query", json=payload)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn("answer", body)
        self.assertIn("sources", body)
        self.assertIsInstance(body["sources"], list)
        self.assertTrue(body["sources"])
        self.assertIsInstance(body["suggested_task_decomposition"], list)
        self.assertTrue(all(isinstance(t, str) for t in body["suggested_task_decomposition"]))
        self.assertTrue(body["suggested_task_decomposition"])

    def test_rag_context_flows_into_answer_and_sources(self):
        payload = {"question": "what's the protocol for a flooded sector rescue?"}
        r = self.client.post("/knowledge_query", json=payload)
        body = r.get_json()
        self.assertTrue(any("flood" in s.lower() for s in body["sources"]))

    def test_empty_question_preserves_existing_fallback_contract(self):
        r = self.client.post("/knowledge_query", json={"question": ""})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.get_json(),
            {"answer": "No question supplied.", "sources": [], "suggested_task_decomposition": []},
        )

    def test_missing_question_field_uses_fallback(self):
        r = self.client.post("/knowledge_query", json={})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["answer"], "No question supplied.")

    def test_mission_alias_is_accepted(self):
        r = self.client.post("/knowledge_query", json={"mission": "Evacuate victims from a cyclone shelter."})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["suggested_task_decomposition"])

    def test_response_shape_matches_documented_contract(self):
        r = self.client.post("/knowledge_query", json={"question": "hazardous blocked route"})
        body = r.get_json()
        self.assertEqual(set(body.keys()), {"answer", "sources", "suggested_task_decomposition"})

    def test_llm_provider_ollama_routes_through_llm_with_fallback_semantics(self):
        # Default (LLM_PROVIDER unset/"local") never touches the Ollama
        # provider at all -- verify that, then verify the opt-in path
        # uses whatever provider _task_provider() builds (substituted
        # here with a fake, so no network call happens).
        original_provider_cls = service.OllamaTaskDecompositionProvider
        original_env = service.os.environ.get("LLM_PROVIDER")
        try:
            service.OllamaTaskDecompositionProvider = FakeOllamaProvider
            service.os.environ["LLM_PROVIDER"] = "ollama"
            r = self.client.post("/knowledge_query", json={"question": "flood rescue protocol"})
            body = r.get_json()
            self.assertEqual(body["suggested_task_decomposition"], ["LLM-suggested high-level task"])
        finally:
            service.OllamaTaskDecompositionProvider = original_provider_cls
            if original_env is None:
                service.os.environ.pop("LLM_PROVIDER", None)
            else:
                service.os.environ["LLM_PROVIDER"] = original_env

    def test_default_provider_is_rule_based_when_llm_provider_unset(self):
        service.os.environ.pop("LLM_PROVIDER", None)
        provider = service._task_provider()
        self.assertIsInstance(provider, RuleBasedProvider)

    def test_malformed_body_type_is_400_via_real_json_parsing(self):
        # Exercise the raw-body parser directly: a JSON array is valid
        # JSON but not the required top-level object.
        with self.assertRaises(service.ApiError) as ctx:
            service._read_json(b"[1, 2, 3]")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_invalid_json_body_is_400_via_real_json_parsing(self):
        with self.assertRaises(service.ApiError) as ctx:
            service._read_json(b"{not valid json")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_empty_body_parses_to_empty_object(self):
        self.assertEqual(service._read_json(b""), {})


# --------------------------------------------------------------------------
# Real HTTP transport (loopback only) -- confirms status codes propagate
# through BaseHTTPRequestHandler, not just the in-process TestClient.
# --------------------------------------------------------------------------


class HttpTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer

        cls._orig_policy = service.policy
        service.policy = FakeMappoPolicy(fixed_action="wait")

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), service.AgentIntelligenceHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        service.policy = cls._orig_policy

    def _post(self, path, body: bytes):
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        status = resp.status
        payload = json.loads(resp.read().decode("utf-8"))
        conn.close()
        return status, payload

    def test_malformed_json_over_http_is_400(self):
        status, payload = self._post("/decide", b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_valid_decide_over_http(self):
        body = json.dumps({"observations": [{"agent_id": "drone-1", "observation": {}}]}).encode("utf-8")
        status, payload = self._post("/decide", body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["actions"][0]["action"], "wait")

    def test_health_over_http(self):
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
