# Frontend ⇄ Engine contract (draft)

This is the shape the frontend already consumes from its mock engine
(`frontend/src/lib/mockEngine.js`). Treat it as the target for the real simulation engine's
WebSocket feed — matching it means swapping the frontend's data source is a one-line change.

## `world_state` (pushed every simulation tick)

```jsonc
{
  "tick": 128,
  "scenario": "flood",              // "flood" | "cyclone" | "earthquake"
  "clock_seconds": 384,
  "grid": { "width": 40, "height": 26 },
  "hazards": [                       // sparse list of cells with active/forecast hazard
    { "x": 12, "y": 4, "intensity": 0.8, "type": "water", "forecast": [0.8, 0.85, 0.9, 0.95] }
  ],
  "blocked_routes": [
    { "from": [10, 4], "to": [11, 4], "reason": "debris" }
  ],
  "victims": [
    {
      "id": "v-014",
      "pos": [14, 6],
      "status": "detected",          // "hidden" | "detected" | "assigned" | "rescued"
      "severity": 0.7,               // 0-1, feeds mission priority
      "detected_by": "drone-2",
      "confidence": 0.91
    }
  ],
  "agents": [
    {
      "id": "drone-2",
      "type": "drone",               // "drone" | "rescue_team" | "vehicle"
      "pos": [13.5, 5.5],
      "battery": 0.62,
      "status": "scanning",          // "idle" | "scanning" | "moving" | "delivering" | "returning"
      "route": [[13.5,5.5],[14,6],[14,7]],
      "carrying": null                // or { "type": "medkit", "qty": 2 }
    }
  ],
  "missions": [
    {
      "id": "m-009",
      "victim_id": "v-014",
      "priority": 0.83,               // MCTS/planner output
      "assigned_agent": "rescue_team-1",
      "eta_seconds": 96,
      "status": "in_progress"
    }
  ],
  "resources": {
    "medkits": { "available": 12, "total": 20 },
    "water": { "available": 30, "total": 40 },
    "evac_capacity": { "available": 4, "total": 6 }
  },
  "metrics": {
    "victims_rescued": 7,
    "victims_total": 22,
    "avg_response_seconds": 142,
    "resource_efficiency": 0.74
  }
}
```

## `knowledge_query` (frontend → engine/LLM service, request/response)

Request:
```json
{ "question": "what's the protocol for a partially collapsed structure with trapped survivors?" }
```

Response:
```json
{
  "answer": "Stabilize before entry; establish two-person minimum with spotter; ...",
  "sources": ["FEMA USAR field ops guide, ch. 4"],
  "suggested_task_decomposition": [
    "Hold supply drop until structural stabilization confirmed",
    "Reroute rescue-team-1 via cleared path at sector C4"
  ]
}
```

## Commands (frontend → engine)

```json
{ "type": "set_scenario", "scenario": "cyclone" }
{ "type": "pause" }
{ "type": "resume" }
{ "type": "set_speed", "multiplier": 2 }
{ "type": "reprioritize_mission", "mission_id": "m-009", "priority": 0.95 }
```

Open questions for whoever wires the real transport: WebSocket vs SSE, auth (probably none needed
for a local demo), and whether `knowledge_query` is synchronous or streamed token-by-token.

---

## Internal service contract (gateway ⇄ simulation ⇄ agent intelligence)

Everything above is what the **frontend** sees. Internally, the gateway is not required to talk to
the other two services in any particular language — this is just the JSON shape.

### Simulation service (Avinandan) → Gateway

`GET /health` → `{ "ok": true }`

`GET /tick` (or push over its own WebSocket) → the same `world_state` shape documented above,
plus a `pending_observations` array the gateway forwards to the agent intelligence service:

```jsonc
{
  "world_state": { /* ...as above... */ },
  "pending_observations": [
    { "agent_id": "drone-2", "observation": { "local_grid": "...", "battery": 0.62, "nearby_agents": [...] } }
  ]
}
```

### Agent Intelligence service (Saatwik) → Simulation service / Gateway

`GET /health` → `{ "ok": true }`

`POST /decide` — request: `{ "observations": [ { "agent_id": "drone-2", "observation": {...} } ] }`
response:
```json
{
  "actions": [
    { "agent_id": "drone-2", "action": "move", "target": [14, 6] }
  ]
}
```

`POST /knowledge_query` — same request/response shape as the frontend-facing `knowledge_query`
above; the gateway just proxies to this endpoint directly.

### Wiring order

1. Gateway ships first with its own mock tick loop (no dependency on the other two).
2. Once the simulation service exposes `GET /tick`, the gateway swaps its mock loop for polling
   (or subscribing to) that endpoint — configured via `SIMULATION_SERVICE_URL` env var.
3. Once the agent intelligence service exposes `/decide` and `/knowledge_query`, the simulation
   service calls `/decide` each tick instead of moving agents randomly, and the gateway proxies
   `knowledge_query` to it via `AGENT_SERVICE_URL`.
