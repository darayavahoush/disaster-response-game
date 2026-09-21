# Task breakdown

Ground rule: build against **mock data / mock APIs** first so nobody blocks on anybody else.
The frontend already ships with a fake simulation engine (`frontend/src/lib/mockEngine.js`) that
produces exactly the JSON shape described in `api-contract.md` — treat that file as the spec.

## Backend split (3 services, one per person)

The backend isn't one blob — it's three small services with a clear contract between them, so
each of us owns a service end-to-end instead of all three fighting over one codebase.

```
 Frontend  ──WS/REST──▶  Gateway service (Ananya)
                              │  proxies knowledge queries, forwards commands
                              ▼
                    Simulation service (Avinandan)
                     world/env + CNN + LSTM + A*/MCTS
                              │  per-tick: "here's what each agent observes"
                              ▼
                  Agent Intelligence service (Saatwik)
                    MAPPO/MADDPG policies + LLM/RAG
                              │  returns: actions per agent + doctrine answers
                              ▲──────────────────────────┘
```

| Service | Owner | Responsibility |
|---|---|---|
| `backend-gateway/` | **Ananya** | The one door the frontend talks to: WebSocket broadcast of `world_state`, REST endpoints for commands (`pause`, `set_scenario`, `reprioritize_mission`, ...), proxies `knowledge_query` to Saatwik's service. Ships today with an authoritative mock world loop so the full stack runs end-to-end before the other two services exist — swapping in the real simulation service later is a one-file change. |
| `simulation-engine/` (+ service entrypoint) | **Avinandan** | Wraps the world/env, CNN perception, and LSTM hazard forecast behind an internal service that produces the real `world_state` tick and per-agent observations, replacing the gateway's mock loop. (Route/mission planning now lives with Ananya, see below — your job is to produce correct `hazards`/`blocked_routes`/`victims`, hers consumes them.) |
| `agents-rl-llm/` (+ service entrypoint) | **Saatwik** | Wraps MAPPO/MADDPG policies and the LLM/RAG knowledge base behind an internal `/decide` (per-agent action) and `/knowledge_query` (doctrine Q&A + task decomposition) API that the simulation service and gateway call into. |

See `docs/api-contract.md` for both the frontend-facing contract and the internal
service-to-service contract below it.

## Avinandan — World, Perception & Planning (`simulation-engine/`)

**1. World & disaster engine (`env/`)**
- Grid-based world (e.g. 40x40 cells): terrain, roads, buildings, elevation.
- Disaster generators: flood (rising water level + spread), cyclone (wind field + debris/blocked
  roads), earthquake (initial damage + aftershock events).
- A step loop that advances time, updates hazard cells, and exposes world state as JSON.
- Victim placement (with difficulty: some in view, some hidden behind rubble/water).

**2. CNN perception (`perception-cnn/`)**
- Input: simulated "sensor tiles" (rasterized local view a drone/vehicle sees — can literally be a
  crop of the world grid rendered to a small image/array).
- Output: victim detections (location + confidence), blocked-route classification per road segment.
- Start with a small CNN (e.g. 3-4 conv layers) trained on procedurally generated tiles before
  worrying about anything fancier — this is a perception *module*, not the main event.

**3. LSTM hazard forecasting (`hazard-lstm/`)**
- Input: sequence of past hazard-grid snapshots (water level / debris / damage per cell).
- Output: predicted hazard grid N steps ahead — this is what feeds the frontend's hazard-timeline
  and lets planners avoid routes that *will* flood, not just ones that already have.

**Suggested order:** world engine skeleton → expose state as JSON (unblocks frontend/gateway
integration, and unblocks Ananya's planner which already expects this exact shape) → CNN
perception → LSTM hazard forecasting, with the forecast array feeding straight into the planner's
`lookahead` option (`planning-astar-mcts/src/astar.js`) so routes can plan against predicted, not
just current, hazard.

> Route planning (A*) and mission sequencing (MCTS-flavored Monte Carlo assignment) moved to
> Ananya — see `planning-astar-mcts/` — since it plugs directly into the gateway she already
> owns. It's built and live already, consuming `world.hazards[].forecast` and
> `world.blocked_routes` in exactly the shape your env module needs to produce. Nothing changes
> for you here except: you don't need to build a planner, just make sure your world state matches
> `docs/api-contract.md`.

## Saatwik — Agent Intelligence (`agents-rl-llm/`)

**1. MAPPO/MADDPG coordination (`mappo-maddpg/`)**
- Define the multi-agent env interface (can wrap Avinandan's `env/` once it exists; use a stub env
  in the meantime so training code isn't blocked).
- Observation space per agent: local perception, own state (battery/fuel/capacity), shared mission
  board.
- Action space: move, scan, pick up/drop supply, request reinforcement.
- Reward shaping: victims rescued (weighted by urgency), time penalty, resource-waste penalty,
  collision/overlap penalty (to actually push toward *coordination*, not independent agents that
  ignore each other).
- Start with MAPPO (simpler, on-policy, more stable) and treat MADDPG as a stretch goal / comparison.

**2. LLM/RAG knowledge layer (`llm-rag/`)**
- Small retrieval corpus of real emergency-response doctrine (FEMA/NDMA-style triage rules,
  search-and-rescue protocols, supply-priority guidelines — public documents only).
- RAG pipeline: given the current situation summary (from the sim engine), retrieve relevant
  doctrine and have the LLM produce a **high-level task decomposition** ("prioritize the collapsed
  structure at sector C4, hold supply drop until route is confirmed clear") that gets handed down
  to the MAPPO policies as a soft objective/constraint, and to the frontend's knowledge panel as an
  explanation.

**Suggested order:** stub env + MAPPO skeleton training loop on a toy version of the task → reward
shaping iteration → RAG corpus + retrieval → LLM task-decomposition prompt → feed decomposition
into the policy layer as auxiliary reward/constraint → swap stub env for Avinandan's real env.

## Everyone — service entrypoints

Whatever language/framework you use internally, each service should expose:
- a small HTTP server with a `/health` endpoint (the gateway polls this to know if you're up)
- the tick/decision endpoints described in `api-contract.md`

That's the whole coupling surface. Everything else inside your service is yours to design.

## Ananya — Frontend, Gateway & Planning — done, see repo

**Frontend (`frontend/`):** command-center UI — tactical map, agent roster/HUD, mission queue,
resource allocation, hazard forecast timeline, LLM knowledge panel, scenario + playback controls.

**Gateway (`backend-gateway/`):** the service the frontend actually talks to — WebSocket
broadcast + REST commands + knowledge-query proxy, per `docs/api-contract.md`. Ships with an
authoritative mock world loop so the whole stack runs today without waiting on the other two
services; swapping in Avinandan's real engine and Saatwik's real agent service are both one env
var each (`SIMULATION_SERVICE_URL`, `AGENT_SERVICE_URL`).

**Planning (`planning-astar-mcts/`):** hazard-aware A* routing (cost-weighted by hazard intensity,
with an optional `lookahead` into the LSTM forecast array once that's real) + Monte Carlo mission
assignment (rollouts over victim/responder pairings, scored by severity + wait time + route cost —
the lightweight cousin of full MCTS at this scale). Already wired live into the gateway's tick
loop, not just a standalone module — run `npm test` in that folder for the smoke tests.

## Integration checkpoints

1. **Week 1:** Frontend renders mock data (done). Engine exposes real world-state JSON matching
   the contract. RL stub env exists.
2. **Week 2:** Frontend points at real engine's WebSocket instead of the mock. A* routes and
   hazard forecast are real. MAPPO trains on stub env with sane reward curves.
3. **Week 3:** MAPPO/MCTS output flows into the real engine's agents. LLM/RAG decomposition shows
   up in the frontend's knowledge panel. Polish, record a demo.
