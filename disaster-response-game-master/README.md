# Aegis — Multi-Agent Disaster Response Simulation

A game/simulation where autonomous **drones**, **rescue teams**, and **vehicles** coordinate
to save lives after floods, cyclones, and earthquakes — under real time and resource pressure.

Built by **Ananya**, **Avinandan**, and **Saatwik**.

## The idea

A disaster hits a city grid. Roads collapse, water rises, buildings weaken over time.
Victims are scattered and hidden. A fleet of heterogeneous agents has to:

1. **See** — read noisy satellite/sensor tiles to find victims and hazards (CNN perception).
2. **Foresee** — predict how the flood/fire/aftershock zone will evolve in the next N minutes (LSTM).
3. **Plan** — find routes through a shifting, partially-blocked road network and sequence missions (A* / MCTS).
4. **Reason** — pull real emergency-response doctrine to decide *what* to do, not just *how* (LLM + RAG).
5. **Coordinate** — many agents sharing limited supplies and time, learning to cooperate instead of colliding (MAPPO / MADDPG).
6. **Show it** — a command-center UI where you watch (and nudge) all of this happen live.

## Architecture

```
                         ┌───────────────────────────┐
                         │        Frontend (Ananya)   │
                         │  Tactical map · agent HUD  │
                         │  mission queue · knowledge │
                         │  panel · hazard timeline   │
                         └─────────────▲──────────────┘
                                       │ WebSocket / REST (sim state, commands)
                         ┌─────────────┴──────────────┐
                         │   Simulation Engine (Avi)   │
                         │  world/grid state, physics  │
                         │  ┌────────────┐ ┌─────────┐ │
                         │  │ CNN percept.│ │  LSTM   │ │
                         │  │ (victims,   │ │ hazard  │ │
                         │  │ blocked rds)│ │ forecast│ │
                         │  └────────────┘ └─────────┘ │
                         │  ┌────────────────────────┐ │
                         │  │  A* / MCTS route +      │ │
                         │  │  mission planner        │ │
                         │  └────────────────────────┘ │
                         └─────────────▲──────────────┘
                                       │ agent observations / actions
                         ┌─────────────┴──────────────┐
                         │   Agent Intelligence (Saatwik)│
                         │  ┌────────────┐ ┌─────────┐ │
                         │  │ MAPPO /     │ │ LLM+RAG │ │
                         │  │ MADDPG      │ │ task    │ │
                         │  │ policies    │ │ decomp. │ │
                         │  └────────────┘ └─────────┘ │
                         └─────────────────────────────┘
```

The three pieces talk over a small JSON contract (see `docs/api-contract.md`, to be filled in as
the sim engine's endpoints firm up) so all three of us can build in parallel against mock data
before wiring things together.

## Repo layout

```
disaster-response-game/
├── frontend/                  → Ananya — command-center UI
├── backend-gateway/           → Ananya — gateway service: WS/REST to the frontend, mock world loop
├── planning-astar-mcts/       → Ananya — hazard-aware A* routing + Monte Carlo mission assignment
│                                 (already wired live into backend-gateway's tick loop)
├── simulation-engine/         → Avinandan — world state, perception, hazard forecasting
│   ├── env/                   world/grid, agents, disaster events, step loop
│   ├── perception-cnn/        CNN over simulated map/sensor tiles → victims + blocked routes
│   └── hazard-lstm/           LSTM hazard spread forecasting
├── agents-rl-llm/             → Saatwik — decision-making layer
│   ├── mappo-maddpg/          multi-agent RL coordination policies
│   └── llm-rag/               emergency-response knowledge base + task decomposition
├── docs/                      architecture notes, API contract, scenario specs
└── .github/workflows/         CI
```

## Task division

| Track | Owner | Covers |
|---|---|---|
| **Frontend, Gateway & Planning** | **Ananya** | Tactical map, agent HUD, mission/priority queue, resource allocation view, hazard-forecast timeline, LLM knowledge panel; the gateway backend service (WebSocket/REST, mock world loop); hazard-aware **A\*** routing and Monte-Carlo (**MCTS-flavored**) mission assignment — `planning-astar-mcts/` |
| **World & Perception** | **Avinandan** | Grid/world simulation, disaster event generator, CNN perception (victim/route detection from map tiles), LSTM hazard-spread forecasting, and the simulation backend service that serves all of it |
| **Agent Intelligence** | **Saatwik** | MAPPO/MADDPG multi-agent coordination policies, reward shaping, LLM/RAG emergency-doctrine knowledge base, high-level task decomposition, and the agent-intelligence backend service |

See `docs/TASKS.md` for the detailed, week-by-week breakdown for Avinandan and Saatwik, and
`docs/api-contract.md` for the state/action JSON shape the frontend expects from the engine so
everyone can build against a stable mock before the real thing exists.

## Getting started (frontend)

```bash
cd frontend
npm install
npm run dev
```

Runs on mock simulation data out of the box — no backend needed yet. Swap `src/lib/mockEngine.js`
for a real WebSocket client once the simulation engine exposes one (see `docs/api-contract.md`).

## Status

- [x] Repo structure
- [x] Frontend command-center UI, now on a real satellite/street map (Chennai coastline anchor)
- [x] Simulation engine world/physics (terrain, flood/cyclone/earthquake, victim placement) —
      `simulation-engine/`. CNN perception and LSTM forecasting still stubs; see its README.
- [x] Gateway wired to the real simulation engine (`SIMULATION_SERVICE_URL`) — merges real
      hazards/blocked_routes/victims each tick, still owns agents/missions/resources locally
- [ ] Agent intelligence (MAPPO/MADDPG, LLM/RAG) — Saatwik
- [x] Frontend connects live over WebSocket automatically (`frontend/src/lib/useSimulation.js`)
      when the gateway is reachable, falling back to its own offline mock otherwise — nothing
      further needed here, just run all three services (see `simulation-engine/README.md`)
