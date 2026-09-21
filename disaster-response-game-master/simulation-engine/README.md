# Simulation Engine — owner: Avinandan

World state, disaster generation, CNN perception, LSTM hazard forecasting.
See `../docs/TASKS.md` for the detailed breakdown and `../docs/api-contract.md` for the JSON
shape this needs to expose downstream.

## Status

- [x] World engine: terrain (elevation/roads/buildings), flood/cyclone/earthquake physics,
      victim placement — `env/world.js`
- [x] Service entrypoint (`GET /health`, `GET /tick`, `POST /command`) — `service/server.js`
- [ ] CNN perception (`perception-cnn/`) — still a stub; victim detection is currently a
      distance+random stand-in living in the gateway's mock loop
- [ ] LSTM hazard forecasting (`hazard-lstm/`) — not needed yet in the sense that `env/world.js`
      already produces a *real* forecast array per hazard cell (same deterministic physics
      evaluated a few ticks ahead), so the frontend's hazard-timeline already has something
      genuine to show. An LSTM would replace that closed-form forecast with a learned one once
      the physics gets complex enough that a closed form stops being tractable (e.g. real
      diffusion/rainfall-runoff instead of a single rising water-level scalar).

## Running it

```bash
cd simulation-engine
npm install
npm run dev          # http://localhost:4100
```

Try it standalone:

```bash
curl http://localhost:4100/tick | jq
curl -X POST http://localhost:4100/command -H "Content-Type: application/json" \
  -d '{"type":"set_scenario","scenario":"cyclone"}'
```

To wire it into the full stack, run this alongside `backend-gateway` with
`SIMULATION_SERVICE_URL=http://localhost:4100`:

```bash
cd backend-gateway
SIMULATION_SERVICE_URL=http://localhost:4100 npm run dev
```

The gateway merges this service's `hazards`/`blocked_routes`/`grid` into its own world state
every tick, and reseeds `victims` from here whenever the scenario changes — it keeps owning
`agents`/`missions`/`resources` itself since that's where the A*/MCTS planner and mission
lifecycle already live. See `mergeEngineTick()` in `backend-gateway/src/engine/mockWorld.js`.

## Layout
- `env/world.js` — terrain generation (deterministic, seeded — same city every run), disaster
  physics (flood = rising water level vs. elevation; cyclone = moving wind-field storm center;
  earthquake = epicenter + random aftershocks), road-graph-derived `blocked_routes`, victim
  placement
- `service/server.js` — the HTTP entrypoint described above
- `perception-cnn/` — victim + blocked-route detection from simulated sensor tiles (stub)
- `hazard-lstm/` — learned hazard-spread forecasting, if/when the closed-form physics above
  stops being enough (stub)

Route planning (A*) and mission assignment (Monte Carlo / MCTS-flavored) live in
`../planning-astar-mcts/` (Ananya) — it's already wired into the gateway and consumes exactly the
`hazards`, `blocked_routes`, and `victims` shape this service produces.

## Design note: why `agents: []`

This service's `world_state.agents` is intentionally empty. Per `docs/TASKS.md`, agent lifecycle
(movement, mission assignment, battery/resource use) is owned by the gateway's planner until an
agent-intelligence service (Saatwik's MAPPO/MADDPG + LLM/RAG) exists to actually decide what
agents do. This engine's job is the *physical world* they operate in — terrain, hazards, blocked
routes, and where victims are — not the agents themselves.
