# Simulation Engine — owner: Avinandan

World state, disaster generation, CNN perception, LSTM hazard forecasting.
See `../docs/TASKS.md` for the detailed breakdown and `../docs/api-contract.md` for the JSON
shape this needs to expose downstream.

## Layout
- `env/` — grid world, disaster event generators, step loop
- `perception-cnn/` — victim + blocked-route detection from simulated sensor tiles
- `hazard-lstm/` — hazard-spread forecasting

Route planning (A*) and mission assignment (Monte Carlo / MCTS-flavored) live in
`../planning-astar-mcts/` (Ananya) — it's already wired into the gateway and consumes exactly the
`hazards`, `blocked_routes`, and `victims` shape this service needs to produce. You don't need to
build a planner; just match `docs/api-contract.md` and hers plugs straight in.
