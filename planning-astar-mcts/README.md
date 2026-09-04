# Route & Mission Planning — owner: Ananya

Hazard-aware A* routing + Monte Carlo mission assignment. Originally scoped as part of the
simulation engine (Avinandan); moved here because it plugs directly into the gateway service
Ananya already owns, and both are JS — no cross-language boundary to maintain.

Already wired live into `../backend-gateway/src/engine/mockWorld.js` — this isn't a standalone
exercise, it's running in the actual tick loop today.

## `src/astar.js`

Grid A* where each cell's traversal cost is bumped by nearby hazard intensity (rasterized from
the sparse `world.hazards` list), so routes bend around danger instead of just taking the
geometrically shortest path. `blocked_routes` cells are hard-excluded.

`astar(start, goal, world, { lookahead })` — `lookahead` (0-3) picks which step of a hazard's
`forecast` array to plan against instead of its current `intensity`. Once Avinandan's LSTM is
producing real forecasts, bumping `lookahead` above 0 makes routes plan against where a flood
*will be*, not just where it is now — no other code needs to change.

## `src/missionPlanner.js`

Full multi-step MCTS over mission *sequences* is overkill for a handful of idle responders and
newly-detected victims per tick. `chooseAssignment` gets the same effect more cheaply: Monte Carlo
rollouts over victim-processing order, each rolled out with a greedy nearest-by-A*-cost pairing,
scored by `severity + wait time` weighted against route ETA, keeping the best rollout found.

`replan(agent, victim, world)` re-runs A* for an agent already mid-mission — call this when a new
hazard or blocked route makes the current route stale (the gateway already does, whenever an
agent's route runs out or looks empty).

## Testing

```bash
npm test
```

Smoke tests cover: pathing across open ground with a hazard in the way, correctly returning `null`
when the goal is unreachable, and that the assignment step pairs the closer agent with the nearby
victim rather than something arbitrary.

## Stretch goals

- Real tree-search MCTS over multi-tick mission sequences once there are enough simultaneous
  missions for greedy rollouts to start leaving value on the table.
- Weight the hazard cost grid by agent type (a drone should barely care about flood water; a
  ground vehicle should care a lot).
- Feed `suggested_task_decomposition` from Saatwik's LLM/RAG service in as a soft constraint on
  assignment (e.g. "hold vehicle dispatch" removes vehicles from the idle-responder pool for a
  mission until the constraint clears).
