import express from "express";
import cors from "cors";

import { createWorld, step, toPublicState, applyCommand } from "../env/world.js";

const PORT = process.env.PORT || 4100;

const world = createWorld("flood");

const app = express();
app.use(cors());
app.use(express.json());

app.get("/health", (_req, res) => res.json({ ok: true }));

// GET /tick — advances the world by one step and returns the latest state, per the internal
// service contract in docs/api-contract.md. Pull-based (advances only when polled) so the
// gateway's own tick loop stays the single clock for the whole system.
app.get("/tick", (_req, res) => {
  step(world);
  res.json({
    world_state: toPublicState(world),
    pending_observations: [], // no CNN/agent-intelligence service yet — see perception-cnn/
  });
});

app.post("/command", (req, res) => {
  const cmd = req.body;
  if (!cmd || !cmd.type) return res.status(400).json({ ok: false, error: "expected { type, ...params }" });
  const result = applyCommand(world, cmd);
  res.status(result.ok ? 200 : 400).json(result);
});

app.listen(PORT, () => {
  console.log(`Aegis simulation-engine listening on http://localhost:${PORT}`);
  console.log(`  GET  /health`);
  console.log(`  GET  /tick      — steps the world once, returns world_state`);
  console.log(`  POST /command   — { type: "set_scenario" | "pause" | "resume" | "set_speed", ... }`);
});
