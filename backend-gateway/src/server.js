import express from "express";
import cors from "cors";
import { createServer } from "http";
import { WebSocketServer } from "ws";

import { createWorld, step, toPublicState } from "./engine/mockWorld.js";
import { makeCommandsRoute } from "./routes/commands.js";
import { knowledgeQuery } from "./routes/knowledge.js";

const PORT = process.env.PORT || 4000;
const TICK_MS = 1000;

// SIMULATION_SERVICE_URL, once set, means: stop running the local mock loop and instead poll
// Avinandan's real simulation service for world state. Not wired yet — see docs/api-contract.md
// for the swap-over plan.
const SIMULATION_SERVICE_URL = process.env.SIMULATION_SERVICE_URL || null;

const world = createWorld("flood");

const app = express();
app.use(cors());
app.use(express.json());

app.get("/health", (_req, res) => res.json({ ok: true, mode: SIMULATION_SERVICE_URL ? "real-engine" : "mock" }));
app.get("/world_state", (_req, res) => res.json(toPublicState(world)));
app.post("/command", makeCommandsRoute(world));
app.post("/knowledge_query", knowledgeQuery);

const httpServer = createServer(app);
const wss = new WebSocketServer({ server: httpServer, path: "/ws" });

function broadcast(payload) {
  const msg = JSON.stringify(payload);
  for (const client of wss.clients) {
    if (client.readyState === client.OPEN) client.send(msg);
  }
}

wss.on("connection", (ws) => {
  // send current state immediately so a new client doesn't wait a full tick
  ws.send(JSON.stringify({ type: "world_state", payload: toPublicState(world) }));
});

async function tickLoop() {
  if (SIMULATION_SERVICE_URL) {
    try {
      const r = await fetch(`${SIMULATION_SERVICE_URL}/tick`);
      const data = await r.json();
      broadcast({ type: "world_state", payload: data.world_state });
      return;
    } catch (err) {
      console.error("simulation service unreachable, falling back to mock tick:", err.message);
    }
  }
  step(world);
  broadcast({ type: "world_state", payload: toPublicState(world) });
}

setInterval(tickLoop, TICK_MS);

httpServer.listen(PORT, () => {
  console.log(`Aegis gateway listening on http://localhost:${PORT}`);
  console.log(`  WebSocket:    ws://localhost:${PORT}/ws`);
  console.log(`  REST:         GET /world_state, POST /command, POST /knowledge_query`);
  console.log(`  Mode:         ${SIMULATION_SERVICE_URL ? `proxying ${SIMULATION_SERVICE_URL}` : "local mock engine"}`);
});
