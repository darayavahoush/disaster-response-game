import express from "express";
import cors from "cors";
import { createServer } from "http";
import { WebSocketServer } from "ws";

import { createWorld, step, toPublicState, mergeEngineTick } from "./engine/mockWorld.js";
import { makeCommandsRoute } from "./routes/commands.js";
import { knowledgeQuery } from "./routes/knowledge.js";

const PORT = process.env.PORT || 4000;
const TICK_MS = 1000;

// SIMULATION_SERVICE_URL, once set, means: stop generating hazards/blocked_routes/victims
// locally and instead poll the real simulation-engine for them each tick (see mergeEngineTick
// in ./engine/mockWorld.js). Agents/missions/resources stay local either way.
const SIMULATION_SERVICE_URL = process.env.SIMULATION_SERVICE_URL || null;

const world = createWorld("flood");

const app = express();
app.use(cors());
app.use(express.json());

app.get("/health", (_req, res) => res.json({ ok: true, mode: SIMULATION_SERVICE_URL ? "real-engine" : "mock" }));
app.get("/world_state", (_req, res) => res.json(toPublicState(world)));
app.post("/command", async (req, res, next) => {
  if (SIMULATION_SERVICE_URL) {
    try {
      await fetch(`${SIMULATION_SERVICE_URL}/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
    } catch (err) {
      console.error("failed to forward command to simulation service:", err.message);
    }
  }
  return makeCommandsRoute(world)(req, res, next);
});
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
      mergeEngineTick(world, data.world_state);
    } catch (err) {
      console.error("simulation service unreachable, falling back to local physics for this tick:", err.message);
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
