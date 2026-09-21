// Lightweight client-side fallback so the UI has something real to render even if
// backend-gateway isn't running. Produces the same shape as docs/api-contract.md, just with
// simpler physics (no real A*/hazard-cost routing — that lives server-side in
// planning-astar-mcts). This is a demo convenience, not a second implementation to maintain.

const GRID = { width: 40, height: 26 };

const AGENT_TEMPLATES = [
  { id: "drone-1", type: "drone" },
  { id: "drone-2", type: "drone" },
  { id: "drone-3", type: "drone" },
  { id: "rescue_team-1", type: "rescue_team" },
  { id: "rescue_team-2", type: "rescue_team" },
  { id: "vehicle-1", type: "vehicle" },
  { id: "vehicle-2", type: "vehicle" },
];

const rand = (min, max) => min + Math.random() * (max - min);
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export function createMockWorld(scenario = "flood") {
  const agents = AGENT_TEMPLATES.map((t) => ({
    ...t,
    pos: [rand(2, GRID.width - 2), rand(2, GRID.height - 2)],
    battery: rand(0.7, 1),
    status: "idle",
    route: [],
    carrying: null,
  }));

  const victims = Array.from({ length: 16 }, (_, i) => ({
    id: `v-${String(i).padStart(3, "0")}`,
    pos: [rand(1, GRID.width - 1), rand(1, GRID.height - 1)],
    status: "hidden",
    severity: rand(0.2, 1),
    detected_by: null,
    confidence: 0,
  }));

  const hazards = Array.from({ length: 20 }, () => {
    const intensity = rand(0.2, 0.6);
    return {
      x: rand(0, GRID.width),
      y: rand(0, GRID.height),
      intensity,
      type: scenario === "flood" ? "water" : scenario === "cyclone" ? "wind" : "structural",
      forecast: [0, 1, 2, 3].map(() => clamp(intensity + rand(-0.1, 0.25), 0, 1)),
    };
  });

  return {
    tick: 0,
    scenario,
    clock_seconds: 0,
    speed: 1,
    paused: false,
    grid: GRID,
    hazards,
    blocked_routes: [],
    victims,
    agents,
    missions: [],
    resources: {
      medkits: { available: 20, total: 20 },
      water: { available: 40, total: 40 },
      evac_capacity: { available: 6, total: 6 },
    },
    metrics: { victims_rescued: 0, victims_total: victims.length, avg_response_seconds: 0, resource_efficiency: 1 },
    _responseTimes: [],
  };
}

export function stepMockWorld(world) {
  if (world.paused) return world;
  world.tick += 1;
  world.clock_seconds += world.speed;

  for (const h of world.hazards) {
    h.intensity = clamp(h.intensity + (h.forecast[0] - h.intensity) * 0.1, 0, 1);
  }

  if (Math.random() < 0.05 && world.hazards.length) {
    const h = world.hazards[Math.floor(rand(0, world.hazards.length))];
    world.blocked_routes.push({
      from: [Math.round(h.x), Math.round(h.y)],
      to: [Math.round(h.x) + 1, Math.round(h.y)],
      reason: h.type === "water" ? "flooded" : h.type === "wind" ? "debris" : "collapsed",
    });
    if (world.blocked_routes.length > 8) world.blocked_routes.shift();
  }

  for (const a of world.agents) {
    if (a.type === "drone") {
      if (a.status === "idle") a.status = "scanning";
      for (const v of world.victims) {
        if (v.status !== "hidden") continue;
        if (dist(a.pos, v.pos) < 3 && Math.random() < 0.08) {
          v.status = "detected";
          v.detected_by = a.id;
          v.confidence = Number(rand(0.75, 0.98).toFixed(2));
        }
      }
      a.pos = [clamp(a.pos[0] + rand(-0.6, 0.6), 0, GRID.width), clamp(a.pos[1] + rand(-0.6, 0.6), 0, GRID.height)];
      a.battery = clamp(a.battery - 0.0015, 0, 1);
    }
  }

  for (const v of world.victims) {
    if (v.status !== "detected") continue;
    if (world.missions.find((m) => m.victim_id === v.id)) continue;
    const responder = world.agents.find((a) => (a.type === "rescue_team" || a.type === "vehicle") && a.status === "idle");
    if (!responder) continue;
    world.missions.push({
      id: `m-${world.missions.length}-${v.id}`,
      victim_id: v.id,
      priority: Number((v.severity * 0.75 + rand(0, 0.25)).toFixed(2)),
      assigned_agent: responder.id,
      eta_seconds: Math.round(dist(responder.pos, v.pos) * 10),
      status: "in_progress",
      _startedAt: world.clock_seconds,
    });
    responder.status = "moving";
    v.status = "assigned";
  }
  world.missions.sort((a, b) => b.priority - a.priority);

  for (const m of world.missions) {
    if (m.status !== "in_progress") continue;
    const agent = world.agents.find((a) => a.id === m.assigned_agent);
    const victim = world.victims.find((v) => v.id === m.victim_id);
    if (!agent || !victim) continue;
    const d = dist(agent.pos, victim.pos);
    if (d > 0.3) {
      const step = Math.min(0.5, d);
      agent.pos = [agent.pos[0] + ((victim.pos[0] - agent.pos[0]) / d) * step, agent.pos[1] + ((victim.pos[1] - agent.pos[1]) / d) * step];
      agent.route = [agent.pos, victim.pos];
      m.eta_seconds = Math.max(0, Math.round(d * 10));
    } else {
      m.status = "complete";
      victim.status = "rescued";
      agent.status = "idle";
      agent.route = [];
      world._responseTimes.push(world.clock_seconds - m._startedAt);
      world.metrics.victims_rescued += 1;
      world.resources.medkits.available = Math.max(0, world.resources.medkits.available - 1);
      world.resources.water.available = Math.max(0, world.resources.water.available - 1);
    }
  }
  world.missions = world.missions.filter((m) => m.status === "in_progress");

  if (world._responseTimes.length) {
    world.metrics.avg_response_seconds = Math.round(world._responseTimes.reduce((s, x) => s + x, 0) / world._responseTimes.length);
  }

  return { ...world };
}

const CANNED_KNOWLEDGE = [
  {
    match: /collaps|structur|rubble|trapped/i,
    answer:
      "Stabilize the structure before entry. Require a two-person minimum with a dedicated spotter watching for secondary collapse.",
    sources: ["FEMA US&R Field Operations Guide, ch. 4"],
    suggested_task_decomposition: ["Hold vehicle dispatch until structural stabilization is confirmed"],
  },
  {
    match: /flood|water|drown|submerg/i,
    answer:
      "Prioritize victims in rising-water zones over static ones even at lower severity — water hazard is time-dilating risk, not fixed damage.",
    sources: ["NDMA Flood Response Guidelines"],
    suggested_task_decomposition: ["Reroute ground vehicles around cells with forecast intensity above 0.6"],
  },
];

export async function mockKnowledgeQuery(question) {
  await new Promise((r) => setTimeout(r, 500));
  const hit = CANNED_KNOWLEDGE.find((c) => c.match.test(question));
  return (
    hit || {
      answer: "No doctrine match for that yet (offline demo mode — connect the gateway for real RAG answers).",
      sources: [],
      suggested_task_decomposition: [],
    }
  );
}
