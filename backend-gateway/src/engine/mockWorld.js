// Authoritative mock world engine.
//
// Stands in for the real simulation-engine service (Avinandan — world/env, CNN perception,
// LSTM hazard forecast) until it exposes GET /tick per docs/api-contract.md. Swap point: see
// server.js `getWorldState()`. Mission/route planning, however, is the *real* thing already —
// see ../../../planning-astar-mcts (Ananya): hazard-aware A* + Monte Carlo mission assignment.
//
// The state shape produced here is exactly the `world_state` object from
// docs/api-contract.md, so nothing downstream (frontend, or a future real engine)
// needs to change when this gets replaced.

import { chooseAssignment, replan } from "../../../planning-astar-mcts/src/missionPlanner.js";

const GRID = { width: 40, height: 26 };

const SCENARIOS = ["flood", "cyclone", "earthquake"];

const AGENT_TEMPLATES = [
  { id: "drone-1", type: "drone" },
  { id: "drone-2", type: "drone" },
  { id: "drone-3", type: "drone" },
  { id: "rescue_team-1", type: "rescue_team" },
  { id: "rescue_team-2", type: "rescue_team" },
  { id: "vehicle-1", type: "vehicle" },
  { id: "vehicle-2", type: "vehicle" },
];

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export function createWorld(scenario = "flood") {
  const agents = AGENT_TEMPLATES.map((t) => ({
    ...t,
    pos: [rand(2, GRID.width - 2), rand(2, GRID.height - 2)],
    battery: rand(0.7, 1),
    status: "idle",
    route: [],
    carrying: null,
  }));

  const victims = Array.from({ length: 18 }, (_, i) => ({
    id: `v-${String(i).padStart(3, "0")}`,
    pos: [rand(1, GRID.width - 1), rand(1, GRID.height - 1)],
    status: "hidden",
    severity: rand(0.2, 1),
    detected_by: null,
    confidence: 0,
  }));

  const hazards = Array.from({ length: 24 }, () => ({
    x: Math.floor(rand(0, GRID.width)),
    y: Math.floor(rand(0, GRID.height)),
    intensity: rand(0.2, 0.6),
    type: scenario === "flood" ? "water" : scenario === "cyclone" ? "wind" : "structural",
    forecast: [],
  }));
  hazards.forEach((h) => {
    let v = h.intensity;
    h.forecast = Array.from({ length: 4 }, () => {
      v = clamp(v + rand(-0.05, 0.12), 0, 1);
      return Number(v.toFixed(2));
    });
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
    metrics: {
      victims_rescued: 0,
      victims_total: victims.length,
      avg_response_seconds: 0,
      resource_efficiency: 1,
    },
    _responseTimes: [],
  };
}

export function step(world) {
  if (world.paused) return world;

  const dt = 1 * world.speed;
  world.tick += 1;
  world.clock_seconds += dt;

  // hazard drift: intensities creep along their forecast curve
  for (const h of world.hazards) {
    if (h.forecast.length) {
      h.intensity = clamp(h.intensity + (h.forecast[0] - h.intensity) * 0.1, 0, 1);
      if (Math.random() < 0.05) {
        h.forecast = h.forecast.slice(1).concat(clamp(h.forecast[h.forecast.length - 1] + rand(-0.08, 0.1), 0, 1));
      }
    }
  }

  // occasionally spawn a blocked route near a high hazard cell (stand-in for CNN route classification)
  if (Math.random() < 0.04 && world.hazards.length) {
    const h = world.hazards[Math.floor(rand(0, world.hazards.length))];
    world.blocked_routes.push({
      from: [Math.round(h.x), Math.round(h.y)],
      to: [Math.round(h.x) + 1, Math.round(h.y)],
      reason: h.type === "water" ? "flooded" : h.type === "wind" ? "debris" : "collapsed",
    });
    if (world.blocked_routes.length > 10) world.blocked_routes.shift();
  }

  // drones scan and detect victims (stand-in for CNN perception)
  for (const a of world.agents) {
    if (a.type !== "drone") continue;
    if (a.status === "idle") a.status = "scanning";
    for (const v of world.victims) {
      if (v.status !== "hidden") continue;
      if (dist(a.pos, v.pos) < 3 && Math.random() < 0.08) {
        v.status = "detected";
        v.detected_by = a.id;
        v.confidence = Number(rand(0.75, 0.98).toFixed(2));
      }
    }
    // wander while scanning
    a.pos = [
      clamp(a.pos[0] + rand(-0.6, 0.6), 0, GRID.width),
      clamp(a.pos[1] + rand(-0.6, 0.6), 0, GRID.height),
    ];
    a.battery = clamp(a.battery - 0.0015, 0, 1);
  }

  // track how long each detected victim has been waiting for assignment (feeds mission priority)
  for (const v of world.victims) {
    if (v.status === "detected") v._waited = (v._waited || 0) + dt;
  }

  // mission planner (real A*/MCTS — planning-astar-mcts, owned by Ananya): rank detected
  // victims and idle responders together, and route each assignment through the hazard-aware
  // A* graph rather than just picking whoever happens to be nearest.
  const idleResponders = world.agents.filter(
    (a) => (a.type === "rescue_team" || a.type === "vehicle") && a.status === "idle"
  );
  const unassignedVictims = world.victims.filter((v) => v.status === "detected");
  if (idleResponders.length && unassignedVictims.length) {
    const assignment = chooseAssignment(idleResponders, unassignedVictims, world);
    for (const { agent, victim, path, eta } of assignment) {
      const priority = Number(
        (victim.severity * 0.75 + Math.min((victim._waited || 0) / 120, 1) * 0.25).toFixed(2)
      );
      world.missions.push({
        id: `m-${world.missions.length}-${victim.id}`,
        victim_id: victim.id,
        priority,
        assigned_agent: agent.id,
        eta_seconds: eta,
        status: "in_progress",
        _startedAt: world.clock_seconds,
      });
      agent.status = "moving";
      agent.route = path;
      victim.status = "assigned";
    }
  }
  world.missions.sort((a, b) => b.priority - a.priority);

  // advance missions along their planned A* route, replanning if a new hazard/blockage
  // has made the current route stale
  for (const m of world.missions) {
    if (m.status !== "in_progress") continue;
    const agent = world.agents.find((a) => a.id === m.assigned_agent);
    const victim = world.victims.find((v) => v.id === m.victim_id);
    if (!agent || !victim) continue;

    if (!agent.route || agent.route.length < 2) {
      const r = replan(agent, victim, world);
      if (r) {
        agent.route = r.path;
        m.eta_seconds = r.eta;
      }
    }

    const d = dist(agent.pos, victim.pos);
    if (d > 0.3) {
      const waypoint = agent.route && agent.route.length > 1 ? agent.route[1] : victim.pos;
      const wd = dist(agent.pos, waypoint) || 1;
      const step = Math.min(0.5, wd);
      agent.pos = [
        agent.pos[0] + ((waypoint[0] - agent.pos[0]) / wd) * step,
        agent.pos[1] + ((waypoint[1] - agent.pos[1]) / wd) * step,
      ];
      if (wd < 0.3 && agent.route && agent.route.length > 1) agent.route = agent.route.slice(1);
      m.eta_seconds = Math.max(0, Math.round(dist(agent.pos, victim.pos) * 12));
    } else {
      m.status = "complete";
      victim.status = "rescued";
      agent.status = "idle";
      agent.route = [];
      const responseTime = world.clock_seconds - m._startedAt;
      world._responseTimes.push(responseTime);
      world.metrics.victims_rescued += 1;
      world.resources.medkits.available = Math.max(0, world.resources.medkits.available - 1);
      world.resources.water.available = Math.max(0, world.resources.water.available - 1);
    }
  }

  world.missions = world.missions.filter((m) => m.status === "in_progress");

  if (world._responseTimes.length) {
    world.metrics.avg_response_seconds = Math.round(
      world._responseTimes.reduce((s, x) => s + x, 0) / world._responseTimes.length
    );
  }
  const used =
    world.resources.medkits.total -
    world.resources.medkits.available +
    (world.resources.water.total - world.resources.water.available);
  const usedCap = world.resources.medkits.total + world.resources.water.total;
  world.metrics.resource_efficiency = Number(
    (world.metrics.victims_rescued / Math.max(1, used) || 1).toFixed(2)
  );

  return world;
}

export function toPublicState(world) {
  // strip internal bookkeeping (_startedAt, _responseTimes) before sending over the wire
  const { _responseTimes, missions, ...rest } = world;
  return {
    ...rest,
    missions: missions.map(({ _startedAt, ...m }) => m),
  };
}

export function applyCommand(world, cmd) {
  switch (cmd.type) {
    case "set_scenario": {
      if (!SCENARIOS.includes(cmd.scenario)) return { ok: false, error: "unknown scenario" };
      const fresh = createWorld(cmd.scenario);
      Object.assign(world, fresh);
      return { ok: true };
    }
    case "pause":
      world.paused = true;
      return { ok: true };
    case "resume":
      world.paused = false;
      return { ok: true };
    case "set_speed":
      world.speed = clamp(cmd.multiplier ?? 1, 0.25, 8);
      return { ok: true };
    case "reprioritize_mission": {
      const m = world.missions.find((m) => m.id === cmd.mission_id);
      if (!m) return { ok: false, error: "mission not found" };
      m.priority = clamp(cmd.priority ?? m.priority, 0, 1);
      return { ok: true };
    }
    default:
      return { ok: false, error: "unknown command type" };
  }
}
