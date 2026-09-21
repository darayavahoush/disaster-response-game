// Real world engine: terrain (elevation/roads/buildings), disaster physics (flood/cyclone/
// earthquake), and victim placement. Replaces the random hazard-drift stand-in that currently
// lives in backend-gateway/src/engine/mockWorld.js.
//
// Scope note: this module owns hazards/blocked_routes/victims/terrain — the physical world —
// per docs/TASKS.md ("your job is to produce correct hazards/blocked_routes/victims, hers
// consumes them"). Agents, missions, and resources stay authoritative in the gateway (that's
// where the real A*/MCTS planner already lives) — this engine reports `agents: []` on purpose;
// see simulation-engine/README.md for how the gateway merges this into its own world state.

const GRID = { width: 40, height: 26 };

// ---------- deterministic noise (no deps) ----------

function hash(x, y, seed) {
  const h = Math.sin(x * 127.1 + y * 311.7 + seed * 74.7) * 43758.5453123;
  return h - Math.floor(h);
}

function fade(t) {
  return t * t * (3 - 2 * t);
}

function valueNoise(x, y, seed) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const a = hash(xi, yi, seed);
  const b = hash(xi + 1, yi, seed);
  const c = hash(xi, yi + 1, seed);
  const d = hash(xi + 1, yi + 1, seed);
  const u = fade(xf);
  const v = fade(yf);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// ---------- terrain (fixed regardless of scenario — same city every time) ----------

const TERRAIN_SEED = 7;
const ROAD_SPACING = 4;

function buildTerrain(grid) {
  const elevation = [];
  const isRoad = [];
  const isBuilding = [];

  for (let y = 0; y < grid.height; y++) {
    const elevRow = [];
    const roadRow = [];
    const buildingRow = [];
    for (let x = 0; x < grid.width; x++) {
      // Coastal gradient: our real-world anchor (see frontend/src/lib/geo.js) puts the coast
      // at the east edge (high x) and inland at the west edge (low x) — Chennai's Bay of
      // Bengal coastline. Elevation is high inland, low at the coast, plus organic noise.
      const coastal = 1 - x / grid.width;
      const noise = valueNoise(x * 0.15, y * 0.15, TERRAIN_SEED);
      const elev = clamp(coastal * 0.75 + noise * 0.35, 0, 1);
      elevRow.push(Number(elev.toFixed(3)));

      const road = x % ROAD_SPACING === 0 || y % ROAD_SPACING === 0;
      roadRow.push(road);
      buildingRow.push(!road && hash(x, y, TERRAIN_SEED + 3) < 0.55);
    }
    elevation.push(elevRow);
    isRoad.push(roadRow);
    isBuilding.push(buildingRow);
  }

  // road graph: 4-connected edges between adjacent road cells, used to derive blocked_routes
  const edges = [];
  for (let y = 0; y < grid.height; y++) {
    for (let x = 0; x < grid.width; x++) {
      if (!isRoad[y][x]) continue;
      if (x + 1 < grid.width && isRoad[y][x + 1]) edges.push([[x, y], [x + 1, y]]);
      if (y + 1 < grid.height && isRoad[y + 1][x]) edges.push([[x, y], [x, y + 1]]);
    }
  }

  return { elevation, isRoad, isBuilding, edges };
}

// ---------- disaster physics ----------
// Each returns { hazardType, intensityAt(x, y, tOffset) } — tOffset lets us cheaply compute a
// real forecast (same deterministic formula evaluated at a future tick) instead of faking one.

function floodPhysics(world) {
  const rate = 0.01 * world.speed;
  return {
    hazardType: "water",
    intensityAt(x, y, tOffset = 0) {
      const waterLevel = clamp(world.disaster.waterLevel + rate * tOffset, 0, 0.95);
      const elev = world.terrain.elevation[Math.floor(y)][Math.floor(x)];
      if (waterLevel <= elev) return 0;
      return clamp((waterLevel - elev) / 0.3, 0, 1);
    },
    advance() {
      world.disaster.waterLevel = clamp(world.disaster.waterLevel + rate, 0, 0.95);
    },
  };
}

function cyclonePhysics(world) {
  const speed = 0.15 * world.speed;
  return {
    hazardType: "wind",
    intensityAt(x, y, tOffset = 0) {
      const cx = world.disaster.centerX + Math.cos(world.disaster.angle) * speed * tOffset;
      const cy = world.disaster.centerY + Math.sin(world.disaster.angle) * speed * tOffset;
      const d = Math.hypot(x - cx, y - cy);
      const radius = world.disaster.radius;
      if (d > radius) return 0;
      const core = clamp(1 - d / radius, 0, 1);
      const gust = 0.15 * valueNoise(x * 0.3 + tOffset * 0.2, y * 0.3, 42);
      return clamp(core * 0.85 + gust, 0, 1);
    },
    advance() {
      world.disaster.centerX += Math.cos(world.disaster.angle) * speed;
      world.disaster.centerY += Math.sin(world.disaster.angle) * speed;
    },
  };
}

function earthquakePhysics(world) {
  return {
    hazardType: "structural",
    intensityAt(x, y, tOffset = 0) {
      const d = Math.hypot(x - world.disaster.epicenterX, y - world.disaster.epicenterY);
      const base = clamp(world.disaster.baseDamage - d * 0.05, 0, 1);
      let aftershock = 0;
      for (const a of world.disaster.aftershocks) {
        const age = a.age + tOffset;
        const ad = Math.hypot(x - a.x, y - a.y);
        if (ad < a.radius) {
          aftershock = Math.max(aftershock, clamp((a.strength - age * 0.08) * (1 - ad / a.radius), 0, 1));
        }
      }
      return clamp(base + aftershock, 0, 1);
    },
    advance() {
      for (const a of world.disaster.aftershocks) a.age += 1;
      world.disaster.aftershocks = world.disaster.aftershocks.filter((a) => a.age < a.strength / 0.08);
      if (Math.random() < 0.03 * world.speed) {
        world.disaster.aftershocks.push({
          x: Math.floor(Math.random() * world.grid.width),
          y: Math.floor(Math.random() * world.grid.height),
          strength: 0.4 + Math.random() * 0.4,
          radius: 3 + Math.random() * 3,
          age: 0,
        });
      }
    },
  };
}

function makeDisasterState(scenario, grid) {
  if (scenario === "cyclone") {
    return { centerX: -3, centerY: grid.height / 2, radius: 9, angle: 0.15 };
  }
  if (scenario === "earthquake") {
    return {
      epicenterX: Math.floor(grid.width * 0.5),
      epicenterY: Math.floor(grid.height * 0.5),
      baseDamage: 0.55,
      aftershocks: [],
    };
  }
  return { waterLevel: 0.15 }; // flood (default)
}

function physicsFor(scenario, world) {
  if (scenario === "cyclone") return cyclonePhysics(world);
  if (scenario === "earthquake") return earthquakePhysics(world);
  return floodPhysics(world);
}

// ---------- victims ----------

function placeVictims(terrain, grid, count = 18) {
  const victims = [];
  let tries = 0;
  while (victims.length < count && tries < count * 40) {
    tries++;
    const x = Math.floor(Math.random() * grid.width);
    const y = Math.floor(Math.random() * grid.height);
    if (terrain.isBuilding[y][x]) continue; // victims are near/inside structures, placed just outside them
    victims.push({
      id: `v-${String(victims.length).padStart(3, "0")}`,
      pos: [x + 0.5, y + 0.5],
      status: "hidden", // CNN perception (not built yet) is what would flip this — see perception-cnn/
      severity: Number((0.2 + Math.random() * 0.8).toFixed(2)),
      detected_by: null,
      confidence: 0,
    });
  }
  return victims;
}

// ---------- world lifecycle ----------

let terrainCache = null;
function getTerrain(grid) {
  if (!terrainCache) terrainCache = buildTerrain(grid);
  return terrainCache;
}

export function createWorld(scenario = "flood") {
  const grid = GRID;
  const terrain = getTerrain(grid);
  return {
    tick: 0,
    clock_seconds: 0,
    scenario,
    speed: 1,
    paused: false,
    grid,
    terrain,
    disaster: makeDisasterState(scenario, grid),
    victims: placeVictims(terrain, grid),
  };
}

const HAZARD_STRIDE = 2; // sample every 2nd cell to keep the sparse hazard list a sane size
const HAZARD_THRESHOLD = 0.1;
const BLOCK_THRESHOLD = 0.5;
const FORECAST_STEPS = 4;

function computeHazards(world, physics) {
  const hazards = [];
  for (let y = 0; y < world.grid.height; y += HAZARD_STRIDE) {
    for (let x = 0; x < world.grid.width; x += HAZARD_STRIDE) {
      const intensity = physics.intensityAt(x, y, 0);
      if (intensity < HAZARD_THRESHOLD) continue;
      const forecast = Array.from({ length: FORECAST_STEPS }, (_, i) =>
        Number(physics.intensityAt(x, y, i + 1).toFixed(2))
      );
      hazards.push({ x, y, intensity: Number(intensity.toFixed(2)), type: physics.hazardType, forecast });
    }
  }
  return hazards;
}

function computeBlockedRoutes(world, physics) {
  const reason = physics.hazardType === "water" ? "flooded" : physics.hazardType === "wind" ? "debris" : "collapsed";
  const blocked = [];
  for (const [from, to] of world.terrain.edges) {
    // both endpoints are integer grid cells (edges only ever join adjacent road cells) — take
    // the worse of the two rather than an in-between fractional cell that isn't a real index.
    const intensity = Math.max(physics.intensityAt(from[0], from[1], 0), physics.intensityAt(to[0], to[1], 0));
    if (intensity >= BLOCK_THRESHOLD) {
      blocked.push({ from, to, reason });
    }
  }
  return blocked;
}

export function step(world) {
  if (world.paused) return world;
  world.tick += 1;
  world.clock_seconds += 1 * world.speed;

  const physics = physicsFor(world.scenario, world);
  physics.advance();

  world._hazards = computeHazards(world, physics);
  world._blockedRoutes = computeBlockedRoutes(world, physics);
  return world;
}

export function toPublicState(world) {
  return {
    tick: world.tick,
    scenario: world.scenario,
    clock_seconds: world.clock_seconds,
    grid: world.grid,
    hazards: world._hazards || [],
    blocked_routes: world._blockedRoutes || [],
    victims: world.victims,
    // Owned by the gateway's planner (planning-astar-mcts) — see module comment at top.
    agents: [],
    missions: [],
    resources: null,
    metrics: null,
  };
}

export function applyCommand(world, cmd) {
  switch (cmd.type) {
    case "set_scenario": {
      if (!["flood", "cyclone", "earthquake"].includes(cmd.scenario)) {
        return { ok: false, error: "unknown scenario" };
      }
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
    default:
      return { ok: false, error: "unknown command type" };
  }
}
