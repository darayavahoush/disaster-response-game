// Grid A* pathfinding, cost-weighted by hazard intensity so routes bend away from
// dangerous cells instead of just taking the geometrically shortest path.
//
// Consumes the same `world_state` shape everyone else uses (docs/api-contract.md):
// world.grid, world.hazards (sparse, with per-cell intensity + forecast), world.blocked_routes.

const DIRS = [
  [1, 0], [-1, 0], [0, 1], [0, -1],
  [1, 1], [1, -1], [-1, 1], [-1, -1],
];

function key(x, y) {
  return `${x},${y}`;
}

function octile(x1, y1, x2, y2) {
  const dx = Math.abs(x1 - x2);
  const dy = Math.abs(y1 - y2);
  return Math.max(dx, dy) + (Math.SQRT2 - 1) * Math.min(dx, dy);
}

/**
 * Rasterizes the sparse hazard list into a per-cell extra-cost grid.
 * Each hazard cell radiates a cost penalty that falls off with distance, so A*
 * naturally prefers routes that skirt the edge of a hazard zone over the middle of it.
 * `lookahead` (0-3) picks which forecast step to plan against — 0 = current intensity,
 * higher = plan against where the hazard is *predicted* to be (LSTM output, once wired in).
 */
export function hazardCostGrid(world, { lookahead = 0, radius = 3, weight = 8 } = {}) {
  const { width, height } = world.grid;
  const cost = Array.from({ length: height }, () => new Array(width).fill(1));

  for (const h of world.hazards) {
    const intensity = lookahead > 0 && h.forecast?.length ? h.forecast[Math.min(lookahead - 1, h.forecast.length - 1)] : h.intensity;
    const cx = Math.round(h.x);
    const cy = Math.round(h.y);
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const x = cx + dx;
        const y = cy + dy;
        if (x < 0 || y < 0 || x >= width || y >= height) continue;
        const d = Math.hypot(dx, dy);
        if (d > radius) continue;
        cost[y][x] += intensity * (1 - d / radius) * weight;
      }
    }
  }
  return cost;
}

function blockedSet(world) {
  const s = new Set();
  for (const b of world.blocked_routes || []) {
    s.add(key(Math.round(b.to[0]), Math.round(b.to[1])));
  }
  return s;
}

function reconstructPath(nodes, endKey) {
  const path = [];
  let cur = endKey;
  while (cur) {
    const n = nodes.get(cur);
    path.push([n.x, n.y]);
    cur = n.parent;
  }
  return path.reverse();
}

/**
 * Returns { path: [[x,y], ...], cost: number } or null if unreachable.
 * `start`/`goal` are [x, y] in grid coordinates (fractional is fine, gets rounded).
 */
export function astar(start, goal, world, opts = {}) {
  const { width, height } = world.grid;
  const costGrid = hazardCostGrid(world, opts);
  const blocked = blockedSet(world);

  const sx = Math.round(start[0]);
  const sy = Math.round(start[1]);
  const gx = Math.round(goal[0]);
  const gy = Math.round(goal[1]);
  const goalKey = key(gx, gy);

  if (blocked.has(goalKey)) return null;

  const nodes = new Map();
  const startNode = { x: sx, y: sy, g: 0, f: octile(sx, sy, gx, gy), parent: null };
  nodes.set(key(sx, sy), startNode);

  const open = new Map([[key(sx, sy), startNode]]);
  const closed = new Set();
  let iterations = 0;

  while (open.size && iterations++ < 6000) {
    let curKey = null;
    let cur = null;
    for (const [k, n] of open) {
      if (!cur || n.f < cur.f) {
        cur = n;
        curKey = k;
      }
    }
    if (curKey === goalKey) {
      return { path: reconstructPath(nodes, curKey), cost: cur.g };
    }
    open.delete(curKey);
    closed.add(curKey);

    for (const [dx, dy] of DIRS) {
      const nx = cur.x + dx;
      const ny = cur.y + dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const nk = key(nx, ny);
      if (closed.has(nk) || blocked.has(nk)) continue;

      const diagonal = dx !== 0 && dy !== 0;
      const stepCost = costGrid[ny][nx] * (diagonal ? Math.SQRT2 : 1);
      const g = cur.g + stepCost;
      const existing = nodes.get(nk);
      if (!existing || g < existing.g) {
        const node = { x: nx, y: ny, g, f: g + octile(nx, ny, gx, gy), parent: curKey };
        nodes.set(nk, node);
        open.set(nk, node);
      }
    }
  }
  return null; // unreachable within iteration budget
}
