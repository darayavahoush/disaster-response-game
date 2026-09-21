// Mission-level planning: *which* responder goes to *which* victim, in what order.
//
// True MCTS would build a search tree over the full multi-step mission sequence. At the
// scale of one tick (a handful of idle responders, a handful of newly-detected victims)
// a full tree is overkill, so this does the same thing in spirit at a cheaper cost: Monte
// Carlo rollouts over candidate assignments, scored by a rollout that accounts for time
// (via A* route cost) and priority (severity + wait time), keeping the best one found.
// Swap-in point for a real MCTS over multi-tick mission sequences later: `chooseAssignment`.

import { astar } from "./astar.js";

const ROLLOUTS = 24;

function eta(agent, victim, world) {
  const result = astar(agent.pos, victim.pos, world);
  if (!result) return { eta: Infinity, path: null };
  // ~12 "seconds" per unit of A* cost — matches the mock engine's existing pacing
  return { eta: Math.round(result.cost * 12), path: result.path };
}

function priorityScore(victim, waitedSeconds) {
  // severity dominates, but a victim that's been sitting detected-but-unassigned for a
  // while should climb the queue even if their severity score is middling
  return victim.severity * 0.75 + Math.min(waitedSeconds / 120, 1) * 0.25;
}

/**
 * Scores one full assignment (list of {agent, victim} pairs) by summing, for each pair,
 * priority / max(eta, 1) — i.e. reward high-priority victims reached quickly, penalize
 * sending anyone on a long detour when a closer responder was available for someone else.
 */
function scoreAssignment(pairs, etas) {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    const { victim } = pairs[i];
    const { eta: e } = etas[i];
    if (e === Infinity) continue;
    total += priorityScore(victim, victim._waited || 0) / Math.max(e, 1);
  }
  return total;
}

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * Greedily pairs agents to victims for one random ordering, so each is used at most once.
 */
function greedyPairForOrder(agents, victims, world) {
  const remainingAgents = agents.slice();
  const pairs = [];
  const etas = [];
  for (const victim of victims) {
    if (!remainingAgents.length) break;
    let bestIdx = -1;
    let best = null;
    for (let i = 0; i < remainingAgents.length; i++) {
      const r = eta(remainingAgents[i], victim, world);
      if (!best || r.eta < best.eta) {
        best = r;
        bestIdx = i;
      }
    }
    if (bestIdx === -1 || best.eta === Infinity) continue;
    pairs.push({ agent: remainingAgents[bestIdx], victim, path: best.path });
    etas.push(best);
    remainingAgents.splice(bestIdx, 1);
  }
  return { pairs, etas };
}

/**
 * Runs a handful of Monte Carlo rollouts over victim processing order and keeps the
 * highest-scoring resulting assignment. Returns [{ agent, victim, path, eta }].
 */
export function chooseAssignment(idleAgents, unassignedVictims, world) {
  if (!idleAgents.length || !unassignedVictims.length) return [];

  let bestPairs = null;
  let bestScore = -Infinity;

  for (let r = 0; r < ROLLOUTS; r++) {
    const order = shuffle(unassignedVictims);
    const { pairs, etas } = greedyPairForOrder(idleAgents, order, world);
    const score = scoreAssignment(pairs, etas);
    if (score > bestScore) {
      bestScore = score;
      bestPairs = pairs.map((p, i) => ({ ...p, eta: etas[i].eta }));
    }
  }

  return bestPairs || [];
}

/**
 * Recomputes a route for an agent already en route, e.g. when a new hazard forecast
 * or blocked route appears — call this instead of chooseAssignment for agents mid-mission.
 */
export function replan(agent, victim, world) {
  const result = astar(agent.pos, victim.pos, world);
  if (!result) return null;
  return { path: result.path, eta: Math.round(result.cost * 12) };
}
