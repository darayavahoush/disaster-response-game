import { astar } from "../src/astar.js";
import { chooseAssignment } from "../src/missionPlanner.js";

function assert(cond, msg) {
  if (!cond) {
    console.error("FAIL:", msg);
    process.exitCode = 1;
  } else {
    console.log("ok  -", msg);
  }
}

const world = {
  grid: { width: 10, height: 10 },
  hazards: [{ x: 5, y: 5, intensity: 0.9, forecast: [0.9, 0.95, 0.97, 0.99] }],
  blocked_routes: [],
};

const r = astar([0, 0], [9, 9], world);
assert(r && r.path.length > 0, "A* finds a path across an open grid with one hazard");
assert(r.path[0][0] === 0 && r.path[0][1] === 0, "path starts at the start cell");
assert(r.path[r.path.length - 1][0] === 9 && r.path[r.path.length - 1][1] === 9, "path ends at the goal cell");

const blockedWorld = { ...world, blocked_routes: [{ from: [8, 9], to: [9, 9], reason: "test" }] };
const r2 = astar([0, 0], [9, 9], blockedWorld);
assert(r2 === null, "A* returns null when the only approach to the goal is blocked");

const agents = [
  { id: "a1", pos: [0, 0] },
  { id: "a2", pos: [9, 0] },
];
const victims = [
  { id: "v1", pos: [1, 1], severity: 0.9, _waited: 0 },
  { id: "v2", pos: [8, 1], severity: 0.4, _waited: 0 },
];
const assignment = chooseAssignment(agents, victims, world);
assert(assignment.length === 2, "mission planner assigns both victims when both are reachable");
const a1Pair = assignment.find((p) => p.agent.id === "a1");
assert(a1Pair && a1Pair.victim.id === "v1", "closer agent gets paired with the nearby victim, not a random one");

console.log(process.exitCode ? "\nsome checks failed" : "\nall checks passed");
