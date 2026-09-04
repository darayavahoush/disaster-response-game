import { applyCommand } from "../engine/mockWorld.js";

// POST /command  — body matches one of the command shapes in docs/api-contract.md
export function makeCommandsRoute(world) {
  return (req, res) => {
    const cmd = req.body;
    if (!cmd || !cmd.type) {
      return res.status(400).json({ ok: false, error: "expected { type, ...params }" });
    }
    const result = applyCommand(world, cmd);
    res.status(result.ok ? 200 : 400).json(result);
  };
}
