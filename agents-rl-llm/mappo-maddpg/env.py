from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


DEFAULT_ACTION_SPACE = ["move", "scan", "rescue", "deliver", "return", "wait"]


@dataclass
class DisasterResponseEnv:
    agent_ids: List[str] = field(default_factory=lambda: ["drone-1", "rescue_team-1", "vehicle-1"])
    num_victims: int = 5
    action_space: List[str] = field(default_factory=lambda: DEFAULT_ACTION_SPACE.copy())
    obs_dim: int = 12
    _agents: Dict[str, Dict[str, object]] = field(default_factory=dict, init=False)
    _victims: List[Dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.action_space is None:
            self.action_space = DEFAULT_ACTION_SPACE.copy()
        self._victims = [
            {
                "id": f"v-{idx:02d}",
                "severity": 0.35 + (idx % 4) * 0.15,
                "pos": [idx + 2, idx % 3 + 2],
                "status": "detected",
            }
            for idx in range(self.num_victims)
        ]
        self._agents = {}
        for agent_id in self.agent_ids:
            agent_type = agent_id.split("-")[0] if "-" in agent_id else "drone"
            if agent_type == "rescue":
                agent_type = "rescue_team"
            self._agents[agent_id] = {
                "id": agent_id,
                "type": agent_type,
                "battery": 0.72 if agent_type == "drone" else 0.6 if agent_type == "rescue_team" else 0.8,
                "status": "idle",
                "mission_priority": 0.5,
                "hazard_risk": 0.15,
                "victims_nearby": 0,
                "supply": 1.0,
                "pos": [1.0, 1.0],
            }

    @property
    def victims(self):
        return self._victims

    def encode_observation(self, agent_id: str) -> List[float]:
        state = self._agents.get(agent_id, {})
        battery = float(state.get("battery", 0.5))
        hazard_risk = float(state.get("hazard_risk", 0.1))
        mission_priority = float(state.get("mission_priority", 0.5))
        victims_nearby = float(state.get("victims_nearby", 0.0))
        supply = float(state.get("supply", 0.5))
        x, y = state.get("pos", [0.0, 0.0])
        type_index = {"drone": 0.0, "rescue_team": 1.0, "vehicle": 2.0}.get(str(state.get("type", "drone")), 0.0)
        status = 1.0 if state.get("status") in {"idle", "scanning"} else 0.0
        route_pressure = 0.5 + (0.5 if victims_nearby > 0 else 0.0)
        return [
            battery,
            hazard_risk,
            mission_priority,
            victims_nearby,
            supply,
            status,
            type_index,
            float(x),
            float(y),
            route_pressure,
            1.0 if battery < 0.3 else 0.0,
            1.0 if hazard_risk > 0.75 else 0.0,
        ]

    def step(self, actions_by_agent: Optional[Dict[str, str]] = None):
        actions_by_agent = actions_by_agent or {}
        for agent_id in self.agent_ids:
            action = actions_by_agent.get(agent_id, "wait")
            state = self._agents[agent_id]
            if action == "return":
                state["battery"] = max(0.0, float(state["battery"]) - 0.05)
            elif action == "scan":
                state["hazard_risk"] = max(0.0, float(state["hazard_risk"]) - 0.08)
            elif action == "rescue":
                state["victims_nearby"] = max(0, int(state["victims_nearby"]) - 1)
                state["mission_priority"] = min(1.0, float(state["mission_priority"]) + 0.1)
            elif action == "move":
                state["battery"] = max(0.0, float(state["battery"]) - 0.04)
            elif action == "deliver":
                state["supply"] = max(0.0, float(state["supply"]) - 0.1)
            elif action == "wait":
                state["battery"] = max(0.0, float(state["battery"]) + 0.01)
        return {"agents": self._agents, "victims": self._victims}
