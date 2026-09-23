from __future__ import annotations

from typing import Any, Dict


DEFAULT_REWARD_WEIGHTS = {
    "rescued": 150.0,
    "target_reached": 40.0,
    "high_priority": 80.0,
    "delivery": 35.0,
    "cooperation": 20.0,
    "mission_complete": 50.0,
    "collision": -35.0,
    "hazard": -50.0,
    "waste": -25.0,
    "movement": -8.0,
    "timeout": -15.0,
    "failed_rescue": -30.0,
    "duplicate_assignment": -22.0,
}


def compute_reward(env: Any, agent_id: str, action: str, outcome: Dict[str, Any], weights: Dict[str, float] | None = None) -> float:
    config = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
    reward = 0.0
    reward += config["rescued"] * float(outcome.get("rescue", 0.0))
    reward += config["target_reached"] * float(outcome.get("target_reached", 0.0))
    reward += config["high_priority"] * float(outcome.get("high_priority", 0.0))
    reward += config["delivery"] * float(outcome.get("delivery", 0.0))
    reward += config["cooperation"] * float(outcome.get("cooperation", 0.0))
    reward += config["mission_complete"] * float(outcome.get("mission_complete", 0.0))
    reward += config["collision"] * float(outcome.get("collision", 0.0))
    reward += config["hazard"] * float(outcome.get("danger", 0.0))
    reward += config["waste"] * float(outcome.get("waste", 0.0))
    reward += config["movement"] * float(outcome.get("movement", 0.0))
    reward += config["timeout"] * float(outcome.get("timeout", 0.0))
    reward += config["failed_rescue"] * float(outcome.get("failed_rescue", 0.0))
    reward += config["duplicate_assignment"] * float(outcome.get("duplicate_assignment", 0.0))
    if action == "wait":
        reward -= 2.5
    if action == "move" and float(outcome.get("distance", 0.0)) > 1.5:
        reward += config["movement"] * min(1.0, float(outcome.get("distance", 0.0)) / 3.0)
    return float(reward)
