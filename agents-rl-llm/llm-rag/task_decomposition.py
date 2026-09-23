from __future__ import annotations

import re
from typing import List


def decompose_task(mission: str) -> List[str]:
    text = (mission or "").strip()
    if not text:
        return ["Assess the current situation and identify the most urgent objective."]
    lowered = text.lower()
    tasks = []
    if re.search(r"flood|water|submerg", lowered):
        tasks.append("Identify detected victims in the flooded sector")
        tasks.append("Prioritize victims according to severity and accessibility")
        tasks.append("Assign suitable rescue agents and vehicles")
        tasks.append("Select safe routes that avoid rising-water hazards")
        tasks.append("Rescue victims and move them to a safe evacuation point")
        tasks.append("Return remaining agents to a safe staging location")
        return tasks
    if re.search(r"collapse|structur|rubble|trapped", lowered):
        tasks.append("Inspect the structure and confirm the presence of trapped survivors")
        tasks.append("Stabilize the area before entry")
        tasks.append("Assign a rescue team with spotter support")
        tasks.append("Clear a safe access route and mark hazard zones")
        tasks.append("Extract survivors to a safe triage point")
        return tasks
    if re.search(r"cyclone|storm|wind", lowered):
        tasks.append("Identify shelters, safe staging points, and route disruptions")
        tasks.append("Prioritize evacuation of vulnerable populations")
        tasks.append("Stage medical supplies and reserve equipment")
        tasks.append("Dispatch teams only along cleared routes")
        tasks.append("Monitor hazard changes and adjust rescue plans")
        return tasks
    if re.search(r"earthquake|quake", lowered):
        tasks.append("Survey damaged areas and identify trapped victims")
        tasks.append("Prioritize unstable zones that require immediate rescue")
        tasks.append("Deploy drones for rapid reconnaissance")
        tasks.append("Route rescue teams through stable paths")
        tasks.append("Deliver medical support and evacuate survivors")
        return tasks
    tasks.extend([
        "Identify detected victims relevant to the mission objective",
        "Prioritize victims according to severity, access, and urgency",
        "Assign appropriate rescue agents and resources",
        "Choose safe routes and avoid hazards",
        "Execute rescue, delivery, or evacuation tasks",
        "Return teams to a safe staging position when complete",
    ])
    return tasks[:6]


def task_decomposition_to_dict(mission: str) -> dict:
    tasks = decompose_task(mission)
    return {"objective": mission.strip() or "Emergency response mission", "subtasks": tasks}
