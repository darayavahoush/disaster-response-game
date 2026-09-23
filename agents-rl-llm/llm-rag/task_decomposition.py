"""Task decomposition for the Agent Intelligence module (Step 4).

Given a free-text mission / question, this module breaks it down into an
ordered pipeline of concrete response tasks. It is deterministic and
rule-based only -- no LLM or network calls are made here (that is a later
step, layered on top via `agents_rl_llm.llm_rag.ollama_provider`, which
falls back to `RuleBasedProvider` from this module on any failure).

Public API
----------
decompose_task(mission) -> List[str]
    Backward-compatible flat list of task descriptions.

task_decomposition_to_dict(mission) -> Dict[str, Any]
    Backward-compatible dictionary view of the decomposition.

decompose_task_structured(mission, provider=None, context=None) -> List[Dict[str, Any]]
    Structured decomposition. Each task dict has:
        task_id, task, priority, dependencies, disaster_type

TaskDecompositionProvider
    Base interface for anything that can turn a mission into structured
    tasks (rule-based today; LLM-backed providers, e.g. Ollama, plug in
    later behind the same interface).

RuleBasedProvider
    Deterministic, keyword-based implementation of the interface above.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DISASTER_TYPES: Sequence[str] = ("flood", "earthquake", "cyclone", "generic")
PRIORITIES: Sequence[str] = ("critical", "high", "medium", "low")

_DEFAULT_DISASTER_TYPE = "generic"
_DEFAULT_PRIORITY = "medium"

# Keyword matching is checked in this order, so the first disaster type
# whose keywords appear in the mission text wins. "generic" is not listed
# here -- it is always the fallback when nothing else matches.
_KEYWORDS: Dict[str, Sequence[str]] = {
    "flood": (
        "flood",
        "flooding",
        "flooded",
        "inundat",
        "water level",
        "levee",
        "storm drain",
    ),
    "earthquake": (
        "earthquake",
        "quake",
        "seismic",
        "aftershock",
        "collapsed building",
        "rubble",
        "tremor",
    ),
    "cyclone": (
        "cyclone",
        "hurricane",
        "typhoon",
        "storm surge",
        "high wind",
        "tropical storm",
    ),
}

# --------------------------------------------------------------------------
# Rule-based task pipelines
#
# Each pipeline is an ordered list of task templates. Order encodes a
# sensible default sequence of operations; `dependencies` additionally
# encodes which earlier task_ids must be considered complete before a
# given task starts (a task may depend on more than just "the previous
# one", e.g. a report step depending on several branches).
# --------------------------------------------------------------------------

_PIPELINES: Dict[str, List[Dict[str, Any]]] = {
    "flood": [
        {
            "task_id": "flood_assess",
            "task": "Assess flood extent and identify affected zones",
            "priority": "critical",
            "dependencies": [],
            "disaster_type": "flood",
        },
        {
            "task_id": "flood_rescue",
            "task": "Deploy rescue teams to flooded areas with trapped victims",
            "priority": "critical",
            "dependencies": ["flood_assess"],
            "disaster_type": "flood",
        },
        {
            "task_id": "flood_evacuate",
            "task": "Coordinate evacuation of residents from rising water zones",
            "priority": "high",
            "dependencies": ["flood_assess"],
            "disaster_type": "flood",
        },
        {
            "task_id": "flood_medical",
            "task": "Set up medical triage stations for flood-related injuries",
            "priority": "high",
            "dependencies": ["flood_rescue"],
            "disaster_type": "flood",
        },
        {
            "task_id": "flood_infrastructure",
            "task": "Inspect and secure critical infrastructure (levees, drainage, power)",
            "priority": "medium",
            "dependencies": ["flood_assess"],
            "disaster_type": "flood",
        },
        {
            "task_id": "flood_supply",
            "task": "Distribute emergency supplies (clean water, food, shelter) to displaced residents",
            "priority": "medium",
            "dependencies": ["flood_evacuate"],
            "disaster_type": "flood",
        },
        {
            "task_id": "flood_report",
            "task": "Compile flood situation report and update command center",
            "priority": "low",
            "dependencies": ["flood_medical", "flood_supply", "flood_infrastructure"],
            "disaster_type": "flood",
        },
    ],
    "earthquake": [
        {
            "task_id": "eq_assess",
            "task": "Assess structural damage and identify collapsed buildings",
            "priority": "critical",
            "dependencies": [],
            "disaster_type": "earthquake",
        },
        {
            "task_id": "eq_search_rescue",
            "task": "Conduct search and rescue operations in collapsed structures",
            "priority": "critical",
            "dependencies": ["eq_assess"],
            "disaster_type": "earthquake",
        },
        {
            "task_id": "eq_medical",
            "task": "Establish field triage and medical care for the injured",
            "priority": "high",
            "dependencies": ["eq_search_rescue"],
            "disaster_type": "earthquake",
        },
        {
            "task_id": "eq_aftershock_monitor",
            "task": "Monitor for aftershocks and cordon off hazardous structures",
            "priority": "high",
            "dependencies": ["eq_assess"],
            "disaster_type": "earthquake",
        },
        {
            "task_id": "eq_evacuate",
            "task": "Evacuate residents from unsafe or structurally unstable buildings",
            "priority": "high",
            "dependencies": ["eq_assess"],
            "disaster_type": "earthquake",
        },
        {
            "task_id": "eq_supply",
            "task": "Distribute emergency shelter, food, and water to survivors",
            "priority": "medium",
            "dependencies": ["eq_evacuate"],
            "disaster_type": "earthquake",
        },
        {
            "task_id": "eq_report",
            "task": "Compile damage assessment report for command center",
            "priority": "low",
            "dependencies": ["eq_medical", "eq_supply"],
            "disaster_type": "earthquake",
        },
    ],
    "cyclone": [
        {
            "task_id": "cyclone_track",
            "task": "Track cyclone path/intensity and issue advance warnings",
            "priority": "critical",
            "dependencies": [],
            "disaster_type": "cyclone",
        },
        {
            "task_id": "cyclone_evacuate",
            "task": "Evacuate residents from high-risk coastal and low-lying areas",
            "priority": "critical",
            "dependencies": ["cyclone_track"],
            "disaster_type": "cyclone",
        },
        {
            "task_id": "cyclone_shelter",
            "task": "Prepare and staff emergency shelters for evacuees",
            "priority": "high",
            "dependencies": ["cyclone_evacuate"],
            "disaster_type": "cyclone",
        },
        {
            "task_id": "cyclone_secure",
            "task": "Secure critical infrastructure against high winds and storm surge",
            "priority": "high",
            "dependencies": ["cyclone_track"],
            "disaster_type": "cyclone",
        },
        {
            "task_id": "cyclone_rescue",
            "task": "Deploy rescue teams once winds subside to assist stranded residents",
            "priority": "high",
            "dependencies": ["cyclone_secure"],
            "disaster_type": "cyclone",
        },
        {
            "task_id": "cyclone_supply",
            "task": "Distribute emergency supplies to shelters and affected areas",
            "priority": "medium",
            "dependencies": ["cyclone_shelter"],
            "disaster_type": "cyclone",
        },
        {
            "task_id": "cyclone_report",
            "task": "Compile impact assessment report for command center",
            "priority": "low",
            "dependencies": ["cyclone_rescue", "cyclone_supply"],
            "disaster_type": "cyclone",
        },
    ],
    "generic": [
        {
            "task_id": "generic_assess",
            "task": "Assess the situation and gather initial information",
            "priority": "critical",
            "dependencies": [],
            "disaster_type": "generic",
        },
        {
            "task_id": "generic_prioritize",
            "task": "Identify and prioritize immediate risks to life and safety",
            "priority": "critical",
            "dependencies": ["generic_assess"],
            "disaster_type": "generic",
        },
        {
            "task_id": "generic_deploy",
            "task": "Deploy available response teams to affected areas",
            "priority": "high",
            "dependencies": ["generic_prioritize"],
            "disaster_type": "generic",
        },
        {
            "task_id": "generic_medical",
            "task": "Provide medical assistance to injured or affected individuals",
            "priority": "high",
            "dependencies": ["generic_deploy"],
            "disaster_type": "generic",
        },
        {
            "task_id": "generic_supply",
            "task": "Coordinate distribution of essential supplies",
            "priority": "medium",
            "dependencies": ["generic_deploy"],
            "disaster_type": "generic",
        },
        {
            "task_id": "generic_report",
            "task": "Compile situation report and update command center",
            "priority": "low",
            "dependencies": ["generic_medical", "generic_supply"],
            "disaster_type": "generic",
        },
    ],
}


# --------------------------------------------------------------------------
# Disaster type detection
# --------------------------------------------------------------------------


def detect_disaster_type(mission: Any) -> str:
    """Best-effort keyword detection of the disaster type for `mission`.

    Defensive by design: any non-string or unmatched input resolves to
    "generic" rather than raising.
    """
    text = mission.lower() if isinstance(mission, str) else ""
    for disaster_type, keywords in _KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return disaster_type
    return _DEFAULT_DISASTER_TYPE


# --------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------


class TaskDecompositionProvider(ABC):
    """Interface for anything that turns a mission into structured tasks.

    Implementations (rule-based today; LLM-backed later, e.g. an Ollama
    provider) must return a list of dicts. Output does not need to be
    perfectly well-formed -- `decompose_task_structured` normalizes and
    validates whatever comes back, so a malformed/partial provider result
    degrades gracefully instead of breaking the API.
    """

    @abstractmethod
    def decompose(self, mission: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a list of task dicts for `mission`.

        Each dict should ideally contain: task_id, task, priority,
        dependencies, disaster_type -- but callers must not assume that;
        use `decompose_task_structured` to get a normalized result.
        """
        raise NotImplementedError


class RuleBasedProvider(TaskDecompositionProvider):
    """Deterministic, keyword-based task decomposition.

    No external dependencies, no network calls, no LLM. This is the
    default provider and the safe fallback for any other provider.
    """

    def decompose(self, mission: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        disaster_type = detect_disaster_type(mission)
        pipeline = _PIPELINES.get(disaster_type, _PIPELINES[_DEFAULT_DISASTER_TYPE])
        # Return copies so callers can't mutate the module-level templates.
        return [
            {
                "task_id": task["task_id"],
                "task": task["task"],
                "priority": task["priority"],
                "dependencies": list(task["dependencies"]),
                "disaster_type": task["disaster_type"],
            }
            for task in pipeline
        ]


# --------------------------------------------------------------------------
# Normalization / validation helpers
#
# Provider output is untrusted (especially once LLM-backed providers are
# introduced), so it is normalized defensively: malformed entries are
# fixed up where possible and dropped only when they can't be salvaged
# (e.g. no usable task text at all).
# --------------------------------------------------------------------------


def _normalize_task_entry(raw: Any, fallback_index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    task_text = raw.get("task")
    if not isinstance(task_text, str) or not task_text.strip():
        return None
    task_text = task_text.strip()

    task_id = raw.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        task_id = f"task_{fallback_index}"
    else:
        task_id = task_id.strip()

    priority = raw.get("priority")
    if not isinstance(priority, str) or priority.strip().lower() not in PRIORITIES:
        priority = _DEFAULT_PRIORITY
    else:
        priority = priority.strip().lower()

    dependencies_raw = raw.get("dependencies")
    if isinstance(dependencies_raw, (list, tuple)):
        dependencies = [dep.strip() for dep in dependencies_raw if isinstance(dep, str) and dep.strip()]
    else:
        dependencies = []

    disaster_type = raw.get("disaster_type")
    if not isinstance(disaster_type, str) or disaster_type.strip().lower() not in DISASTER_TYPES:
        disaster_type = _DEFAULT_DISASTER_TYPE
    else:
        disaster_type = disaster_type.strip().lower()

    return {
        "task_id": task_id,
        "task": task_text,
        "priority": priority,
        "dependencies": dependencies,
        "disaster_type": disaster_type,
    }


def normalize_tasks(raw_tasks: Any) -> List[Dict[str, Any]]:
    """Coerce arbitrary provider output into a list of well-formed task
    dicts. Anything that can't be salvaged (not a list, not a dict, no
    usable task text) is dropped rather than raising, so a malformed or
    partially-malformed provider response never breaks the API."""
    if not isinstance(raw_tasks, (list, tuple)):
        return []

    normalized: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_tasks):
        entry = _normalize_task_entry(raw, fallback_index=index)
        if entry is not None:
            normalized.append(entry)
    return normalized


def _dedupe_preserve_order(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate tasks (by case-insensitive task text), keeping the
    first occurrence and preserving overall order.

    Any dependency that pointed at a since-removed duplicate is remapped
    to the surviving task_id for that same task text, so dependency
    semantics are preserved rather than silently dropped. Dependencies
    are then pruned to only reference task_ids that survived dedup (and
    never reference themselves), so a duplicate removal can never leave a
    dangling/self dependency behind.
    """
    deduped: List[Dict[str, Any]] = []
    seen_task_text: Dict[str, str] = {}  # task text -> surviving task_id
    seen_task_ids: set = set()
    id_remap: Dict[str, str] = {}  # removed/duplicate task_id -> surviving task_id

    for task in tasks:
        text_key = task["task"].strip().lower()
        original_id = task["task_id"]

        if text_key in seen_task_text:
            id_remap[original_id] = seen_task_text[text_key]
            continue

        task_id = original_id
        if task_id in seen_task_ids:
            # Duplicate id from a different task's text: keep the task but
            # make its id unique so downstream dependency references stay
            # unambiguous.
            suffix = 2
            new_id = f"{task_id}_{suffix}"
            while new_id in seen_task_ids:
                suffix += 1
                new_id = f"{task_id}_{suffix}"
            id_remap[original_id] = new_id
            task = {**task, "task_id": new_id}
            task_id = new_id

        seen_task_text[text_key] = task_id
        seen_task_ids.add(task_id)
        deduped.append(task)

    valid_ids = {task["task_id"] for task in deduped}
    for task in deduped:
        remapped = (id_remap.get(dep, dep) for dep in task["dependencies"])
        task["dependencies"] = [dep for dep in remapped if dep in valid_ids and dep != task["task_id"]]

    return deduped


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _coerce_mission_text(mission: Any) -> str:
    if mission is None:
        return ""
    if isinstance(mission, str):
        return mission
    return str(mission)


def decompose_task_structured(
    mission: Any,
    provider: Optional[TaskDecompositionProvider] = None,
    context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Decompose `mission` into an ordered list of structured task dicts.

    Each task dict has: task_id, task, priority, dependencies,
    disaster_type. Defensive against: empty/None/non-string missions,
    a provider that raises, and a provider that returns malformed or
    duplicate tasks -- none of these ever raise out of this function.
    """
    safe_mission = _coerce_mission_text(mission)
    active_provider = provider if provider is not None else RuleBasedProvider()

    raw_tasks: Any = None
    try:
        raw_tasks = active_provider.decompose(safe_mission, context=context)
    except Exception as exc:  # noqa: BLE001 - a misbehaving provider must not break the API
        logger.warning("TaskDecompositionProvider %r raised %s; falling back to rule-based output", active_provider, exc)
        raw_tasks = None

    normalized = normalize_tasks(raw_tasks)

    if not normalized:
        # Either the provider returned nothing usable, or it raised.
        # Fall back to the deterministic rule-based pipeline so the API
        # always returns something sensible.
        if not isinstance(active_provider, RuleBasedProvider):
            normalized = normalize_tasks(RuleBasedProvider().decompose(safe_mission, context=context))

    return _dedupe_preserve_order(normalized)


def decompose_task(mission: Any) -> List[str]:
    """Backward-compatible flat decomposition: ordered list of task
    description strings only (no priorities/dependencies/disaster type).
    """
    structured = decompose_task_structured(mission)
    return [task["task"] for task in structured]


def task_decomposition_to_dict(mission: Any) -> Dict[str, Any]:
    """Backward-compatible dictionary view of the decomposition.

    Shape:
        {
            "mission": <original mission as text>,
            "disaster_type": <detected disaster type>,
            "tasks": [<task description>, ...],
            "task_count": <int>,
        }
    """
    structured = decompose_task_structured(mission)
    disaster_type = structured[0]["disaster_type"] if structured else _DEFAULT_DISASTER_TYPE
    tasks = [task["task"] for task in structured]
    return {
        "mission": _coerce_mission_text(mission),
        "disaster_type": disaster_type,
        "tasks": tasks,
        "task_count": len(tasks),
    }
