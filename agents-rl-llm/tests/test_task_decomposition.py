"""Unit tests for `agents-rl-llm/llm-rag/task_decomposition.py` (Step 4).

Run from the repository root, e.g.:
    python3 -m pytest agents-rl-llm/tests/test_task_decomposition.py -v

The repository's `agents-rl-llm/` and `llm-rag/` directories use hyphens,
which are not valid Python package/module name characters, so
`task_decomposition.py` cannot be reached via a normal dotted import
(there is no `agents_rl_llm` package). Instead, the module is loaded
directly from its file path, relative to this test file's own location,
so the tests work regardless of the current working directory or
PYTHONPATH and without adding any new package to the repository.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TASK_DECOMPOSITION_PATH = os.path.join(
    _THIS_DIR, "..", "llm-rag", "task_decomposition.py"
)

_spec = importlib.util.spec_from_file_location(
    "task_decomposition", _TASK_DECOMPOSITION_PATH
)
_task_decomposition = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task_decomposition)

DISASTER_TYPES = _task_decomposition.DISASTER_TYPES
PRIORITIES = _task_decomposition.PRIORITIES
RuleBasedProvider = _task_decomposition.RuleBasedProvider
TaskDecompositionProvider = _task_decomposition.TaskDecompositionProvider
decompose_task = _task_decomposition.decompose_task
decompose_task_structured = _task_decomposition.decompose_task_structured
detect_disaster_type = _task_decomposition.detect_disaster_type
task_decomposition_to_dict = _task_decomposition.task_decomposition_to_dict

REQUIRED_FIELDS = {"task_id", "task", "priority", "dependencies", "disaster_type"}


class DisasterDetectionTests(unittest.TestCase):
    def test_flood_detection(self):
        self.assertEqual(detect_disaster_type("What's the protocol for a flooded sector rescue?"), "flood")
        self.assertEqual(detect_disaster_type("Rising flood waters near the levee"), "flood")

    def test_earthquake_detection(self):
        self.assertEqual(
            detect_disaster_type("Rescue victims trapped in a collapsed building after an earthquake."),
            "earthquake",
        )
        self.assertEqual(detect_disaster_type("Aftershock reported near the epicenter"), "earthquake")

    def test_cyclone_detection(self):
        self.assertEqual(detect_disaster_type("Evacuate victims from a cyclone shelter."), "cyclone")
        self.assertEqual(detect_disaster_type("Hurricane making landfall with storm surge"), "cyclone")

    def test_generic_fallback(self):
        self.assertEqual(detect_disaster_type("hazardous blocked route"), "generic")
        self.assertEqual(detect_disaster_type(""), "generic")
        self.assertEqual(detect_disaster_type(None), "generic")


class StructuredOutputTests(unittest.TestCase):
    def test_structured_output_fields(self):
        tasks = decompose_task_structured("Rescue victims after an earthquake.")
        self.assertTrue(tasks)
        for task in tasks:
            self.assertEqual(set(task.keys()), REQUIRED_FIELDS)
            self.assertIsInstance(task["task_id"], str) and self.assertTrue(task["task_id"])
            self.assertIsInstance(task["task"], str) and self.assertTrue(task["task"])
            self.assertIn(task["priority"], PRIORITIES)
            self.assertIsInstance(task["dependencies"], list)
            self.assertIn(task["disaster_type"], DISASTER_TYPES)

    def test_structured_output_disaster_type_matches_detection(self):
        tasks = decompose_task_structured("Evacuate victims from a cyclone shelter.")
        self.assertTrue(all(task["disaster_type"] == "cyclone" for task in tasks))

    def test_task_ordering_is_deterministic(self):
        first = decompose_task_structured("Flood rescue protocol")
        second = decompose_task_structured("Flood rescue protocol")
        self.assertEqual([t["task_id"] for t in first], [t["task_id"] for t in second])
        # First task in a flood pipeline should be an assessment task.
        self.assertEqual(first[0]["task_id"], "flood_assess")

    def test_dependencies_reference_earlier_valid_task_ids(self):
        tasks = decompose_task_structured("Earthquake rescue mission")
        seen_ids = set()
        for task in tasks:
            for dep in task["dependencies"]:
                # Every dependency must point at a task that appears
                # earlier in the pipeline (no forward/cyclic references)
                # and must be a real task_id in this decomposition.
                self.assertIn(dep, seen_ids)
            seen_ids.add(task["task_id"])

    def test_first_task_has_no_dependencies(self):
        tasks = decompose_task_structured("Cyclone response plan")
        self.assertEqual(tasks[0]["dependencies"], [])


class DuplicateRemovalTests(unittest.TestCase):
    def test_duplicate_tasks_removed_preserving_order(self):
        class DuplicateProvider(TaskDecompositionProvider):
            def decompose(self, mission, context=None):
                return [
                    {"task_id": "a", "task": "Assess the situation", "priority": "critical",
                     "dependencies": [], "disaster_type": "generic"},
                    {"task_id": "b", "task": "Assess the situation", "priority": "high",
                     "dependencies": [], "disaster_type": "generic"},
                    {"task_id": "c", "task": "Deploy response teams", "priority": "high",
                     "dependencies": ["a"], "disaster_type": "generic"},
                ]

        tasks = decompose_task_structured("anything", provider=DuplicateProvider())
        task_texts = [t["task"] for t in tasks]
        self.assertEqual(task_texts, ["Assess the situation", "Deploy response teams"])
        self.assertEqual(len(tasks), 2)

    def test_dependencies_pruned_after_dedup_removes_referenced_id(self):
        class ProviderWithDanglingDep(TaskDecompositionProvider):
            def decompose(self, mission, context=None):
                return [
                    {"task_id": "a", "task": "Do thing one", "priority": "high",
                     "dependencies": [], "disaster_type": "generic"},
                    {"task_id": "a2", "task": "Do thing one", "priority": "high",
                     "dependencies": [], "disaster_type": "generic"},
                    {"task_id": "b", "task": "Do thing two", "priority": "high",
                     "dependencies": ["a2"], "disaster_type": "generic"},
                ]

        tasks = decompose_task_structured("anything", provider=ProviderWithDanglingDep())
        by_id = {t["task_id"]: t for t in tasks}
        self.assertIn("a", by_id)
        self.assertEqual(by_id["b"]["dependencies"], ["a"])


class InvalidInputTests(unittest.TestCase):
    def test_none_mission_does_not_raise(self):
        tasks = decompose_task_structured(None)
        self.assertTrue(tasks)
        self.assertTrue(all(t["disaster_type"] == "generic" for t in tasks))

    def test_empty_string_mission_does_not_raise(self):
        tasks = decompose_task_structured("")
        self.assertTrue(tasks)

    def test_non_string_mission_is_coerced_safely(self):
        tasks = decompose_task_structured(12345)
        self.assertTrue(tasks)

    def test_malformed_provider_output_is_dropped_not_raised(self):
        class MalformedProvider(TaskDecompositionProvider):
            def decompose(self, mission, context=None):
                return [
                    "not a dict",
                    {"task": ""},  # empty task text -> dropped
                    {"no_task_field": True},  # missing task -> dropped
                    {"task": "Valid task", "priority": "not-a-real-priority",
                     "dependencies": "not-a-list", "disaster_type": "not-a-real-type"},
                ]

        tasks = decompose_task_structured("anything", provider=MalformedProvider())
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task"], "Valid task")
        self.assertEqual(tasks[0]["priority"], "medium")  # normalized default
        self.assertEqual(tasks[0]["dependencies"], [])
        self.assertEqual(tasks[0]["disaster_type"], "generic")

    def test_provider_returning_non_list_falls_back_to_rule_based(self):
        class BrokenProvider(TaskDecompositionProvider):
            def decompose(self, mission, context=None):
                return None

        tasks = decompose_task_structured("Flood rescue protocol", provider=BrokenProvider())
        self.assertTrue(tasks)
        self.assertTrue(all(t["disaster_type"] == "flood" for t in tasks))

    def test_provider_raising_exception_falls_back_to_rule_based(self):
        class ExplodingProvider(TaskDecompositionProvider):
            def decompose(self, mission, context=None):
                raise RuntimeError("boom")

        tasks = decompose_task_structured("Earthquake rescue mission", provider=ExplodingProvider())
        self.assertTrue(tasks)
        self.assertTrue(all(t["disaster_type"] == "earthquake" for t in tasks))


class BackwardCompatibilityTests(unittest.TestCase):
    def test_decompose_task_returns_list_of_strings(self):
        tasks = decompose_task("Rescue victims trapped in a collapsed building after an earthquake.")
        self.assertIsInstance(tasks, list)
        self.assertTrue(tasks)
        self.assertTrue(all(isinstance(t, str) for t in tasks))

    def test_decompose_task_matches_structured_task_text(self):
        mission = "Evacuate victims from a cyclone shelter."
        flat = decompose_task(mission)
        structured = decompose_task_structured(mission)
        self.assertEqual(flat, [t["task"] for t in structured])

    def test_decompose_task_handles_empty_input(self):
        tasks = decompose_task("")
        self.assertIsInstance(tasks, list)
        self.assertTrue(tasks)

    def test_decompose_task_handles_none_input(self):
        tasks = decompose_task(None)
        self.assertIsInstance(tasks, list)
        self.assertTrue(tasks)

    def test_task_decomposition_to_dict_shape(self):
        result = task_decomposition_to_dict("what's the protocol for a flooded sector rescue?")
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"mission", "disaster_type", "tasks", "task_count"})
        self.assertEqual(result["disaster_type"], "flood")
        self.assertIsInstance(result["tasks"], list)
        self.assertEqual(result["task_count"], len(result["tasks"]))

    def test_task_decomposition_to_dict_handles_invalid_input(self):
        result = task_decomposition_to_dict(None)
        self.assertEqual(result["mission"], "")
        self.assertEqual(result["disaster_type"], "generic")
        self.assertTrue(result["tasks"])


class CustomProviderSupportTests(unittest.TestCase):
    def test_custom_provider_output_is_used(self):
        class FixedProvider(TaskDecompositionProvider):
            def decompose(self, mission, context=None):
                return [
                    {
                        "task_id": "llm_task",
                        "task": "LLM-suggested high-level task",
                        "priority": "high",
                        "dependencies": [],
                        "disaster_type": "generic",
                    }
                ]

        tasks = decompose_task_structured("flood rescue protocol", provider=FixedProvider())
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task"], "LLM-suggested high-level task")

    def test_context_argument_is_accepted_without_error(self):
        tasks = decompose_task_structured(
            "Evacuate victims from a cyclone shelter.",
            provider=RuleBasedProvider(),
            context="Some RAG-retrieved grounding context.",
        )
        self.assertTrue(tasks)

    def test_rule_based_provider_is_instance_of_base_interface(self):
        self.assertIsInstance(RuleBasedProvider(), TaskDecompositionProvider)


if __name__ == "__main__":
    unittest.main()
