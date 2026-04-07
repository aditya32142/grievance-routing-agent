import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from inference import choose_action, create_llm_client_from_env
from server.grievance_routing_environment import (
    GrievanceRoutingEnvironment,
    grade_task,
    list_available_tasks,
)
from models import GrievanceRoutingAction


TEST_CASES = [
    {
        "complaint": "Streetlight not working near Main Street.",
        "difficulty": "easy",
        "expected": {
            "department": "electricity",
            "priority": "medium",
            "action": "send_team",
        },
    },
    {
        "complaint": "Garbage has not been collected for 5 days.",
        "difficulty": "easy",
        "expected": {
            "department": "sanitation",
            "priority": "high",
            "action": "send_team",
        },
    },
    {
        "complaint": "Pothole on Highway 9 causing accidents.",
        "difficulty": "easy",
        "expected": {
            "department": "roads",
            "priority": "high",
            "action": "send_team",
        },
    },
    {
        "complaint": "Water pipe leaking for 3 days, damaging road.",
        "difficulty": "medium",
        "expected": {
            "department": "water",
            "priority": "high",
            "action": "send_team",
        },
    },
    {
        "complaint": "Power outage in entire Block C since morning.",
        "difficulty": "medium",
        "expected": {
            "department": "electricity",
            "priority": "critical",
            "action": "escalate",
        },
    },
    {
        "complaint": "Mosquito breeding in stagnant water near park.",
        "difficulty": "medium",
        "expected": {
            "department": "health",
            "priority": "medium",
            "action": "send_team",
        },
    },
    {
        "complaint": "Garbage dump near school causing illness in children.",
        "difficulty": "hard",
        "expected": {
            "department": "sanitation",
            "priority": "critical",
            "action": "escalate",
        },
    },
    {
        "complaint": "Suspected food poisoning at restaurant, 10 people affected.",
        "difficulty": "hard",
        "expected": {
            "department": "health",
            "priority": "critical",
            "action": "escalate",
        },
    },
    {
        "complaint": "Street brawl ongoing, police not responding.",
        "difficulty": "hard",
        "expected": {
            "department": "police",
            "priority": "critical",
            "action": "escalate",
        },
    },
    {
        "complaint": "Minor crack in road, no immediate danger.",
        "difficulty": "hard",
        "expected": {
            "department": "roads",
            "priority": "low",
            "action": "log_complaint",
        },
    },
]

PRIORITIES = ["low", "medium", "high", "critical"]
DIFFICULTY_TARGET_SCORES = {
    "easy": 0.8,
    "medium": 0.9,
    "hard": 1.0,
}


def choose_silently(complaint: str):
    with redirect_stdout(io.StringIO()):
        return choose_action(complaint)


def score_decision(action, expected, difficulty: str) -> float:
    # Mirrors the active server reward function in server/grievance_routing_environment.py.
    raw_reward = 0.0

    if action["department"] == expected["department"]:
        raw_reward += 0.4
    else:
        related_groups = [
            {"electricity", "water", "roads"},
            {"sanitation", "health"},
        ]
        if any(action["department"] in group and expected["department"] in group for group in related_groups):
            raw_reward -= 0.15
        else:
            raw_reward -= 0.3

    if action["priority"] == expected["priority"]:
        raw_reward += 0.3
    else:
        action_idx = PRIORITIES.index(action["priority"]) if action["priority"] in PRIORITIES else -1
        expected_idx = PRIORITIES.index(expected["priority"]) if expected["priority"] in PRIORITIES else -1
        if abs(action_idx - expected_idx) == 1:
            raw_reward += 0.1
        else:
            raw_reward -= 0.1

    if action["action"] == expected["action"]:
        raw_reward += 0.3
    else:
        if expected["priority"] == "critical" and {action["action"], expected["action"]} == {"send_team", "escalate"}:
            raw_reward += 0.05
        elif expected["priority"] == "low" and {action["action"], expected["action"]} == {"log_complaint", "send_team"}:
            raw_reward += 0.0
        else:
            raw_reward -= 0.05

    max_raw_reward = 1.0
    if difficulty == "hard" and action.get("reasoning"):
        raw_reward += 0.1
        max_raw_reward = 1.1

    target_score = DIFFICULTY_TARGET_SCORES.get(difficulty, 1.0)
    return max(0.0, round((max(raw_reward, 0.0) / max_raw_reward) * target_score, 2))


class InferenceRegressionTests(unittest.TestCase):
    def test_all_known_complaints_route_correctly(self):
        for case in TEST_CASES:
            with self.subTest(complaint=case["complaint"]):
                decision = choose_silently(case["complaint"])
                self.assertEqual(decision["department"], case["expected"]["department"])
                self.assertEqual(decision["priority"], case["expected"]["priority"])
                self.assertEqual(decision["action"], case["expected"]["action"])

    def test_known_dataset_reaches_perfect_reward(self):
        total_reward = 0.0

        for case in TEST_CASES:
            decision = choose_silently(case["complaint"])
            total_reward += score_decision(decision, case["expected"], case["difficulty"])

        self.assertAlmostEqual(total_reward, 9.1, places=2)

    def test_related_department_mismatch_is_not_over_penalized(self):
        reward = score_decision(
            {"department": "electricity", "priority": "high", "action": "send_team", "reasoning": "test"},
            {"department": "roads", "priority": "high", "action": "send_team"},
            "easy",
        )
        self.assertEqual(reward, 0.36)

    def test_perfect_rewards_vary_by_difficulty(self):
        easy_reward = score_decision(
            {"department": "sanitation", "priority": "high", "action": "send_team", "reasoning": "ok"},
            {"department": "sanitation", "priority": "high", "action": "send_team"},
            "easy",
        )
        medium_reward = score_decision(
            {"department": "water", "priority": "high", "action": "send_team", "reasoning": "ok"},
            {"department": "water", "priority": "high", "action": "send_team"},
            "medium",
        )
        hard_reward = score_decision(
            {"department": "police", "priority": "critical", "action": "escalate", "reasoning": "ok"},
            {"department": "police", "priority": "critical", "action": "escalate"},
            "hard",
        )
        self.assertEqual((easy_reward, medium_reward, hard_reward), (0.8, 0.9, 1.0))

    def test_environment_metadata_includes_scored_complaint_and_breakdown(self):
        env = GrievanceRoutingEnvironment()
        env._pool = [
            {
                "complaint": "Pothole on Highway 9 causing accidents.",
                "difficulty": "easy",
                "expected": {"department": "roads", "priority": "high", "action": "send_team"},
            },
            {
                "complaint": "Streetlight not working near Main Street.",
                "difficulty": "easy",
                "expected": {"department": "electricity", "priority": "medium", "action": "send_team"},
            },
        ]
        env._index = 0
        env._total_reward = 0.0
        env._load_current()

        result = env.step(
            GrievanceRoutingAction(
                department="electricity",
                priority="high",
                action="send_team",
                reasoning="test",
            )
        )

        self.assertEqual(result.metadata["scored_complaint"], "Pothole on Highway 9 causing accidents.")
        self.assertEqual(result.metadata["next_complaint"], "Streetlight not working near Main Street.")
        self.assertIn("reward_breakdown", result.metadata)
        self.assertIn("submitted_action", result.metadata)

    def test_create_llm_client_uses_required_proxy_env_vars(self):
        with patch("inference.API_BASE_URL", "https://proxy.example/v1"):
            with patch("inference.API_KEY", "proxy-key"):
                with patch("inference.OpenAI") as mock_openai:
                    create_llm_client_from_env(require=True)

        mock_openai.assert_called_once_with(
            base_url="https://proxy.example/v1",
            api_key="proxy-key",
        )

    def test_create_llm_client_requires_proxy_env_vars_when_requested(self):
        with patch("inference.API_BASE_URL", None):
            with patch("inference.API_KEY", None):
                with self.assertRaises(RuntimeError):
                    create_llm_client_from_env(require=True)

    def test_choose_action_builds_client_from_proxy_env_when_missing(self):
        fake_client = MagicMock()

        with patch("inference.API_BASE_URL", "https://proxy.example/v1"):
            with patch("inference.API_KEY", "proxy-key"):
                with patch("inference.create_llm_client_from_env", return_value=fake_client) as mock_create_client:
                    with patch("inference.ask_llm", return_value=None) as mock_ask_llm:
                        choose_action("Streetlight not working near Main Street.")

        mock_create_client.assert_called_once_with(require=False)
        mock_ask_llm.assert_called_once_with(fake_client, "Streetlight not working near Main Street.", "easy")

    def test_llm_call_uses_timeout(self):
        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=MagicMock(content='{"department":"roads","priority":"medium","action":"send_team","reasoning":"ok"}'))]
        fake_client.chat.completions.create.return_value = fake_response

        from inference import ask_llm

        ask_llm(fake_client, "Pothole on Highway 9 causing accidents.", "easy")

        _, kwargs = fake_client.chat.completions.create.call_args
        self.assertEqual(kwargs["timeout"], 30)

    def test_task_catalog_exposes_three_graded_tasks(self):
        tasks = list_available_tasks()

        self.assertGreaterEqual(len(tasks), 3)
        self.assertEqual({task["difficulty"] for task in tasks}, {"easy", "medium", "hard"})
        self.assertTrue(all(task["grader"] for task in tasks))

    def test_task_grader_scores_stay_strictly_between_zero_and_one(self):
        for task_id in ("easy-routing", "medium-routing", "hard-routing"):
            with self.subTest(task_id=task_id):
                result = grade_task(task_id)
                self.assertGreater(result["score"], 0.0)
                self.assertLess(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
