"""
Grievance Routing Environment Implementation.
An RL environment where an agent learns to route citizen complaints
to the correct government department.
"""
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
import random
from typing import Optional

try:
    from ..models import GrievanceRoutingAction, GrievanceRoutingObservation
except ImportError:
    from models import GrievanceRoutingAction, GrievanceRoutingObservation

DEPARTMENTS = ["sanitation", "electricity", "water", "roads", "health", "police"]
PRIORITIES = ["low", "medium", "high", "critical"]
ACTIONS = ["log_complaint", "send_team", "escalate", "close_resolved"]
RELATED_DEPARTMENT_GROUPS = [
    {"electricity", "water", "roads"},
    {"sanitation", "health"},
]
DIFFICULTY_TARGET_SCORES = {
    "easy": 0.8,
    "medium": 0.9,
    "hard": 1.0,
}

GRIEVANCE_DATASET = [
    {
        "complaint": "Streetlight not working near Main Street.",
        "difficulty": "easy",
        "expected": {"department": "electricity", "priority": "medium", "action": "send_team"},
    },
    {
        "complaint": "Garbage has not been collected for 5 days.",
        "difficulty": "easy",
        "expected": {"department": "sanitation", "priority": "high", "action": "send_team"},
    },
    {
        "complaint": "Pothole on Highway 9 causing accidents.",
        "difficulty": "easy",
        "expected": {"department": "roads", "priority": "high", "action": "send_team"},
    },
    {
        "complaint": "Water pipe leaking for 3 days, damaging road.",
        "difficulty": "medium",
        "expected": {"department": "water", "priority": "high", "action": "send_team"},
    },
    {
        "complaint": "Power outage in entire Block C since morning.",
        "difficulty": "medium",
        "expected": {"department": "electricity", "priority": "critical", "action": "escalate"},
    },
    {
        "complaint": "Mosquito breeding in stagnant water near park.",
        "difficulty": "medium",
        "expected": {"department": "health", "priority": "medium", "action": "send_team"},
    },
    {
        "complaint": "Garbage dump near school causing illness in children.",
        "difficulty": "hard",
        "expected": {"department": "sanitation", "priority": "critical", "action": "escalate"},
    },
    {
        "complaint": "Suspected food poisoning at restaurant, 10 people affected.",
        "difficulty": "hard",
        "expected": {"department": "health", "priority": "critical", "action": "escalate"},
    },
    {
        "complaint": "Street brawl ongoing, police not responding.",
        "difficulty": "hard",
        "expected": {"department": "police", "priority": "critical", "action": "escalate"},
    },
    {
        "complaint": "Minor crack in road, no immediate danger.",
        "difficulty": "hard",
        "expected": {"department": "roads", "priority": "low", "action": "log_complaint"},
    },
]

def _department_penalty(predicted: str, expected: str) -> tuple[float, str]:
    if predicted == expected:
        return 0.4, "+0.4 correct department"

    for group in RELATED_DEPARTMENT_GROUPS:
        if predicted in group and expected in group:
            return -0.15, f"-0.15 related department mismatch ({predicted} vs {expected})"

    return -0.3, f"-0.3 wrong department ({predicted} vs {expected})"


def _priority_score(predicted: str, expected: str) -> tuple[float, str]:
    if predicted == expected:
        return 0.3, "+0.3 correct priority"

    predicted_index = PRIORITIES.index(predicted) if predicted in PRIORITIES else -1
    expected_index = PRIORITIES.index(expected) if expected in PRIORITIES else -1
    if abs(predicted_index - expected_index) == 1:
        return 0.1, f"+0.1 close priority ({predicted} vs {expected})"

    return -0.1, f"-0.1 wrong priority ({predicted} vs {expected})"


def _action_score(predicted: str, expected: str, priority: str) -> tuple[float, str]:
    if predicted == expected:
        return 0.3, "+0.3 correct action"

    if priority == "critical" and {predicted, expected} == {"send_team", "escalate"}:
        return 0.05, f"+0.05 near-miss action on critical issue ({predicted} vs {expected})"

    if priority == "low" and {predicted, expected} == {"log_complaint", "send_team"}:
        return 0.0, f"+0.0 conservative action on low-priority issue ({predicted} vs {expected})"

    return -0.05, f"-0.05 wrong action ({predicted} vs {expected})"


def calculate_reward(action, expected, difficulty):
    raw_reward = 0.0
    breakdown = {}

    department_score, department_note = _department_penalty(action.department, expected["department"])
    raw_reward += department_score
    breakdown["department"] = department_note

    priority_score, priority_note = _priority_score(action.priority, expected["priority"])
    raw_reward += priority_score
    breakdown["priority"] = priority_note

    action_score, action_note = _action_score(action.action, expected["action"], expected["priority"])
    raw_reward += action_score
    breakdown["action"] = action_note

    max_raw_reward = 1.0
    if difficulty == "hard" and action.reasoning:
        raw_reward += 0.1
        breakdown["reasoning"] = "+0.1 reasoning provided for hard task"
        max_raw_reward = 1.1
    elif difficulty == "hard":
        breakdown["reasoning"] = "+0.0 no reasoning bonus"

    target_score = DIFFICULTY_TARGET_SCORES.get(difficulty, 1.0)
    reward = max(0.0, round((max(raw_reward, 0.0) / max_raw_reward) * target_score, 2))
    breakdown["raw_total"] = round(raw_reward, 2)
    breakdown["target_score"] = target_score
    breakdown["total"] = reward
    return reward, breakdown


class GrievanceRoutingEnvironment(Environment):
    """
    RL Environment for Public Grievance Routing.
    Agent reads citizen complaints and learns to:
    - Route to correct government department
    - Assign correct priority
    - Suggest correct action
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._pool = GRIEVANCE_DATASET.copy()
        self._index = 0
        self._total_reward = 0.0
        self._current_grievance = None
        random.shuffle(self._pool)
        self._load_current()

    def reset(self) -> GrievanceRoutingObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._pool = GRIEVANCE_DATASET.copy()
        self._index = 0
        self._total_reward = 0.0
        random.shuffle(self._pool)
        self._load_current()
        return GrievanceRoutingObservation(
            complaint_id=self._index,
            complaint_text=self._current_grievance["complaint"],
            difficulty=self._current_grievance["difficulty"],
            reward=0.0,
            done=False,
        )

    def step(self, action: GrievanceRoutingAction) -> GrievanceRoutingObservation:
        self._state.step_count += 1
        scored_grievance = self._current_grievance
        expected = self._current_grievance["expected"]
        difficulty = self._current_grievance["difficulty"]

        reward, breakdown = calculate_reward(action, expected, difficulty)
        self._total_reward += reward
        self._index += 1

        done = self._index >= len(self._pool)
        if not done:
            self._load_current()

        return GrievanceRoutingObservation(
            complaint_id=self._index,
            complaint_text=self._current_grievance["complaint"] if not done else "",
            difficulty=self._current_grievance["difficulty"] if not done else "",
            reward=reward,
            done=done,
            metadata={
                "scored_complaint": scored_grievance["complaint"],
                "scored_difficulty": difficulty,
                "expected": expected,
                "submitted_action": {
                    "department": action.department,
                    "priority": action.priority,
                    "action": action.action,
                    "reasoning": action.reasoning,
                },
                "reward_breakdown": breakdown,
                "total_reward": round(self._total_reward, 2),
                "step": self._state.step_count,
                "next_complaint": self._current_grievance["complaint"] if not done else "",
            },
        )

    @property
    def state(self) -> State:
        return self._state

    def _load_current(self):
        if self._index < len(self._pool):
            self._current_grievance = self._pool[self._index]
