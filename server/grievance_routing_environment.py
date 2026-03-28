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

def calculate_reward(action, expected, difficulty):
    reward = 0.0

    if action.department == expected["department"]:
        reward += 0.4
    else:
        reward -= 0.5

    if action.priority == expected["priority"]:
        reward += 0.3
    else:
        p_idx = PRIORITIES.index(action.priority) if action.priority in PRIORITIES else -1
        e_idx = PRIORITIES.index(expected["priority"]) if expected["priority"] in PRIORITIES else -1
        if abs(p_idx - e_idx) == 1:
            reward += 0.1
        else:
            reward -= 0.2

    if action.action == expected["action"]:
        reward += 0.3
    else:
        reward -= 0.1

    if difficulty == "hard" and action.reasoning:
        reward += 0.1

    return max(-1.0, min(1.0, round(reward, 2)))


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
        expected = self._current_grievance["expected"]
        difficulty = self._current_grievance["difficulty"]

        reward = calculate_reward(action, expected, difficulty)
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
                "expected": expected,
                "total_reward": round(self._total_reward, 2),
                "step": self._state.step_count,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    def _load_current(self):
        if self._index < len(self._pool):
            self._current_grievance = self._pool[self._index]
