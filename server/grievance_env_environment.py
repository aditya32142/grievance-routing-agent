
"""
Decentralized Public Grievance Routing Environment
grievance_routing/server/grievance_env_environment.py
"""

import random
from dataclasses import dataclass, field
from typing import Optional

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
        "complaint": "Water pipe leaking for 3 days, damaging road.",
        "difficulty": "medium",
        "expected": {"department": "water", "priority": "high", "action": "send_team"},
    },
    {
        "complaint": "Garbage dump near school causing illness in children.",
        "difficulty": "hard",
        "expected": {"department": "sanitation", "priority": "critical", "action": "escalate"},
    },
]

@dataclass
class Observation:
    complaint_id: int
    complaint_text: str
    difficulty: str

@dataclass
class Action:
    department: str
    priority: str
    action: str
    reasoning: Optional[str] = None

@dataclass
class StepResult:
    observation: Observation
    action: Action
    reward: float
    done: bool
    info: dict = field(default_factory=dict)

def calculate_reward(action, expected, difficulty):
    reward = 0.0
    breakdown = {}

    if action.department == expected["department"]:
        reward += 0.4
        breakdown["department"] = "+0.4 (correct)"
    else:
        reward -= 0.5
        breakdown["department"] = f"-0.5 (wrong: got {action.department})"

    if action.priority == expected["priority"]:
        reward += 0.3
        breakdown["priority"] = "+0.3 (correct)"
    else:
        reward -= 0.2
        breakdown["priority"] = f"-0.2 (wrong: got {action.priority})"

    if action.action == expected["action"]:
        reward += 0.3
        breakdown["action"] = "+0.3 (correct)"
    else:
        reward -= 0.1
        breakdown["action"] = f"-0.1 (wrong: got {action.action})"

    reward = max(-1.0, min(1.0, round(reward, 2)))
    breakdown["total"] = reward
    return reward, breakdown

class GrievanceEnv:
    def __init__(self, difficulty="all", shuffle=True):
        if difficulty == "all":
            self._pool = GRIEVANCE_DATASET.copy()
        else:
            self._pool = [g for g in GRIEVANCE_DATASET if g["difficulty"] == difficulty]

        self.difficulty = difficulty
        self.shuffle = shuffle
        self._index = 0
        self._total_reward = 0.0
        self._step_count = 0
        self._history = []
        self._current_grievance = None

        if self.shuffle:
            random.shuffle(self._pool)

    def reset(self):
        self._index = 0
        self._total_reward = 0.0
        self._step_count = 0
        self._history = []
        if self.shuffle:
            random.shuffle(self._pool)
        return self._get_observation()

    def step(self, action):
        expected = self._current_grievance["expected"]
        difficulty = self._current_grievance["difficulty"]

        reward, breakdown = calculate_reward(action, expected, difficulty)
        self._total_reward += reward
        self._step_count += 1
        self._index += 1

        done = self._index >= len(self._pool)

        obs = self._get_observation() if not done else self._current_observation

        result = StepResult(
            observation=obs,
            action=action,
            reward=reward,
            done=done,
            info={
                "expected": expected,
                "reward_breakdown": breakdown,
                "total_reward": round(self._total_reward, 2),
                "step": self._step_count,
            },
        )
        self._history.append(result)
        return result

    def summary(self):
        correct = sum(1 for r in self._history if r.reward > 0.5)
        return {
            "total_steps": self._step_count,
            "total_reward": round(self._total_reward, 2),
            "avg_reward": round(self._total_reward / max(self._step_count, 1), 2),
            "correct_decisions": correct,
            "accuracy": round(correct / max(self._step_count, 1), 2),
        }

    def _get_observation(self):
        if self._index >= len(self._pool):
            return self._current_observation
        self._current_grievance = self._pool[self._index]
        obs = Observation(
            complaint_id=self._index,
            complaint_text=self._current_grievance["complaint"],
            difficulty=self._current_grievance["difficulty"],
        )
        self._current_observation = obs
        return obs

if __name__ == "__main__":
    print("=" * 50)
    print("  GrievanceEnv — Smoke Test")
    print("=" * 50)

    env = GrievanceEnv(difficulty="all", shuffle=False)
    obs = env.reset()

    while True:
        print(f"\nComplaint : {obs.complaint_text}")
        print(f"Difficulty: {obs.difficulty}")

        action = Action(
            department="sanitation",
            priority="high",
            action="send_team",
        )

        result = env.step(action)
        print(f"Reward    : {result.reward}")
        print(f"Breakdown : {result.info['reward_breakdown']}")

        if result.done:
            break
        obs = result.observation

    print("\n── Summary ──")
    for k, v in env.summary().items():
        print(f"  {k}: {v}")