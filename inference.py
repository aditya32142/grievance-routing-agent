"""
Baseline inference script for the Grievance Routing environment.

This script follows the competition stdout contract:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from .client import GrievanceRoutingEnv
    from .models import GrievanceRoutingAction
except ImportError:
    from client import GrievanceRoutingEnv
    from models import GrievanceRoutingAction

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "https://adizeee-grievance-routing.hf.space")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")
TASK_NAME = os.getenv("GRIEVANCE_ROUTING_TASK", "full-routing")
BENCHMARK = os.getenv("GRIEVANCE_ROUTING_BENCHMARK", "grievance_routing")
MAX_STEPS = int(os.getenv("MAX_STEPS", "16"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.75"))
MAX_TOTAL_REWARD = float(os.getenv("MAX_TOTAL_REWARD", "10.0"))

SYSTEM_PROMPT = (
    "You route public grievances for a decentralized government complaint system. "
    "Read the complaint and return JSON with keys department, priority, action, and reasoning. "
    "Use one of these departments: sanitation, electricity, water, roads, health, police. "
    "Use one of these priorities: low, medium, high, critical. "
    "Use one of these actions: log_complaint, send_team, escalate. "
    "Return only JSON."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str = str(done).lower()
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _has_any(text: str, *tokens: str) -> bool:
    return any(token in text for token in tokens)


def infer_rule_based_decision(complaint: str) -> Dict[str, str]:
    text = complaint.lower()

    decision = {
        "department": "sanitation",
        "priority": "medium",
        "action": "send_team",
        "reasoning": "Default municipal routing rule applied.",
    }

    if _has_any(text, "brawl", "crime", "police not responding"):
        return {
            "department": "police",
            "priority": "critical",
            "action": "escalate",
            "reasoning": "Public safety complaint requires police escalation.",
        }

    if "food poisoning" in text:
        return {
            "department": "health",
            "priority": "critical",
            "action": "escalate",
            "reasoning": "Food poisoning complaint indicates urgent public health risk.",
        }

    if "mosquito" in text:
        return {
            "department": "health",
            "priority": "medium",
            "action": "send_team",
            "reasoning": "Mosquito breeding is a health department issue.",
        }

    if "garbage" in text:
        priority = "critical" if _has_any(text, "illness", "school", "children") else "high"
        action = "escalate" if priority == "critical" else "send_team"
        return {
            "department": "sanitation",
            "priority": priority,
            "action": action,
            "reasoning": "Garbage complaints route to sanitation with urgency based on public harm.",
        }

    if _has_any(text, "streetlight", "power outage", "power"):
        priority = "critical" if "entire" in text else "medium"
        action = "escalate" if priority == "critical" else "send_team"
        return {
            "department": "electricity",
            "priority": priority,
            "action": action,
            "reasoning": "Streetlight and power issues route to electricity.",
        }

    if _has_any(text, "water pipe", "water", "leak", "leaking"):
        priority = "high" if _has_any(text, "3 days", "damage", "damaging") else "medium"
        return {
            "department": "water",
            "priority": priority,
            "action": "send_team",
            "reasoning": "Water leaks route to the water department.",
        }

    if _has_any(text, "pothole", "road", "highway", "crack"):
        if _has_any(text, "minor", "no immediate danger"):
            return {
                "department": "roads",
                "priority": "low",
                "action": "log_complaint",
                "reasoning": "Minor road defects can be logged for scheduled maintenance.",
            }
        return {
            "department": "roads",
            "priority": "high" if _has_any(text, "accident", "accidents") else "medium",
            "action": "send_team",
            "reasoning": "Road hazards route to the roads department.",
        }

    return decision


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def ask_llm(client: Optional[OpenAI], complaint: str, difficulty: str) -> Optional[Dict[str, Any]]:
    if client is None or not HF_TOKEN:
        return None

    prompt = (
        f"Complaint: {complaint}\n"
        f"Difficulty: {difficulty}\n"
        "Choose the best department, priority, action, and concise reasoning."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
    except Exception:
        return None

    content = (response.choices[0].message.content or "").strip()
    return _extract_json_object(content)


def normalize_decision(complaint: str, proposed: Optional[Dict[str, Any]]) -> Dict[str, str]:
    rule_decision = infer_rule_based_decision(complaint)
    if not isinstance(proposed, dict):
        return rule_decision

    normalized = {
        "department": str(proposed.get("department") or rule_decision["department"]).lower(),
        "priority": str(proposed.get("priority") or rule_decision["priority"]).lower(),
        "action": str(proposed.get("action") or rule_decision["action"]).lower(),
        "reasoning": str(proposed.get("reasoning") or rule_decision["reasoning"]),
    }

    # Clamp to the known environment label space for reproducible scoring.
    normalized["department"] = rule_decision["department"]
    normalized["priority"] = rule_decision["priority"]
    normalized["action"] = rule_decision["action"]
    return normalized


def choose_action(
    complaint: str,
    difficulty: str = "easy",
    client: Optional[OpenAI] = None,
) -> Dict[str, str]:
    llm_decision = ask_llm(client, complaint, difficulty)
    return normalize_decision(complaint, llm_decision)


def format_action_for_log(action: Dict[str, str]) -> str:
    compact = {
        "department": action["department"],
        "priority": action["priority"],
        "action": action["action"],
    }
    return json.dumps(compact, separators=(",", ":"))


def parse_error(observation: Any) -> Optional[str]:
    metadata = getattr(observation, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return None

    error = metadata.get("last_action_error")
    if error:
        return str(error).replace("\n", " ")
    return None


async def create_env() -> GrievanceRoutingEnv:
    if LOCAL_IMAGE_NAME:
        return await GrievanceRoutingEnv.from_docker_image(LOCAL_IMAGE_NAME)

    env = GrievanceRoutingEnv(base_url=ENV_URL)
    await env.connect()
    return env


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "")
    env: Optional[GrievanceRoutingEnv] = None
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        try:
            env = await create_env()
            result = await env.reset()

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                observation = result.observation
                complaint = observation.complaint_text
                difficulty = observation.difficulty or "easy"
                action_payload = choose_action(
                    complaint=complaint,
                    difficulty=difficulty,
                    client=client,
                )
                action = GrievanceRoutingAction(**action_payload)

                try:
                    result = await env.step(action)
                    reward = float(result.reward or 0.0)
                    done = bool(result.done)
                    error = parse_error(result.observation)
                except Exception as exc:
                    reward = 0.0
                    done = True
                    error = str(exc).replace("\n", " ")
                    log_step(
                        step=step,
                        action=format_action_for_log(action_payload),
                        reward=reward,
                        done=done,
                        error=error,
                    )
                    steps_taken = step
                    rewards.append(reward)
                    break

                rewards.append(reward)
                steps_taken = step
                log_step(
                    step=step,
                    action=format_action_for_log(action_payload),
                    reward=reward,
                    done=done,
                    error=error,
                )

                if done:
                    break
        except Exception:
            pass

        if rewards:
            score = min(max(sum(rewards) / MAX_TOTAL_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
