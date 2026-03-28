import os
import json
import requests
import random
import re
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# RL Environment (HF Space)
ENV_URL = os.environ.get(
    "ENV_URL",
    "https://adizeee-grievance-routing.hf.space"
)

# LLM (Groq via OpenAI-compatible API)
LLM_BASE = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("MODEL_NAME", "llama3-8b-8192")
API_KEY = os.environ.get("HF_TOKEN")

client = OpenAI(
    base_url=LLM_BASE,
    api_key=API_KEY
)

Q = {}
epsilon = 0.1


# ─────────────────────────────────────────────
# LLM DECISION (ROBUST)
# ─────────────────────────────────────────────

def ask_llm(complaint):

    prompt = f"""
Classify this complaint:

{complaint}

Return ONLY JSON:
{{
  "department": "...",
  "priority": "...",
  "action": "..."
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content

        # Extract JSON safely
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())

    except:
        pass

    return None


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

def get_state(text):

    text = text.lower()

    if "power" in text or "electric" in text or "streetlight" in text:
        return "electricity"

    if "pothole" in text or "crack" in text or "road" in text:
        return "roads"

    if "garbage" in text or "waste" in text:
        return "sanitation"

    if "mosquito" in text:
        return "sanitation"

    if "food" in text or "poison" in text:
        return "health"

    if "water" in text or "leak" in text:
        return "water"

    if "brawl" in text or "crime" in text or "fight" in text:
        return "police"

    return "other"


# ─────────────────────────────────────────────
# FALLBACK POLICY
# ─────────────────────────────────────────────

def fallback(state):

    options = {
        "roads": [
            {"department": "roads", "priority": "low", "action": "log_complaint"},
            {"department": "roads", "priority": "high", "action": "send_team"}
        ],
        "sanitation": [
            {"department": "sanitation", "priority": "medium", "action": "send_team"}
        ],
        "electricity": [
            {"department": "electricity", "priority": "low", "action": "log_complaint"},
            {"department": "electricity", "priority": "high", "action": "send_team"}
        ],
        "health": [
            {"department": "health", "priority": "critical", "action": "send_team"}
        ],
        "water": [
            {"department": "water", "priority": "medium", "action": "send_team"}
        ],
        "police": [
            {"department": "police", "priority": "high", "action": "send_team"}
        ]
    }

    return random.choice(options.get(state, [
        {"department": "sanitation", "priority": "low", "action": "log_complaint"}
    ]))


# ─────────────────────────────────────────────
# ACTION SELECTION
# ─────────────────────────────────────────────

def choose_action(state, complaint):

    # 1. Try LLM first
    decision = ask_llm(complaint)

    if isinstance(decision, dict):
        return decision

    # 2. Learned policy
    if state in Q and Q[state]:
        best = max(Q[state], key=Q[state].get)
        return json.loads(best)

    # 3. fallback
    return fallback(state)


# ─────────────────────────────────────────────
# Q-LEARNING UPDATE
# ─────────────────────────────────────────────

def update_q(state, decision, reward):

    key = json.dumps(decision)

    if state not in Q:
        Q[state] = {}

    old = Q[state].get(key, 0)

    # Slight penalty amplification
    if reward < 0:
        reward *= 1.2

    Q[state][key] = old + 0.3 * (reward - old)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():

    obs = requests.post(f"{ENV_URL}/reset").json().get("observation", {})

    total = 0

    for step in range(30):

        complaint = obs.get("complaint_text", "")

        print(f"\nStep {step + 1}")
        print("Complaint:", complaint)

        state = get_state(complaint)
        decision = choose_action(state, complaint)

        print("Decision:", decision)

        result = requests.post(f"{ENV_URL}/step", json={"action": decision}).json()

        reward = result.get("reward", 0)
        print("Reward:", reward)

        update_q(state, decision, reward)
        total += reward

        if result.get("done"):
            break

        obs = result.get("observation", {})

    print("\nFinal Score:", total)


# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()