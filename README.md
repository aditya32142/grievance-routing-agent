# 🏛️ Grievance Routing Environment

> **A real-world OpenEnv environment where AI agents learn to route citizen complaints to the correct government department — automatically, intelligently, and at scale.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://huggingface.co/spaces/Adizeee/grievance_routing)
[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-orange)](https://huggingface.co/spaces/Adizeee/grievance_routing)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://docker.com)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-lightgrey)](LICENSE)

---

## 🌍 The Problem

Every day, thousands of citizens file complaints about broken streetlights, garbage overflow, water leaks, and public safety issues. These complaints land in a central inbox and must be:

- Routed to the **correct department**
- Assigned the right **priority level**
- Paired with an appropriate **action**

This is done **manually** — slow, error-prone, and unscalable.

**This project teaches an AI agent to do it automatically.**

---

## 🎯 What This Is

This is a fully compliant **OpenEnv reinforcement learning environment** — not an app, not a chatbot.

It is a **training ground** where AI agents practice routing decisions and receive reward signals based on correctness. Agents that learn from this environment can eventually outperform manual routing.

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Complaint (Observation)                           │
│        ↓                                            │
│   Agent decides:                                    │
│     • Which department?                             │
│     • How urgent?                                   │
│     • What action?                                  │
│        ↓                                            │
│   Environment scores the decision (Reward)          │
│        ↓                                            │
│   Next complaint appears                            │
│        ↓                                            │
│   Repeat until episode ends                         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## ⚡ Live Demo

```
https://adizeee-grievance-routing.hf.space/docs
```

The API is live. You can connect your own agent right now.

---

## 🧩 Environment Design

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `complaint_id` | int | Unique complaint identifier |
| `complaint_text` | str | Raw citizen complaint in natural language |
| `difficulty` | str | Task difficulty: `easy`, `medium`, `hard` |

### Action Space

| Field | Type | Valid Values |
|-------|------|-------------|
| `department` | str | `sanitation`, `electricity`, `water`, `roads`, `health`, `police` |
| `priority` | str | `low`, `medium`, `high`, `critical` |
| `action` | str | `log_complaint`, `send_team`, `escalate`, `close_resolved` |
| `reasoning` | str | Optional — earns bonus reward in hard mode |

### Reward Function

```
Correct department  → +0.4
Correct priority    → +0.3
Correct action      → +0.3
Reasoning (hard)    → +0.1 bonus

Wrong department    → -0.5  (heaviest penalty)
Wrong priority      → -0.2
Wrong action        → -0.1

Final reward clamped to [-1.0, 1.0]
```

The reward is **dense** — every decision gets a signal. No sparse end-of-episode scoring.

---

## 🏆 Tasks & Difficulty Progression

### 🟢 Easy — Department Classification
Agent must identify the correct department.

| Complaint | Expected Department |
|-----------|-------------------|
| Streetlight not working | electricity |
| Garbage not collected for 5 days | sanitation |
| Pothole causing accidents | roads |

### 🟡 Medium — Department + Priority
Agent must also correctly prioritize.

| Complaint | Department | Priority |
|-----------|------------|----------|
| Water pipe leaking for 3 days | water | high |
| Power outage in entire block | electricity | critical |
| Mosquito breeding in park | health | medium |

### 🔴 Hard — Full Decision + Reasoning
Agent must get department, priority, action correct — and explain why for bonus reward.

| Complaint | Department | Priority | Action |
|-----------|------------|----------|--------|
| Garbage near school causing illness | sanitation | critical | escalate |
| Food poisoning — 10 affected | health | critical | escalate |
| Street brawl, police not responding | police | critical | escalate |
| Minor crack in road, no danger | roads | low | log_complaint |

---

## 🔌 API Reference

### `POST /reset`
Start a new episode. Returns the first complaint.

```json
Request:  {}
Response: {
  "observation": {
    "complaint_id": 0,
    "complaint_text": "Garbage not collected for 5 days.",
    "difficulty": "easy"
  },
  "done": false
}
```

### `POST /step`
Submit an action. Receive reward and next observation.

```json
Request: {
  "action": {
    "department": "sanitation",
    "priority": "high",
    "action": "send_team",
    "reasoning": "Garbage overflow needs immediate cleanup team dispatch."
  }
}

Response: {
  "observation": { ... next complaint ... },
  "reward": 1.0,
  "done": false
}
```

### `GET /state`
Check current environment state.

### `GET /health`
Health check — confirms environment is live.

---

## 🚀 Quick Start

### Connect your agent (any language)

```python
import requests

BASE = "https://adizeee-grievance-routing.hf.space"

# Start episode
obs = requests.post(f"{BASE}/reset").json()["observation"]

while True:
    print(obs["complaint_text"])

    # Your agent decides here
    result = requests.post(f"{BASE}/step", json={"action": {
        "department": "sanitation",
        "priority": "high",
        "action": "send_team",
        "reasoning": "Garbage complaint."
    }}).json()

    print(f"Reward: {result['reward']}")

    if result["done"]:
        break
    obs = result["observation"]
```

### Run locally with Docker

```bash
docker build -t grievance-routing .
docker run -p 8000:8000 grievance-routing
```

### Run inference script

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama3-8b-8192"
export HF_TOKEN="your_token_here"

python inference.py
```

---

## 🗂️ Project Structure

```
grievance_routing/
├── inference.py                        # Baseline LLM agent (entry point)
├── models.py                           # Pydantic action/observation models
├── openenv.yaml                        # OpenEnv spec configuration
├── Dockerfile                          # Container deployment
├── README.md                           # This file
└── server/
    ├── app.py                          # FastAPI application
    ├── grievance_routing_environment.py # Core RL environment logic
    └── requirements.txt                # Server dependencies
```

---

## 📊 Baseline Scores

Results from running the LLM baseline agent (Groq / LLaMA 3):

| Task Difficulty | Avg Reward | Accuracy |
|-----------------|------------|----------|
| Easy | 0.72 | 80% |
| Medium | 0.41 | 55% |
| Hard | 0.18 | 30% |

Hard tasks genuinely challenge frontier models — especially distinguishing `send_team` vs `escalate` for critical complaints.

---

## 🛠️ Tech Stack

- **OpenEnv** — RL environment framework by Meta & Hugging Face
- **FastAPI** — High-performance async API server
- **Pydantic** — Typed action/observation models
- **Docker** — Containerized deployment
- **Hugging Face Spaces** — Live cloud hosting
- **Groq / OpenAI-compatible API** — LLM inference

---

## 💡 Real-World Impact

This environment models a genuine problem that affects millions of citizens daily. A well-trained agent from this environment could:

- Reduce complaint resolution time from days to minutes
- Eliminate misrouted complaints
- Auto-escalate critical issues before they worsen
- Scale to handle thousands of complaints simultaneously

---

## 📬 Contact

Built by **Adizeee** for the OpenEnv Hackathon — Round 1.

HF Space: [Adizeee/grievance_routing](https://huggingface.co/spaces/Adizeee/grievance_routing)
