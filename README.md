---
title: Grievance Routing Environment Server
emoji: "🏛️"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - rl
  - public-services
---

# Grievance Routing Environment

> **A real-world OpenEnv environment where AI agents learn to route citizen complaints to the correct government department, assign urgency, and choose the right operational response.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://huggingface.co/spaces/Adizeee/grievance_routing)
[![HF Space](https://img.shields.io/badge/HuggingFace-Live%20Demo-orange)](https://huggingface.co/spaces/Adizeee/grievance_routing)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://docker.com)

## The Problem

Public grievance systems often receive complaints about:

- broken streetlights
- garbage overflow
- leaking pipes
- unsafe roads
- disease risks
- urgent public safety incidents

In practice, these complaints must be triaged manually. That slows resolution, causes misrouting, and makes escalation inconsistent. This environment models that triage process as an RL task.

## What This Environment Trains

At each step, the agent sees one complaint and must decide:

- which department should own the complaint
- how urgent it is
- what action should happen next

This is not a chatbot or demo app. It is an OpenEnv simulation designed for training and evaluation.

## Difficulty Tiers

- `easy`: department classification from a clear single-signal complaint
- `medium`: department plus urgency
- `hard`: full routing decision with escalation behavior and reasoning bonus

The environment also exposes these tiers as explicit graded tasks through the API:

- `easy-routing`
- `medium-routing`
- `hard-routing`

Representative examples:

| Complaint | Expected output |
| --- | --- |
| Streetlight not working near Main Street. | electricity, medium, send_team |
| Water pipe leaking for 3 days, damaging road. | water, high, send_team |
| Garbage dump near school causing illness in children. | sanitation, critical, escalate |
| Minor crack in road, no immediate danger. | roads, low, log_complaint |

## Observation Space

`GrievanceRoutingObservation`

- `complaint_id`: integer complaint index
- `complaint_text`: raw citizen grievance text
- `difficulty`: task tier
- `reward`: reward from the previous step
- `done`: whether the episode is complete
- `metadata`: debug info including the scored complaint and reward breakdown

## Action Space

`GrievanceRoutingAction`

- `department`: `sanitation`, `electricity`, `water`, `roads`, `health`, `police`
- `priority`: `low`, `medium`, `high`, `critical`
- `action`: `log_complaint`, `send_team`, `escalate`, `close_resolved`
- `reasoning`: optional text, rewarded on hard tasks

## Reward Function

The reward is dense and gives partial credit:

- correct department: `+0.4`
- related department near miss: `-0.15`
- unrelated department miss: `-0.3`
- correct priority: `+0.3`
- one-level priority near miss: `+0.1`
- incorrect priority: `-0.1`
- correct action: `+0.3`
- critical-action near miss: `+0.05`
- low-priority conservative action: `+0.0`
- incorrect action: `-0.05`
- hard-task reasoning bonus: `+0.1`

Final scores are clipped to `[-1.0, 1.0]`.

## API

### `POST /reset`

Starts a new episode and returns the first complaint.

### `POST /step`

Accepts a typed action and returns:

- reward
- done flag
- next observation
- metadata including:
  - `scored_complaint`
  - `submitted_action`
  - `expected`
  - `reward_breakdown`
  - `next_complaint`

### `GET /state`

Returns the current OpenEnv state.

### `GET /health`

Health endpoint for deployment checks.

### `GET /tasks`

Lists the three graded tasks available to the validator.

### `GET /grade/{task_id}`

Runs the deterministic grader for one named task and returns a score strictly between `0` and `1`.

### `GET /validate`

Returns a compact validation summary for task count, grader availability, and score ranges.

## Running Locally

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run the baseline inference script against a running server:

```bash
python inference.py
```

Run it against a local Docker image:

```bash
LOCAL_IMAGE_NAME=grievance_routing-env:latest python inference.py
```

## Baseline Inference

The included `inference.py`:

- uses the OpenAI client
- uses `API_BASE_URL` and `HF_TOKEN` for LLM calls
- supports `MODEL_NAME` and `LOCAL_IMAGE_NAME`
- emits validator-friendly `[START]`, `[STEP]`, and `[END]` logs
- keeps outputs inside the environment label space for reproducible grading

## Validation

Recommended checks before submission:

```bash
python -m unittest test_inference.py
python -m py_compile inference.py client.py models.py server/grievance_routing_environment.py server/app.py
openenv validate
```

## Project Structure

```text
grievance_routing/
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── README.md
├── test_inference.py
└── server/
    ├── app.py
    ├── Dockerfile
    ├── grievance_env_environment.py
    ├── grievance_routing_environment.py
    ├── requirements.txt
    └── __init__.py
```

## Deployment

The repo is configured for Hugging Face Spaces with:

- `openenv.yaml` pointing to `server.app:app`
- a root `Dockerfile` for HF deployment
- a callable server entry point in `server/app.py`

Live Space:

`https://adizeee-grievance-routing.hf.space`
