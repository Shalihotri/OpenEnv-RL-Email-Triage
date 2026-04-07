---
title: Email Triage Rl Environment Server
emoji: 📥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# OpenEnv RL Email Triage

An OpenEnv environment for benchmarking and training RL policies on realistic email triage. Each episode presents a fixed inbox of business emails. The agent must classify urgency, choose the correct category, decide the next action, route the message to the right team, and detect phishing.

## What The Environment Contains

- A finite-horizon inbox of six realistic business emails
- Security, billing, meeting, support, personal, and phishing scenarios
- Dense per-step reward with immediate feedback
- Docker-ready OpenEnv server
- A root `inference.py` script that follows the sample OpenEnv inference contract

## Observation Space

Each step exposes one email with:

- `email_id`
- `sender`
- `subject`
- `body_preview`
- `hours_since_received`
- `customer_tier`
- `thread_length`
- `remaining_emails`
- `completed`
- `previous_action_feedback`

## Action Space

The agent must emit one structured action per email:

- `priority`: `low | medium | high | urgent`
- `category`: `sales | support | billing | security | meeting | personal`
- `response`: `archive | reply | forward | escalate | schedule`
- `assign_to`: `none | sales | support | finance | security | exec_assistant`
- `mark_phishing`: `true | false`

## Reward Design

Reward is dense and computed per email:

- `+1.5` for correct priority
- `+1.0` for correct category
- `+1.0` for correct response
- `+0.5` for correct assignment
- `+1.0` for correct phishing detection on phishing emails
- `+0.25` for correctly not flagging safe emails
- Negative reward for incorrect decisions

The maximum total reward per episode is `31.5`, and `inference.py` normalizes score into `[0, 1]`.

## Project Layout

```text
email_triage_rl/
├── Dockerfile
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── client.py
├── models.py
└── server/
    ├── app.py
    ├── Dockerfile
    ├── email_triage_rl_environment.py
    └── requirements.txt
```

## Local Setup

```bash
uv sync
```

Run the server locally:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Docker

Build from the repo root:

```bash
docker build -t email_triage_rl_environment.py:latest .
```

Run the container:

```bash
docker run -p 8000:8000 email_triage_rl_environment.py:latest
```

## Inference Script

The root [inference.py](./inference.py) follows the contract from the OpenEnv sample script:

- Uses the `OpenAI` client for model calls
- Emits `[START]`, `[STEP]`, and `[END]` lines
- Supports either a running environment via `ENV_BASE_URL` or a local Docker image via `LOCAL_IMAGE_NAME`
- Falls back to a deterministic heuristic policy when no API key is configured

Example:

```bash
ENV_BASE_URL=http://127.0.0.1:8000 python inference.py
```

Optional environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `API_KEY`
- `LOCAL_IMAGE_NAME`
- `ENV_BASE_URL`
- `TASK_NAME`
- `BENCHMARK`

## Validation

This environment was prepared to satisfy the same checks used by the OpenEnv pre-validation flow:

- `openenv validate`
- `docker build <repo_dir>`
- Python syntax checks for environment modules and `inference.py`

## Example Usage

```python
from email_triage_rl import EmailTriageRlAction, EmailTriageRlEnv

with EmailTriageRlEnv(base_url="http://localhost:8000") as env:
    obs = env.reset().observation
    print(obs.subject)

    result = env.step(
        EmailTriageRlAction(
            priority="urgent",
            category="security",
            response="escalate",
            assign_to="security",
            mark_phishing=False,
        )
    )
    print(result.reward)
    print(result.observation.previous_action_feedback)
```

## Status

The environment passes:

- `openenv validate`
- repo-level Docker build
- direct `inference.py` execution against a live container
