"""Inference entrypoint for the email triage RL environment."""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from email_triage_rl.client import EmailTriageRlEnv
    from email_triage_rl.models import EmailTriageRlAction
except ImportError:
    from client import EmailTriageRlEnv
    from models import EmailTriageRlAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv(
    "IMAGE_NAME", "email_triage_rl_environment.py:latest"
)
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("ENV_HTTP_URL")
TASK_NAME = os.getenv("TASK_NAME", "email-triage")
BENCHMARK = os.getenv("BENCHMARK", "email_triage_rl")
MAX_STEPS = 6
SUCCESS_SCORE_THRESHOLD = 0.8
MAX_TOTAL_REWARD = 6 * 5.25

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are solving an email triage task.
    For the current email, choose exactly one value for:
    - priority
    - category
    - response
    - assign_to
    - mark_phishing

    Reply with strict JSON only, using this schema:
    {
      "priority": "low|medium|high|urgent",
      "category": "sales|support|billing|security|meeting|personal",
      "response": "archive|reply|forward|escalate|schedule",
      "assign_to": "none|sales|support|finance|security|exec_assistant",
      "mark_phishing": true
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(observation) -> str:
    labels = observation.metadata.get("available_labels", {})
    return textwrap.dedent(
        f"""
        Email ID: {observation.email_id}
        Sender: {observation.sender}
        Subject: {observation.subject}
        Body Preview: {observation.body_preview}
        Hours Since Received: {observation.hours_since_received}
        Customer Tier: {observation.customer_tier}
        Thread Length: {observation.thread_length}
        Remaining Emails: {observation.remaining_emails}
        Previous Feedback: {observation.previous_action_feedback}

        Allowed labels:
        {json.dumps(labels, indent=2)}
        """
    ).strip()


def normalize_action(payload: dict) -> EmailTriageRlAction:
    return EmailTriageRlAction(
        priority=str(payload.get("priority", "medium")).lower(),
        category=str(payload.get("category", "support")).lower(),
        response=str(payload.get("response", "reply")).lower(),
        assign_to=str(payload.get("assign_to", "none")).lower(),
        mark_phishing=bool(payload.get("mark_phishing", False)),
    )


def fallback_action(observation) -> EmailTriageRlAction:
    subject = observation.subject.lower()
    body = observation.body_preview.lower()
    text = f"{subject} {body}"

    if "password" in text or "credential" in text or "verify" in text:
        return EmailTriageRlAction(
            priority="urgent" if "credential" in text else "high",
            category="security",
            response="escalate",
            assign_to="security",
            mark_phishing="attached html" in text or "verify your password" in text,
        )
    if "invoice" in text or "bill" in text or "payment" in text:
        return EmailTriageRlAction(
            priority="high",
            category="billing",
            response="forward",
            assign_to="finance",
            mark_phishing=False,
        )
    if "meeting" in text or "sync" in text or "review" in text or "tuesday" in text:
        return EmailTriageRlAction(
            priority="medium",
            category="meeting",
            response="schedule",
            assign_to="exec_assistant",
            mark_phishing=False,
        )
    if "locked" in text or "cannot" in text or "help restoring access" in text:
        return EmailTriageRlAction(
            priority="urgent",
            category="support",
            response="escalate",
            assign_to="support",
            mark_phishing=False,
        )
    if "reminder" in text or "reimbursement" in text:
        return EmailTriageRlAction(
            priority="low",
            category="personal",
            response="archive",
            assign_to="none",
            mark_phishing=False,
        )
    return EmailTriageRlAction(
        priority="medium",
        category="support",
        response="reply",
        assign_to="support",
        mark_phishing=False,
    )


def get_model_action(client: OpenAI, observation) -> EmailTriageRlAction:
    if not API_KEY:
        return fallback_action(observation)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(observation)},
            ],
            temperature=0.0,
            max_tokens=200,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        action = normalize_action(json.loads(content))
        return action
    except Exception:
        return fallback_action(observation)


def format_action(action: EmailTriageRlAction) -> str:
    return (
        f"priority={action.priority},category={action.category},response={action.response},"
        f"assign_to={action.assign_to},mark_phishing={str(action.mark_phishing).lower()}"
    )


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing")
    env = None
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    fatal_error: Optional[str] = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = (
            EmailTriageRlEnv(base_url=ENV_BASE_URL)
            if ENV_BASE_URL
            else await EmailTriageRlEnv.from_docker_image(LOCAL_IMAGE_NAME)
        )
        result = await env.reset()
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, result.observation)
            result = await env.step(action)
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=format_action(action),
                reward=reward,
                done=result.done,
                error=None,
            )

            if result.done:
                break

        score = min(max(sum(rewards) / MAX_TOTAL_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        fatal_error = str(exc).replace("\n", " ")
    finally:
        try:
            if env is not None:
                await env.close()
        finally:
            if fatal_error:
                print(f"fatal_error={fatal_error}", file=sys.stderr, flush=True)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
