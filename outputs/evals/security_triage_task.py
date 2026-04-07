"""Inspect AI task for security-focused email triage."""

from __future__ import annotations

import json
from typing import Any, Dict

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, accuracy, scorer
from inspect_ai.solver import generate


def _prompt(sample: Dict[str, Any]) -> str:
    return (
        "Classify this email as JSON with keys priority, category, response, "
        "assign_to, mark_phishing.\n"
        f"Sender: {sample['sender']}\n"
        f"Subject: {sample['subject']}\n"
        f"Body: {sample['body_preview']}\n"
    )


SECURITY_DATASET = [
    {
        "input": {
            "sender": "ciso@contoso-security.com",
            "subject": "Multiple employees received credential reset prompts",
            "body_preview": "Investigate suspicious Microsoft 365 reset requests immediately.",
        },
        "target": {
            "priority": "urgent",
            "category": "security",
            "response": "escalate",
            "assign_to": "security",
            "mark_phishing": False,
        },
    }
]


@scorer(metrics=[accuracy()])
def security_triage_grader():
    async def score(state, target):
        try:
            output = json.loads(state.output.completion)
        except Exception:
            return Score(value=0.0, explanation="invalid_json")

        expected = target
        value = 1.0 if all(output.get(k) == v for k, v in expected.items()) else 0.0
        return Score(value=value, explanation="exact_match" if value else "mismatch")

    return score


@task
def security_triage_task():
    samples = [
        Sample(input=_prompt(item["input"]), target=item["target"])
        for item in SECURITY_DATASET
    ]
    return Task(dataset=samples, solver=[generate()], scorer=security_triage_grader())
