"""Inspect AI task for phishing-focused email triage."""

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


PHISHING_DATASET = [
    {
        "input": {
            "sender": "it-helpdesk@companny-support.co",
            "subject": "Mandatory password verification today",
            "body_preview": "Verify your password immediately using the attached HTML file.",
        },
        "target": {
            "priority": "high",
            "category": "security",
            "response": "escalate",
            "assign_to": "security",
            "mark_phishing": True,
        },
    }
]


@scorer(metrics=[accuracy()])
def phishing_triage_grader():
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
def phishing_triage_task():
    samples = [
        Sample(input=_prompt(item["input"]), target=item["target"])
        for item in PHISHING_DATASET
    ]
    return Task(dataset=samples, solver=[generate()], scorer=phishing_triage_grader())
