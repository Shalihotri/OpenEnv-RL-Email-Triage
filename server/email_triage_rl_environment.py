# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email triage RL environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EmailTriageRlAction, EmailTriageRlObservation
except ImportError:
    from models import EmailTriageRlAction, EmailTriageRlObservation


@dataclass(frozen=True)
class EmailCase:
    email_id: str
    sender: str
    subject: str
    body_preview: str
    hours_since_received: int
    customer_tier: str
    thread_length: int
    priority: str
    category: str
    response: str
    assign_to: str
    phishing: bool = False


INBOX: list[EmailCase] = [
    EmailCase(
        email_id="msg-001",
        sender="ciso@contoso-security.com",
        subject="Multiple employees received credential reset prompts",
        body_preview=(
            "We are seeing suspicious Microsoft 365 reset requests across finance. "
            "Please investigate immediately and confirm containment steps."
        ),
        hours_since_received=1,
        customer_tier="enterprise",
        thread_length=4,
        priority="urgent",
        category="security",
        response="escalate",
        assign_to="security",
    ),
    EmailCase(
        email_id="msg-002",
        sender="ap@vendor-payments.net",
        subject="Invoice discrepancy on March platform bill",
        body_preview=(
            "The invoice total appears 18% above our contract. Please send a corrected "
            "statement before tomorrow's payment run."
        ),
        hours_since_received=6,
        customer_tier="midmarket",
        thread_length=2,
        priority="high",
        category="billing",
        response="forward",
        assign_to="finance",
    ),
    EmailCase(
        email_id="msg-003",
        sender="events@northstar.ai",
        subject="Can we lock a roadmap review for next Tuesday?",
        body_preview=(
            "Our team would like a 30 minute sync next Tuesday afternoon to review "
            "integration milestones and launch readiness."
        ),
        hours_since_received=18,
        customer_tier="enterprise",
        thread_length=1,
        priority="medium",
        category="meeting",
        response="schedule",
        assign_to="exec_assistant",
    ),
    EmailCase(
        email_id="msg-004",
        sender="support@shopper-mail.com",
        subject="Need help restoring access for 200 retail agents",
        body_preview=(
            "A broken SSO rollout locked our floor staff out of the dashboard. "
            "Stores open in three hours and agents cannot clock in."
        ),
        hours_since_received=2,
        customer_tier="enterprise",
        thread_length=5,
        priority="urgent",
        category="support",
        response="escalate",
        assign_to="support",
    ),
    EmailCase(
        email_id="msg-005",
        sender="benefits@wellhub.example",
        subject="Reminder: wellness reimbursement form",
        body_preview=(
            "Friendly reminder that reimbursement forms for last quarter are due by "
            "Friday. Reply only if you need help with the portal."
        ),
        hours_since_received=30,
        customer_tier="internal",
        thread_length=1,
        priority="low",
        category="personal",
        response="archive",
        assign_to="none",
    ),
    EmailCase(
        email_id="msg-006",
        sender="it-helpdesk@companny-support.co",
        subject="Mandatory password verification today",
        body_preview=(
            "To avoid mailbox suspension, verify your password immediately using the "
            "attached HTML file and send the result to the IT desk."
        ),
        hours_since_received=3,
        customer_tier="internal",
        thread_length=1,
        priority="high",
        category="security",
        response="escalate",
        assign_to="security",
        phishing=True,
    ),
]


class EmailTriageRlEnvironment(Environment):
    """Finite-horizon environment for learning inbox triage policies."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._cursor = 0
        self._score = 0.0
        self._last_feedback = "Episode initialized."

    def reset(self) -> EmailTriageRlObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._cursor = 0
        self._score = 0.0
        self._last_feedback = "Classify the first email."
        return self._build_observation(status="ready", reward=0.0, done=False)

    def step(self, action: EmailTriageRlAction) -> EmailTriageRlObservation:  # type: ignore[override]
        self._state.step_count += 1
        if self._cursor >= len(INBOX):
            return EmailTriageRlObservation(
                status="done",
                done=True,
                reward=0.0,
                previous_action_feedback="Episode already completed. Call reset().",
                metadata={"total_reward": self._score},
            )

        target = INBOX[self._cursor]
        reward, feedback_parts = self._score_action(action, target)
        self._score += reward
        self._last_feedback = " ".join(feedback_parts)
        self._cursor += 1
        done = self._cursor >= len(INBOX)
        status = "done" if done else "in_progress"
        return self._build_observation(status=status, reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    def _score_action(
        self, action: EmailTriageRlAction, target: EmailCase
    ) -> tuple[float, list[str]]:
        reward = 0.0
        feedback: list[str] = []

        if action.priority == target.priority:
            reward += 1.5
            feedback.append("Priority correct.")
        else:
            reward -= 0.5
            feedback.append(f"Priority should be {target.priority}.")

        if action.category == target.category:
            reward += 1.0
            feedback.append("Category correct.")
        else:
            reward -= 0.5
            feedback.append(f"Category should be {target.category}.")

        if action.response == target.response:
            reward += 1.0
            feedback.append("Response correct.")
        else:
            reward -= 0.5
            feedback.append(f"Best action is {target.response}.")

        if action.assign_to == target.assign_to:
            reward += 0.5
            feedback.append("Assignment correct.")
        else:
            reward -= 0.25
            feedback.append(f"Assign to {target.assign_to}.")

        if action.mark_phishing == target.phishing:
            reward += 1.0 if target.phishing else 0.25
            feedback.append("Phishing flag correct.")
        else:
            reward -= 1.0
            feedback.append(
                "This message should be flagged as phishing."
                if target.phishing
                else "This message is not phishing."
            )

        return reward, feedback

    def _build_observation(
        self, status: str, reward: float, done: bool
    ) -> EmailTriageRlObservation:
        if done:
            return EmailTriageRlObservation(
                status=status,
                email_id="",
                sender="",
                subject="",
                body_preview="",
                hours_since_received=0,
                customer_tier="",
                thread_length=0,
                remaining_emails=0,
                completed=len(INBOX),
                previous_action_feedback=self._last_feedback,
                done=True,
                reward=reward,
                metadata={
                    "reset_count": self._reset_count,
                    "total_reward": self._score,
                    "max_reward": len(INBOX) * 5.25,
                    "emails_processed": len(INBOX),
                },
            )

        current = INBOX[self._cursor]
        return EmailTriageRlObservation(
            status=status,
            email_id=current.email_id,
            sender=current.sender,
            subject=current.subject,
            body_preview=current.body_preview,
            hours_since_received=current.hours_since_received,
            customer_tier=current.customer_tier,
            thread_length=current.thread_length,
            remaining_emails=len(INBOX) - self._cursor - 1,
            completed=self._cursor,
            previous_action_feedback=self._last_feedback,
            done=False,
            reward=reward,
            metadata={
                "reset_count": self._reset_count,
                "total_reward": self._score,
                "available_labels": {
                    "priority": ["low", "medium", "high", "urgent"],
                    "category": [
                        "sales",
                        "support",
                        "billing",
                        "security",
                        "meeting",
                        "personal",
                    ],
                    "response": [
                        "archive",
                        "reply",
                        "forward",
                        "escalate",
                        "schedule",
                    ],
                    "assign_to": [
                        "none",
                        "sales",
                        "support",
                        "finance",
                        "security",
                        "exec_assistant",
                    ],
                },
            },
        )
