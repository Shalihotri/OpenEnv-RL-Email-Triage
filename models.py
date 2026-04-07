# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the email triage RL environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


PriorityLabel = Literal["low", "medium", "high", "urgent"]
CategoryLabel = Literal[
    "sales", "support", "billing", "security", "meeting", "personal"
]
ResponseLabel = Literal["archive", "reply", "forward", "escalate", "schedule"]
AssigneeLabel = Literal[
    "none", "sales", "support", "finance", "security", "exec_assistant"
]


class EmailTriageRlAction(Action):
    """A single triage decision for the current email."""

    priority: PriorityLabel = Field(..., description="Chosen priority.")
    category: CategoryLabel = Field(..., description="Chosen routing category.")
    response: ResponseLabel = Field(..., description="Best next action.")
    assign_to: AssigneeLabel = Field(
        default="none", description="Mailbox or team to route the email to."
    )
    mark_phishing: bool = Field(
        default=False, description="Whether the email should be flagged as phishing."
    )


class EmailTriageRlObservation(Observation):
    """Current inbox state and feedback for the latest decision."""

    status: str = Field(default="ready")
    email_id: str = Field(default="")
    sender: str = Field(default="")
    subject: str = Field(default="")
    body_preview: str = Field(default="")
    hours_since_received: int = Field(default=0, ge=0)
    customer_tier: str = Field(default="")
    thread_length: int = Field(default=0, ge=0)
    remaining_emails: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    previous_action_feedback: str = Field(default="")
