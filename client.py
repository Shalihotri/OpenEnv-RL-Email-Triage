# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email triage RL environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EmailTriageRlAction, EmailTriageRlObservation


class EmailTriageRlEnv(EnvClient[EmailTriageRlAction, EmailTriageRlObservation, State]):
    """Client for the email triage RL environment."""

    def _step_payload(self, action: EmailTriageRlAction) -> Dict:
        return {
            "priority": action.priority,
            "category": action.category,
            "response": action.response,
            "assign_to": action.assign_to,
            "mark_phishing": action.mark_phishing,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EmailTriageRlObservation]:
        obs_data = payload.get("observation", {})
        observation = EmailTriageRlObservation(
            status=obs_data.get("status", "ready"),
            email_id=obs_data.get("email_id", ""),
            sender=obs_data.get("sender", ""),
            subject=obs_data.get("subject", ""),
            body_preview=obs_data.get("body_preview", ""),
            hours_since_received=obs_data.get("hours_since_received", 0),
            customer_tier=obs_data.get("customer_tier", ""),
            thread_length=obs_data.get("thread_length", 0),
            remaining_emails=obs_data.get("remaining_emails", 0),
            completed=obs_data.get("completed", 0),
            previous_action_feedback=obs_data.get("previous_action_feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
