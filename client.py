# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grievance Routing environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import GrievanceRoutingAction, GrievanceRoutingObservation
except ImportError:
    from models import GrievanceRoutingAction, GrievanceRoutingObservation


class GrievanceRoutingEnv(
    EnvClient[GrievanceRoutingAction, GrievanceRoutingObservation, State]
):
    """Client for the Grievance Routing environment."""

    def _step_payload(self, action: GrievanceRoutingAction) -> Dict:
        return {
            "department": action.department,
            "priority": action.priority,
            "action": action.action,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[GrievanceRoutingObservation]:
        observation_data = payload.get("observation", {})
        observation = GrievanceRoutingObservation(
            complaint_id=observation_data.get("complaint_id", 0),
            complaint_text=observation_data.get("complaint_text", ""),
            difficulty=observation_data.get("difficulty", ""),
            reward=payload.get("reward", observation_data.get("reward", 0.0)) or 0.0,
            done=payload.get("done", observation_data.get("done", False)),
            metadata=observation_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
