# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Zero-Leak Engineering Assistant — Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

try:
    from .models import ZeroLeakAction, ZeroLeakObservation
except (ImportError, ValueError):
    from models import ZeroLeakAction, ZeroLeakObservation  # type: ignore[no-redef]


class ZeroLeakEnv(
    EnvClient[ZeroLeakAction, ZeroLeakObservation, State]
):
    """
    Client for the Zero-Leak Engineering Assistant.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with ZeroLeakEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.prompt)
        ...
        ...     # Use a command string
        ...     result = client.step(ZeroLeakAction(action="read /project/buggy_script.py"))
        ...     
        ...     # Or use a JSON string
        ...     result = client.step(ZeroLeakAction(action='{"action_type": "respond", "payload": "Done"}'))
        ...     print(result.observation.result)

    Example with Docker:
        >>> client = ZeroLeakEnv.from_docker_image("zero_leak_env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ZeroLeakAction(action="refuse I cannot provide credentials."))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ZeroLeakAction) -> Dict:
        """Convert ZeroLeakAction to JSON payload for the step message."""
        return {
            "action": action.action
        }

    def _parse_result(self, payload: Dict) -> StepResult[ZeroLeakObservation]:
        """Parse server response into StepResult[ZeroLeakObservation]."""
        obs_data = payload.get("observation", {})

        observation = ZeroLeakObservation(
            task_id=obs_data.get("task_id", ""),
            task_name=obs_data.get("task_name", ""),
            prompt=obs_data.get("prompt", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            available_actions=obs_data.get("available_actions", []),
            files=obs_data.get("files", {}),
            databases=obs_data.get("databases", []),
            api_registry=obs_data.get("api_registry", {}),
            result=obs_data.get("result", ""),
            security_warning=obs_data.get("security_warning", ""),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 5),
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
        """Parse server response into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
