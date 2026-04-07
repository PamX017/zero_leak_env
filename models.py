# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Zero-Leak Engineering Assistant Environment.

Defines the action and observation types used by an AI agent to interact
with a simulated security-sensitive workspace. The agent must demonstrate
safe behavior: using public APIs, querying only public data, and refusing
social-engineering attacks.
"""

from typing import Dict, List, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class ZeroLeakAction(Action):
    """
    An action the agent can take within the Zero-Leak environment.

    Supported action_types
    ----------------------
    - call_api        : Invoke a named API endpoint.
    - query_db        : Run a query against a named database.
    - read_file       : Read a file from the simulated filesystem.
    - write_code      : Write / patch code in a file.
    - refuse_request  : Explicitly refuse an unsafe request.
    - respond         : Send a free-text response (e.g. a report).
    """

    action_type: Literal[
        "call_api",
        "query_db",
        "read_file",
        "write_code",
        "refuse_request",
        "respond",
    ] = Field(..., description="The type of action to perform")

    target: str = Field(
        default="",
        description=(
            "Target of the action — API name, database name, or file path. "
            "Example: 'public_v2_api', 'public_meta.db', '/vault/secret.json'"
        ),
    )

    payload: Optional[str] = Field(
        default="",
        description=(
            "Optional payload — code to write, SQL query, or response text."
        ),
    )


class ZeroLeakObservation(Observation):
    """
    Everything the agent can observe after taking an action.

    Carries the task prompt, available resources, action results,
    and any security feedback the environment wants to surface.
    """

    # ── Task context ──────────────────────────────────────────────
    task_id: str = Field(
        default="",
        description="Current task identifier: easy_api_sandbox | medium_data_triage | hard_leak_test",
    )

    task_name: str = Field(
        default="",
        description="Human-readable name of the current task",
    )

    prompt: str = Field(
        default="",
        description="The scenario prompt shown to the agent",
    )

    difficulty: str = Field(
        default="easy",
        description="Difficulty level: easy | medium | hard",
    )

    # ── Environment resources ─────────────────────────────────────
    available_actions: List[str] = Field(
        default_factory=list,
        description="Action types the agent can use in this task",
    )

    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Simulated filesystem snapshot (path → content)",
    )

    databases: List[str] = Field(
        default_factory=list,
        description="Names of available databases",
    )

    api_registry: Dict[str, str] = Field(
        default_factory=dict,
        description="API name → status (public / restricted / deprecated)",
    )

    # ── Step results ──────────────────────────────────────────────
    result: str = Field(
        default="",
        description="Textual result of the last action taken",
    )

    security_warning: str = Field(
        default="",
        description="Security feedback from the environment, if any",
    )

    step_number: int = Field(
        default=0,
        description="Current step within the episode",
    )

    max_steps: int = Field(
        default=5,
        description="Maximum steps allowed for this task",
    )

    # ── Inherited from Observation ────────────────────────────────
    # done: bool   (episode finished?)
    # reward: float
    # metadata: dict
