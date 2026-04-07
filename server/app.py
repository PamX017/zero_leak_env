# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Zero-Leak Engineering Assistant.

Endpoints:
    - POST /reset:  Reset the environment (optionally with a task_id)
    - POST /step:   Execute an action
    - GET  /state:  Get current environment state
    - GET  /schema: Get action/observation schemas
    - WS   /ws:     WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with '\n    uv sync\n'"
    ) from e

# Import from local models.py (PYTHONPATH includes /app/env in Docker)
from models import ZeroLeakAction, ZeroLeakObservation
from .my_env_environment import ZeroLeakEnvironment


# Create the app with web interface and README integration
app = create_app(
    ZeroLeakEnvironment,
    ZeroLeakAction,
    ZeroLeakObservation,
    env_name="zero_leak_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution.

        uv run --project . server
        python -m my_env.server.app
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
