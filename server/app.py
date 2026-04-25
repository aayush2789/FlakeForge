# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the FlakeForge environment server."""

import sys
import os
from pathlib import Path

# Add project root to sys.path to resolve 'agent' and 'models' as top-level modules
project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import FlakeForgeAction, FlakeForgeObservation
    from .FlakeForge_environment import FlakeForgeEnvironment
except ImportError:
    try:
        from FlakeForge.models import FlakeForgeAction, FlakeForgeObservation
        from FlakeForge.server.FlakeForge_environment import FlakeForgeEnvironment
    except ImportError:
        from models import FlakeForgeAction, FlakeForgeObservation  # type: ignore
        from server.FlakeForge_environment import FlakeForgeEnvironment  # type: ignore


app = create_app(FlakeForgeEnvironment, FlakeForgeAction, FlakeForgeObservation)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m FlakeForge.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn FlakeForge.server.app:app --workers 4
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=port)
    args, _ = parser.parse_known_args()

    uvicorn.run(app, host=host, port=args.port)


if __name__ == "__main__":
    main()
