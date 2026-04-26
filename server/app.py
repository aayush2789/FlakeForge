"""FastAPI application for the FlakeForge environment server.

Mounts:
- OpenEnv core routes (/reset, /step) for RL training compatibility
- FlakeForge API (/api/*) for the web UI and external integrations
- Static homepage (GET /)
- CORS middleware for frontend development
"""

import sys
import os
import logging
from pathlib import Path

project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
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


from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute

logger = logging.getLogger("flakeforge")

# ── Create the base OpenEnv app ─────────────────────────────────────────────
app = create_app(FlakeForgeEnvironment, FlakeForgeAction, FlakeForgeObservation)

# Ensure our custom homepage wins over OpenEnv's default playground at "/".
app.router.routes = [
    route
    for route in app.router.routes
    if not (
        isinstance(route, APIRoute)
        and route.path == "/"
        and "GET" in route.methods
    )
]

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount the FlakeForge API router ─────────────────────────────────────────
try:
    from .api import router as api_router
except ImportError:
    try:
        from server.api import router as api_router
    except ImportError:
        from FlakeForge.server.api import router as api_router  # type: ignore

app.include_router(api_router)

# ── Homepage ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def homepage():
    """Serve the FlakeForge marketing / demo UI (Hugging Face App tab at `/`)."""
    html_path = Path(__file__).resolve().parents[1] / "templates" / "index.html"
    if not html_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Missing UI template: {html_path.as_posix()}",
        )
    return html_path.read_text(encoding="utf-8")


# ── Startup / shutdown events ────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    logger.info("FlakeForge server starting...")
    logger.info("  Project root:  %s", project_root)
    logger.info("  API docs:      http://localhost:8000/docs")
    logger.info("  Homepage:      http://localhost:8000/")
    logger.info("  Health check:  http://localhost:8000/api/health")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("FlakeForge server shutting down.")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution.

    Usage:
        uv run --project . server
        uv run --project . server --port 8001
        python -m FlakeForge.server.app
    """
    import argparse
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="FlakeForge Environment Server")
    parser.add_argument("--host", type=str, default=host, help="Host address")
    parser.add_argument("--port", type=int, default=port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args, _ = parser.parse_known_args()

    logger.info("Starting FlakeForge server on %s:%d", args.host, args.port)
    uvicorn.run(
        "server.app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
