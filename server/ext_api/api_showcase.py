# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read-only demo + streaming endpoints for the FlakeForge Space UI (no Gradio)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from server.deep_flakiness import build_deep_observation_signals
except ImportError:  # pragma: no cover
    from ..deep_flakiness import build_deep_observation_signals

router = APIRouter(prefix="/api", tags=["showcase"])


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(repo: Optional[str]) -> Path:
    """Resolve a repo to scan. Defaults to the bundled demo; rejects traversal."""
    if repo:
        p = (Path(repo) if Path(repo).is_absolute() else _project_root() / repo).resolve()
        try:
            p.relative_to(_project_root().resolve())
        except ValueError as e:
            raise HTTPException(400, "repo_path must stay under the project root") from e
        if not p.is_dir():
            raise HTTPException(400, f"not a directory: {p}")
        return p
    env = os.environ.get("FF_REPO_PATH")
    if env:
        return (Path(env) if Path(env).is_absolute() else _project_root() / env).resolve()
    return (_project_root() / "test_repos" / "timing_race_minimal").resolve()


# Demo graph (matches the primary episode visualizer in templates/index.html)
CAUSAL_DEMO: Dict[str, Any] = {
    "id": "timing_race_counter",
    "label": "Demo: test → threads → non-atomic global",
    "nodes": [
        {"id": "test", "label": "test_counter.py", "x": 0.05, "y": 0.5},
        {"id": "runner", "label": "run_threads()", "x": 0.22, "y": 0.5},
        {"id": "thread", "label": "Thread ×N", "x": 0.4, "y": 0.25},
        {"id": "inc", "label": "increment()", "x": 0.58, "y": 0.5},
        {"id": "global", "label": "counter (global)", "x": 0.78, "y": 0.5},
        {"id": "race", "label": "RACE", "x": 0.78, "y": 0.2},
    ],
    "edges": [
        {"from": "test", "to": "runner"},
        {"from": "runner", "to": "thread"},
        {"from": "thread", "to": "inc"},
        {"from": "inc", "to": "global"},
        {"from": "global", "to": "race"},
    ],
}


@router.get("/showcase", response_class=JSONResponse)
def get_showcase() -> Dict[str, Any]:
    """Metadata for the Space UI: what the env exposes and which HTTP routes to call."""
    return {
        "name": "FlakeForge (OpenEnv)",
        "description": "POMDP for flaky test repair: reset → act → verify with verifiable reward.",
        "core_routes": {
            "GET /health": "Liveness (Docker / Space health).",
            "GET /docs": "OpenAPI (if enabled).",
            "POST /reset": "Start episode. Body: test_identifier, repo_path (optional), max_steps, num_runs.",
            "POST /step": "Agent step. Body: raw_response, think_text, patch_text, predicted_*, etc.",
        },
        "showcase_routes": {
            "GET /api/causal_graph/demo": "Static causal graph for the live canvas.",
            "GET /api/deep_flakiness/signals?repo=…": "AST deep-flakiness bundle for the current repo path.",
            "GET /api/stream/causal_graph": "SSE: incremental graph build (demo animation API).",
            "GET /api/stream/deep_flakiness?repo=…": "SSE: stream deep signal groups one after another.",
        },
    }


@router.get("/causal_graph/demo", response_class=JSONResponse)
def get_causal_demo() -> Dict[str, Any]:
    return CAUSAL_DEMO


@router.get("/deep_flakiness/signals", response_class=JSONResponse)
def get_deep_signals(repo: Optional[str] = Query(None, description="Repo path relative to project root or absolute under project")) -> JSONResponse:
    root = _project_root()
    try:
        path = _resolve_repo_path(repo)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"repo not found: {path}", "project_root": str(root)},
        )
    signals = build_deep_observation_signals(path)
    return JSONResponse(
        {
            "repo_path": str(path),
            "signals": signals,
        }
    )


def _causal_stream_events() -> List[Dict[str, Any]]:
    ev: List[Dict[str, Any]] = []
    for n in CAUSAL_DEMO["nodes"]:
        ev.append({"type": "node", "node": n})
    for i, e in enumerate(CAUSAL_DEMO["edges"]):
        ev.append({"type": "edge", "index": i, "edge": e})
    ev.append({"type": "done"})
    return ev


@router.get("/stream/causal_graph")
async def stream_causal_graph() -> StreamingResponse:
    """Server-Sent Events: one JSON object per `data:` line (demo of incremental graph)."""

    async def gen() -> AsyncIterator[str]:
        for payload in _causal_stream_events():
            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            await asyncio.sleep(0.12)

    return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8")


@router.get("/stream/deep_flakiness")
async def stream_deep_flakiness(
    repo: Optional[str] = Query(None, description="Repo to scan; defaults to FF_REPO_PATH or demo repo"),
) -> StreamingResponse:
    """Stream deep signals in digestible chunks (labels + a few example lines)."""

    try:
        path = _resolve_repo_path(repo)
    except HTTPException as e:
        async def err() -> AsyncIterator[str]:
            yield f"data: {json.dumps({'type': 'error', 'detail': e.detail})}\n\n"

        return StreamingResponse(err(), media_type="text/event-stream; charset=utf-8")

    if not path.exists():
        async def not_found() -> AsyncIterator[str]:
            yield f"data: {json.dumps({'type': 'error', 'detail': f'repo not found: {path}'})}\n\n"

        return StreamingResponse(not_found(), media_type="text/event-stream; charset=utf-8")

    data = build_deep_observation_signals(path)
    # Order matters for UX
    key_labels = {
        "module_cache_violations": "Module cache & mutable defaults",
        "fixture_scope_risks": "Fixture scope leak risks",
        "mock_residue_sites": "Monkeypatch cleanup gaps",
        "import_side_effect_files": "Import-time side effect signals",
        "async_contamination_alive": "Async/thread stragglers",
    }

    async def gen() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'start', 'repo_path': str(path)}, separators=(',', ':'))}\n\n"
        await asyncio.sleep(0.05)
        for key, label in key_labels.items():
            val = data.get(key)
            chunk: Dict[str, Any] = {"type": "signal", "key": key, "label": label}
            if key == "async_contamination_alive":
                chunk["value"] = bool(val)
            else:
                items = list(val) if isinstance(val, list) else []
                chunk["count"] = len(items)
                chunk["preview"] = items[:3]
            yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
            await asyncio.sleep(0.18)
        yield f"data: {json.dumps({'type': 'done', 'raw_keys': list(data.keys())}, separators=(',', ':'))}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8")
