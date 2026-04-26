"""FlakeForge REST API — all /api/* routes."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .api_models import (
    ChallengeRequest,
    ChallengeResponse,
    EpisodeStartRequest,
    EpisodeStartResponse,
    EpisodeStatusResponse,
    EpisodeStepRequest,
    EpisodeStepResponse,
    HealthResponse,
    ProjectInfo,
    RepoInfo,
    RepoListResponse,
    RunEpisodeRequest,
    RunEpisodeResponse,
    StepResult,
    TrainingStats,
    TrainingStatsResponse,
)
from .challenge_engine import analyze_challenge
from .episode_runner import get_episode_runner

logger = logging.getLogger(__name__)

_START_TIME = time.time()
_PROJECT_ROOT = Path(__file__).parents[1]

router = APIRouter(prefix="/api", tags=["api"])


# ── Health & Info ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(time.time() - _START_TIME, 1),
        environment_ready=True,
    )


@router.get("/info", response_model=ProjectInfo)
def project_info():
    try:
        from models import ROOT_CAUSE_TYPES
    except ImportError:
        from ..models import ROOT_CAUSE_TYPES

    repos = _scan_test_repos()
    return ProjectInfo(
        name="FlakeForge",
        version="0.1.0",
        description="RL Agent for Flaky Test Repair — GRPO + Causal Reasoning + Verifiable Rewards",
        root_cause_categories=list(ROOT_CAUSE_TYPES),
        total_test_repos=len(repos),
        max_steps_per_episode=int(os.environ.get("INFERENCE_MAX_STEPS", "8")),
        reward_signals=[
            "format", "compile", "stability", "causal_proximity",
            "failure_entropy", "anti_hack", "regression", "reasoning_consistency",
            "oracle_reasoning", "patch_validation", "noop_patch", "think_history",
            "terminal_bonus",
        ],
    )


# ── Repos ────────────────────────────────────────────────────────────────────

def _scan_test_repos() -> List[RepoInfo]:
    """Scan test_repos/ directory for available scenarios."""
    repos: List[RepoInfo] = []
    test_repos_dir = _PROJECT_ROOT / "test_repos"

    if not test_repos_dir.exists():
        return repos

    for category_dir in sorted(test_repos_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue

        if category_dir.name == "synthetic":
            for repo_dir in sorted(category_dir.iterdir()):
                if not repo_dir.is_dir():
                    continue
                repo = _build_repo_info(repo_dir, parent_category="synthetic")
                if repo:
                    repos.append(repo)
        else:
            repo = _build_repo_info(category_dir)
            if repo:
                repos.append(repo)

    return repos


def _build_repo_info(repo_dir: Path, parent_category: str = "") -> Optional[RepoInfo]:
    """Build RepoInfo from a repo directory."""
    manifest_path = repo_dir / "flake_manifest.json"
    manifest = None
    category = parent_category or "demo"
    test_id = ""

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            category = manifest.get("flake_category", category)
            test_id = manifest.get("test_identifier", "")
        except Exception:
            pass

    if not test_id:
        test_files = list(repo_dir.glob("tests/test_*.py")) + list(repo_dir.glob("test_*.py"))
        if test_files:
            rel = test_files[0].relative_to(repo_dir)
            test_id = str(rel).replace("\\", "/")

    return RepoInfo(
        name=repo_dir.name,
        path=str(repo_dir.relative_to(_PROJECT_ROOT)).replace("\\", "/"),
        category=category,
        test_identifier=test_id,
        has_manifest=manifest is not None,
        manifest=manifest,
    )


@router.get("/repos", response_model=RepoListResponse)
def list_repos(category: str = ""):
    repos = _scan_test_repos()
    if category:
        repos = [r for r in repos if r.category == category]
    return RepoListResponse(repos=repos, total=len(repos))


@router.get("/repos/{repo_name}")
def get_repo(repo_name: str):
    repos = _scan_test_repos()
    for r in repos:
        if r.name == repo_name:
            return r
    raise HTTPException(404, f"Repo '{repo_name}' not found")


@router.get("/repos/{repo_name}/manifest")
def get_repo_manifest(repo_name: str):
    repos = _scan_test_repos()
    for r in repos:
        if r.name == repo_name:
            if r.manifest:
                return r.manifest
            raise HTTPException(404, f"Repo '{repo_name}' has no manifest")
    raise HTTPException(404, f"Repo '{repo_name}' not found")


@router.get("/repos/{repo_name}/source")
def get_repo_source(repo_name: str, file: str = "source.py"):
    """Read a source file from a test repo."""
    repos = _scan_test_repos()
    for r in repos:
        if r.name == repo_name:
            source_path = _PROJECT_ROOT / r.path / file
            if source_path.exists() and source_path.is_file():
                return {"file": file, "content": source_path.read_text(encoding="utf-8", errors="ignore")[:10000]}
            raise HTTPException(404, f"File '{file}' not found in repo '{repo_name}'")
    raise HTTPException(404, f"Repo '{repo_name}' not found")


# ── Episode Management ───────────────────────────────────────────────────────

_env_instances: Dict[str, Any] = {}


@router.post("/episode/start", response_model=EpisodeStartResponse)
def start_episode(req: EpisodeStartRequest):
    """Initialize a new episode without running steps. Returns observation."""
    import uuid
    from .FlakeForge_environment import FlakeForgeEnvironment
    from .docker_runner import DockerTestRunner

    repo_path = req.repo_path or os.environ.get(
        "FF_REPO_PATH",
        str(_PROJECT_ROOT / "test_repos" / "timing_race_minimal"),
    )
    test_id = req.test_identifier or os.environ.get(
        "FF_TEST_ID", "tests/test_flaky.py::test_fetch_should_complete"
    )

    if not repo_path.startswith("/") and not repo_path[1:2] == ":":
        repo_path = str(_PROJECT_ROOT / repo_path)

    runner = DockerTestRunner(repo_path)
    env = FlakeForgeEnvironment(
        repo_path=repo_path,
        test_identifier=test_id,
        max_steps=req.max_steps,
        num_runs=req.num_runs,
        runner=runner,
    )

    episode_id = str(uuid.uuid4())[:8]
    observation = env.reset(episode_id=episode_id)
    _env_instances[episode_id] = env

    obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else {}

    return EpisodeStartResponse(
        episode_id=episode_id,
        status="initialized",
        observation=obs_dict,
        baseline_pass_rate=observation.baseline_pass_rate,
        env_type=observation.env_type,
        should_train=observation.should_train,
    )


@router.post("/episode/{episode_id}/step", response_model=EpisodeStepResponse)
def step_episode(episode_id: str, req: EpisodeStepRequest):
    """Take one step in an active episode."""
    env = _env_instances.get(episode_id)
    if env is None:
        raise HTTPException(404, f"Episode '{episode_id}' not found or expired")

    try:
        from models import FlakeForgeAction
    except ImportError:
        from ..models import FlakeForgeAction

    action = FlakeForgeAction(
        raw_response=req.raw_response,
        think_text=req.think_text,
        patch_text=req.patch_text,
        predicted_category=req.predicted_category,
        predicted_confidence=req.predicted_confidence,
    )

    observation = env.step(action)
    state = env.state

    step_result = StepResult(
        step=state.step_count,
        category=req.predicted_category,
        confidence=req.predicted_confidence,
        reward=round(observation.reward, 4),
        reward_breakdown=observation.reward_breakdown,
        pass_rate_before=state.baseline_pass_rate,
        pass_rate_after=round(observation.current_pass_rate, 4),
        patch_applied=bool(observation.patch_result.get("success")),
        patch_files=observation.patch_result.get("files_modified", []),
        think_summary=req.think_text[:200],
        done=observation.done,
        done_reason=observation.done_reason,
    )

    obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else {}

    if observation.done:
        _env_instances.pop(episode_id, None)

    return EpisodeStepResponse(step_result=step_result, observation=obs_dict)


@router.post("/episode/run", response_model=RunEpisodeResponse)
def run_full_episode(req: RunEpisodeRequest):
    """Run a complete episode end-to-end (agent loop). Blocking."""
    runner = get_episode_runner()

    repo_path = req.repo_path
    if repo_path and not repo_path.startswith("/") and not (len(repo_path) > 1 and repo_path[1] == ":"):
        repo_path = str(_PROJECT_ROOT / repo_path)

    return runner.run_episode_sync(
        repo_path=repo_path,
        test_identifier=req.test_identifier,
        max_steps=req.max_steps,
        num_runs=req.num_runs,
    )


@router.post("/episode/run-async")
async def run_episode_background(req: RunEpisodeRequest, background_tasks: BackgroundTasks):
    """Start an episode in the background. Returns immediately with episode_id."""
    import uuid
    episode_id = str(uuid.uuid4())[:8]
    runner = get_episode_runner()

    repo_path = req.repo_path
    if repo_path and not repo_path.startswith("/") and not (len(repo_path) > 1 and repo_path[1] == ":"):
        repo_path = str(_PROJECT_ROOT / repo_path)

    runner._active_episodes[episode_id] = {
        "episode_id": episode_id,
        "status": "queued",
        "current_step": 0,
        "max_steps": req.max_steps,
        "pass_rate": 0.0,
        "total_reward": 0.0,
        "done": False,
    }

    background_tasks.add_task(
        runner.run_episode_sync,
        repo_path=repo_path,
        test_identifier=req.test_identifier,
        max_steps=req.max_steps,
        num_runs=req.num_runs,
    )

    return {"episode_id": episode_id, "status": "queued"}


@router.get("/episode/{episode_id}/status", response_model=EpisodeStatusResponse)
def episode_status(episode_id: str):
    runner = get_episode_runner()
    status = runner.get_status(episode_id)
    if status is None:
        if episode_id in _env_instances:
            env = _env_instances[episode_id]
            state = env.state
            return EpisodeStatusResponse(
                episode_id=episode_id,
                status="active",
                current_step=state.step_count,
                max_steps=8,
                pass_rate=state.current_pass_rate,
                total_reward=0.0,
                done=state.done,
            )
        raise HTTPException(404, f"Episode '{episode_id}' not found")
    return EpisodeStatusResponse(**status)


@router.get("/episode/{episode_id}/result", response_model=RunEpisodeResponse)
def episode_result(episode_id: str):
    runner = get_episode_runner()
    result = runner.get_result(episode_id)
    if result is None:
        raise HTTPException(404, f"No completed result for episode '{episode_id}'")
    return result


# ── Challenge ────────────────────────────────────────────────────────────────

@router.post("/challenge/analyze", response_model=ChallengeResponse)
def challenge_analyze(req: ChallengeRequest):
    """Analyze a user-submitted flaky test."""
    analysis = analyze_challenge(
        code=req.code,
        test_code=req.test_code,
        preset=req.preset,
    )
    return ChallengeResponse(status="analyzed", analysis=analysis)


# ── Training Stats ───────────────────────────────────────────────────────────

@router.get("/training/stats", response_model=TrainingStatsResponse)
def training_stats():
    """Return training statistics. Uses completed episodes from the runner."""
    runner = get_episode_runner()
    completed = runner._completed_episodes

    if not completed:
        return TrainingStatsResponse(stats=TrainingStats())

    total = len(completed)
    rewards = [ep.total_reward for ep in completed.values()]
    avg_reward = round(sum(rewards) / max(total, 1), 4)

    fixes = sum(1 for ep in completed.values() if ep.final_pass_rate >= 1.0)
    fix_rate = round(fixes / max(total, 1), 4)

    steps_to_fix = [
        len(ep.steps) for ep in completed.values() if ep.final_pass_rate >= 1.0
    ]
    avg_steps = round(sum(steps_to_fix) / max(len(steps_to_fix), 1), 2) if steps_to_fix else 0.0

    cat_breakdown: Dict[str, int] = {}
    for ep in completed.values():
        for step in ep.steps:
            cat_breakdown[step.category] = cat_breakdown.get(step.category, 0) + 1

    return TrainingStatsResponse(
        stats=TrainingStats(
            total_episodes=total,
            avg_reward=avg_reward,
            fix_rate=fix_rate,
            avg_steps_to_fix=avg_steps,
            category_breakdown=cat_breakdown,
            reward_history=rewards[-50:],
        )
    )


# ── WebSocket for live episode streaming ─────────────────────────────────────

@router.websocket("/ws/episode")
async def websocket_episode(ws: WebSocket):
    """Stream episode execution over WebSocket.

    Send a JSON message to start: {"repo_path": "...", "test_identifier": "...", "max_steps": 8}
    Receives step-by-step results as JSON messages.
    """
    await ws.accept()
    try:
        config = await ws.receive_json()
        runner = get_episode_runner()

        repo_path = config.get("repo_path", "")
        if repo_path and not repo_path.startswith("/") and not (len(repo_path) > 1 and repo_path[1] == ":"):
            repo_path = str(_PROJECT_ROOT / repo_path)

        await ws.send_json({"type": "started", "data": {"status": "initializing"}})

        async for msg in runner.stream_episode(
            repo_path=repo_path,
            test_identifier=config.get("test_identifier", ""),
            max_steps=config.get("max_steps", 8),
            num_runs=config.get("num_runs", 10),
        ):
            await ws.send_json(msg)

        await ws.send_json({"type": "done", "data": {}})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await ws.send_json({"type": "error", "data": {"message": str(exc)}})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
