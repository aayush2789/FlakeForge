"""Pydantic request/response models for the FlakeForge REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Health & Info ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    uptime_seconds: float = 0.0
    environment_ready: bool = True


class ProjectInfo(BaseModel):
    name: str = "FlakeForge"
    version: str = "0.1.0"
    description: str = "RL Agent for Flaky Test Repair"
    root_cause_categories: List[str] = Field(default_factory=list)
    total_test_repos: int = 0
    max_steps_per_episode: int = 8
    reward_signals: List[str] = Field(default_factory=list)


# ── Repos ────────────────────────────────────────────────────────────────────

class RepoInfo(BaseModel):
    name: str
    path: str
    category: str = "unknown"
    test_identifier: str = ""
    has_manifest: bool = False
    manifest: Optional[Dict[str, Any]] = None


class RepoListResponse(BaseModel):
    repos: List[RepoInfo]
    total: int


# ── Episode ──────────────────────────────────────────────────────────────────

class EpisodeStartRequest(BaseModel):
    repo_path: str = ""
    test_identifier: str = ""
    max_steps: int = 8
    num_runs: int = 10


class EpisodeStartResponse(BaseModel):
    episode_id: str
    status: str = "initialized"
    observation: Dict[str, Any] = Field(default_factory=dict)
    baseline_pass_rate: float = 0.0
    env_type: str = "unknown"
    should_train: bool = True


class EpisodeStepRequest(BaseModel):
    raw_response: str = ""
    think_text: str = ""
    patch_text: str = ""
    predicted_category: str = "unknown"
    predicted_confidence: float = 0.5


class StepResult(BaseModel):
    step: int
    action: str = ""
    category: str = "unknown"
    confidence: float = 0.0
    reward: float = 0.0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    pass_rate_before: float = 0.0
    pass_rate_after: float = 0.0
    patch_applied: bool = False
    patch_files: List[str] = Field(default_factory=list)
    think_summary: str = ""
    done: bool = False
    done_reason: str = ""


class EpisodeStepResponse(BaseModel):
    step_result: StepResult
    observation: Dict[str, Any] = Field(default_factory=dict)


class RunEpisodeRequest(BaseModel):
    repo_path: str = ""
    test_identifier: str = ""
    max_steps: int = 8
    num_runs: int = 10
    backend: str = "nvidia"


class RunEpisodeResponse(BaseModel):
    episode_id: str
    status: str = "completed"
    steps: List[StepResult] = Field(default_factory=list)
    total_reward: float = 0.0
    final_pass_rate: float = 0.0
    baseline_pass_rate: float = 0.0
    done_reason: str = ""
    causal_graph: Optional[Dict[str, Any]] = None


class EpisodeStatusResponse(BaseModel):
    episode_id: str
    status: str = "idle"
    current_step: int = 0
    max_steps: int = 8
    pass_rate: float = 0.0
    total_reward: float = 0.0
    done: bool = False


# ── Challenge ────────────────────────────────────────────────────────────────

class ChallengeRequest(BaseModel):
    code: str
    test_code: str = ""
    preset: str = ""


class ChallengeAnalysis(BaseModel):
    detected_category: str = "unknown"
    confidence: float = 0.0
    root_cause_file: str = ""
    root_cause_function: str = ""
    causal_chain: List[str] = Field(default_factory=list)
    infrastructure_sensitive: bool = False
    suggested_fix: str = ""
    patch_diff: str = ""
    explanation: str = ""
    estimated_reward: float = 0.0


class ChallengeResponse(BaseModel):
    status: str = "analyzed"
    analysis: ChallengeAnalysis


# ── Training ─────────────────────────────────────────────────────────────────

class TrainingStats(BaseModel):
    total_episodes: int = 0
    avg_reward: float = 0.0
    fix_rate: float = 0.0
    avg_steps_to_fix: float = 0.0
    category_breakdown: Dict[str, int] = Field(default_factory=dict)
    reward_history: List[float] = Field(default_factory=list)
    baseline_history: List[float] = Field(default_factory=list)


class TrainingStatsResponse(BaseModel):
    stats: TrainingStats


# ── WebSocket messages ───────────────────────────────────────────────────────

class WSMessage(BaseModel):
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)
