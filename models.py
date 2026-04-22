# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the FlakeForge Gym environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

ACTION_TYPES = Literal[
    # ── V1 Actions ────────────────────────────────────────────────────
    "GATHER_EVIDENCE",
    "ADD_TIMING_GUARD",
    "ADD_SYNCHRONIZATION",
    "MOCK_DEPENDENCY",
    "RESET_STATE",
    "ADD_RETRY",
    "REVERT_LAST_PATCH",
    "SEED_RANDOMNESS",
    # ── V2 Deep-Surgery Actions ───────────────────────────────────────
    "DIAGNOSE_BOUNDARY",       # Pull source across a detected service/DB boundary
    "REFACTOR_CONCURRENCY",    # Swap threading primitive for a safer alternative
    "ISOLATE_BOUNDARY",        # Wrap external call in circuit-breaker / timeout
    "EXTRACT_ASYNC_SCOPE",     # Move sync code out of async context or vice versa
    "HARDEN_IDEMPOTENCY",      # Add idempotency guard to a state-mutating function
    "CHAOS_PROBE",             # Run under chaos profile to gather evidence
    # ── Finalist alias actions ───────────────────────────────────────
    "detect_flakiness",
    "analyze_logs",
    "add_sleep",
    "add_lock",
    "mock_dependency",
    "isolate_state",
    "reorder_execution",
    "retry_test",
]

ROOT_CAUSE_TYPES = Literal[
    "TIMING_RACE",
    "SHARED_STATE",
    "EXTERNAL_DEPENDENCY",
    "ORDER_DEPENDENCY",
    "RESOURCE_LEAK",
    "NONDETERMINISM",
    # ── V2 deep categories ────────────────────────────────────────────
    "ASYNC_DEADLOCK",           # threading.Lock / blocking call inside async event loop
    "INFRASTRUCTURE_SENSITIVE", # Only reproducible under CPU/memory/network pressure
    # ── Finalist normalized categories ───────────────────────────────
    "timing",
    "race",
    "shared_state",
    "network",
    "order",
    "unknown",
]


class FlakeForgeAction(Action):
    """Single-step action sent by the agent."""

    action_type: ACTION_TYPES
    parameters: Dict[str, Any] = Field(default_factory=dict)
    hypothesis: Optional["HypothesisPayload"] = None
    judge_feedback: Optional["JudgeFeedbackPayload"] = None
    # Improvement 1: agent's predicted outcome before env evaluates the action.
    # Penalised by reward shaping when wrong — accelerates credit assignment.
    predicted_pass_rate_after: Optional[float] = Field(
        default=None,
        description="Agent's predicted pass rate after this action (0.0-1.0).",
    )
    justification: Optional[str] = None
    expected_outcome: Optional[str] = None
    risk_assessment: Optional[str] = None
    fallback_plan: Optional[str] = None

    @field_validator("predicted_pass_rate_after", mode="before")
    @classmethod
    def _clamp_predicted_pass_rate(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode="after")
    def _validate_action_payload(self) -> "FlakeForgeAction":
        alias_action_map = {
            "detect_flakiness": "GATHER_EVIDENCE",
            "analyze_logs": "GATHER_EVIDENCE",
            "add_sleep": "ADD_TIMING_GUARD",
            "add_lock": "ADD_SYNCHRONIZATION",
            "mock_dependency": "MOCK_DEPENDENCY",
            "isolate_state": "RESET_STATE",
            "reorder_execution": "RESET_STATE",
            "retry_test": "ADD_RETRY",
        }
        canonical_action_type = alias_action_map.get(self.action_type, self.action_type)

        params = self.parameters or {}

        if canonical_action_type == "GATHER_EVIDENCE":
            allowed = {"injection_target"}
            if self.action_type in {"detect_flakiness", "analyze_logs"} and "injection_target" not in params:
                params["injection_target"] = "test"
            self._check_only_allowed(params, allowed)
            if params.get("injection_target") not in {"test", "source"}:
                raise ValueError("GATHER_EVIDENCE.injection_target must be 'test' or 'source'")

        elif canonical_action_type == "ADD_TIMING_GUARD":
            allowed = {"delay_ms"}
            if self.action_type == "add_sleep" and "delay_ms" not in params:
                params["delay_ms"] = 100
            self._check_only_allowed(params, allowed)
            delay = params.get("delay_ms")
            if delay not in {50, 100, 200, 500}:
                raise ValueError("ADD_TIMING_GUARD.delay_ms must be one of 50, 100, 200, 500")

        elif canonical_action_type == "ADD_SYNCHRONIZATION":
            allowed = {"primitive"}
            if self.action_type == "add_lock" and "primitive" not in params:
                params["primitive"] = "lock"
            self._check_only_allowed(params, allowed)
            if params.get("primitive") not in {"lock", "event", "barrier", "semaphore"}:
                raise ValueError(
                    "ADD_SYNCHRONIZATION.primitive must be one of lock, event, barrier, semaphore"
                )

        elif canonical_action_type == "MOCK_DEPENDENCY":
            allowed = {"target"}
            if self.action_type == "mock_dependency" and "target" not in params:
                params["target"] = "requests.get"
            self._check_only_allowed(params, allowed)
            target = params.get("target")
            if not isinstance(target, str) or "." not in target.strip("."):
                raise ValueError("MOCK_DEPENDENCY.target must be a dotted import path (e.g. requests.get)")

        elif canonical_action_type == "RESET_STATE":
            allowed = {"scope"}
            if self.action_type in {"isolate_state", "reorder_execution"} and "scope" not in params:
                params["scope"] = "function"
            self._check_only_allowed(params, allowed)
            if params.get("scope") not in {"function", "class", "module"}:
                raise ValueError("RESET_STATE.scope must be one of function, class, module")

        elif canonical_action_type == "ADD_RETRY":
            allowed = {"max_attempts", "backoff_ms"}
            if self.action_type == "retry_test":
                params.setdefault("max_attempts", 2)
                params.setdefault("backoff_ms", 100)
            self._check_only_allowed(params, allowed)
            if params.get("max_attempts") not in {2, 3, 5}:
                raise ValueError("ADD_RETRY.max_attempts must be one of 2, 3, 5")
            if params.get("backoff_ms") not in {100, 500}:
                raise ValueError("ADD_RETRY.backoff_ms must be one of 100, 500")

        elif canonical_action_type == "REVERT_LAST_PATCH":
            if params:
                raise ValueError("REVERT_LAST_PATCH.parameters must be empty")

        elif canonical_action_type == "SEED_RANDOMNESS":
            allowed = {"library"}
            self._check_only_allowed(params, allowed)
            if params.get("library") not in {"random", "numpy", "both"}:
                raise ValueError("SEED_RANDOMNESS.library must be one of random, numpy, both")

        # ── V2 Deep-Action Validators ──────────────────────────────────
        elif canonical_action_type == "DIAGNOSE_BOUNDARY":
            allowed = {"boundary_node"}
            self._check_only_allowed(params, allowed)
            if not isinstance(params.get("boundary_node"), str):
                raise ValueError("DIAGNOSE_BOUNDARY.boundary_node must be a dotted path string")

        elif canonical_action_type == "REFACTOR_CONCURRENCY":
            allowed = {"from_primitive", "to_primitive", "target_function"}
            self._check_only_allowed(params, allowed)
            valid_from = {"threading.Lock", "threading.RLock", "bare", "asyncio.Lock"}
            valid_to = {"asyncio.Lock", "threading.RLock", "asyncio.Semaphore", "asyncio.Event"}
            if params.get("from_primitive") not in valid_from:
                raise ValueError(f"REFACTOR_CONCURRENCY.from_primitive must be one of {sorted(valid_from)}")
            if params.get("to_primitive") not in valid_to:
                raise ValueError(f"REFACTOR_CONCURRENCY.to_primitive must be one of {sorted(valid_to)}")
            if not isinstance(params.get("target_function"), str):
                raise ValueError("REFACTOR_CONCURRENCY.target_function must be a string")

        elif canonical_action_type == "ISOLATE_BOUNDARY":
            allowed = {"boundary_call", "pattern"}
            self._check_only_allowed(params, allowed)
            valid_patterns = {"circuit_breaker", "timeout_wrapper", "bulkhead"}
            if not isinstance(params.get("boundary_call"), str):
                raise ValueError("ISOLATE_BOUNDARY.boundary_call must be a dotted path string")
            if params.get("pattern") not in valid_patterns:
                raise ValueError(f"ISOLATE_BOUNDARY.pattern must be one of {sorted(valid_patterns)}")

        elif canonical_action_type == "EXTRACT_ASYNC_SCOPE":
            allowed = {"target_function", "direction"}
            self._check_only_allowed(params, allowed)
            valid_directions = {"make_async", "make_sync", "offload_to_thread"}
            if not isinstance(params.get("target_function"), str):
                raise ValueError("EXTRACT_ASYNC_SCOPE.target_function must be a string")
            if params.get("direction") not in valid_directions:
                raise ValueError(f"EXTRACT_ASYNC_SCOPE.direction must be one of {valid_directions}")

        elif canonical_action_type == "HARDEN_IDEMPOTENCY":
            allowed = {"state_target", "key_strategy"}
            self._check_only_allowed(params, allowed)
            valid_strategies = {"uuid", "content_hash", "composite_key"}
            if not isinstance(params.get("state_target"), str):
                raise ValueError("HARDEN_IDEMPOTENCY.state_target must be a string")
            if params.get("key_strategy") not in valid_strategies:
                raise ValueError(f"HARDEN_IDEMPOTENCY.key_strategy must be one of {valid_strategies}")

        elif canonical_action_type == "CHAOS_PROBE":
            allowed = {"profile", "n_runs"}
            self._check_only_allowed(params, allowed)
            valid_profiles = {"cpu", "mem", "net", "compound"}
            if params.get("profile") not in valid_profiles:
                raise ValueError(f"CHAOS_PROBE.profile must be one of {valid_profiles}")
            n_runs = params.get("n_runs", 10)
            if not isinstance(n_runs, int) or not (1 <= n_runs <= 20):
                raise ValueError("CHAOS_PROBE.n_runs must be an integer between 1 and 20")

        return self

    @staticmethod
    def _check_only_allowed(params: Dict[str, Any], allowed: set[str]) -> None:
        unexpected = set(params.keys()) - allowed
        missing = allowed - set(params.keys())
        if unexpected:
            raise ValueError(f"Unexpected parameter keys: {sorted(unexpected)}")
        if missing:
            raise ValueError(f"Missing required parameter keys: {sorted(missing)}")


@dataclass
class RunRecord:
    passed: bool
    duration_ms: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stderr_excerpt: Optional[str] = None

    def __post_init__(self) -> None:
        if self.error_message:
            self.error_message = self.error_message[:200]
        if self.stderr_excerpt:
            self.stderr_excerpt = self.stderr_excerpt[:500]


@dataclass
class FailurePattern:
    pass_rate: float
    most_common_error: Optional[str]
    error_distribution: Dict[str, int]
    duration_mean: float
    duration_std: float
    flakiness_score: float


@dataclass
class Hypothesis:
    root_cause_category: ROOT_CAUSE_TYPES
    confidence: float
    evidence: List[str]
    suggested_action: Optional[str] = None
    reasoning_steps: List[str] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    next_best_action: Optional[str] = None
    predicted_effect: Optional[str] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Hypothesis.confidence must be between 0 and 1")
        if len(self.evidence) > 5:
            raise ValueError("Hypothesis.evidence can contain at most 5 entries")


class HypothesisPayload(BaseModel):
    root_cause_category: ROOT_CAUSE_TYPES
    confidence: float
    evidence: List[str] = Field(default_factory=list)
    suggested_action: Optional[str] = None
    reasoning_steps: List[str] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    next_best_action: Optional[str] = None
    predicted_effect: Optional[str] = None


class JudgeFeedbackPayload(BaseModel):
    judge_hypothesis_score: int = 0
    judge_patch_score: int = 0
    # Improvement 3 (Reflexion): verbal critique fed back into Fixer's next prompt.
    critique: str = ""
    prediction_error: str = ""

    @field_validator("judge_hypothesis_score", "judge_patch_score")
    @classmethod
    def _clamp_scores(cls, value: int) -> int:
        return max(0, min(5, int(value)))


@dataclass
class PatchRecord:
    action_taken: str
    target_file: str
    lines_changed: int
    pass_rate_after: float
    judge_patch_score: float


@dataclass
class ASTSummary:
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[str]
    global_vars: List[str]
    threading_primitives: List[str]
    external_calls: List[str]


class FlakeForgeObservation(Observation):
    episode_id: str
    test_identifier: str
    step: int
    steps_remaining: int
    test_function_source: str
    source_under_test: str
    relevant_imports: List[str] = Field(default_factory=list)
    file_tree: List[str] = Field(default_factory=list)
    async_markers: List[str] = Field(default_factory=list)
    run_history: List[RunRecord] = Field(default_factory=list)
    current_hypothesis: Optional[Hypothesis] = None
    patches_applied: List[PatchRecord] = Field(default_factory=list)
    log_snippets: List[str] = Field(default_factory=list)
    current_pass_rate: float = 0.0
    baseline_pass_rate: float = 0.0
    total_diff_lines: int = 0
    reward: float = 0.0
    done: bool = False
    # ── V2 Fields ─────────────────────────────────────────────────────
    causal_graph: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cross-repo call graph: nodes, edges, boundary warnings (Pillar 1)",
    )
    chaos_pass_rate: Optional[float] = Field(
        default=None,
        description="Pass rate under chaos conditions (Pillar 2). None = not yet probed.",
    )
    chaos_baseline_pass_rate: Optional[float] = Field(
        default=None,
        description="Baseline pass rate under the same chaos profile.",
    )
    perf_sentinel_status: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Performance regression sentinel result (Pillar 4): {regression, median_ratio}",
    )
    infrastructure_sensitive: bool = Field(
        default=False,
        description="True when chaos baseline reveals infrastructure-dependent flakiness.",
    )
    # Improvement 4: timing statistics from baseline runs, used to boost
    # initial hypothesis confidence and reduce wasted GATHER_EVIDENCE steps.
    duration_fingerprint: Optional[Dict[str, float]] = Field(
        default=None,
        description="{mean_ms, std_ms, cv, flakiness_score} computed from baseline runs.",
    )
    # Improvement 5: competing hypothesis when primary confidence < 0.5.
    secondary_hypothesis: Optional[Hypothesis] = Field(
        default=None,
        description="Runner-up root-cause hypothesis for low-confidence situations.",
    )
    last_actions: List[str] = Field(default_factory=list)
    last_outcomes: List[Dict[str, Any]] = Field(default_factory=list)
    prediction_error_history: List[float] = Field(default_factory=list)
    failure_pattern_summary: Optional[Dict[str, Any]] = None
    causal_hints: List[str] = Field(default_factory=list)
    reflection: Optional[Dict[str, Any]] = None

    @field_validator("run_history")
    @classmethod
    def _limit_history(cls, value: List[RunRecord]) -> List[RunRecord]:
        return value[-10:]


class FlakeForgeState(State):
    episode_id: str
    step_count: int
    done: bool = False
    current_pass_rate: float = 0.0
    baseline_pass_rate: float = 0.0
    regression_detected: bool = False
    judge_scores: List[Dict[str, Any]] = Field(default_factory=list)
    # ── V2 Fields ─────────────────────────────────────────────────────
    chaos_pass_rate: Optional[float] = None
    chaos_baseline_pass_rate: Optional[float] = None
    perf_regression_detected: bool = False
    perf_median_ratio: float = 1.0
    infrastructure_sensitive: bool = False
    prediction_error_history: List[float] = Field(default_factory=list)


# Backward compatible aliases for template-generated names.
FlakeforgeAction = FlakeForgeAction
FlakeforgeObservation = FlakeForgeObservation
