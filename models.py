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
    "GATHER_EVIDENCE",
    "ADD_TIMING_GUARD",
    "ADD_SYNCHRONIZATION",
    "MOCK_DEPENDENCY",
    "RESET_STATE",
    "ADD_RETRY",
    "REVERT_LAST_PATCH",
]

ROOT_CAUSE_TYPES = Literal[
    "TIMING_RACE",
    "SHARED_STATE",
    "EXTERNAL_DEPENDENCY",
    "ORDER_DEPENDENCY",
    "RESOURCE_LEAK",
    "NONDETERMINISM",
]


class FlakeForgeAction(Action):
    """Single-step action sent by the agent."""

    action_type: ACTION_TYPES
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_action_payload(self) -> "FlakeForgeAction":
        params = self.parameters or {}

        if self.action_type == "GATHER_EVIDENCE":
            allowed = {"injection_target"}
            self._check_only_allowed(params, allowed)
            if params.get("injection_target") not in {"test", "source"}:
                raise ValueError("GATHER_EVIDENCE.injection_target must be 'test' or 'source'")

        elif self.action_type == "ADD_TIMING_GUARD":
            allowed = {"delay_ms"}
            self._check_only_allowed(params, allowed)
            delay = params.get("delay_ms")
            if delay not in {50, 100, 200, 500}:
                raise ValueError("ADD_TIMING_GUARD.delay_ms must be one of 50, 100, 200, 500")

        elif self.action_type == "ADD_SYNCHRONIZATION":
            allowed = {"primitive"}
            self._check_only_allowed(params, allowed)
            if params.get("primitive") not in {"lock", "event", "barrier", "semaphore"}:
                raise ValueError(
                    "ADD_SYNCHRONIZATION.primitive must be one of lock, event, barrier, semaphore"
                )

        elif self.action_type == "MOCK_DEPENDENCY":
            allowed = {"target"}
            self._check_only_allowed(params, allowed)
            target = params.get("target")
            if not isinstance(target, str) or "." not in target.strip("."):
                raise ValueError("MOCK_DEPENDENCY.target must be a dotted import path (e.g. requests.get)")

        elif self.action_type == "RESET_STATE":
            allowed = {"scope"}
            self._check_only_allowed(params, allowed)
            if params.get("scope") not in {"function", "class", "module"}:
                raise ValueError("RESET_STATE.scope must be one of function, class, module")

        elif self.action_type == "ADD_RETRY":
            allowed = {"max_attempts", "backoff_ms"}
            self._check_only_allowed(params, allowed)
            if params.get("max_attempts") not in {2, 3, 5}:
                raise ValueError("ADD_RETRY.max_attempts must be one of 2, 3, 5")
            if params.get("backoff_ms") not in {100, 500}:
                raise ValueError("ADD_RETRY.backoff_ms must be one of 100, 500")

        elif self.action_type == "REVERT_LAST_PATCH":
            if params:
                raise ValueError("REVERT_LAST_PATCH.parameters must be empty")

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

    def __post_init__(self) -> None:
        if self.error_message:
            self.error_message = self.error_message[:200]


@dataclass
class Hypothesis:
    root_cause_category: ROOT_CAUSE_TYPES
    confidence: float
    evidence: List[str]
    suggested_action: Optional[str] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Hypothesis.confidence must be between 0 and 1")
        if len(self.evidence) > 5:
            raise ValueError("Hypothesis.evidence can contain at most 5 entries")


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


# Backward compatible aliases for template-generated names.
FlakeforgeAction = FlakeForgeAction
FlakeforgeObservation = FlakeForgeObservation
