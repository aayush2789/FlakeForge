"""FlakeForge data models — actions, observations, state, and reward breakdown."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator
from pydantic.dataclasses import dataclass


ROOT_CAUSE_TYPES = [
    "async_wait",
    "concurrency",
    "test_order_dependency",
    "resource_leak",
    "shared_state",
    "network",
    "platform_dependency",
    "nondeterminism",
    "import_side_effect",
    "module_cache_pollution",
    "fixture_scope_leak",
    "mock_residue",
    "unknown",
]

RELATED_CATEGORIES = {
    "async_wait": {"concurrency", "platform_dependency"},
    "concurrency": {"async_wait", "shared_state"},
    "test_order_dependency": {"shared_state", "resource_leak"},
    "resource_leak": {"shared_state", "test_order_dependency"},
    "shared_state": {"test_order_dependency", "resource_leak", "module_cache_pollution"},
    "network": {"platform_dependency", "async_wait"},
    "platform_dependency": {"network", "async_wait"},
    "nondeterminism": {"concurrency", "async_wait"},
    "import_side_effect": {"module_cache_pollution", "shared_state"},
    "module_cache_pollution": {"import_side_effect", "shared_state"},
    "fixture_scope_leak": {"shared_state", "test_order_dependency"},
    "mock_residue": {"shared_state", "fixture_scope_leak"},
    "unknown": set(),
}


ClaimPolarity = Literal["present", "absent"]
ClaimVerdict = Literal["confirmed", "inconclusive", "refuted", "unverified"]


class ThinkClaim(BaseModel):
    """One structured assertion about a potential flakiness root cause."""

    claim_id: str = Field(description="Unique ID within this think block, e.g. 'c1'")
    category: str = Field(description="Root cause category from ROOT_CAUSE_TYPES")
    entity: str = Field(description="Name of the function/class/variable involved")
    location: str = Field(
        description="Fully-qualified location: 'path/to/file.py::ClassName.method_name'"
    )
    ast_node_type: str = Field(
        default="",
        description="Optional: expected libcst node type, e.g. 'FunctionDef', 'Decorator'",
    )
    polarity: ClaimPolarity = Field(
        description="'present' = bug exists; 'absent' = bug was removed by the fix"
    )
    predicted_effect: str = Field(
        default="",
        description="One-sentence prediction of the expected pass-rate change after fix",
    )
    reason: str = Field(description="Short (≤40 words) causal justification")

    verdict: ClaimVerdict = Field(default="unverified")
    oracle_score: float = Field(default=0.0)


class StructuredThink(BaseModel):
    """Structured think block: claims list + overall confidence."""

    claims: List[ThinkClaim] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    format_penalty: float = Field(default=0.0, ge=-1.0, le=0.0)

    @property
    def primary_category(self) -> str:
        if not self.claims:
            return "unknown"
        return self.claims[0].category


class PatchHunk(BaseModel):
    """One search/replace hunk expressed as structured fields.

    Modes (inference may set ``file`` from the observation; models may omit it):
    - **Search/replace:** non-empty ``search`` + ``replace`` (fuzzy / indent matching at apply time).
    - **Line replace:** ``line_number`` (1-based) + ``replace`` = full new line; ``search`` may be
      empty until the runtime expands it from disk for the legacy patch string.
    """

    hunk_id: str = Field(description="Unique ID within this patch block, e.g. 'h1'")
    file: str = Field(
        default="",
        description="Repo-relative path; leave empty for inference to use observation.source_file",
    )
    search: str = Field(
        default="",
        description="Key line(s) or block to find (approximate is OK; applier can fuzzy-match)",
    )
    line_number: Optional[int] = Field(
        default=None,
        description="1-based line index to replace in target file; alternative to search block",
    )
    replace: str = Field(
        description="Lines that replace the search block (may be empty string to delete), or the full new line when using line_number",
    )
    rationale: str = Field(
        default="",
        description="One sentence explaining why this specific change fixes the root cause",
    )
    addresses_claim: str = Field(
        default="",
        description="claim_id from StructuredThink.claims that this hunk resolves",
    )

    applied: bool = False
    apply_error: str = ""


class StructuredPatch(BaseModel):
    """Structured patch block: a list of hunks."""

    hunks: List[PatchHunk] = Field(default_factory=list)
    format_penalty: float = Field(default=0.0, ge=-1.0, le=0.0)

    @property
    def files_targeted(self) -> List[str]:
        return list({h.file for h in self.hunks})


class FlakeForgeAction(Action):
    """Unified action: the agent produces structured think + patch JSON."""

    raw_response: str = ""
    think_text: str = ""
    patch_text: str = ""
    structured_think: Optional[StructuredThink] = None
    structured_patch: Optional[StructuredPatch] = None
    predicted_category: str = "unknown"
    predicted_confidence: float = 0.0

    action_type: str = "UNIFIED_PATCH"
    parameters: Dict[str, Any] = Field(default_factory=dict)


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
class PatchRecord:
    patch_text: str
    target_files: List[str]
    lines_changed: int
    pass_rate_after: float
    applied_successfully: bool = True


@dataclass
class ASTSummary:
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[str]
    global_vars: List[str]
    threading_primitives: List[str]
    external_calls: List[str]


class FlakeForgeObservation(Observation):
    """Observation with deep flakiness signals from SE research."""

    episode_id: str
    test_identifier: str
    step: int
    steps_remaining: int

    test_function_source: str
    source_under_test: str
    source_file: str = Field(
        default="",
        description="Repo-relative path to the primary file under test (SUT), for patches without a model-chosen file",
    )
    relevant_imports: List[str] = Field(default_factory=list)
    file_tree: List[str] = Field(default_factory=list)
    async_markers: List[str] = Field(default_factory=list)

    run_history: List[RunRecord] = Field(default_factory=list)
    current_pass_rate: float = 0.0
    baseline_pass_rate: float = 0.0
    env_type: str = Field(
        default="unknown",
        description="Preflight classification: stable, flaky, deterministic_bug, infra_broken, unknown",
    )
    should_train: bool = Field(
        default=True,
        description="False when preflight says this environment should be skipped for flaky-fixer training",
    )
    preflight_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Three-stage Sanity/Determinism/Flakiness gate summary",
    )

    patches_applied: List[PatchRecord] = Field(default_factory=list)
    total_diff_lines: int = 0

    think_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per-step summary dicts: {step, categories, entities, reason_signatures, "
            "oracle_score, pass_rate_after, reward}.  Used by the reward system to "
            "penalise repeated hypotheses and by the prompt builder to warn the agent."
        ),
    )

    module_cache_violations: List[str] = Field(
        default_factory=list,
        description="Files with @lru_cache, mutable defaults, global state mutations",
    )
    fixture_scope_risks: List[str] = Field(
        default_factory=list,
        description="Session/module scoped fixtures returning mutable objects without teardown",
    )
    mock_residue_sites: List[str] = Field(
        default_factory=list,
        description="Uncleaned patches (patch() without with-context or .stop())",
    )
    import_side_effect_files: List[str] = Field(
        default_factory=list,
        description="Top-level module code with non-constant expressions",
    )
    async_contamination_alive: bool = Field(
        default=False,
        description="Async tasks / threads survived past test boundary",
    )

    failure_frontier: str = Field(
        default="",
        description="Deepest user-code frame in failing stack trace",
    )
    call_chain_to_frontier: List[str] = Field(
        default_factory=list,
        description="Caller chain from test entry to failure site",
    )
    boundary_crossings: List[str] = Field(
        default_factory=list,
        description="HTTP/DB/queue/gRPC boundaries crossed in call chain",
    )

    order_dependency_detected: bool = Field(
        default=False,
        description="Reverse-order run produced different result",
    )
    infrastructure_sensitive: bool = Field(
        default=False,
        description="Chaos run changed outcome significantly",
    )

    causal_graph: Optional[Dict[str, Any]] = None
    causal_hints: List[str] = Field(default_factory=list)

    failure_pattern_summary: Optional[Dict[str, Any]] = None
    duration_fingerprint: Optional[Dict[str, float]] = None
    failing_stack_trace: str = ""

    last_think_text: str = ""
    last_patch_text: str = ""
    last_reward: float = 0.0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    patch_result: Dict[str, Any] = Field(default_factory=dict)
    done_reason: str = ""
    reward: float = 0.0
    done: bool = False

    @field_validator("run_history")
    @classmethod
    def _limit_history(cls, value: List[RunRecord]) -> List[RunRecord]:
        return value[-20:]


class FlakeForgeState(State):
    """Server-side episode state."""

    episode_id: str
    step_count: int
    done: bool = False
    current_pass_rate: float = 0.0
    baseline_pass_rate: float = 0.0
    regression_detected: bool = False
    env_type: str = "unknown"
    should_train: bool = True


@dataclass
class RewardBreakdown:
    """Multi-signal reward breakdown for transparency and debugging."""
    format_reward: float = 0.0
    compile_reward: float = 0.0
    stability_reward: float = 0.0
    causal_proximity_reward: float = 0.0
    failure_entropy_reward: float = 0.0
    anti_hack_penalty: float = 0.0
    regression_penalty: float = 0.0
    reasoning_consistency_reward: float = 0.0
    oracle_reasoning_reward: float = 0.0
    patch_validation_signal: float = 0.0
    noop_patch_penalty: float = 0.0
    think_history_penalty: float = 0.0
    terminal_bonus: float = 0.0
    total_reward: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "format": self.format_reward,
            "compile": self.compile_reward,
            "stability": self.stability_reward,
            "causal_proximity": self.causal_proximity_reward,
            "failure_entropy": self.failure_entropy_reward,
            "anti_hack": self.anti_hack_penalty,
            "regression": self.regression_penalty,
            "reasoning_consistency": self.reasoning_consistency_reward,
            "oracle_reasoning": self.oracle_reasoning_reward,
            "patch_validation": self.patch_validation_signal,
            "noop_patch": self.noop_patch_penalty,
            "think_history": self.think_history_penalty,
            "terminal_bonus": self.terminal_bonus,
            "total": self.total_reward,
        }


def failure_mode_entropy(run_records: List[RunRecord]) -> float:
    """Shannon entropy of error types. Lower = more deterministic = better."""
    errors = [r.error_type for r in run_records if not r.passed and r.error_type]
    if not errors:
        return 0.0
    counts = Counter(errors)
    total = len(errors)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_e = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return round(entropy / max_e, 4) if max_e > 0 else 0.0


# Backward compatible aliases for template-generated names.
FlakeforgeAction = FlakeForgeAction
FlakeforgeObservation = FlakeForgeObservation
