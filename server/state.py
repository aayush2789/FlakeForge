import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from ..models import Hypothesis
except ImportError:
    from models import Hypothesis

@dataclass
class EpisodeState:
    episode_id: str
    # Keep these optional for compatibility with older call sites.
    target_function_source: str = ""
    source_under_test: str = ""
    test_identifier: str = ""
    step: int = 0
    step_count: int = 0
    max_steps: int = 10
    total_reward: float = 0.0
    done: bool = False

    baseline_pass_rate: float = 0.0
    current_pass_rate: float = 0.0
    chaos_pass_rate: float = 0.0
    chaos_baseline_pass_rate: Optional[float] = None
    regression_detected: bool = False
    perf_regression_detected: bool = False
    perf_median_ratio: float = 1.0
    total_diff_lines: int = 0

    reward_history: List[float] = field(default_factory=list)
    last_actions: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    hypothesis_confidence_at_each_step: List[float] = field(default_factory=list)
    hypothesis_history: List[Dict[str, Any]] = field(default_factory=list)
    last_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    prediction_error_history: List[float] = field(default_factory=list)
    actions_tried: List[str] = field(default_factory=list)
    judge_scores: List[Dict[str, Any]] = field(default_factory=list)

    current_hypothesis: Optional[Hypothesis] = None
    secondary_hypothesis: Optional[Hypothesis] = None
    reflection: Optional[Dict[str, Any]] = None
    run_history: List[Any] = field(default_factory=list)
    patches_applied: List[Any] = field(default_factory=list)
    log_snippets: List[str] = field(default_factory=list)

    # ── V2 Dynamic Features
    causal_graph_dict: Optional[Dict[str, Any]] = None
    boundary_warnings: List[str] = field(default_factory=list)
    infrastructure_sensitive: bool = False
    duration_fingerprint: Optional[Dict[str, float]] = None

    failure_pattern_summary: Dict[str, Any] = field(default_factory=dict)
    causal_hints: List[str] = field(default_factory=list)
    async_markers: List[str] = field(default_factory=list)

    # ── RLVR Hybrid: Chain-of-Thought trajectory & Teacher Judge ──────────────
    # Each step appends one dict: {step, action_type, justification,
    # reasoning_steps, predicted_pass_rate_after, actual_pass_rate}
    cot_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # Set once at episode end by _finalize_episode(); range 0.0–10.0
    teacher_judge_score: Optional[float] = None
    # Verbal critique from the Teacher Judge LLM
    teacher_judge_critique: str = ""

    # ── Baseline regression snapshot: True if non-target tests were ALREADY
    # failing before the agent took any action. Prevents test_flaky_simple
    # (randomly fails 30%) from counting as an agent-introduced regression.
    baseline_regression_status: bool = False

    start_time: float = field(default_factory=time.time)


    def is_done(self) -> bool:
        return self.done or self.step_count >= self.max_steps or self.current_pass_rate >= 0.95
