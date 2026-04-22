from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


def compute_reward(
    episode_state: Any,
    step_result: Dict[str, Any],
    judge_scores: Dict[str, int],
) -> Tuple[float, Dict[str, float]]:
    """Compute scalar reward from environment and judge signals."""

    current_pass_rate = float(step_result.get("current_pass_rate", episode_state.current_pass_rate))
    baseline_pass_rate = float(getattr(episode_state, "baseline_pass_rate", 0.0))

    r_stability = (current_pass_rate - baseline_pass_rate) * 10.0
    
    # v2: Add chaos stability reward - compare against chaos_baseline, not clean_baseline
    chaos_pass_rate = float(step_result.get("chaos_pass_rate", current_pass_rate))
    chaos_baseline_pass_rate = float(getattr(episode_state, "chaos_baseline_pass_rate", None) or 0.0)
    
    # When chaos hasn't been run yet, return 0.0 instead of defaulting to current_pass_rate
    if chaos_baseline_pass_rate == 0.0 and chaos_pass_rate == current_pass_rate:
        r_chaos_stability = 0.0
    else:
        r_chaos_stability = (chaos_pass_rate - chaos_baseline_pass_rate) * 5.0

    judge_hypothesis_score = int(judge_scores.get("judge_hypothesis_score", 0))
    judge_patch_score = int(judge_scores.get("judge_patch_score", 0))
    r_judge = ((judge_hypothesis_score / 5.0) + (judge_patch_score / 5.0)) * 1.5

    r_efficiency = 0.0
    confidence_hist = getattr(episode_state, "hypothesis_confidence_at_each_step", [])
    action_hist = getattr(episode_state, "actions_taken", [])
    for idx, confidence in enumerate(confidence_hist):
        if idx < len(action_hist) and action_hist[idx] == "GATHER_EVIDENCE" and confidence > 0.8:
            r_efficiency -= 0.3

    p_regression = 15.0 if step_result.get("regression_detected", False) else 0.0
    
    # v2: Use sentinel's penalty() method for performance regression
    # If perf_regression_detected is True, use 10 × log(median_ratio)
    p_perf_regression = 0.0
    if step_result.get("perf_regression_detected", False):
        median_ratio = float(step_result.get("perf_median_ratio", 1.0))
        if median_ratio > 1.0:
            p_perf_regression = min(25.0, 10.0 * math.log(median_ratio))

    action_taken = step_result.get("action_taken", "")
    p_retry_abuse = 2.0 if action_taken == "ADD_RETRY" else 0.0

    ast_diff = step_result.get("ast_diff", {}) or {}
    semantic_footprint = len(ast_diff.get("functions_modified", []))
    r_semantic_efficiency = -0.1 * max(0, semantic_footprint - 1)

    # ── Improvement 1: prediction-error shaping ─────────────────────────────
    # When the Fixer includes predicted_pass_rate_after we penalise the delta
    # between prediction and reality (up to -2.0). This forces the model to
    # internalise outcome uncertainty rather than being unconditionally optimistic.
    predicted_pass_rate: Optional[float] = step_result.get("predicted_pass_rate_after")
    r_prediction_accuracy = 0.0
    if predicted_pass_rate is not None:
        p_error = abs(predicted_pass_rate - current_pass_rate)
        r_prediction_accuracy = -round(p_error * 2.0, 4)  # penalty ∈ [-2, 0]

    reward = (
        r_stability
        + r_chaos_stability
        + r_judge
        + r_efficiency
        + r_semantic_efficiency
        + r_prediction_accuracy
        - p_regression
        - p_perf_regression
        - p_retry_abuse
    )

    terminal_bonus = 0.0
    terminal_timeout_penalty = 0.0

    if step_result.get("done", False):
        success = current_pass_rate >= 1.0 and not step_result.get("regression_detected", False)
        timeout = step_result.get("timed_out", False) or (
            episode_state.step_count >= episode_state.max_steps and current_pass_rate < 0.9
        )
        if success:
            terminal_bonus = 5.0
            reward += terminal_bonus
        elif timeout:
            terminal_timeout_penalty = 5.0
            reward -= terminal_timeout_penalty

    breakdown = {
        "r_stability": float(r_stability),
        "r_chaos_stability": float(r_chaos_stability),
        "r_judge": float(r_judge),
        "r_efficiency": float(r_efficiency),
        "r_semantic_efficiency": float(r_semantic_efficiency),
        "r_prediction_accuracy": float(r_prediction_accuracy),
        "p_regression": float(p_regression),
        "p_perf_regression": float(p_perf_regression),
        "p_retry_abuse": float(p_retry_abuse),
        "terminal_bonus": float(terminal_bonus),
        "terminal_timeout_penalty": float(terminal_timeout_penalty),
    }
    return float(reward), breakdown
