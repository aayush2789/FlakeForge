from __future__ import annotations

from typing import Any, Dict, Tuple


def compute_reward(
    episode_state: Any,
    step_result: Dict[str, Any],
    judge_scores: Dict[str, int],
) -> Tuple[float, Dict[str, float]]:
    """Compute scalar reward from environment and judge signals."""

    current_pass_rate = float(step_result.get("current_pass_rate", episode_state.current_pass_rate))
    baseline_pass_rate = float(getattr(episode_state, "baseline_pass_rate", 0.0))

    r_stability = (current_pass_rate - baseline_pass_rate) * 10.0
    
    # v2: Add chaos stability reward
    chaos_pass_rate = float(step_result.get("chaos_pass_rate", current_pass_rate))
    r_chaos_stability = (chaos_pass_rate - baseline_pass_rate) * 5.0

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
    
    # v2: Add performance regression penalty
    p_perf_regression = 5.0 if step_result.get("perf_regression_detected", False) else 0.0

    action_taken = step_result.get("action_taken", "")
    p_retry_abuse = 2.0 if action_taken == "ADD_RETRY" else 0.0

    ast_diff = step_result.get("ast_diff", {}) or {}
    semantic_footprint = len(ast_diff.get("functions_modified", []))
    r_semantic_efficiency = -0.1 * max(0, semantic_footprint - 1)

    reward = r_stability + r_chaos_stability + r_judge + r_efficiency + r_semantic_efficiency - p_regression - p_perf_regression - p_retry_abuse

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
        "p_regression": float(p_regression),
        "p_perf_regression": float(p_perf_regression),
        "p_retry_abuse": float(p_retry_abuse),
        "terminal_bonus": float(terminal_bonus),
        "terminal_timeout_penalty": float(terminal_timeout_penalty),
    }
    return float(reward), breakdown
