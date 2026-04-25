from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_canonical_action(action_taken: str) -> str:
    """Normalise alias action names to their canonical form."""
    _ALIASES: Dict[str, str] = {
        "detect_flakiness": "GATHER_EVIDENCE",
        "analyze_logs": "GATHER_EVIDENCE",
        "add_sleep": "ADD_TIMING_GUARD",
        "add_lock": "ADD_SYNCHRONIZATION",
        "mock_dependency": "MOCK_DEPENDENCY",
        "isolate_state": "RESET_STATE",
        "reorder_execution": "RESET_STATE",
        "retry_test": "ADD_RETRY",
    }
    return _ALIASES.get(action_taken, action_taken)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    episode_state: Any,
    step_result: Dict[str, Any],
    manifest_oracle: Dict[str, Any],
    teacher_judge_score: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Compute scalar reward from environment signals and the manifest oracle.

    The inline Judge LLM has been removed.  All per-step rewards are now
    deterministic, grounded in ``manifest_oracle`` (the ``flake_manifest.json``
    loaded at episode start).  A single ``teacher_judge_score`` (0–10) is
    supplied at episode end by ``FlakeForge_environment._finalize_episode()``.

    Parameters
    ----------
    episode_state:
        The ``EpisodeState`` dataclass instance.
    step_result:
        Dict returned by the environment step containing runtime metrics.
    manifest_oracle:
        The full parsed ``flake_manifest.json`` for this episode's repo.
    teacher_judge_score:
        0.0 for all intra-episode steps; set to the LLM score (0–10) when
        ``done=True`` by ``_finalize_episode``.
    """

    # ── 1. Basic pass-rate stability (keep from V1/V2) ───────────────────────
    current_pass_rate = float(step_result.get("current_pass_rate", episode_state.current_pass_rate))
    baseline_pass_rate = float(getattr(episode_state, "baseline_pass_rate", 0.0))
    r_stability = (current_pass_rate - baseline_pass_rate) * 10.0

    # ── 2. Chaos stability (keep from V2) ────────────────────────────────────
    v_chaos = step_result.get("chaos_pass_rate")
    chaos_pass_rate = float(v_chaos if v_chaos is not None else current_pass_rate)
    v_chaos_bl = getattr(episode_state, "chaos_baseline_pass_rate", None)
    chaos_baseline_pass_rate = float(v_chaos_bl if v_chaos_bl is not None else 0.0)

    if chaos_baseline_pass_rate == 0.0 and chaos_pass_rate == current_pass_rate:
        r_chaos_stability = 0.0
    else:
        r_chaos_stability = (chaos_pass_rate - chaos_baseline_pass_rate) * 5.0

    # ── 3. Action-Match Bonus (NEW — manifest-grounded) ──────────────────────
    # +3.0 if the action taken is one of manifest's correct_actions.
    # Additional –1.5 penalty if the agent chose a band-aid (ADD_RETRY /
    # ADD_TIMING_GUARD) when the manifest demands a structural fix.
    action_taken = step_result.get("action_taken", "")
    canonical = _get_canonical_action(action_taken)
    correct_actions: List[str] = manifest_oracle.get("correct_actions", [])
    band_aid_actions = {"ADD_RETRY", "ADD_TIMING_GUARD", "retry_test", "add_sleep"}

    r_action_match = 0.0
    if correct_actions:
        if canonical in correct_actions or action_taken in correct_actions:
            r_action_match = 3.0
        elif canonical in band_aid_actions and any(
            a not in band_aid_actions for a in correct_actions
        ):
            # Structural fix needed but agent chose a band-aid
            r_action_match = -1.5

    # ── 4. Correct-Primitive Match (NEW) ─────────────────────────────────────
    # +1.5 if the agent's chosen primitive matches manifest's correct_primitives.to
    # Checked only for patching actions (those that carry parameters).
    correct_primitives: Dict[str, Any] = manifest_oracle.get("correct_primitives", {})
    expected_to_primitive: Optional[str] = correct_primitives.get("to")
    r_primitive_match = 0.0
    if expected_to_primitive and canonical in {
        "ADD_SYNCHRONIZATION", "REFACTOR_CONCURRENCY", "EXTRACT_ASYNC_SCOPE",
        "ISOLATE_BOUNDARY", "HARDEN_IDEMPOTENCY",
    }:
        params = step_result.get("action_parameters", {}) or {}
        agent_primitive = params.get("to_primitive") or params.get("primitive") or ""
        # Normalised substring match (e.g. "asyncio.Lock" inside the full string)
        if agent_primitive and expected_to_primitive in agent_primitive:
            r_primitive_match = 1.5

    # ── 5. Pass-Rate Gate (NEW — activates previously doc-only fields) ────────
    # If manifest defines min_pass_rate_after_fix and agent's patch doesn't
    # reach it, apply a scaled penalty proportional to the shortfall.
    eval_criteria: Dict[str, Any] = manifest_oracle.get("eval_criteria", {})
    min_pass_rate_required: float = float(eval_criteria.get("min_pass_rate_after_fix", 0.0))
    r_pass_rate_gate = 0.0
    if min_pass_rate_required > 0.0 and canonical not in {
        "GATHER_EVIDENCE", "CHAOS_PROBE", "DIAGNOSE_BOUNDARY",
    }:
        shortfall = min_pass_rate_required - current_pass_rate
        if shortfall > 0:
            r_pass_rate_gate = -min(8.0, shortfall * 8.0)

    # ── 6. Prediction Accuracy vs Manifest (NEW) ──────────────────────────────
    # Compares agent's predicted_pass_rate_after against the manifest's
    # expected_pass_rate_after_fix (the objective oracle).
    expected_pass_rate_after_fix: Optional[float] = manifest_oracle.get("expected_pass_rate_after_fix")
    predicted_pass_rate: Optional[float] = step_result.get("predicted_pass_rate_after")
    r_prediction_vs_manifest = 0.0
    if expected_pass_rate_after_fix is not None and predicted_pass_rate is not None:
        delta = abs(predicted_pass_rate - expected_pass_rate_after_fix)
        if delta < 0.05:
            r_prediction_vs_manifest = 0.5   # bonus for near-perfect prediction
        else:
            r_prediction_vs_manifest = -round(delta * 3.0, 4)  # penalty ∈ [−3, 0]

    # ── 7. Root-Cause Category & File Match (NEW) ─────────────────────────────
    # Reward correct diagnosis independent of whether the patch succeeds.
    flake_category: str = manifest_oracle.get("flake_category", "")
    root_cause_file: str = manifest_oracle.get("root_cause_file", "")
    r_root_cause_match = 0.0
    current_hyp = getattr(episode_state, "current_hypothesis", None)
    if current_hyp and flake_category:
        hyp_category = getattr(current_hyp, "root_cause_category", "") or ""
        if hyp_category.upper() == flake_category.upper():
            r_root_cause_match += 1.0
        # Secondary: check if any evidence string mentions the right file
        if root_cause_file:
            evidence: List[str] = list(getattr(current_hyp, "evidence", []) or [])
            if any(root_cause_file in ev for ev in evidence):
                r_root_cause_match += 0.5

    # ── 8. Chaos Sensitivity Match (NEW) ─────────────────────────────────────
    # +1.0 if manifest says infra-sensitive and agent actually ran CHAOS_PROBE.
    is_infra_sensitive: bool = bool(manifest_oracle.get("is_infrastructure_sensitive", False))
    r_chaos_sensitivity_match = 0.0
    if is_infra_sensitive and canonical == "CHAOS_PROBE":
        r_chaos_sensitivity_match = 1.0

    # ── 9. Efficiency penalties (keep from V1/V2) ─────────────────────────────
    r_efficiency = 0.0
    confidence_hist = getattr(episode_state, "hypothesis_confidence_at_each_step", [])
    action_hist = getattr(episode_state, "actions_taken", [])
    for idx, confidence in enumerate(confidence_hist):
        if idx < len(action_hist) and _get_canonical_action(action_hist[idx]) == "GATHER_EVIDENCE" and confidence > 0.8:
            r_efficiency -= 0.3

    p_regression = 15.0 if step_result.get("regression_detected", False) else 0.0

    p_perf_regression = 0.0
    if step_result.get("perf_regression_detected", False):
        v_ratio = step_result.get("perf_median_ratio")
        median_ratio = float(v_ratio if v_ratio is not None else 1.0)
        if median_ratio > 1.0:
            p_perf_regression = min(25.0, 10.0 * math.log(median_ratio))

    p_retry_abuse = 2.0 if canonical in {"ADD_RETRY", "retry_test"} else 0.0

    p_repeat_action = 0.0
    repeat_count = int(step_result.get("repeat_action_count", 1))
    if repeat_count > 2:
        p_repeat_action = 1.5 * (repeat_count - 2)

    lines_changed = int(step_result.get("lines_changed", 0))
    improvement = current_pass_rate - baseline_pass_rate
    p_large_patch_no_gain = 0.0
    if lines_changed >= 20 and improvement <= 0:
        p_large_patch_no_gain = min(6.0, lines_changed * 0.1)

    p_false_fix = 0.0
    if improvement < -0.1:
        p_false_fix = min(8.0, abs(improvement) * 12.0)

    ast_diff = step_result.get("ast_diff", {}) or {}
    semantic_footprint = len(ast_diff.get("functions_modified", []))
    r_semantic_efficiency = -0.1 * max(0, semantic_footprint - 1)

    # ── 10. End-of-Episode Teacher Judge (NEW) ───────────────────────────────
    # teacher_judge_score is 0.0 for all intra-episode steps.
    # When done=True _finalize_episode() calls this again with the actual score.
    r_teacher_judge = 0.0
    if teacher_judge_score > 0.0:
        # Scale 0–10 → [−2, +4]; penalise if score < 4
        r_teacher_judge = (teacher_judge_score / 10.0) * 4.0
        if teacher_judge_score < 4.0:
            r_teacher_judge -= (4.0 - teacher_judge_score) * 0.5

    # ── Total ─────────────────────────────────────────────────────────────────
    reward = (
        r_stability
        + r_chaos_stability
        + r_action_match
        + r_primitive_match
        + r_pass_rate_gate
        + r_prediction_vs_manifest
        + r_root_cause_match
        + r_chaos_sensitivity_match
        + r_efficiency
        + r_semantic_efficiency
        + r_teacher_judge
        - p_regression
        - p_perf_regression
        - p_retry_abuse
        - p_repeat_action
        - p_large_patch_no_gain
        - p_false_fix
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

    breakdown: Dict[str, float] = {
        # ── New RLVR signals ──────────────────────────────────────────────────
        "r_action_match": float(r_action_match),
        "r_primitive_match": float(r_primitive_match),
        "r_pass_rate_gate": float(r_pass_rate_gate),
        "r_prediction_vs_manifest": float(r_prediction_vs_manifest),
        "r_root_cause_match": float(r_root_cause_match),
        "r_chaos_sensitivity_match": float(r_chaos_sensitivity_match),
        "r_teacher_judge": float(r_teacher_judge),
        # ── Kept V1/V2 signals ────────────────────────────────────────────────
        "r_stability": float(r_stability),
        "r_chaos_stability": float(r_chaos_stability),
        "r_efficiency": float(r_efficiency),
        "r_semantic_efficiency": float(r_semantic_efficiency),
        "p_regression": float(p_regression),
        "p_perf_regression": float(p_perf_regression),
        "p_retry_abuse": float(p_retry_abuse),
        "p_repeat_action": float(p_repeat_action),
        "p_large_patch_no_gain": float(p_large_patch_no_gain),
        "p_false_fix": float(p_false_fix),
        "terminal_bonus": float(terminal_bonus),
        "terminal_timeout_penalty": float(terminal_timeout_penalty),
        # ── Aggregate groupings (for dashboards) ──────────────────────────────
        "stability_reward": float(r_stability + r_chaos_stability),
        "ground_truth_reward": float(
            r_action_match + r_primitive_match + r_pass_rate_gate
            + r_prediction_vs_manifest + r_root_cause_match + r_chaos_sensitivity_match
        ),
        "efficiency_penalty": float(max(0.0, -r_efficiency) + p_repeat_action + p_retry_abuse),
        "regression_penalty": float(p_regression + p_perf_regression),
        "false_fix_penalty": float(p_false_fix + p_large_patch_no_gain),
        # Removed: "judge_score" — no per-step judge any more.
        "total_reward": 0.0,
    }
    breakdown["total_reward"] = float(reward)
    return float(reward), breakdown
