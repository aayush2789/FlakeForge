from __future__ import annotations

from pathlib import Path

import pytest

from models import FlakeForgeAction, FlakeForgeObservation
from server.action_executor import build_patch_spec
from server.tools import apply_ast_patch


def test_add_sleep_injects_default_delay_ms() -> None:
    action = FlakeForgeAction(action_type="add_sleep", parameters={})
    assert action.parameters["delay_ms"] == 100


# ── New RLVR reward unit tests ────────────────────────────────────────────────

def _make_episode_state():
    """Minimal EpisodeState for reward tests."""
    try:
        from server.state import EpisodeState
    except ImportError:
        from state import EpisodeState  # type: ignore
    return EpisodeState(episode_id="test-ep", max_steps=14, test_identifier="t")


def _compute(step_result, oracle, teacher_score=0.0):
    try:
        from server.reward import compute_reward
    except ImportError:
        from reward import compute_reward  # type: ignore
    ep = _make_episode_state()
    ep.current_pass_rate = step_result.get("current_pass_rate", 0.0)
    ep.baseline_pass_rate = step_result.get("current_pass_rate", 0.0)  # neutral stability
    return compute_reward(ep, step_result, oracle, teacher_score)


def _base_step(action="ADD_SYNCHRONIZATION", pass_rate=1.0):
    return {
        "action_taken": action,
        "action_parameters": {"primitive": "lock"},
        "current_pass_rate": pass_rate,
        "regression_detected": False,
        "done": False,
        "repeat_action_count": 1,
        "lines_changed": 5,
    }


_ORACLE_SYNC = {
    "correct_actions": ["ADD_SYNCHRONIZATION"],
    "correct_primitives": {"from": None, "to": "threading.Lock"},
    "expected_pass_rate_after_fix": 1.0,
    "eval_criteria": {"min_pass_rate_after_fix": 0.95},
    "is_infrastructure_sensitive": False,
    "flake_category": "TIMING_RACE",
    "root_cause_file": "counter.py",
}


def test_action_match_reward_fires() -> None:
    """Correct action → +3.0 action match bonus."""
    _, bd = _compute(_base_step("ADD_SYNCHRONIZATION"), _ORACLE_SYNC)
    assert bd["r_action_match"] == 3.0


def test_band_aid_penalty_when_structural_fix_needed() -> None:
    """ADD_RETRY when manifest requires structural fix → −1.5."""
    _, bd = _compute(_base_step("ADD_RETRY"), _ORACLE_SYNC)
    assert bd["r_action_match"] == -1.5


def test_wrong_action_gives_zero_action_match() -> None:
    """Action outside correct_actions but not a band-aid → 0.0."""
    _, bd = _compute(_base_step("GATHER_EVIDENCE"), _ORACLE_SYNC)
    assert bd["r_action_match"] == 0.0


def test_prediction_vs_manifest_bonus_exact() -> None:
    """Exact prediction vs manifest → +0.5 bonus."""
    sr = _base_step()
    sr["predicted_pass_rate_after"] = 1.0
    _, bd = _compute(sr, _ORACLE_SYNC)
    assert bd["r_prediction_vs_manifest"] == 0.5


def test_prediction_vs_manifest_penalty() -> None:
    """Predicted 0.7, manifest expects 1.0 → delta=0.3, penalty≈−0.9."""
    sr = _base_step()
    sr["predicted_pass_rate_after"] = 0.7
    _, bd = _compute(sr, _ORACLE_SYNC)
    assert bd["r_prediction_vs_manifest"] == pytest.approx(-0.9, abs=0.01)


def test_pass_rate_gate_penalty_fires() -> None:
    """Pass rate 0.80, gate requires 0.95 → shortfall=0.15 → penalty=−1.2."""
    sr = _base_step(pass_rate=0.80)
    _, bd = _compute(sr, _ORACLE_SYNC)
    # Shortfall = 0.95 − 0.80 = 0.15, penalty = 0.15 * 8 = 1.2
    assert bd["r_pass_rate_gate"] == pytest.approx(-1.2, abs=0.01)


def test_pass_rate_gate_no_penalty_when_met() -> None:
    """Pass rate ≥ gate → no gate penalty."""
    sr = _base_step(pass_rate=1.0)
    _, bd = _compute(sr, _ORACLE_SYNC)
    assert bd["r_pass_rate_gate"] == 0.0


def test_chaos_sensitivity_match_reward() -> None:
    """Infra-sensitive manifest + CHAOS_PROBE action → +1.0."""
    oracle = dict(_ORACLE_SYNC, is_infrastructure_sensitive=True)
    sr = _base_step("CHAOS_PROBE")
    sr["action_parameters"] = {"profile": "cpu", "n_runs": 10}
    _, bd = _compute(sr, oracle)
    assert bd["r_chaos_sensitivity_match"] == 1.0


def test_no_judge_score_in_breakdown() -> None:
    """Removed 'judge_score' key must not appear in breakdown."""
    _, bd = _compute(_base_step(), _ORACLE_SYNC)
    assert "judge_score" not in bd


def test_ground_truth_reward_key_present() -> None:
    """New aggregate 'ground_truth_reward' key must be in breakdown."""
    _, bd = _compute(_base_step(), _ORACLE_SYNC)
    assert "ground_truth_reward" in bd