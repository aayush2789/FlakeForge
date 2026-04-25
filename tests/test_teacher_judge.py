"""Tests for the RLVR Hybrid End-of-Episode Teacher Judge.

All tests mock _call_teacher_judge so no actual API calls are made.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest


# ── Fixtures & helpers ────────────────────────────────────────────────────────

def _make_env(repo_path: str = ""):
    """Create a FlakeForgeEnvironment with a temporary repo path (no real FS)."""
    try:
        from server.FlakeForge_environment import FlakeForgeEnvironment
    except ImportError:
        from FlakeForge_environment import FlakeForgeEnvironment  # type: ignore

    env = FlakeForgeEnvironment.__new__(FlakeForgeEnvironment)
    # Inject minimal state without running __init__
    from pathlib import Path
    env.repo_path = Path(repo_path) if repo_path else Path(".")
    env._manifest_oracle = {
        "flake_category": "TIMING_RACE",
        "correct_actions": ["ADD_SYNCHRONIZATION"],
        "correct_primitives": {"from": None, "to": "threading.Lock"},
        "root_cause_file": "counter.py",
        "root_cause_function": "increment",
        "expected_pass_rate_after_fix": 1.0,
        "is_infrastructure_sensitive": True,
        "eval_criteria": {"min_pass_rate_after_fix": 0.95},
        "teacher_judge_context": {
            "expected_reasoning_steps": [
                "Run CHAOS_PROBE to confirm CPU sensitivity",
                "Apply ADD_SYNCHRONIZATION with threading.Lock",
            ],
            "anti_patterns": ["ADD_RETRY as first response"],
        },
    }

    try:
        from server.state import EpisodeState
    except ImportError:
        from state import EpisodeState  # type: ignore

    env._episode = EpisodeState(
        episode_id="ep-test",
        max_steps=14,
        test_identifier="tests/test_flaky.py::test_case",
        current_pass_rate=1.0,
    )
    env._episode.cot_trajectory = [
        {
            "step": 1,
            "action_type": "CHAOS_PROBE",
            "justification": "Checking infrastructure sensitivity",
            "reasoning_steps": ["Baseline pass rate 0.80 suggests load-dependent race"],
            "hypothesis_category": "TIMING_RACE",
            "predicted_pass_rate_after": None,
            "actual_pass_rate": 0.80,
        },
        {
            "step": 2,
            "action_type": "ADD_SYNCHRONIZATION",
            "justification": "Wrap increment in threading.Lock()",
            "reasoning_steps": [
                "CPU chaos confirmed race; threading.Lock will serialise increment"
            ],
            "hypothesis_category": "TIMING_RACE",
            "predicted_pass_rate_after": 1.0,
            "actual_pass_rate": 1.0,
        },
    ]
    return env


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_teacher_judge_prompt_contains_trajectory() -> None:
    """_call_teacher_judge receives a prompt that contains CoT steps and oracle."""
    env = _make_env()
    captured_prompts: List[str] = []

    async def fake_call(prompt: str, api_key: str):
        captured_prompts.append(prompt)
        return 8.0, "Good reasoning chain."

    with patch.object(
        type(env), "_call_teacher_judge", staticmethod(fake_call)
    ):
        import os
        os.environ.setdefault("NVIDIA_API_KEY", "test-key")
        # Manually call _finalize_episode to avoid full env init
        delta = env._finalize_episode()

    assert len(captured_prompts) == 1, "Teacher Judge should be called exactly once"
    payload = json.loads(captured_prompts[0])
    assert payload["task"] == "grade_reasoning_trajectory"
    # Trajectory present
    assert len(payload["agent_trajectory"]) == 2
    assert payload["agent_trajectory"][0]["action_type"] == "CHAOS_PROBE"
    # Oracle present
    assert payload["manifest_oracle"]["flake_category"] == "TIMING_RACE"
    assert "expected_reasoning_steps" in payload["manifest_oracle"]


@pytest.mark.asyncio
async def test_teacher_judge_score_applied_to_reward_delta() -> None:
    """Score of 8/10 → delta = (8/10)*4 = 3.2."""
    env = _make_env()

    async def fake_call(prompt: str, api_key: str):
        return 8.0, "Solid reasoning."

    with patch.object(type(env), "_call_teacher_judge", staticmethod(fake_call)):
        import os
        os.environ.setdefault("NVIDIA_API_KEY", "test-key")
        delta = env._finalize_episode()

    expected = (8.0 / 10.0) * 4.0  # = 3.2
    assert abs(delta - expected) < 0.01
    assert env._episode.teacher_judge_score == 8.0
    assert "Solid" in env._episode.teacher_judge_critique


@pytest.mark.asyncio
async def test_teacher_judge_penalises_low_score() -> None:
    """Score below 4 gets extra penalty: (3/10)*4 − (4−3)*0.5 = 1.2 − 0.5 = 0.7."""
    env = _make_env()
    # Bad reasoning: correct action but empty justification
    env._episode.cot_trajectory[0]["reasoning_steps"] = []
    env._episode.cot_trajectory[0]["justification"] = ""

    async def fake_call(prompt: str, api_key: str):
        return 3.0, "Correct action but no causal reasoning provided."

    with patch.object(type(env), "_call_teacher_judge", staticmethod(fake_call)):
        import os
        os.environ.setdefault("NVIDIA_API_KEY", "test-key")
        delta = env._finalize_episode()

    expected = (3.0 / 10.0) * 4.0 - (4.0 - 3.0) * 0.5  # 1.2 - 0.5 = 0.7
    assert abs(delta - expected) < 0.01


@pytest.mark.asyncio
async def test_teacher_judge_no_api_key_returns_zero() -> None:
    """Without an API key, Teacher Judge gracefully returns 0.0 delta."""
    env = _make_env()

    import os
    original = os.environ.pop("NVIDIA_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)

    try:
        delta = env._finalize_episode()
    finally:
        if original:
            os.environ["NVIDIA_API_KEY"] = original

    assert delta == 0.0
    assert env._episode.teacher_judge_score == 0.0


@pytest.mark.asyncio
async def test_teacher_judge_critique_stored_on_episode() -> None:
    """Critique from Teacher Judge is persisted on episode state."""
    env = _make_env()

    async def fake_call(prompt: str, api_key: str):
        return 7.0, "Correctly identified sync primitive but skipped boundary analysis."

    with patch.object(type(env), "_call_teacher_judge", staticmethod(fake_call)):
        import os
        os.environ.setdefault("NVIDIA_API_KEY", "test-key")
        env._finalize_episode()

    assert "boundary" in env._episode.teacher_judge_critique


def test_reward_breakdown_has_no_judge_score_key() -> None:
    """The legacy 'judge_score' key must not appear in reward breakdown at all."""
    try:
        from server.reward import compute_reward
        from server.state import EpisodeState
    except ImportError:
        from reward import compute_reward  # type: ignore
        from state import EpisodeState  # type: ignore

    ep = EpisodeState(episode_id="x", max_steps=14, test_identifier="t")
    ep.current_pass_rate = 1.0
    ep.baseline_pass_rate = 1.0
    oracle: Dict[str, Any] = {
        "correct_actions": ["ADD_SYNCHRONIZATION"],
        "correct_primitives": {"to": "threading.Lock"},
        "expected_pass_rate_after_fix": 1.0,
        "eval_criteria": {"min_pass_rate_after_fix": 0.95},
        "is_infrastructure_sensitive": False,
        "flake_category": "TIMING_RACE",
    }
    sr = {
        "action_taken": "ADD_SYNCHRONIZATION",
        "action_parameters": {},
        "current_pass_rate": 1.0,
        "regression_detected": False,
        "done": False,
        "repeat_action_count": 1,
        "lines_changed": 5,
    }
    _, bd = compute_reward(ep, sr, oracle)
    assert "judge_score" not in bd, "Per-step judge score must be removed from breakdown"
    assert "r_teacher_judge" in bd
    assert "r_action_match" in bd
    assert "ground_truth_reward" in bd
