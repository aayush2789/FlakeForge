from __future__ import annotations

import pytest

from client import FlakeForgeEnv
from models import FlakeForgeAction, FlakeForgeObservation


def test_add_sleep_injects_default_delay_ms() -> None:
    action = FlakeForgeAction(action_type="add_sleep", parameters={})

    assert action.parameters["delay_ms"] == 100


@pytest.mark.asyncio
async def test_judge_uses_single_request(monkeypatch: pytest.MonkeyPatch) -> None:
    observation = FlakeForgeObservation(
        episode_id="episode-1",
        test_identifier="tests/test_flaky.py::test_fetch_should_complete",
        step=1,
        steps_remaining=3,
        test_function_source="def test_fetch_should_complete():\n    pass\n",
        source_under_test="def fetch():\n    return True\n",
    )

    call_count = 0

    async def fake_call(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        assert '"task": "score_both"' in prompt
        return (
            '{"judge_hypothesis_score": 4, "judge_patch_score": 3, '
            '"critique": "tighten the wait", "prediction_error": "prediction too optimistic"}'
        )

    monkeypatch.setattr(FlakeForgeEnv, "_call_nvidia_judge", staticmethod(fake_call))

    env = FlakeForgeEnv("http://localhost:5000")
    try:
        result = await env._run_judge(observation)
    finally:
        await env.close()

    assert call_count == 1
    assert result["judge_hypothesis_score"] == 4
    assert result["judge_patch_score"] == 3