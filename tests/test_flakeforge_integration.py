from __future__ import annotations

import pytest
import subprocess

from client import FlakeForgeEnv
from models import FlakeForgeAction


@pytest.mark.integration
@pytest.mark.asyncio
async def test_flakeforge_integration_contract() -> None:
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            "flakeforge-env:latest",
            "-f",
            "server/Dockerfile",
            ".",
        ],
        check=True,
    )

    env = await FlakeForgeEnv.from_docker_image("flakeforge-env:latest")
    try:
        reset_result = await env.reset()
        obs = reset_result.observation

        assert 0.0 <= obs.baseline_pass_rate <= 0.9

        gather = await env.step(
            FlakeForgeAction(
                action_type="GATHER_EVIDENCE",
                parameters={"injection_target": "test"},
            )
        )
        assert len(gather.observation.log_snippets) >= 1

        timing = await env.step(
            FlakeForgeAction(
                action_type="ADD_TIMING_GUARD",
                parameters={"delay_ms": 200},
            )
        )
        assert timing.reward > -15
        assert len(timing.observation.patches_applied) == 1

        retry = await env.step(
            FlakeForgeAction(
                action_type="ADD_RETRY",
                parameters={"max_attempts": 2, "backoff_ms": 100},
            )
        )
        retry_breakdown = retry.observation.metadata.get("reward_breakdown", {})
        assert retry_breakdown.get("p_retry_abuse") == 2.0

        revert = await env.step(
            FlakeForgeAction(
                action_type="REVERT_LAST_PATCH",
                parameters={},
            )
        )
        assert len(revert.observation.patches_applied) <= len(retry.observation.patches_applied)
        assert revert.observation.metadata.get("action") == "REVERT_LAST_PATCH"
    finally:
        await env.close()
