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
    finally:
        await env.close()
