# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlakeForge environment client — RLVR Hybrid Architecture.

Key changes vs. previous version:
- Per-step Judge LLM entirely removed. No more _run_judge, _run_judge_safely,
  _call_nvidia_judge, _pending_judge_feedback, or critic injection.
- The Teacher Judge now lives inside the SERVER (FlakeForge_environment.py)
  and is called exactly once at episode end with the full CoT trajectory.
- get_judge_scores() kept as a stub returning [] for backward compatibility.
- get_teacher_judge_result() added to surface the end-of-episode score.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState
except ImportError:
    from models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState


def _log(message: str) -> None:
    print(message, flush=True)


class FlakeForgeEnv(EnvClient[FlakeForgeAction, FlakeForgeObservation, FlakeForgeState]):
    """Async client for the FlakeForge OpenEnv server."""

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__(base_url=base_url, **kwargs)
        # RLVR Hybrid: no per-step judge state needed any more.
        # _teacher_judge_result is populated from the final observation's metadata.
        self._teacher_judge_result: Optional[Dict[str, Any]] = None

    def _step_payload(self, action: FlakeForgeAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FlakeForgeObservation]:
        obs_data = payload.get("observation", payload)
        observation = FlakeForgeObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=float(payload.get("reward", observation.reward or 0.0)),
            done=bool(payload.get("done", observation.done)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> FlakeForgeState:
        return FlakeForgeState.model_validate(payload)

    async def reset(self, **kwargs: Any) -> StepResult[FlakeForgeObservation]:
        self._teacher_judge_result = None
        return await super().reset(**kwargs)

    async def step(self, action: FlakeForgeAction, **kwargs: Any) -> StepResult[FlakeForgeObservation]:
        result = await super().step(action, **kwargs)

        # If the episode is done, capture the Teacher Judge result from metadata.
        if result.done:
            teacher_data = (result.observation.metadata or {}).get("teacher_judge")
            if teacher_data:
                self._teacher_judge_result = teacher_data

        return result

    # ── Backward-compatible stubs ─────────────────────────────────────────────

    def get_judge_scores(self) -> List[Dict[str, Any]]:
        """Kept for backward compatibility. Per-step judge removed; returns []."""
        return []

    def get_teacher_judge_result(self) -> Optional[Dict[str, Any]]:
        """Returns the end-of-episode Teacher Judge result: {score, critique}.

        Available after step() returns done=True. Returns None if the episode
        is not yet complete or if no API key was configured.
        """
        return self._teacher_judge_result

    @classmethod
    async def from_docker_image(cls, image_name: str, **kwargs: Any) -> "FlakeForgeEnv":
        return await super().from_docker_image(image_name, **kwargs)


# Backward compatible alias for template-generated class name.
FlakeforgeEnv = FlakeForgeEnv
