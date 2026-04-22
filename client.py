# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlakeForge environment client."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.judge import FrozenJudge, NVIDIAJudgeBackend

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState, Hypothesis
except ImportError:
    from models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState, Hypothesis


class FlakeForgeEnv(EnvClient[FlakeForgeAction, FlakeForgeObservation, FlakeForgeState]):
    """Async client for the FlakeForge OpenEnv server."""

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__(base_url=base_url, **kwargs)
        self._judge = FrozenJudge(NVIDIAJudgeBackend())
        self._judge_scores: List[Dict[str, int]] = []
        self._pending_judge_feedback: Optional[Dict[str, int]] = None

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
        self._judge_scores = []
        self._pending_judge_feedback = None
        return await super().reset(**kwargs)

    async def step(self, action: FlakeForgeAction, **kwargs: Any) -> StepResult[FlakeForgeObservation]:
        if self._pending_judge_feedback:
            action.judge_feedback = dict(self._pending_judge_feedback)
        result = await super().step(action, **kwargs)
        scores = await self._run_judge(result.observation)
        self._judge_scores.append(scores)
        self._pending_judge_feedback = scores
        return result

    def get_judge_scores(self) -> List[Dict[str, int]]:
        return list(self._judge_scores)

    @classmethod
    async def from_docker_image(cls, image_name: str, **kwargs: Any) -> "FlakeForgeEnv":
        return await super().from_docker_image(image_name, **kwargs)

    async def _run_judge(self, observation: FlakeForgeObservation) -> Dict[str, int]:
        """Judge call parser utilizing centralized FrozenJudge."""
        hypothesis_result = await self._judge.score_hypothesis(
            observation, observation.current_hypothesis
        ) if observation.current_hypothesis else {"score": 0}
        
        patch_result = {"score": 0}
        if observation.patches_applied:
            last_patch = observation.patches_applied[-1]
            last_action = FlakeForgeAction(
                action_type=last_patch.action_taken,
                parameters={} # Simplified for grading context
            )
            # Use empty diff if not provided in patch record, though ideally we'd pass it
            patch_result = await self._judge.score_patch(
                observation, 
                observation.current_hypothesis or Hypothesis(root_cause_category="TIMING_RACE", confidence=0.1, evidence=[]),
                last_action,
                "" 
            )

        return {
            "judge_hypothesis_score": int(hypothesis_result.get("score", 0)),
            "judge_patch_score": int(patch_result.get("score", 0)),
        }


# Backward compatible alias for template-generated class name.
FlakeforgeEnv = FlakeForgeEnv
