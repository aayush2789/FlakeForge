# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlakeForge environment client."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState
except ImportError:
    from models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState


class FlakeForgeEnv(EnvClient[FlakeForgeAction, FlakeForgeObservation, FlakeForgeState]):
    """Async client for the FlakeForge OpenEnv server."""

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__(base_url=base_url, **kwargs)
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
        """Placeholder judge call parser; replace with LLM API call in trainer stack."""
        hypothesis_prompt = self._build_hypothesis_prompt(observation)
        patch_prompt = self._build_patch_prompt(observation)

        hypothesis_response = self._mock_judge_response(hypothesis_prompt)
        patch_response = self._mock_judge_response(patch_prompt)

        return {
            "judge_hypothesis_score": self._parse_judge_score(hypothesis_response),
            "judge_patch_score": self._parse_judge_score(patch_response),
        }

    @staticmethod
    def _build_hypothesis_prompt(observation: FlakeForgeObservation) -> str:
        hypothesis = observation.current_hypothesis.model_dump() if observation.current_hypothesis else None
        return json.dumps(
            {
                "task": "score_hypothesis",
                "hypothesis": hypothesis,
                "run_history": [r.__dict__ for r in observation.run_history],
            }
        )

    @staticmethod
    def _build_patch_prompt(observation: FlakeForgeObservation) -> str:
        patch = observation.patches_applied[-1].__dict__ if observation.patches_applied else None
        return json.dumps(
            {
                "task": "score_patch",
                "patch": patch,
                "current_pass_rate": observation.current_pass_rate,
                "baseline_pass_rate": observation.baseline_pass_rate,
            }
        )

    @staticmethod
    def _mock_judge_response(prompt: str) -> str:
        _ = prompt
        return '{"score": 3}'

    @staticmethod
    def _parse_judge_score(response: str) -> int:
        try:
            parsed = json.loads(response)
            score = int(parsed.get("score", 0))
            return max(0, min(5, score))
        except Exception:
            match = re.search(r"\b([0-5])\b", response)
            return int(match.group(1)) if match else 0


# Backward compatible alias for template-generated class name.
FlakeforgeEnv = FlakeForgeEnv
