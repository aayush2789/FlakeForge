# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlakeForge environment client."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

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
        return await super().reset(**kwargs)

    async def step(self, action: FlakeForgeAction, **kwargs: Any) -> StepResult[FlakeForgeObservation]:
        result = await super().step(action, **kwargs)
        scores = await self._run_judge(result.observation)
        self._judge_scores.append(scores)
        return result

    def get_judge_scores(self) -> List[Dict[str, int]]:
        return list(self._judge_scores)

    @classmethod
    async def from_docker_image(cls, image_name: str, **kwargs: Any) -> "FlakeForgeEnv":
        return await super().from_docker_image(image_name, **kwargs)

    async def _run_judge(self, observation: FlakeForgeObservation) -> Dict[str, int]:
        """Judge call parser utilizing NVIDIA's Minimax API."""
        hypothesis_prompt = self._build_hypothesis_prompt(observation)
        patch_prompt = self._build_patch_prompt(observation)

        hypothesis_response = await self._call_nvidia_judge(hypothesis_prompt)
        patch_response = await self._call_nvidia_judge(patch_prompt)

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
    async def _call_nvidia_judge(prompt: str) -> str:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            return '{"score": 0}'

        client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

        messages = [
            {"role": "system", "content": "You are a senior code reviewer grading a patch/hypothesis. Reply ONLY with JSON containing a single integer key 'score' between 1 and 5. Example: {\"score\": 4}"},
            {"role": "user", "content": prompt}
        ]

        try:
            completion = await client.chat.completions.create(
                model="minimaxai/minimax-m2.7",
                messages=messages,
                temperature=0.2,  # Low temp for more deterministic metric scoring
                top_p=0.95,
                max_tokens=256,
            )
            return completion.choices[0].message.content or '{"score": 0}'
        except Exception as e:
            print(f"Judge API err: {e}")
            return '{"score": 0}'

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
