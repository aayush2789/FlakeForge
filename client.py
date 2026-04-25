# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlakeForge environment client.

Key changes vs. original:
- Judge calls are the ONLY judge path (duplicate FrozenJudge in inference.py removed).
- Reflexion loop (Improvement 3): judge critique is captured and injected as a
  `[JUDGE_CRITIQUE]` log snippet into the *next* step's observation so the
  Fixer agent can act on the verbal feedback without a gradient update.
- predicted_pass_rate_after is forwarded to the step payload so the server-side
  reward function can compute a prediction-error penalty (Improvement 1).
"""

from __future__ import annotations

import asyncio
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


def _log(message: str) -> None:
    print(message, flush=True)


class FlakeForgeEnv(EnvClient[FlakeForgeAction, FlakeForgeObservation, FlakeForgeState]):
    """Async client for the FlakeForge OpenEnv server."""

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__(base_url=base_url, **kwargs)
        self._judge_scores: List[Dict[str, Any]] = []
        self._pending_judge_feedback: Optional[Dict[str, Any]] = None
        # Reflexion: store the last critique to inject into the next observation.
        self._pending_critique: str = ""
        # Speed Optimization: store background judge task
        self._latest_judge_task: Optional[asyncio.Task] = None

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
        self._pending_critique = ""
        self._latest_judge_task = None
        return await super().reset(**kwargs)

    async def step(self, action: FlakeForgeAction, **kwargs: Any) -> StepResult[FlakeForgeObservation]:
        # Attach prior judge feedback so env can track it.
        if self._pending_judge_feedback:
            try:
                from models import JudgeFeedbackPayload  # type: ignore
            except ImportError:
                try:
                    from .models import JudgeFeedbackPayload  # type: ignore
                except ImportError:
                    JudgeFeedbackPayload = None  # type: ignore

            if JudgeFeedbackPayload:
                action.judge_feedback = JudgeFeedbackPayload(**self._pending_judge_feedback)

        # --- env.step (the only network call that blocks) ---
        result = await super().step(action, **kwargs)

        # Reflexion: inject critique from the *previous* step into log_snippets
        # before running the judge for this step. The Fixer will see it on the
        # next call to produce_action().
        if self._pending_critique:
            critique_snippet = json.dumps({"judge_critique": self._pending_critique})
            result.observation.log_snippets = (
                [critique_snippet] + list(result.observation.log_snippets)
            )[-5:]  # keep window tight

        # --- Async judge (detached background task) ---
        self._latest_judge_task = asyncio.create_task(self._run_judge_safely(result.observation))
        
        return result

    async def _run_judge_safely(self, observation: FlakeForgeObservation) -> None:
        try:
            scores = await asyncio.wait_for(self._run_judge(observation), timeout=60.0)
        except Exception as judge_exc:
            import traceback
            print(f"[WARN] Failed to run judge: {judge_exc!r}", flush=True)
            traceback.print_exc()
            scores = {
                "judge_hypothesis_score": 0,
                "judge_patch_score": 0,
                "critique": "",
                "prediction_error": "",
            }

        self._judge_scores.append(scores)
        self._pending_judge_feedback = {
            "judge_hypothesis_score": scores.get("judge_hypothesis_score", 0),
            "judge_patch_score": scores.get("judge_patch_score", 0),
            "critique": scores.get("critique", ""),
            "prediction_error": scores.get("prediction_error", ""),
        }
        self._pending_critique = scores.get("critique", "")

    async def wait_for_previous_judge(self) -> None:
        """Awaits the previously dispatched judge task so the critique is ready for the current step."""
        if self._latest_judge_task and not self._latest_judge_task.done():
            await self._latest_judge_task

    def get_judge_scores(self) -> List[Dict[str, Any]]:
        return list(self._judge_scores)

    @classmethod
    async def from_docker_image(cls, image_name: str, **kwargs: Any) -> "FlakeForgeEnv":
        return await super().from_docker_image(image_name, **kwargs)

    async def _run_judge(self, observation: FlakeForgeObservation) -> Dict[str, Any]:
        """Run one judge prompt that scores both hypothesis and patch together."""
        judge_prompt = json.dumps(
            {
                "task": "score_both",
                "hypothesis": json.loads(self._build_hypothesis_prompt(observation)),
                "patch": json.loads(self._build_patch_prompt(observation)),
            }
        )

        response = await self._call_nvidia_judge(judge_prompt)
        parsed = self._parse_judge_response(response)

        return {
            "judge_hypothesis_score": parsed["judge_hypothesis_score"],
            "judge_patch_score": parsed["judge_patch_score"],
            "critique": parsed["critique"],
            "prediction_error": parsed["prediction_error"],
        }

    @staticmethod
    def _build_hypothesis_prompt(observation: FlakeForgeObservation) -> str:
        if observation.current_hypothesis:
            hypothesis = {
                "root_cause_category": observation.current_hypothesis.root_cause_category,
                "confidence": observation.current_hypothesis.confidence,
                "evidence": list(observation.current_hypothesis.evidence),
                "suggested_action": observation.current_hypothesis.suggested_action,
            }
        else:
            hypothesis = None
        # Include causal graph hint for richer context (Improvement 2).
        causal = observation.causal_graph or {}
        return json.dumps(
            {
                "task": "score_hypothesis",
                "hypothesis": hypothesis,
                "run_history": [r.__dict__ for r in observation.run_history],
                "infrastructure_sensitive": observation.infrastructure_sensitive,
                "boundary_warnings": causal.get("boundary_warnings", [])[:3],
                "duration_fingerprint": observation.duration_fingerprint,
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
                "duration_fingerprint": observation.duration_fingerprint,
            }
        )

    @staticmethod
    async def _call_nvidia_judge(prompt: str) -> str:
        api_key = (os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            return '{"judge_hypothesis_score": 0, "judge_patch_score": 0, "critique": "", "prediction_error": ""}'

        judge_model = os.environ.get("JUDGE_MODEL", "minimaxai/minimax-m2.7").strip()
        judge_timeout = float(os.environ.get("REQUEST_TIMEOUT_S", "45"))

        client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=judge_timeout,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior code reviewer grading a hypothesis and a patch. "
                    'Reply ONLY with JSON: {"judge_hypothesis_score": <0-5>, "judge_patch_score": <0-5>, '
                    '"critique": "<one actionable sentence for the Fixer>", '
                    '"prediction_error": "<what clue or prediction was wrong>"}'
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            completion = await client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=0.2,
                top_p=0.95,
                max_tokens=400,
                timeout=judge_timeout,
            )
            return completion.choices[0].message.content or '{"judge_hypothesis_score": 0, "judge_patch_score": 0, "critique": "", "prediction_error": ""}'
        except Exception as e:
            print(f"[WARN] Client judge API err: {e}")
            return '{"judge_hypothesis_score": 0, "judge_patch_score": 0, "critique": "", "prediction_error": ""}'

    @staticmethod
    def _parse_judge_response(response: str) -> Dict[str, Any]:
        print(f"[DEBUG_JUDGE] response={response!r}", flush=True)
        try:
            # Extract JSON block even if prefaced by <think> block
            json_str = response
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx+1]
                
            parsed = json.loads(json_str)
            return {
                "judge_hypothesis_score": max(0, min(5, int(parsed.get("judge_hypothesis_score", parsed.get("score", 0))))),
                "judge_patch_score": max(0, min(5, int(parsed.get("judge_patch_score", parsed.get("score", 0))))),
                "critique": str(parsed.get("critique", ""))[:300],
                "prediction_error": str(parsed.get("prediction_error", ""))[:200],
            }
        except Exception:
            match = re.search(r"\b([0-5])\b", response)
            return {
                "judge_hypothesis_score": int(match.group(1)) if match else 0,
                "judge_patch_score": int(match.group(1)) if match else 0,
                "critique": "",
                "prediction_error": "",
            }

    @staticmethod
    def _parse_judge_score(response: str) -> int:
        """Legacy helper kept for backward compatibility."""
        try:
            parsed = json.loads(response)
            score = int(parsed.get("score", 0))
            return max(0, min(5, score))
        except Exception:
            match = re.search(r"\b([0-5])\b", response)
            return int(match.group(1)) if match else 0


# Backward compatible alias for template-generated class name.
FlakeforgeEnv = FlakeForgeEnv
