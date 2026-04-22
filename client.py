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

        # --- Async judge (60 s hard cap) ---
        try:
            scores = await asyncio.wait_for(self._run_judge(result.observation), timeout=60.0)
        except Exception:
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
        # Store critique for next step's Reflexion injection.
        self._pending_critique = scores.get("critique", "")
        return result

    def get_judge_scores(self) -> List[Dict[str, Any]]:
        return list(self._judge_scores)

    @classmethod
    async def from_docker_image(cls, image_name: str, **kwargs: Any) -> "FlakeForgeEnv":
        return await super().from_docker_image(image_name, **kwargs)

    async def _run_judge(self, observation: FlakeForgeObservation) -> Dict[str, Any]:
        """Run both judge prompts concurrently; returns scores + critique."""
        hypothesis_prompt = self._build_hypothesis_prompt(observation)
        patch_prompt = self._build_patch_prompt(observation)

        hypothesis_response, patch_response = await asyncio.gather(
            self._call_nvidia_judge(hypothesis_prompt),
            self._call_nvidia_judge(patch_prompt),
        )

        h_parsed = self._parse_judge_response(hypothesis_response)
        p_parsed = self._parse_judge_response(patch_response)

        # Aggregate critiques: prefer patch critique (more action-specific).
        critique = p_parsed.get("critique") or h_parsed.get("critique") or ""
        prediction_error = p_parsed.get("prediction_error") or h_parsed.get("prediction_error") or ""

        return {
            "judge_hypothesis_score": h_parsed["score"],
            "judge_patch_score": p_parsed["score"],
            "critique": critique,
            "prediction_error": prediction_error,
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
        api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not api_key:
            return '{"score": 0, "reasoning": "no_api_key", "critique": "", "prediction_error": ""}'

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
                    "You are a senior code reviewer grading a patch/hypothesis. "
                    'Reply ONLY with JSON: {"score": <0-5>, "reasoning": "<40 words>", '
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
                max_tokens=300,
                timeout=judge_timeout,
            )
            return completion.choices[0].message.content or '{"score": 0}'
        except Exception as e:
            print(f"[WARN] Client judge API err: {e}")
            return '{"score": 0, "reasoning": "api_error", "critique": "", "prediction_error": ""}'

    @staticmethod
    def _parse_judge_response(response: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(response)
            score = max(0, min(5, int(parsed.get("score", 0))))
            return {
                "score": score,
                "reasoning": str(parsed.get("reasoning", ""))[:200],
                "critique": str(parsed.get("critique", ""))[:300],
                "prediction_error": str(parsed.get("prediction_error", ""))[:200],
            }
        except Exception:
            match = re.search(r"\b([0-5])\b", response)
            return {
                "score": int(match.group(1)) if match else 0,
                "reasoning": "fallback_parse",
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
