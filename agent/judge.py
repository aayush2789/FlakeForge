from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Protocol

try:
    from ..models import FlakeForgeAction, FlakeForgeObservation, Hypothesis
except ImportError:
    from models import FlakeForgeAction, FlakeForgeObservation, Hypothesis


class JudgeLLMBackend(Protocol):
    """Backend interface for frozen judge model calls."""

    async def complete(self, prompt: str) -> str:
        ...


class NVIDIAJudgeBackend:
    """Actual NVIDIA Minimax API implementation."""

    def __init__(self, model: str = "minimaxai/minimax-m2.7"):
        from openai import AsyncOpenAI
        import os
        self.model = model
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")
        self.client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )

    async def complete(self, prompt: str) -> str:
        if not self.api_key:
            return '{"score": 0, "reasoning": "Missing API Key"}'
        
        messages = [
            {"role": "system", "content": "You are a senior code reviewer grading a patch/hypothesis. Reply ONLY with JSON containing 'score' (1-5) and 'reasoning'. Example: {\"score\": 4, \"reasoning\": \"Minimal fix\"}"},
            {"role": "user", "content": prompt}
        ]
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
            )
            return completion.choices[0].message.content or '{"score": 0}'
        except Exception as e:
            return json.dumps({"score": 0, "reasoning": f"API Error: {str(e)}"})


@dataclass
class FrozenJudge:
    """Frozen large-model judge called client-side during training."""

    backend: JudgeLLMBackend

    async def score_hypothesis(self, observation: FlakeForgeObservation, hypothesis: Hypothesis) -> Dict[str, Any]:
        compact_observation = self._compact_observation(observation)
        prompt = (
            "You are a senior software engineer reviewing a diagnosis of a flaky test.\n"
            "Given the test code, failure logs, and the agent's hypothesis, score the\n"
            "hypothesis from 0 to 5. Score 5 if the root cause category is correct, the\n"
            "confidence is well-calibrated, and the evidence cited directly supports the\n"
            "hypothesis. Score 0 if the category is wrong or the evidence is irrelevant.\n"
            "Return only a JSON object: {\"score\": <int 0-5>, \"reasoning\": \"<max 50 words>\"}\n\n"
            f"Observation: {json.dumps(compact_observation)}\n"
            f"Hypothesis: {self._hypothesis_json(hypothesis)}"
        )
        raw = await self.backend.complete(prompt)
        return self._parse_score(raw)

    async def score_patch(
        self,
        observation: FlakeForgeObservation,
        hypothesis: Hypothesis,
        action: FlakeForgeAction,
        patch_diff: str,
    ) -> Dict[str, Any]:
        compact_observation = self._compact_observation(observation)
        prompt = (
            "You are a senior engineer doing code review. Given the original test, the\n"
            "diagnosed root cause, and the proposed patch diff, score the patch from 0 to 5.\n"
            "Score 5 if the patch is minimal (under 10 lines), directly addresses the root\n"
            "cause, and would be accepted in a real PR. Penalize if the patch uses retry\n"
            "logic to mask a bug, or changes more than the minimum necessary.\n"
            "Return only JSON: {\"score\": <int 0-5>, \"reasoning\": \"<max 50 words>\"}\n\n"
            f"Observation: {json.dumps(compact_observation)}\n"
            f"Hypothesis: {self._hypothesis_json(hypothesis)}\n"
            f"Action: {action.model_dump_json()}\n"
            f"Diff:\n{patch_diff}"
        )
        raw = await self.backend.complete(prompt)
        return self._parse_score(raw)

    @staticmethod
    def _parse_score(raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw)
            score = max(0, min(5, int(parsed.get("score", 0))))
            reasoning = str(parsed.get("reasoning", ""))[:200]
            return {"score": score, "reasoning": reasoning}
        except Exception:
            match = re.search(r"\b([0-5])\b", raw)
            score = int(match.group(1)) if match else 0
            return {"score": score, "reasoning": "fallback_parse"}

    @staticmethod
    def _hypothesis_json(hypothesis: Hypothesis) -> str:
        return json.dumps(
            {
                "root_cause_category": hypothesis.root_cause_category,
                "confidence": hypothesis.confidence,
                "evidence": hypothesis.evidence,
                "suggested_action": hypothesis.suggested_action,
            }
        )

    @staticmethod
    def _compact_observation(observation: FlakeForgeObservation) -> Dict[str, Any]:
        return {
            "test_identifier": observation.test_identifier,
            "step": observation.step,
            "current_pass_rate": observation.current_pass_rate,
            "baseline_pass_rate": observation.baseline_pass_rate,
            "test_function_source": "\n".join(observation.test_function_source.splitlines()[:50]),
            "run_history": [
                {
                    "passed": record.passed,
                    "error_type": record.error_type,
                }
                for record in observation.run_history[-5:]
            ],
            "current_hypothesis": {
                "root_cause_category": observation.current_hypothesis.root_cause_category,
                "confidence": observation.current_hypothesis.confidence,
                "evidence": observation.current_hypothesis.evidence,
            }
            if observation.current_hypothesis
            else None,
        }
