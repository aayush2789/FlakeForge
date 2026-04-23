from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

try:
    from ..models import FlakeForgeAction, FlakeForgeObservation, Hypothesis
except ImportError:
    from models import FlakeForgeAction, FlakeForgeObservation, Hypothesis

try:
    from .observation_utils import build_compact_observation
except ImportError:
    from observation_utils import build_compact_observation


class JudgeLLMBackend(Protocol):
    """Backend interface for frozen judge model calls."""

    def complete(self, prompt: str) -> str:
        ...


@dataclass
class FrozenJudge:
    """Frozen large-model judge called client-side during training.

    Improvements applied:
    - Prompts now request {"score", "reasoning", "critique", "prediction_error"}
      (Reflexion loop — Shinn et al. 2023).
    - _compact_observation now includes causal graph boundary warnings,
      infrastructure_sensitive, and duration_fingerprint for richer judge context.
    """

    backend: JudgeLLMBackend

    def score_hypothesis(self, observation: FlakeForgeObservation, hypothesis: Hypothesis) -> Dict[str, Any]:
        compact_observation = build_compact_observation(observation, for_judge=True)
        prompt = (
            "You are a senior software engineer reviewing a diagnosis of a flaky test.\n"
            "Given the test code, failure logs, causal graph signals, and the agent's hypothesis,\n"
            "score the hypothesis from 0 to 5. Score 5 if the root cause category is correct, the\n"
            "confidence is well-calibrated, and the evidence cited directly supports the hypothesis.\n"
            "Score 0 if the category is wrong or the evidence is irrelevant.\n"
            "Provide a critique: one concrete sentence the Fixer agent should act on next.\n"
            'Return ONLY JSON: {"score": <int 0-5>, "reasoning": "<max 40 words>", '
            '"critique": "<one actionable sentence>", "prediction_error": "<what clue was missed>"}\n\n'
            f"Observation: {json.dumps(compact_observation)}\n"
            f"Hypothesis: {self._hypothesis_json(hypothesis)}"
        )
        raw = self.backend.complete(prompt)
        return self._parse_score(raw)

    def score_patch(
        self,
        observation: FlakeForgeObservation,
        hypothesis: Hypothesis,
        action: FlakeForgeAction,
        patch_diff: str,
    ) -> Dict[str, Any]:
        compact_observation = build_compact_observation(observation, for_judge=True)

        # Include the agent's own prediction so the judge can flag overconfidence.
        predicted = getattr(action, "predicted_pass_rate_after", None)
        actual = observation.current_pass_rate
        pred_block = (
            f"Agent predicted pass rate: {predicted:.2f}, actual: {actual:.2f}\n"
            if predicted is not None
            else ""
        )

        prompt = (
            "You are a senior engineer doing code review. Given the original test, the\n"
            "diagnosed root cause, and the proposed patch diff, score the patch from 0 to 5.\n"
            "Score 5 if the patch is minimal (under 10 lines), directly addresses the root\n"
            "cause, and would be accepted in a real PR. Penalise if the patch uses retry\n"
            "logic to mask a bug, or changes more than the minimum necessary.\n"
            "Provide a critique: one concrete sentence the Fixer agent should act on next.\n"
            'Return ONLY JSON: {"score": <int 0-5>, "reasoning": "<max 40 words>", '
            '"critique": "<one actionable sentence>", "prediction_error": "<what prediction was wrong>"}\n\n'
            f"Observation: {json.dumps(compact_observation)}\n"
            f"Hypothesis: {self._hypothesis_json(hypothesis)}\n"
            f"Action: {action.model_dump_json()}\n"
            f"{pred_block}"
            f"Diff:\n{patch_diff}"
        )
        raw = self.backend.complete(prompt)
        return self._parse_score(raw)

    @staticmethod
    def _parse_score(raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw)
            score = max(0, min(5, int(parsed.get("score", 0))))
            reasoning = str(parsed.get("reasoning", ""))[:200]
            critique = str(parsed.get("critique", ""))[:300]
            prediction_error = str(parsed.get("prediction_error", ""))[:200]
            return {
                "score": score,
                "reasoning": reasoning,
                "critique": critique,
                "prediction_error": prediction_error,
            }
        except Exception:
            match = re.search(r"\b([0-5])\b", raw)
            score = int(match.group(1)) if match else 0
            return {"score": score, "reasoning": "fallback_parse", "critique": "", "prediction_error": ""}

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


