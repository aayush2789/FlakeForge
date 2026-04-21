from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Protocol

from models import FlakeForgeAction, FlakeForgeObservation, Hypothesis


class JudgeLLMBackend(Protocol):
    """Backend interface for frozen judge model calls."""

    def complete(self, prompt: str) -> str:
        ...


@dataclass
class FrozenJudge:
    """Frozen large-model judge called client-side during training."""

    backend: JudgeLLMBackend

    def score_hypothesis(self, observation: FlakeForgeObservation, hypothesis: Hypothesis) -> Dict[str, Any]:
        prompt = (
            "You are a senior software engineer reviewing a diagnosis of a flaky test.\n"
            "Given the test code, failure logs, and the agent's hypothesis, score the\n"
            "hypothesis from 0 to 5. Score 5 if the root cause category is correct, the\n"
            "confidence is well-calibrated, and the evidence cited directly supports the\n"
            "hypothesis. Score 0 if the category is wrong or the evidence is irrelevant.\n"
            "Return only a JSON object: {\"score\": <int 0-5>, \"reasoning\": \"<max 50 words>\"}\n\n"
            f"Observation: {observation.model_dump_json()}\n"
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
        prompt = (
            "You are a senior engineer doing code review. Given the original test, the\n"
            "diagnosed root cause, and the proposed patch diff, score the patch from 0 to 5.\n"
            "Score 5 if the patch is minimal (under 10 lines), directly addresses the root\n"
            "cause, and would be accepted in a real PR. Penalize if the patch uses retry\n"
            "logic to mask a bug, or changes more than the minimum necessary.\n"
            "Return only JSON: {\"score\": <int 0-5>, \"reasoning\": \"<max 50 words>\"}\n\n"
            f"Observation: {observation.model_dump_json()}\n"
            f"Hypothesis: {self._hypothesis_json(hypothesis)}\n"
            f"Action: {action.model_dump_json()}\n"
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
