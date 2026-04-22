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
        compact_observation = self._compact_observation(observation)
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
        compact_observation = self._compact_observation(observation)

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

    @staticmethod
    def _compact_observation(observation: FlakeForgeObservation) -> Dict[str, Any]:
        """Build a compact observation dict for the judge prompt.

        Includes causal graph warnings, infrastructure sensitivity, and duration
        fingerprint (Improvement 2) so the judge has structural signal beyond
        just pass/fail history.
        """
        # Duration fingerprint from stored field or computed on the fly.
        fp = observation.duration_fingerprint or _compute_duration_fingerprint(
            observation.run_history
        )

        # Causal graph signals.
        causal = observation.causal_graph or {}
        boundary_warnings: List[str] = causal.get("boundary_warnings", [])[:5]
        boundary_nodes: List[str] = causal.get("boundary_nodes", [])[:5]

        return {
            "test_identifier": observation.test_identifier,
            "step": observation.step,
            "current_pass_rate": observation.current_pass_rate,
            "baseline_pass_rate": observation.baseline_pass_rate,
            "infrastructure_sensitive": observation.infrastructure_sensitive,
            "duration_fingerprint": fp,
            "boundary_warnings": boundary_warnings,
            "boundary_nodes": boundary_nodes,
            "test_function_source": "\n".join(observation.test_function_source.splitlines()[:50]),
            "run_history": [
                {
                    "passed": record.passed,
                    "error_type": record.error_type,
                    "duration_ms": record.duration_ms,
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
            "secondary_hypothesis": {
                "root_cause_category": observation.secondary_hypothesis.root_cause_category,
                "confidence": observation.secondary_hypothesis.confidence,
            }
            if observation.secondary_hypothesis
            else None,
        }


def _compute_duration_fingerprint(run_history: list) -> Dict[str, float]:
    """Compute timing statistics used for hypothesis confidence boosting."""
    durations = [r.duration_ms for r in run_history if r.duration_ms is not None]
    if not durations:
        return {"mean_ms": 0.0, "std_ms": 0.0, "cv": 0.0, "flakiness_score": 0.0}
    mean_ms = statistics.mean(durations)
    std_ms = statistics.stdev(durations) if len(durations) > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0
    # Flakiness score: blends pass-rate instability with timing variance.
    pass_rate = sum(1 for r in run_history if r.passed) / len(run_history)
    flakiness_score = round((1.0 - pass_rate) * 0.6 + min(cv, 1.0) * 0.4, 4)
    return {
        "mean_ms": round(mean_ms, 1),
        "std_ms": round(std_ms, 1),
        "cv": round(cv, 4),
        "flakiness_score": flakiness_score,
    }
