from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from pydantic import TypeAdapter

try:
    from ..models import FlakeForgeAction, FlakeForgeObservation, Hypothesis
except ImportError:
    from models import FlakeForgeAction, FlakeForgeObservation, Hypothesis

try:
    from ..server.tools import get_similar_fixes
except Exception:  # pragma: no cover
    try:
        from server.tools import get_similar_fixes
    except Exception:
        get_similar_fixes = None  # type: ignore[assignment]


class ModelBackend(Protocol):
    """Backend interface for shared-base model with switchable LoRA adapters."""

    def generate(self, prompt: str, *, system_prompt: str, adapter_name: str) -> str:
        ...


@dataclass(frozen=True)
class LoRAAdapterSpec:
    name: str
    adapter_path: str


class AnalyzerRole:
    """Analyzer role wrapper around the shared-base model."""

    def __init__(self, backend: ModelBackend, adapter: LoRAAdapterSpec) -> None:
        self.backend = backend
        self.adapter = adapter
        self.system_prompt = (
            "You are a debugging engineer diagnosing flaky CI behavior under uncertainty. "
            "Return ONLY JSON with fields: "
            "root_cause_category, confidence, evidence, reasoning_steps, uncertainty, "
            "next_best_action, predicted_effect. "
            "Allowed root_cause_category values: timing, race, shared_state, network, order, unknown. "
            "reasoning_steps must include observed pattern, hypothesis, and why alternatives are unlikely."
        )

    def produce_hypothesis(self, observation: FlakeForgeObservation) -> Hypothesis:
        prompt = "\n".join(
            [
                "Observation:",
                json.dumps(_compact_observation_payload(observation, include_sources=True), indent=2),
                "Return JSON only.",
            ]
        )
        raw = self.backend.generate(
            prompt,
            system_prompt=self.system_prompt,
            adapter_name=self.adapter.name,
        )

        parsed = _parse_json(raw)
        try:
            category = _normalize_root_cause(parsed.get("root_cause_category", "unknown"))
            return Hypothesis(
                root_cause_category=category,
                confidence=float(parsed.get("confidence", 0.1)),
                evidence=list(parsed.get("evidence", []))[:5],
                suggested_action=parsed.get("suggested_action"),
                reasoning_steps=[str(s) for s in parsed.get("reasoning_steps", [])][:3],
                uncertainty=str(parsed.get("uncertainty", ""))[:240] or None,
                next_best_action=str(parsed.get("next_best_action", ""))[:120] or None,
                predicted_effect=str(parsed.get("predicted_effect", ""))[:240] or None,
            )
        except Exception:
            return Hypothesis(
                root_cause_category="unknown",
                confidence=0.1,
                evidence=[],
                suggested_action=None,
                reasoning_steps=[],
                uncertainty="insufficient evidence",
                next_best_action="detect_flakiness",
                predicted_effect=None,
            )


class FixerRole:
    """Fixer role wrapper around the shared-base model."""

    def __init__(self, backend: ModelBackend, adapter: LoRAAdapterSpec) -> None:
        self.backend = backend
        self.adapter = adapter
        self.system_prompt = (
            "You are a debugging engineer choosing one minimal corrective action under uncertainty. "
            "Use analyzer hypothesis and prior outcomes. "
            "Return ONLY JSON with fields: action_type, parameters, justification, expected_outcome, "
            "predicted_pass_rate_after, risk_assessment, fallback_plan. "
            "Allowed action_type values: detect_flakiness, analyze_logs, add_sleep, add_lock, "
            "mock_dependency, isolate_state, reorder_execution, retry_test. "
            "Predict outcome before acting and avoid random action selection."
        )

    def produce_action(
        self,
        observation: FlakeForgeObservation,
        hypothesis: Hypothesis,
        few_shot_examples: Optional[list[dict[str, str]]] = None,
    ) -> FlakeForgeAction:
        examples = few_shot_examples
        if examples is None:
            examples = _auto_retrieve_examples(observation, hypothesis)

        prompt_parts = [
            "Observation:",
            json.dumps(_compact_observation_payload(observation, include_sources=True), indent=2),
            "Hypothesis:",
            _hypothesis_json(hypothesis),
        ]

        for example in examples or []:
            prompt_parts.extend(
                [
                    "<example>",
                    f"Root cause: {example.get('root_cause', '')}",
                    f"Test: {example.get('original', '')}",
                    f"Fix applied: {example.get('action', '')}",
                    f"Diff: {example.get('diff', '')}",
                    "</example>",
                ]
            )

        prompt_parts.append(
            "Return JSON only: "
            "{action_type, parameters, justification, expected_outcome, predicted_pass_rate_after, risk_assessment, fallback_plan}"
        )
        prompt = "\n".join(prompt_parts)

        raw = self.backend.generate(
            prompt,
            system_prompt=self.system_prompt,
            adapter_name=self.adapter.name,
        )

        parsed = _parse_json(raw)
        try:
            action_type = str(parsed.get("action_type", "analyze_logs"))
            return FlakeForgeAction(
                action_type=action_type,
                parameters=dict(parsed.get("parameters", {})),
                hypothesis=_hypothesis_payload(hypothesis),
                # Improvement 1: capture model's predicted outcome.
                predicted_pass_rate_after=(
                    float(parsed["predicted_pass_rate_after"])
                    if "predicted_pass_rate_after" in parsed
                    else None
                ),
                justification=str(parsed.get("justification", ""))[:500] or None,
                expected_outcome=str(parsed.get("expected_outcome", ""))[:240] or None,
                risk_assessment=str(parsed.get("risk_assessment", ""))[:240] or None,
                fallback_plan=str(parsed.get("fallback_plan", ""))[:240] or None,
            )
        except Exception:
            return FlakeForgeAction(
                action_type="analyze_logs",
                parameters={"injection_target": "test"},
                hypothesis=_hypothesis_payload(hypothesis),
                justification="Need failure signatures before changing code.",
                expected_outcome="Richer failure evidence",
                predicted_pass_rate_after=None,
                risk_assessment="Low risk; no behavioral changes",
                fallback_plan="Try detect_flakiness next",
            )


class FlakeForgeAgentPipeline:
    """Sequential Analyzer -> Fixer pipeline over one shared base model backend."""

    def __init__(self, analyzer: AnalyzerRole, fixer: FixerRole) -> None:
        self.analyzer = analyzer
        self.fixer = fixer

    def run_step(
        self,
        observation: FlakeForgeObservation,
        few_shot_examples: Optional[list[dict[str, str]]] = None,
    ) -> tuple[Hypothesis, FlakeForgeAction]:
        hypothesis = self.analyzer.produce_hypothesis(observation)
        action = self.fixer.produce_action(observation, hypothesis, few_shot_examples)
        return hypothesis, action


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def _hypothesis_json(hypothesis: Hypothesis) -> str:
    return TypeAdapter(Hypothesis).dump_json(hypothesis, indent=2).decode("utf-8")


def _hypothesis_payload(hypothesis: Hypothesis) -> Dict[str, Any]:
    return {
        "root_cause_category": hypothesis.root_cause_category,
        "confidence": hypothesis.confidence,
        "evidence": list(hypothesis.evidence),
        "suggested_action": hypothesis.suggested_action,
        "reasoning_steps": list(hypothesis.reasoning_steps),
        "uncertainty": hypothesis.uncertainty,
        "next_best_action": hypothesis.next_best_action,
        "predicted_effect": hypothesis.predicted_effect,
    }


def _compact_observation_payload(observation: FlakeForgeObservation, include_sources: bool) -> Dict[str, Any]:
    run_history = [
        {
            "passed": r.passed,
            "error_type": r.error_type,
            "duration_ms": r.duration_ms,
        }
        for r in observation.run_history[-5:]
    ]
    payload: Dict[str, Any] = {
        "episode_id": observation.episode_id,
        "test_identifier": observation.test_identifier,
        "step": observation.step,
        "steps_remaining": observation.steps_remaining,
        "current_pass_rate": observation.current_pass_rate,
        "baseline_pass_rate": observation.baseline_pass_rate,
        "async_markers": observation.async_markers[:20],
        "run_history": run_history,
        "current_hypothesis": _hypothesis_payload(observation.current_hypothesis)
        if observation.current_hypothesis
        else None,
        # Improvement 5: secondary hypothesis gives Fixer a hedging option.
        "secondary_hypothesis": {
            "root_cause_category": observation.secondary_hypothesis.root_cause_category,
            "confidence": observation.secondary_hypothesis.confidence,
            "suggested_action": observation.secondary_hypothesis.suggested_action,
        }
        if observation.secondary_hypothesis
        else None,
        "log_snippets": observation.log_snippets[-3:],
        # Improvement 4: duration fingerprint helps Fixer weight timing-race actions.
        "duration_fingerprint": observation.duration_fingerprint,
        "last_actions": list(observation.last_actions[-3:]),
        "last_outcomes": list(observation.last_outcomes[-3:]),
        "prediction_error_history": list(observation.prediction_error_history[-5:]),
        "failure_pattern_summary": observation.failure_pattern_summary,
        "causal_hints": list(observation.causal_hints[-5:]),
        "reflection": observation.reflection,
    }
    if include_sources:
        payload["test_function_source"] = _first_lines(observation.test_function_source, 50)
        payload["source_under_test"] = _first_lines(observation.source_under_test, 50)
    return payload


def _auto_retrieve_examples(observation: FlakeForgeObservation, hypothesis: Hypothesis) -> list[dict[str, str]]:
    # Retrieval uses sentence-transformers/chromadb and can be expensive/unreliable in local runs.
    # Keep it opt-in for inference stability.
    if os.getenv("ENABLE_SIMILAR_FIX_RETRIEVAL", "0").strip().lower() not in {"1", "true", "yes"}:
        return []

    if get_similar_fixes is None:
        return []
    try:
        examples = get_similar_fixes(hypothesis.root_cause_category, observation.test_function_source)
    except Exception:
        return []
    return [
        {
            "root_cause": str(ex.get("root_cause", hypothesis.root_cause_category)),
            "original": _first_lines(str(ex.get("original", "")), 40),
            "action": str(ex.get("action", "")),
            "diff": _first_lines(str(ex.get("diff", "")), 40),
        }
        for ex in examples[:3]
    ]


def _first_lines(text: str, line_count: int) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[:line_count])


def _normalize_root_cause(value: Any) -> str:
    key = str(value or "unknown").strip().lower()
    mapping = {
        "timing_race": "timing",
        "async_deadlock": "race",
        "shared_state": "shared_state",
        "external_dependency": "network",
        "infrastructure_sensitive": "network",
        "order_dependency": "order",
        "resource_leak": "shared_state",
        "nondeterminism": "unknown",
        "timing": "timing",
        "race": "race",
        "network": "network",
        "order": "order",
        "unknown": "unknown",
    }
    return mapping.get(key, "unknown")
