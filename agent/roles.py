from __future__ import annotations

import json
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
            "You are a senior software engineer diagnosing a flaky CI test. "
            "Commit to exactly one root cause category and return only JSON "
            "matching the Hypothesis schema."
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
            return Hypothesis(
                root_cause_category=parsed.get("root_cause_category", "TIMING_RACE"),
                confidence=float(parsed.get("confidence", 0.1)),
                evidence=list(parsed.get("evidence", []))[:5],
                suggested_action=parsed.get("suggested_action"),
            )
        except Exception:
            return Hypothesis(
                root_cause_category="TIMING_RACE",
                confidence=0.1,
                evidence=[],
                suggested_action=None,
            )


class FixerRole:
    """Fixer role wrapper around the shared-base model."""

    def __init__(self, backend: ModelBackend, adapter: LoRAAdapterSpec) -> None:
        self.backend = backend
        self.adapter = adapter
        self.system_prompt = (
            "You are a senior software engineer repairing a flaky CI test. "
            "Select exactly one action from the eight allowed actions and return JSON."
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

        prompt_parts.append("Return JSON only: {action_type, parameters}")
        prompt = "\n".join(prompt_parts)

        raw = self.backend.generate(
            prompt,
            system_prompt=self.system_prompt,
            adapter_name=self.adapter.name,
        )

        parsed = _parse_json(raw)
        try:
            return FlakeForgeAction(
                action_type=parsed.get("action_type", "GATHER_EVIDENCE"),
                parameters=dict(parsed.get("parameters", {})),
                hypothesis=_hypothesis_payload(hypothesis),
            )
        except Exception:
            return FlakeForgeAction(
                action_type="GATHER_EVIDENCE",
                parameters={"injection_target": "test"},
                hypothesis=_hypothesis_payload(hypothesis),
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
        "log_snippets": observation.log_snippets[-3:],
    }
    if include_sources:
        payload["test_function_source"] = _first_lines(observation.test_function_source, 50)
        payload["source_under_test"] = _first_lines(observation.source_under_test, 50)
    return payload


def _auto_retrieve_examples(observation: FlakeForgeObservation, hypothesis: Hypothesis) -> list[dict[str, str]]:
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
