from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from pydantic import TypeAdapter

from models import FlakeForgeAction, FlakeForgeObservation, Hypothesis


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
                observation.model_dump_json(indent=2),
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
            "Select exactly one action from the seven allowed actions and return JSON."
        )

    def produce_action(
        self,
        observation: FlakeForgeObservation,
        hypothesis: Hypothesis,
        few_shot_examples: Optional[list[dict[str, str]]] = None,
    ) -> FlakeForgeAction:
        prompt_parts = [
            "Observation:",
            observation.model_dump_json(indent=2),
            "Hypothesis:",
            _hypothesis_json(hypothesis),
        ]

        for example in few_shot_examples or []:
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
            )
        except Exception:
            return FlakeForgeAction(
                action_type="GATHER_EVIDENCE",
                parameters={"injection_target": "test"},
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
