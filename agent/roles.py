from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from pydantic import TypeAdapter

try:
    from models import FlakeForgeAction, FlakeForgeObservation, Hypothesis
except ImportError:
    from ..models import FlakeForgeAction, FlakeForgeObservation, Hypothesis

try:
    from utils.logger import get_logger
except ImportError:
    from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    from agent.observation_utils import build_compact_observation, _first_lines, _hypothesis_payload
except ImportError:
    from .observation_utils import build_compact_observation, _first_lines, _hypothesis_payload

try:
    from server.tools import get_similar_fixes
except Exception:  # pragma: no cover
    try:
        from .server.tools import get_similar_fixes
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
        "You are a senior debugging engineer analyzing flaky CI failures under uncertainty. Your task is not just to classify the issue, but to reason causally and propose the most informative next action. "

        "Return ONLY valid JSON with the following fields: "
        "root_cause_category, confidence, evidence, reasoning_steps, uncertainty, next_best_action, action_rationale, predicted_effect, counterfactual. "

        "--- "

        "Allowed root_cause_category values: "
        "timing: Operations complete with variable delay; missing waits or timeouts. "
        "race: Parallel execution order affects outcome; async or concurrency issue. "
        "shared_state: Global or persistent state leaks across tests. "
        "network: External dependency variability (latency, failures). "
        "order: Test outcome depends on execution order. "
        "unknown: insufficient or conflicting evidence. "

        "--- "

        "reasoning_steps MUST follow this exact structure: "
        "1. observed_pattern: Describe failure behavior across runs (e.g., intermittent, timing variance, order sensitivity). "
        "2. key_signals: Extract concrete signals (logs, durations, failure modes). "
        "3. hypothesis: State a single dominant causal mechanism (not just label). "
        "4. mechanism_explanation: Explain HOW this cause leads to the observed failure. "
        "5. alternatives_considered: List at least 2 alternative causes and why they are less likely. "
        "6. confidence_justification: Why confidence is high/low given evidence quality. "

        "--- "

        "Action selection rules: "
        "Choose the action that maximally REDUCES uncertainty, not just fixes the issue. "
        "Prefer diagnostic actions over blind fixes when confidence < 0.8. "
        "If multiple causes are plausible, choose action that differentiates them. "

        "--- "

        "Allowed next_best_action values: "
        "GATHER_EVIDENCE, ADD_TIMING_GUARD, ADD_SYNCHRONIZATION, MOCK_DEPENDENCY, RESET_STATE, ADD_RETRY, REVERT_LAST_PATCH. "

        "--- "

        "Action guidelines: "
        "If timing suspected → ADD_TIMING_GUARD. "
        "If race suspected → ADD_SYNCHRONIZATION. "
        "If shared_state suspected → RESET_STATE. "
        "If network suspected → MOCK_DEPENDENCY. "
        "If low confidence → GATHER_EVIDENCE. "
        "If failure is non-deterministic with no signal → ADD_RETRY (last resort). "

        "--- "

        "predicted_effect: Describe expected measurable change (e.g., pass rate increases, variance reduces). "
        "counterfactual: Describe what result would falsify your hypothesis. "

        "--- "

        "STRICT RULES: "
        "Do NOT guess without evidence. "
        "Do NOT output multiple hypotheses. "
        "Do NOT suggest action without causal justification. "
        "Prefer being uncertain over being wrong."
    )

    def produce_hypothesis(self, observation: FlakeForgeObservation) -> Hypothesis:
        prompt = "\n".join(
            [
                "Observation:",
                json.dumps(build_compact_observation(observation, include_sources=True), indent=2),
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
            # Support both old and new field names
            reasoning = parsed.get("reasoning_steps", [])
            if not reasoning and "action_rationale" in parsed:
                reasoning = [str(parsed["action_rationale"])]
            
            # Fix: LLM may return evidence as a raw string instead of a list.
            # Calling list() on a string iterates characters, producing ["T","h","e",...].
            raw_ev = parsed.get("evidence", [])
            if isinstance(raw_ev, str):
                raw_ev = [raw_ev] if raw_ev.strip() else []
            elif not isinstance(raw_ev, list):
                raw_ev = [str(raw_ev)] if raw_ev else []
            return Hypothesis(
                root_cause_category=category,
                confidence=float(parsed.get("confidence", 0.1)),
                evidence=raw_ev[:5],
                suggested_action=parsed.get("next_best_action") or parsed.get("suggested_action"),
                reasoning_steps=list([str(s) for s in reasoning])[:3],
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
        "You are a senior debugging engineer responsible for selecting ONE minimal, high-impact corrective action for a flaky CI failure. "
        "Use the analyzer hypothesis, prior run outcomes, failure patterns, and system signals. "
        "Think like a scientist, but keep the response compact. "

        "Return ONLY valid JSON with these fields: chosen_action, confidence (0 to 1), justification, parameters, predicted_pass_rate_after. "
        "Optional fields: expected_outcome, risk_assessment, fallback_strategy. Keep every string short, preferably one sentence. "

        "--- "

        "Core Principles: "
        "1. Prefer minimal, reversible changes. "
        "2. Prefer actions that TEST the hypothesis over blindly fixing. "
        "3. If confidence < 0.8, prioritize diagnostic actions. "
        "4. Avoid masking the issue (e.g., retries) unless no signal exists. "
        "5. Do NOT apply multiple fixes at once. "
        "6. Every action must have a clear causal justification. "

        "--- "

        "Available Actions and Definitions: "

        "GATHER_EVIDENCE actions: "
        "- detect_flakiness: Run test multiple times to establish baseline pass rate and variance. "
        "- analyze_logs: Inject logging/print statements to observe internal state transitions. "

        "FIX / INTERVENTION actions: "
        "- add_sleep (ADD_TIMING_GUARD): Add fixed delay to wait for async operations. "
        "- add_lock (ADD_SYNCHRONIZATION): Use locks to enforce execution order on shared resources. "
        "- mock_dependency (MOCK_DEPENDENCY): Replace external API/DB with deterministic mock. "
        "- isolate_state (RESET_STATE): Clear/reset DB, cache, or global state before/after test. "
        "- reorder_execution: Change order of setup or dependent steps. "
        "- retry_test (ADD_RETRY): Add retry decorator to mitigate intermittent failures. "

        "ADVANCED actions (use ONLY with strong justification): "
        "- REFACTOR_CONCURRENCY: Change concurrency model (threading ↔ asyncio). "
        "- ISOLATE_BOUNDARY: Add timeouts, circuit breakers, or guards to external calls. "
        "- EXTRACT_ASYNC_SCOPE: Move blocking operations outside event loop. "
        "- CHAOS_PROBE: Introduce CPU/network stress to test sensitivity to timing or load. "

        "--- "

        "Action Selection Strategy: "
        "- If hypothesis is uncertain → choose GATHER_EVIDENCE action. "
        "- If hypothesis strongly indicates timing → add_sleep. "
        "- If hypothesis strongly indicates race → add_lock. "
        "- If hypothesis strongly indicates shared state → isolate_state. "
        "- If hypothesis strongly indicates external instability → mock_dependency. "
        "- If failure depends on order → reorder_execution. "
        "- If no clear signal and highly flaky → retry_test (last resort). "

        "--- "

        "STRICT RULES: "
        "- Output EXACTLY one chosen_action. "
        "- Do NOT combine multiple actions. "
        "- Do NOT apply heavy refactors unless strongly justified. "
        "- Do NOT default to retry unless no other signal exists. "
        "- Do NOT act without linking action to hypothesis. "
        "- Prefer diagnostic clarity over quick fixes."
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
            json.dumps(build_compact_observation(observation, include_sources=True), indent=2),
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
            "{action_type, parameters, justification, predicted_pass_rate_after, expected_outcome, risk_assessment, fallback_plan}"
        )
        prompt = "\n".join(prompt_parts)

        raw = self.backend.generate(
            prompt,
            system_prompt=self.system_prompt,
            adapter_name=self.adapter.name,
        )

        parsed = _parse_json(raw)
        try:
            # Match the new "chosen_action" field from the senior engineer prompt
            raw_action_type = str(parsed.get("chosen_action") or parsed.get("action_type", "analyze_logs"))
            # CRITICAL: Normalize action_type BEFORE constructing FlakeForgeAction.
            # Unknown types (e.g. "FIX", "fix_timeout") cause Pydantic ValidationError
            # on the server side, which drops the WebSocket/HTTP connection entirely.
            action_type = _normalize_fixer_action(raw_action_type, hypothesis)
            justification_text = str(parsed.get("reasoning") or parsed.get("justification", ""))[:500] or None

            logger.info(f"[TOOL_USAGE] [FIXER] Using chosen tool/action <{action_type}> based on root cause hypothesis.")
            if justification_text:
                logger.info(f"[TOOL_USAGE] [FIXER] Reasoning/Purpose for using tool: {justification_text}")

            # Sanitize parameters — strip unexpected keys that would fail server validation
            raw_params = dict(parsed.get("parameters", {}))
            safe_params = _sanitize_params(action_type, raw_params)

            return FlakeForgeAction(
                action_type=action_type,
                parameters=safe_params,
                hypothesis=_hypothesis_payload(hypothesis),
                predicted_pass_rate_after=(
                    float(parsed["predicted_pass_rate_after"])
                    if "predicted_pass_rate_after" in parsed
                    else (0.5 if "expected_outcome" in parsed else None)
                ),
                justification=justification_text,
                expected_outcome=str(parsed.get("expected_outcome", ""))[:240] or None,
                risk_assessment=str(parsed.get("risk_assessment", ""))[:240] or None,
                fallback_plan=str(parsed.get("fallback_strategy") or parsed.get("fallback_plan", ""))[:240] or None,
            )
        except Exception:
            # Smart fallback based on category
            fallback_type = "analyze_logs"
            if hypothesis.root_cause_category in {"race", "timing"}:
                fallback_type = "add_sleep"
            elif hypothesis.root_cause_category == "shared_state":
                fallback_type = "add_lock"
            elif hypothesis.root_cause_category == "network":
                fallback_type = "mock_dependency"

            return FlakeForgeAction(
                action_type=fallback_type,
                parameters=_sanitize_params(fallback_type, {}),
                hypothesis=_hypothesis_payload(hypothesis),
                justification=f"JSON parsing failed; falling back to heuristic {fallback_type}",
                expected_outcome="Fallback recovery",
                predicted_pass_rate_after=None,
                risk_assessment="Low risk",
                fallback_plan="Retry with different prompt",
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


def _normalize_fixer_action(raw: str, hypothesis: "Hypothesis") -> str:
    """Map any LLM-generated action string to a valid ACTION_TYPES value.

    The LLM may output things like 'FIX', 'fix_timeout', 'ADD_SLEEP',
    'increase_timeout', etc. We apply a priority cascade:

    1. Direct match (already a known alias or canonical type).
    2. Keyword-based fuzzy match.
    3. Category-based fallback from hypothesis.
    """
    # All valid values accepted by FlakeForgeAction.action_type
    VALID = {
        "GATHER_EVIDENCE", "ADD_TIMING_GUARD", "ADD_SYNCHRONIZATION",
        "MOCK_DEPENDENCY", "RESET_STATE", "ADD_RETRY", "REVERT_LAST_PATCH",
        "SEED_RANDOMNESS", "DIAGNOSE_BOUNDARY", "REFACTOR_CONCURRENCY",
        "ISOLATE_BOUNDARY", "EXTRACT_ASYNC_SCOPE", "HARDEN_IDEMPOTENCY",
        "CHAOS_PROBE",
        # lowercase aliases
        "detect_flakiness", "analyze_logs", "add_sleep", "add_lock",
        "mock_dependency", "isolate_state", "reorder_execution", "retry_test",
    }

    cleaned = raw.strip()
    # 1. Direct hit
    if cleaned in VALID:
        return cleaned

    # 2. Case-insensitive direct hit
    upper = cleaned.upper()
    lower = cleaned.lower()
    for v in VALID:
        if v.upper() == upper:
            return v

    # 3. Keyword fuzzy mapping
    keyword_map = [
        (["sleep", "timeout", "delay", "wait", "timing"], "add_sleep"),
        (["lock", "sync", "synchron", "barrier"], "add_lock"),
        (["mock", "stub", "patch", "dependency", "external"], "mock_dependency"),
        (["state", "reset", "isolate", "fixture"], "isolate_state"),
        (["retry", "attempt", "flak"], "retry_test"),
        (["log", "evidence", "gather", "detect", "diagnose", "observe"], "analyze_logs"),
        (["reorder", "order", "sequence"], "reorder_execution"),
        (["seed", "random"], "SEED_RANDOMNESS"),
        (["refactor", "concurren", "async"], "REFACTOR_CONCURRENCY"),
        (["chaos", "probe"], "CHAOS_PROBE"),
        (["revert", "undo"], "REVERT_LAST_PATCH"),
    ]
    for keywords, action in keyword_map:
        if any(kw in lower for kw in keywords):
            return action

    # 4. Category-driven fallback
    cat = getattr(hypothesis, "root_cause_category", "unknown")
    category_defaults = {
        "timing": "add_sleep",
        "race": "add_lock",
        "shared_state": "isolate_state",
        "network": "mock_dependency",
        "order": "reorder_execution",
    }
    return category_defaults.get(cat, "analyze_logs")


def _sanitize_params(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of params containing only keys allowed by the server validator.

    This prevents Pydantic ValidationError from 'Unexpected parameter keys'
    that would crash the HTTP request and manifest as a ConnectionClosedError.
    """
    # Map from alias → canonical
    alias_to_canonical = {
        "detect_flakiness": "GATHER_EVIDENCE",
        "analyze_logs": "GATHER_EVIDENCE",
        "add_sleep": "ADD_TIMING_GUARD",
        "add_lock": "ADD_SYNCHRONIZATION",
        "mock_dependency": "MOCK_DEPENDENCY",
        "isolate_state": "RESET_STATE",
        "reorder_execution": "RESET_STATE",
        "retry_test": "ADD_RETRY",
    }
    canonical = alias_to_canonical.get(action_type, action_type)

    allowed_map: Dict[str, set] = {
        "GATHER_EVIDENCE": {"injection_target"},
        "ADD_TIMING_GUARD": {"delay_ms"},
        "ADD_SYNCHRONIZATION": {"primitive"},
        "MOCK_DEPENDENCY": {"target"},
        "RESET_STATE": {"scope"},
        "ADD_RETRY": {"max_attempts", "backoff_ms"},
        "REVERT_LAST_PATCH": set(),
        "SEED_RANDOMNESS": {"library"},
        "DIAGNOSE_BOUNDARY": {"boundary_node"},
        "REFACTOR_CONCURRENCY": {"from_primitive", "to_primitive", "target_function"},
        "ISOLATE_BOUNDARY": {"boundary_call", "pattern"},
        "EXTRACT_ASYNC_SCOPE": {"target_function", "direction"},
        "HARDEN_IDEMPOTENCY": {"state_target", "key_strategy"},
        "CHAOS_PROBE": {"profile", "n_runs"},
    }
    allowed = allowed_map.get(canonical, set())

    # Supply required defaults for certain actions when key is missing
    defaults: Dict[str, Any] = {}
    if canonical == "GATHER_EVIDENCE" and "injection_target" not in params:
        defaults["injection_target"] = "test"
    elif canonical == "ADD_TIMING_GUARD" and "delay_ms" not in params:
        defaults["delay_ms"] = 100
    elif canonical == "ADD_SYNCHRONIZATION" and "primitive" not in params:
        defaults["primitive"] = "lock"
    elif canonical == "MOCK_DEPENDENCY" and "target" not in params:
        defaults["target"] = "requests.get"
    elif canonical == "RESET_STATE" and "scope" not in params:
        defaults["scope"] = "function"
    elif canonical == "ADD_RETRY":
        defaults.setdefault("max_attempts", 2)
        defaults.setdefault("backoff_ms", 100)
    elif canonical == "SEED_RANDOMNESS" and "library" not in params:
        defaults["library"] = "random"

    merged = {**defaults, **{k: v for k, v in params.items() if k in allowed}}
    return merged

