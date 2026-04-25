"""V3 Unified FlakeForge Agent — single model, single forward pass.

Generates <think> + <patch> in one completion. Reasoning is verified
by causal execution outcome, not an LLM judge.

Research basis:
- RLEF: Single 8B model learns code repair via execution feedback
- Open-RS: GRPO with group-relative advantage eliminates value model
- Uni-CoT: Unified reasoning-generation outperforms decoupled pipelines
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Protocol

try:
    from models import FlakeForgeAction, FlakeForgeObservation
except ImportError:
    from ..models import FlakeForgeAction, FlakeForgeObservation

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__)


class ModelBackend(Protocol):
    """Backend interface for unified model calls."""

    def generate(self, prompt: str, *, system_prompt: str) -> str:
        ...


UNIFIED_SYSTEM_PROMPT = """You are FlakeForge, an expert debugging agent that fixes flaky tests.
You receive a deep observation of a flaky test including source code, failure traces, 
run history, and deep codebase signals (module cache violations, fixture scope risks, 
mock residue, import side effects, async contamination).

You MUST respond with EXACTLY two XML blocks and no Markdown fences:

1. <think> block: Your root cause analysis
   - State the root cause category (one of: async_wait, concurrency, test_order_dependency,
     resource_leak, shared_state, network, platform_dependency, nondeterminism,
     import_side_effect, module_cache_pollution, fixture_scope_leak, mock_residue, unknown)
   - State your confidence (0.0 to 1.0)
   - Cite specific evidence from the observation (file names, line numbers, error messages)
   - Explain the causal mechanism: HOW does this cause lead to flakiness?
   - State your fix strategy in one sentence

2. <patch> block: Your code fix using search/replace format
   Each hunk targets a file and replaces exact text:
   --- path/to/file.py
   <<<<<<< SEARCH
   exact lines to find
   =======
   replacement lines
   >>>>>>> REPLACE

RULES:
- The <think> block must contain "Root Cause:" and "confidence:" 
- The <patch> block must contain valid search/replace hunks
- Do NOT wrap your answer in ```xml, ```patch, or any Markdown code fence
- SEARCH text must be copied exactly from the observation, including indentation
- Every SEARCH block must include ======= and >>>>>>> REPLACE
- Do NOT add sleep() calls or retry decorators to mask flakiness
- Do NOT modify unrelated files
- Prefer minimal, surgical fixes that address the root cause
- If the failure is in test code, fix the test. If in source code, fix the source.

Example response:
<think>
Root Cause: async_wait (confidence: 0.85)
Evidence: TimeoutError at test_flaky.py:12, asyncio.wait_for with timeout=0.05s
The coroutine fetch_data_with_race() has a race condition where the lock acquisition
can take longer than 50ms under load, causing intermittent TimeoutError.
Strategy: Increase timeout to accommodate lock contention latency.
</think>

<patch>
--- source.py
<<<<<<< SEARCH
        timeout = 0.05 if random.random() < 0.8 else 0.5
=======
        timeout = 0.5
>>>>>>> REPLACE
</patch>"""


def build_unified_prompt(observation: FlakeForgeObservation) -> str:
    """Build the observation prompt for the unified agent."""
    parts = ["=== FLAKY TEST OBSERVATION ===\n"]

    # Test identity
    parts.append(f"Test: {observation.test_identifier}")
    parts.append(f"Step: {observation.step}/{observation.step + observation.steps_remaining}")
    parts.append(f"Current pass rate: {observation.current_pass_rate:.2f}")
    parts.append(f"Baseline pass rate: {observation.baseline_pass_rate:.2f}")
    parts.append("")

    # Source code
    if observation.test_function_source:
        parts.append("=== TEST SOURCE ===")
        parts.append(observation.test_function_source[:3000])
        parts.append("")

    if observation.source_under_test:
        parts.append("=== SOURCE UNDER TEST ===")
        parts.append(observation.source_under_test[:3000])
        parts.append("")

    # Run history
    if observation.run_history:
        parts.append("=== RUN HISTORY (last 10) ===")
        for r in observation.run_history[-10:]:
            status = "PASS" if r.passed else f"FAIL({r.error_type or 'unknown'})"
            msg = f"  {status} [{r.duration_ms}ms]"
            if r.error_message:
                msg += f" - {r.error_message[:100]}"
            parts.append(msg)
        parts.append("")

    # Failing stack trace
    if observation.failing_stack_trace:
        parts.append("=== FAILING STACK TRACE ===")
        parts.append(observation.failing_stack_trace[:2000])
        parts.append("")

    # Deep flakiness signals
    deep_signals = []
    if observation.module_cache_violations:
        deep_signals.append(f"Module cache violations: {', '.join(observation.module_cache_violations[:5])}")
    if observation.fixture_scope_risks:
        deep_signals.append(f"Fixture scope risks: {', '.join(observation.fixture_scope_risks[:5])}")
    if observation.mock_residue_sites:
        deep_signals.append(f"Mock residue sites: {', '.join(observation.mock_residue_sites[:5])}")
    if observation.import_side_effect_files:
        deep_signals.append(f"Import side effects: {', '.join(observation.import_side_effect_files[:5])}")
    if observation.async_contamination_alive:
        deep_signals.append("ALERT: Async tasks/threads survived past test boundary!")

    if deep_signals:
        parts.append("=== DEEP FLAKINESS SIGNALS ===")
        parts.extend(deep_signals)
        parts.append("")

    # Causal frontier
    if observation.failure_frontier:
        parts.append("=== CAUSAL FRONTIER ===")
        parts.append(f"Failure site: {observation.failure_frontier}")
        if observation.call_chain_to_frontier:
            parts.append(f"Call chain: {' → '.join(observation.call_chain_to_frontier)}")
        if observation.boundary_crossings:
            parts.append(f"Boundary crossings: {', '.join(observation.boundary_crossings)}")
        parts.append("")

    # iDFlakies signals
    if observation.order_dependency_detected:
        parts.append("⚠️ ORDER DEPENDENCY: Test fails when run in reverse order")
    if observation.infrastructure_sensitive:
        parts.append("⚠️ INFRASTRUCTURE SENSITIVE: Test outcome changes under resource pressure")

    # File tree
    if observation.file_tree:
        parts.append("\n=== FILE TREE ===")
        parts.extend(observation.file_tree[:20])

    # Previous patches
    if observation.patches_applied:
        parts.append("\n=== PREVIOUS PATCHES (this episode) ===")
        for p in observation.patches_applied[-3:]:
            status = "✓" if p.applied_successfully else "✗"
            parts.append(f"  {status} {', '.join(p.target_files)} ({p.lines_changed} lines) → pass_rate={p.pass_rate_after:.2f}")

    # Previous reasoning
    if observation.last_think_text:
        parts.append("\n=== PREVIOUS REASONING ===")
        parts.append(observation.last_think_text[:500])

    parts.append("\n=== YOUR TURN ===")
    parts.append("Analyze the root cause and generate a fix patch.")

    return "\n".join(parts)


def _strip_markdown_fence(text: str) -> str:
    """Remove a single Markdown fence around model output if present."""
    stripped = text.strip()
    match = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def extract_think(response: str) -> str:
    """Extract content between <think> and </think> tags."""
    response = _strip_markdown_fence(response)
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback for model outputs that skip the exact XML wrapper.
    return response.strip()


def extract_patch(response: str) -> str:
    """Extract content between <patch> and </patch> tags."""
    response = _strip_markdown_fence(response)
    match = re.search(r"<patch>(.*?)</patch>", response, re.DOTALL)
    if match:
        return _strip_markdown_fence(match.group(1))

    fenced = re.search(r"```(?:patch|diff|xml)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        fenced_text = fenced.group(1).strip()
        patch_match = re.search(r"<patch>(.*?)</patch>", fenced_text, re.DOTALL)
        if patch_match:
            return _strip_markdown_fence(patch_match.group(1))
        return fenced_text

    # Last-resort tolerant parsing for models that emit the hunk without tags.
    hunk_start = response.find("<<<<<<< SEARCH")
    if hunk_start != -1:
        header_start = response.rfind("\n--- ", 0, hunk_start)
        start = header_start + 1 if header_start != -1 else hunk_start
        hunk_end = response.find(">>>>>>> REPLACE", hunk_start)
        if hunk_end != -1:
            return response[start:hunk_end + len(">>>>>>> REPLACE")].strip()

    return ""


def extract_category_from_think(think_text: str) -> str:
    """Extract predicted root cause category from think block."""
    match = re.search(r"Root\s*Cause:\s*(\w+)", think_text, re.IGNORECASE)
    if match:
        category = match.group(1).lower().strip()
        # Normalize common variations
        norm_map = {
            "async": "async_wait",
            "async_wait": "async_wait",
            "asyncwait": "async_wait",
            "timeout": "async_wait",
            "concurrency": "concurrency",
            "race": "concurrency",
            "race_condition": "concurrency",
            "threading": "concurrency",
            "order": "test_order_dependency",
            "test_order": "test_order_dependency",
            "order_dependency": "test_order_dependency",
            "resource": "resource_leak",
            "resource_leak": "resource_leak",
            "leak": "resource_leak",
            "shared": "shared_state",
            "shared_state": "shared_state",
            "state": "shared_state",
            "network": "network",
            "connection": "network",
            "http": "network",
            "platform": "platform_dependency",
            "platform_dependency": "platform_dependency",
            "nondeterminism": "nondeterminism",
            "random": "nondeterminism",
            "import": "import_side_effect",
            "import_side_effect": "import_side_effect",
            "cache": "module_cache_pollution",
            "module_cache": "module_cache_pollution",
            "module_cache_pollution": "module_cache_pollution",
            "fixture": "fixture_scope_leak",
            "fixture_scope": "fixture_scope_leak",
            "fixture_scope_leak": "fixture_scope_leak",
            "mock": "mock_residue",
            "mock_residue": "mock_residue",
            "monkeypatch": "mock_residue",
        }
        return norm_map.get(category, "unknown")
    return "unknown"


def extract_confidence_from_think(think_text: str) -> float:
    """Extract confidence value from think block."""
    match = re.search(r"confidence:\s*([\d.]+)", think_text, re.IGNORECASE)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            return 0.5
    return 0.5


def infer_category_from_patch(patch_text: str) -> str:
    """Infer actual fix category from what the patch modifies."""
    patch_lower = patch_text.lower()

    # Check for sleep/timeout modifications
    if "timeout" in patch_lower or "wait_for" in patch_lower:
        return "async_wait"
    if "sleep" in patch_lower and "asyncio" in patch_lower:
        return "async_wait"

    # Check for lock/synchronization
    if "lock" in patch_lower or "semaphore" in patch_lower or "barrier" in patch_lower:
        return "concurrency"

    # Check for fixture/teardown
    if "fixture" in patch_lower or "teardown" in patch_lower or "yield" in patch_lower:
        return "fixture_scope_leak"

    # Check for mock/patch
    if "mock" in patch_lower or "monkeypatch" in patch_lower or "patch(" in patch_lower:
        return "mock_residue"

    # Check for cache clear
    if "cache_clear" in patch_lower or "lru_cache" in patch_lower:
        return "module_cache_pollution"

    # Check for state reset
    if "clear()" in patch_lower or "reset" in patch_lower:
        return "shared_state"

    # Check for import refactoring
    if "if __name__" in patch_lower or "import" in patch_lower:
        return "import_side_effect"

    # Check for seed/random
    if "seed(" in patch_lower or "random" in patch_lower:
        return "nondeterminism"

    return "unknown"


class UnifiedFlakeForgeAgent:
    """V3 Unified agent — single model, single forward pass.

    Generates <think> + <patch> in one completion. The reasoning block
    is part of the context for patch generation, so the attention mechanism
    learns which reasoning tokens correlate with successful patches.
    """

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self.system_prompt = UNIFIED_SYSTEM_PROMPT

    def generate(self, observation: FlakeForgeObservation) -> FlakeForgeAction:
        """Generate a unified think+patch response from the observation."""
        prompt = build_unified_prompt(observation)

        raw_response = self.backend.generate(
            prompt,
            system_prompt=self.system_prompt,
        )

        think_text = extract_think(raw_response)
        patch_text = extract_patch(raw_response)
        predicted_category = extract_category_from_think(think_text)
        predicted_confidence = extract_confidence_from_think(think_text)

        if "<think>" not in raw_response or "<patch>" not in raw_response:
            logger.warning(
                "[UNIFIED_AGENT] Model output did not include expected tags; falling back to tolerant parsing."
            )

        logger.info(
            "[UNIFIED_AGENT] category=%s confidence=%.2f patch_len=%d",
            predicted_category,
            predicted_confidence,
            len(patch_text),
        )

        return FlakeForgeAction(
            raw_response=raw_response,
            think_text=think_text,
            patch_text=patch_text,
            predicted_category=predicted_category,
            predicted_confidence=predicted_confidence,
            action_type="UNIFIED_PATCH",
            parameters={},
        )
