"""V3 Unified FlakeForge Agent — single model, single forward pass.

The <think> block is now structured JSON (a list of ThinkClaim objects).
The parser tries JSON first; on any parse failure it degrades gracefully
and records a format_penalty on the StructuredThink so the reward system
can penalise malformed output without raising an error.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol

try:
    from models import FlakeForgeAction, FlakeForgeObservation, StructuredThink, ThinkClaim
except ImportError:
    from ..models import (
        FlakeForgeAction, FlakeForgeObservation,
        StructuredThink, ThinkClaim,
        StructuredPatch, PatchHunk,
    )

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__)


# ── Protocol ─────────────────────────────────────────────────────────────────

class ModelBackend(Protocol):
    def generate(self, prompt: str, *, system_prompt: str) -> str: ...


# ── System prompt ─────────────────────────────────────────────────────────────

_CLAIM_SCHEMA = """{
  "claims": [
    {
      "claim_id": "c1",
      "category": "<one of ROOT_CAUSE_TYPES>",
      "entity": "<function / class / variable name>",
      "location": "<path/to/file.py::ClassName.method OR path/to/file.py::function>",
      "ast_node_type": "<optional libcst node type, e.g. FunctionDef>",
      "polarity": "present",
      "predicted_effect": "<one sentence: what pass-rate change do you expect?>",
      "reason": "<≤40 words: causal justification>"
    }
  ],
  "confidence": 0.85
}"""

_HUNK_SCHEMA = """{
  "hunks": [
    {
      "hunk_id": "h1",
      "file": "<repo-relative path, e.g. pybrake/notifier.py>",
      "search": "<exact lines to find, verbatim including indentation>",
      "replace": "<replacement lines (empty string to delete the block)>",
      "rationale": "<one sentence: why does this fix the root cause?>",
      "addresses_claim": "<claim_id from the think block, e.g. c1>"
    }
  ]
}"""

UNIFIED_SYSTEM_PROMPT = f"""You are FlakeForge, an expert debugging agent that identifies and fixes unreliable tests.

You MUST respond with EXACTLY two XML blocks and no Markdown fences:

1. <think> block — MUST be valid JSON matching this schema exactly:
{_CLAIM_SCHEMA}

   ROOT_CAUSE_TYPES: async_wait, concurrency, test_order_dependency,
   resource_leak, shared_state, network, platform_dependency, nondeterminism,
   import_side_effect, module_cache_pollution, fixture_scope_leak, mock_residue, unknown

   Rules for the <think> JSON:
   - claims is a non-empty list.
   - polarity is "present" when you assert the bug exists in the current code.
   - predicted_effect is mandatory; forecast the expected pass-rate change.
   - Do NOT add extra keys; they will be stripped.

2. <patch> block — MUST be valid JSON matching this schema exactly:
{_HUNK_SCHEMA}

   Rules for the <patch> JSON:
   - hunks is a non-empty list of search/replace operations.
   - search must be copied verbatim from the source shown in the observation (preserve indentation).
   - replace is the fixed replacement; use an empty string "" to delete the search block.
   - addresses_claim links this hunk to a claim_id from the <think> block.
   - Do NOT add extra keys.

GLOBAL RULES:
- Do NOT wrap your answer in Markdown fences (no ```).
- Do NOT add sleep() calls, retry decorators, or @pytest.mark.skip.
- Prefer minimal, surgical fixes that address the root cause.
- If uncertain about root cause, use category "unknown" rather than guessing.

PENALTY: Any block that is not valid JSON receives a format penalty that reduces your reward.
"""


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_unified_prompt(observation: FlakeForgeObservation) -> str:
    parts = ["=== FLAKY TEST OBSERVATION ===\n"]

    parts.append(f"Test: {observation.test_identifier}")
    parts.append(f"Step: {observation.step}/{observation.step + observation.steps_remaining}")
    parts.append(f"Current pass rate: {observation.current_pass_rate:.2f}")
    parts.append(f"Baseline pass rate: {observation.baseline_pass_rate:.2f}")
    parts.append("")

    if observation.test_function_source:
        parts.append("=== TEST SOURCE ===")
        parts.append(observation.test_function_source[:3000])
        parts.append("")

    if observation.source_under_test:
        parts.append("=== SOURCE UNDER TEST ===")
        parts.append(observation.source_under_test[:3000])
        parts.append("")

    if observation.run_history:
        parts.append("=== RUN HISTORY (last 10) ===")
        for r in observation.run_history[-10:]:
            status = "PASS" if r.passed else f"FAIL({r.error_type or 'unknown'})"
            msg = f"  {status} [{r.duration_ms}ms]"
            if r.error_message:
                msg += f" - {r.error_message[:100]}"
            parts.append(msg)
        parts.append("")

    if observation.failing_stack_trace:
        parts.append("=== FAILING STACK TRACE ===")
        parts.append(observation.failing_stack_trace[:2000])
        parts.append("")

    deep: List[str] = []
    if observation.module_cache_violations:
        deep.append(f"Module cache violations: {', '.join(observation.module_cache_violations[:5])}")
    if observation.fixture_scope_risks:
        deep.append(f"Fixture scope risks: {', '.join(observation.fixture_scope_risks[:5])}")
    if observation.mock_residue_sites:
        deep.append(f"Mock residue sites: {', '.join(observation.mock_residue_sites[:5])}")
    if observation.import_side_effect_files:
        deep.append(f"Import side effects: {', '.join(observation.import_side_effect_files[:5])}")
    if observation.async_contamination_alive:
        deep.append("ALERT: Async tasks/threads survived past test boundary!")
    if deep:
        parts.append("=== DEEP FLAKINESS SIGNALS ===")
        parts.extend(deep)
        parts.append("")

    if observation.failure_frontier:
        parts.append("=== CAUSAL FRONTIER ===")
        parts.append(f"Failure site: {observation.failure_frontier}")
        if observation.call_chain_to_frontier:
            parts.append(f"Call chain: {' → '.join(observation.call_chain_to_frontier)}")
        if observation.boundary_crossings:
            parts.append(f"Boundary crossings: {', '.join(observation.boundary_crossings)}")
        parts.append("")

    if observation.order_dependency_detected:
        parts.append("⚠️ ORDER DEPENDENCY: Test fails when run in reverse order")
    if observation.infrastructure_sensitive:
        parts.append("⚠️ INFRASTRUCTURE SENSITIVE: Test outcome changes under resource pressure")

    if observation.file_tree:
        parts.append("\n=== FILE TREE ===")
        parts.extend(observation.file_tree[:20])

    if observation.patches_applied:
        parts.append("\n=== PREVIOUS PATCHES (this episode) ===")
        for p in observation.patches_applied[-3:]:
            s = "✓" if p.applied_successfully else "✗"
            parts.append(
                f"  {s} {', '.join(p.target_files)} ({p.lines_changed} lines)"
                f" → pass_rate={p.pass_rate_after:.2f}"
            )

    if observation.last_think_text:
        parts.append("\n=== PREVIOUS REASONING (JSON) ===")
        parts.append(observation.last_think_text[:800])

    parts.append("\n=== YOUR TURN ===")
    parts.append(
        "Output <think> JSON claims then <patch> hunks. No Markdown fences."
    )
    return "\n".join(parts)


# ── Patch extraction (unchanged search/replace format) ────────────────────────

def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    m = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", stripped, re.DOTALL)
    return m.group(1).strip() if m else stripped


def extract_patch(response: str) -> str:
    """Extract content between <patch> and </patch> tags."""
    response = _strip_markdown_fence(response)

    m = re.search(r"<patch>(.*?)</patch>", response, re.DOTALL)
    if m:
        return _strip_markdown_fence(m.group(1))

    fenced = re.search(r"```(?:patch|diff|xml)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        inner = fenced.group(1).strip()
        pm = re.search(r"<patch>(.*?)</patch>", inner, re.DOTALL)
        if pm:
            return _strip_markdown_fence(pm.group(1))
        return inner

    hunk_start = response.find("<<<<<<< SEARCH")
    if hunk_start != -1:
        header_start = response.rfind("\n--- ", 0, hunk_start)
        start = header_start + 1 if header_start != -1 else hunk_start
        hunk_end = response.find(">>>>>>> REPLACE", hunk_start)
        if hunk_end != -1:
            return response[start : hunk_end + len(">>>>>>> REPLACE")].strip()
    return ""


# ── Structured think parsing ──────────────────────────────────────────────────

_VALID_CATEGORIES = {
    "async_wait", "concurrency", "test_order_dependency", "resource_leak",
    "shared_state", "network", "platform_dependency", "nondeterminism",
    "import_side_effect", "module_cache_pollution", "fixture_scope_leak",
    "mock_residue", "unknown",
}

_NORM_MAP: Dict[str, str] = {
    "async": "async_wait", "asyncwait": "async_wait", "timeout": "async_wait",
    "race": "concurrency", "race_condition": "concurrency", "threading": "concurrency",
    "order": "test_order_dependency", "test_order": "test_order_dependency",
    "order_dependency": "test_order_dependency",
    "resource": "resource_leak", "leak": "resource_leak",
    "shared": "shared_state", "state": "shared_state",
    "connection": "network", "http": "network",
    "platform": "platform_dependency",
    "random": "nondeterminism",
    "import": "import_side_effect",
    "cache": "module_cache_pollution", "module_cache": "module_cache_pollution",
    "fixture": "fixture_scope_leak", "fixture_scope": "fixture_scope_leak",
    "mock": "mock_residue", "monkeypatch": "mock_residue",
}

_ALLOWED_HUNK_KEYS = {
    "hunk_id", "file", "search", "replace", "rationale", "addresses_claim",
}


def _normalise_category(raw: str) -> str:
    c = raw.lower().strip()
    if c in _VALID_CATEGORIES:
        return c
    return _NORM_MAP.get(c, "unknown")


def _parse_structured_think(think_raw: str) -> StructuredThink:
    """Try to parse think_raw as JSON StructuredThink.

    On any failure: return a StructuredThink with format_penalty=-1.0 and
    whatever claims could be salvaged.  Never raises.
    """
    text = think_raw.strip()

    # Model sometimes wraps in a code fence inside the think block.
    text = _strip_markdown_fence(text)

    # Some models add a trailing comment or text after the closing brace.
    # Try to extract the outermost JSON object.
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return StructuredThink(format_penalty=-1.0)

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return StructuredThink(format_penalty=-1.0)

    claims: List[ThinkClaim] = []
    raw_claims = data.get("claims", [])

    if not isinstance(raw_claims, list):
        return StructuredThink(format_penalty=-1.0)

    partial_penalty = 0.0
    for i, rc in enumerate(raw_claims):
        if not isinstance(rc, dict):
            partial_penalty = -0.3
            continue
        # Strip unknown keys (rather than erroring).
        clean = {k: v for k, v in rc.items() if k in _ALLOWED_CLAIM_KEYS}
        # Normalise category.
        if "category" in clean:
            clean["category"] = _normalise_category(clean["category"])
        # Ensure polarity is valid.
        if clean.get("polarity") not in ("present", "absent"):
            clean["polarity"] = "present"
            partial_penalty = min(partial_penalty, -0.1)
        # Auto-fill missing optional fields.
        clean.setdefault("claim_id", f"c{i+1}")
        clean.setdefault("entity", "")
        clean.setdefault("location", "")
        clean.setdefault("ast_node_type", "")
        clean.setdefault("predicted_effect", "")
        clean.setdefault("reason", "")
        try:
            claims.append(ThinkClaim(**clean))
        except Exception:
            partial_penalty = -0.3
            continue

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5
        partial_penalty = min(partial_penalty, -0.1)

    if not claims:
        return StructuredThink(confidence=confidence, format_penalty=-1.0)

    return StructuredThink(claims=claims, confidence=confidence, format_penalty=partial_penalty)


_ALLOWED_CLAIM_KEYS = {
    "claim_id", "category", "entity", "location",
    "ast_node_type", "polarity", "predicted_effect", "reason",
}


def _parse_structured_patch(patch_raw: str) -> "StructuredPatch":
    """Try to parse patch_raw as JSON StructuredPatch.

    On any failure: return a StructuredPatch with format_penalty=-1.0 and
    whatever hunks could be salvaged.  Never raises.
    """
    text = patch_raw.strip()
    text = _strip_markdown_fence(text)

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return StructuredPatch(format_penalty=-1.0)

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return StructuredPatch(format_penalty=-1.0)

    hunks: List[PatchHunk] = []
    raw_hunks = data.get("hunks", [])

    if not isinstance(raw_hunks, list):
        return StructuredPatch(format_penalty=-1.0)

    partial_penalty = 0.0
    for i, rh in enumerate(raw_hunks):
        if not isinstance(rh, dict):
            partial_penalty = -0.3
            continue
        clean = {k: v for k, v in rh.items() if k in _ALLOWED_HUNK_KEYS}
        clean.setdefault("hunk_id", f"h{i+1}")
        clean.setdefault("rationale", "")
        clean.setdefault("addresses_claim", "")
        # Mandatory fields — skip hunk if missing.
        if "file" not in clean or "search" not in clean or "replace" not in clean:
            partial_penalty = min(partial_penalty, -0.3)
            continue
        try:
            hunks.append(PatchHunk(**clean))
        except Exception:
            partial_penalty = -0.3
            continue

    if not hunks:
        return StructuredPatch(format_penalty=-1.0)

    return StructuredPatch(hunks=hunks, format_penalty=partial_penalty)



def extract_think(response: str) -> str:
    """Extract raw text between <think> and </think> tags (for logging / prompts)."""
    response = _strip_markdown_fence(response)
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip()


# ── Backward-compat helpers (still used by reward.py / tests) ─────────────────

def extract_category_from_think(think_text: str) -> str:
    """Extract predicted root cause from structured JSON or legacy free-form text."""
    st = _parse_structured_think(think_text)
    if st.claims and st.format_penalty == 0.0:
        return st.primary_category
    # Legacy path: regex on free-form text.
    m = re.search(r"Root\s*Cause:\s*(\w+)", think_text, re.IGNORECASE)
    if m:
        return _normalise_category(m.group(1))
    return "unknown"


def extract_confidence_from_think(think_text: str) -> float:
    """Extract confidence from structured JSON or legacy free-form text."""
    st = _parse_structured_think(think_text)
    if st.claims and st.format_penalty == 0.0:
        return st.confidence
    # Legacy regex fallback.
    m = re.search(r"confidence:\s*([\d.]+)", think_text, re.IGNORECASE)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    return 0.5


def infer_category_from_patch(patch_text: str) -> str:
    """Infer actual fix category from what the patch modifies (keyword scan)."""
    p = patch_text.lower()
    if "timeout" in p or "wait_for" in p:
        return "async_wait"
    if "sleep" in p and "asyncio" in p:
        return "async_wait"
    if "lock" in p or "semaphore" in p or "barrier" in p:
        return "concurrency"
    if "fixture" in p or "teardown" in p or "yield" in p:
        return "fixture_scope_leak"
    if "mock" in p or "monkeypatch" in p or "patch(" in p:
        return "mock_residue"
    if "cache_clear" in p or "lru_cache" in p:
        return "module_cache_pollution"
    if "clear()" in p or "reset" in p:
        return "shared_state"
    if "if __name__" in p or "import" in p:
        return "import_side_effect"
    if "seed(" in p or "random" in p:
        return "nondeterminism"
    return "unknown"


# ── Agent ─────────────────────────────────────────────────────────────────────

class UnifiedFlakeForgeAgent:
    """V3 Unified agent — one model, one forward pass.

    The <think> block is expected to be structured JSON.  If the model emits
    well-formed JSON the structured_think field on the returned action is
    populated and the oracle can verify it.  If not, a format_penalty is
    recorded and backward-compat regex parsers are used as a fallback so the
    episode does not crash.
    """

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self.system_prompt = UNIFIED_SYSTEM_PROMPT

    def generate(self, observation: FlakeForgeObservation) -> FlakeForgeAction:
        prompt = build_unified_prompt(observation)
        raw_response = self.backend.generate(prompt, system_prompt=self.system_prompt)

        think_raw = extract_think(raw_response)
        patch_raw = extract_patch(raw_response)

        # Try structured JSON parse for both blocks; never raises.
        structured_think = _parse_structured_think(think_raw)
        structured_patch = _parse_structured_patch(patch_raw)

        # Derive scalar fields from structured parse, falling back to regex.
        predicted_category = (
            structured_think.primary_category
            if structured_think.claims
            else extract_category_from_think(think_raw)
        )
        predicted_confidence = (
            structured_think.confidence
            if structured_think.claims
            else extract_confidence_from_think(think_raw)
        )

        if "<think>" not in raw_response or "<patch>" not in raw_response:
            logger.warning(
                "[UNIFIED_AGENT] Model output missing expected XML tags; using tolerant parsing."
            )

        if structured_think.format_penalty < 0.0:
            logger.warning(
                "[UNIFIED_AGENT] <think> block is not valid JSON (penalty=%.1f); "
                "falling back to legacy parsing.",
                structured_think.format_penalty,
            )

        if structured_patch.format_penalty < 0.0:
            logger.warning(
                "[UNIFIED_AGENT] <patch> block is not valid JSON (penalty=%.1f); "
                "patch will not have structured hunk fields.",
                structured_patch.format_penalty,
            )

        logger.info(
            "[UNIFIED_AGENT] category=%s confidence=%.2f patch_hunks=%d think_json=%s patch_json=%s",
            predicted_category,
            predicted_confidence,
            len(structured_patch.hunks),
            structured_think.format_penalty == 0.0,
            structured_patch.format_penalty == 0.0,
        )

        return FlakeForgeAction(
            raw_response=raw_response,
            think_text=think_raw,
            patch_text=patch_raw,  # raw fallback
            structured_think=structured_think,
            structured_patch=structured_patch,
            predicted_category=predicted_category,
            predicted_confidence=predicted_confidence,
            action_type="UNIFIED_PATCH",
            parameters={},
        )
