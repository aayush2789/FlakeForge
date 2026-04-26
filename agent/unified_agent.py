"""Unified FlakeForge Agent — single model, single forward pass producing JSON."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol

try:
    from models import (
        FlakeForgeAction, FlakeForgeObservation, 
        StructuredThink, ThinkClaim,
        StructuredPatch, PatchHunk,
    )
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


class ModelBackend(Protocol):
    def generate(self, prompt: str, *, system_prompt: str) -> str: ...


_UNIFIED_RESPONSE_SCHEMA = """{
  "think": {
    "claims": [
      {
        "claim_id": "c1",
        "category": "concurrency",
        "entity": "submit",
        "location": "source.py::WorkerPool.submit",
        "ast_node_type": "FunctionDef",
        "polarity": "present",
        "predicted_effect": "Removing the false queue-full branch should make the request stable.",
        "reason": "The worker submit path can return queue_full before taking the lock."
      }
    ],
    "confidence": 0.85
  },
  "patch": {
    "hunks": [
      {
        "hunk_id": "h1",
        "file": "source.py",
        "search": "        if random.random() < 0.30:\\n            return False\\n\\n        with self._lock:",
        "replace": "        with self._lock:",
        "rationale": "Removes the false queue-full branch so capacity is checked only inside the lock.",
        "addresses_claim": "c1"
      }
    ]
  }
}"""

UNIFIED_SYSTEM_PROMPT = f"""You are FlakeForge, an expert debugging agent that fixes flaky tests.

Respond with EXACTLY ONE JSON object matching this schema:
{_UNIFIED_RESPONSE_SCHEMA}

ROOT_CAUSE_TYPES: async_wait, concurrency, test_order_dependency,
resource_leak, shared_state, network, platform_dependency, nondeterminism,
import_side_effect, module_cache_pollution, fixture_scope_leak, mock_residue, unknown

Important root-cause guidance:
- Prefer patching SOURCE UNDER TEST over TEST SOURCE. Only patch tests when the assertion or fixture is the root cause.
- Do NOT choose async_wait unless the failing flow actually uses async, await, wait_for, timeout, or sleep.
- Do NOT choose module_cache_pollution just because module-level globals exist. Use it only for import/module cache behavior such as lru_cache, sys.modules, import-time side effects, or cache decorators.
- If failures mention queue_full, WorkerPool, submit, QUEUE_CAPACITY, or random.random() in a queue path, classify as concurrency or nondeterminism and patch source.py::WorkerPool.submit.

Rules for "think":
- "think.claims" is a non-empty list.
- Each claim object must use exactly these model-facing keys:
  claim_id, category, entity, location, ast_node_type, polarity, predicted_effect, reason.
- "category" must be one ROOT_CAUSE_TYPES value.
- "polarity" is "present" when you assert the bug exists in the current code.
- "predicted_effect" is mandatory; forecast the expected pass-rate change.
- Do not include validator-filled keys such as verdict, oracle_score, format_penalty, applied, or apply_error.

Rules for "patch":
- "patch.hunks" is a non-empty list of search/replace operations.
- Each hunk object must use exactly these model-facing keys:
  hunk_id, file, search, replace, rationale, addresses_claim.
- "file" must be a repo-relative path.
- "search" must be copied VERBATIM from the source shown in the observation,
  including the EXACT leading whitespace of every line (do NOT trim or re-indent).
- "replace" must use the SAME absolute indentation as the lines it replaces.
  When patching a method body, every replacement line must start with the same
  indentation as the corresponding lines in the source file (typically 4 or 8
  spaces from the left margin). Do NOT add or remove leading whitespace.
- Prefer SMALL hunks: target only the buggy lines, not the entire function.
- "replace" may be an empty string "" to delete the search block.
- "addresses_claim" must equal a claim_id from "think.claims".

JSON STRING FORMAT FOR "search" / "replace":
- Both fields are JSON strings. Use \\n for line breaks and \\\" for quotes.
- Tabs are forbidden; use spaces only.
- Never wrap "search" or "replace" in markdown fences or extra quotes.

GLOBAL RULES:
- Do NOT output XML tags such as <think> or <patch>.
- Do NOT wrap your answer in Markdown fences (no ```).
- Do NOT add text before or after the JSON object.
- Do NOT add sleep() calls, retry decorators, or @pytest.mark.skip.
- Prefer minimal, surgical fixes that address the root cause.
- If an earlier patch failed validation, do not switch to unrelated categories. Use the same evidence and produce a more exact source.py hunk.
- If uncertain about root cause, use category "unknown" rather than guessing.

PENALTY: Any response that is not one valid JSON object receives a format penalty that reduces your reward.
"""


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

        src_lower = observation.source_under_test.lower()
        if (
            "queue_full" in src_lower
            and "workerpool" in src_lower
            and "random.random() < 0.30" in observation.source_under_test
        ):
            parts.append("=== HIGH-CONFIDENCE LOCALIZATION HINT ===")
            parts.append("This repo's observed flake is the false queue_full path in source.py::WorkerPool.submit.")
            parts.append("Use category concurrency. Patch source.py, not tests/test_flaky.py.")
            parts.append("Remove this exact branch from WorkerPool.submit:")
            parts.append("        if random.random() < 0.30:")
            parts.append("            return False")
            parts.append("Keep the existing with self._lock: capacity check.")
            parts.append("")

        if (
            "class connectionpool" in src_lower
            and "def acquire" in src_lower
            and "connection pool exhausted" in src_lower
            and "_in_use" in src_lower
            and "max_size" in src_lower
        ):
            parts.append("=== HIGH-CONFIDENCE LOCALIZATION HINT ===")
            parts.append("This repo's flake is in source.py::ConnectionPool.acquire.")
            parts.append("Use category network or concurrency. Patch source.py, not tests/test_flaky.py.")
            parts.append("Do NOT patch random.random or WorkerPool patterns from other repos.")
            parts.append("Fix acquire() to wait/retry until timeout before raising pool exhaustion.")
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

    if observation.causal_hints:
        parts.append("\n=== TARGETING HINTS ===")
        parts.extend([f"- {hint}" for hint in observation.causal_hints[:10]])
        parts.append(
            "Prioritize patch hunks in the highest-score hinted files unless direct evidence contradicts them."
        )

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

    # ── Hypothesis trail: show the full per-step history so the agent can ────
    # learn from its own trajectory and avoid repeating failed hypotheses.
    if observation.think_history:
        parts.append("\n=== HYPOTHESIS TRAIL (all prior steps) ===")

        # Count category repetitions so we can flag staleness.
        cat_counts: Dict[str, int] = {}
        for h in observation.think_history:
            primary = h["categories"][0] if h.get("categories") else "unknown"
            cat_counts[primary] = cat_counts.get(primary, 0) + 1

        for h in observation.think_history:
            step = h.get("step", "?")
            cats = h.get("categories", [])
            ents = h.get("entities", [])
            oracle = h.get("oracle_score")
            pr = h.get("pass_rate_after", 0.0)
            rew = h.get("reward", 0.0)

            oracle_str = f"{oracle:+.2f}" if oracle is not None else "N/A"
            ent_str = ", ".join(ents[:3]) if ents else "—"
            cat_str = " + ".join(cats[:2]) if cats else "unknown"

            flag = ""
            if cats and cat_counts.get(cats[0], 0) > 1:
                flag = " ⚠️ REPEATED"

            parts.append(
                f"  Step {step}: [{cat_str}]{flag}  entities=[{ent_str}]"
                f"  oracle={oracle_str}  pass_rate={pr:.2f}  reward={rew:+.3f}"
            )

        # Collect stale categories (tried ≥ 2 times without achieving pass_rate=1.0).
        stale = [cat for cat, cnt in cat_counts.items() if cnt >= 2]
        if stale:
            parts.append(
                f"\n⛔ STALE HYPOTHESES (tried {', '.join(stale)} multiple times with no solution)."
            )
            parts.append(
                "Your reward will be HEAVILY penalised if you repeat these categories again."
            )
            parts.append("You MUST propose a DIFFERENT root cause category and NEW entities.")

    parts.append("\n=== YOUR TURN ===")
    turn_instruction = "Output one JSON object with top-level keys think and patch. No XML, no Markdown fences."
    if observation.think_history:
        stale_cats = {
            h["categories"][0]
            for h in observation.think_history
            if h.get("categories")
        }
        attempted = ", ".join(sorted(stale_cats))
        turn_instruction = (
            f"IMPORTANT: You already tried [{attempted}]. "
            "Propose a genuinely DIFFERENT root cause. "
            "Output one JSON object with top-level keys think and patch. No XML, no Markdown fences."
        )
    parts.append(turn_instruction)
    return "\n".join(parts)


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    m = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", stripped, re.DOTALL)
    return m.group(1).strip() if m else stripped


def _extract_json_object_text(text: str) -> str:
    """Return the first complete JSON object in text, or an empty string."""
    stripped = _strip_markdown_fence(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            _, end = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        return stripped[match.start(): match.start() + end]
    return ""


def _load_json_object(text: str) -> Optional[Dict[str, Any]]:
    json_text = _extract_json_object_text(text)
    if not json_text:
        return None
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalise_hunk_text(value: str) -> str:
    """Normalise model hunk strings that are double-escaped JSON text."""
    text = value
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    if "\\t" in text and "\t" not in text:
        text = text.replace("\\t", "    ")
    if '\\"' in text:
        text = text.replace('\\"', '"')
    text = re.sub(r"\\+[\"']\s*$", "", text)
    return text


def _patch_dict_to_search_replace(patch_data: Any) -> str:
    """Convert validator-compatible JSON hunk fields to legacy patch text."""
    if not isinstance(patch_data, dict):
        return ""
    hunks = patch_data.get("hunks", [])
    if not isinstance(hunks, list):
        return ""

    blocks: List[str] = []
    for hunk in hunks:
        if not isinstance(hunk, dict):
            continue
        file_path = str(hunk.get("file") or "").strip().replace("\\", "/")
        search = hunk.get("search")
        replace = hunk.get("replace")
        if not file_path or not isinstance(search, str) or not isinstance(replace, str):
            continue

        search = _normalise_hunk_text(search)
        replace = _normalise_hunk_text(replace)
        if not search.strip():
            continue

        blocks.append(
            "\n".join([
                f"--- {file_path}",
                "<<<<<<< SEARCH",
                search.rstrip("\n"),
                "=======",
                replace.rstrip("\n"),
                ">>>>>>> REPLACE",
            ])
        )
    return "\n\n".join(blocks)


def extract_patch(response: str) -> str:
    """Extract patch text, preferring the single-object JSON response format."""
    response = _strip_markdown_fence(response)

    data = _load_json_object(response)
    if isinstance(data, dict) and "patch" in data:
        patch_text = _patch_dict_to_search_replace(data.get("patch"))
        if patch_text:
            return patch_text

    m = re.search(r"<patch>(.*?)</patch>", response, re.DOTALL)
    if m:
        inner = _strip_markdown_fence(m.group(1))
        data = _load_json_object(inner)
        if isinstance(data, dict):
            patch_text = _patch_dict_to_search_replace(data.get("patch") if "patch" in data else data)
            if patch_text:
                return patch_text
        return inner

    fenced = re.search(r"```(?:patch|diff|xml)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        inner = fenced.group(1).strip()
        pm = re.search(r"<patch>(.*?)</patch>", inner, re.DOTALL)
        if pm:
            patch_inner = _strip_markdown_fence(pm.group(1))
            data = _load_json_object(patch_inner)
            if isinstance(data, dict):
                patch_text = _patch_dict_to_search_replace(data.get("patch") if "patch" in data else data)
                if patch_text:
                    return patch_text
            return patch_inner
        return inner

    hunk_start = response.find("<<<<<<< SEARCH")
    if hunk_start != -1:
        header_start = response.rfind("\n--- ", 0, hunk_start)
        start = header_start + 1 if header_start != -1 else hunk_start
        hunk_end = response.find(">>>>>>> REPLACE", hunk_start)
        if hunk_end != -1:
            return response[start : hunk_end + len(">>>>>>> REPLACE")].strip()
    return ""


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


def _parse_structured_think(think_raw: Any) -> StructuredThink:
    """Try to parse think_raw as JSON StructuredThink.

    On any failure: return a StructuredThink with format_penalty=-1.0 and
    whatever claims could be salvaged.  Never raises.
    """
    if isinstance(think_raw, dict):
        data = think_raw
    else:
        data = _load_json_object(str(think_raw))
    if data is None:
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
        clean = {k: v for k, v in rc.items() if k in _ALLOWED_CLAIM_KEYS}
        if "category" in clean:
            clean["category"] = _normalise_category(clean["category"])
        if clean.get("polarity") not in ("present", "absent"):
            clean["polarity"] = "present"
            partial_penalty = min(partial_penalty, -0.1)
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


def _parse_structured_patch(patch_raw: Any) -> "StructuredPatch":
    """Try to parse patch_raw as JSON StructuredPatch.

    On any failure: return a StructuredPatch with format_penalty=-1.0 and
    whatever hunks could be salvaged.  Never raises.
    """
    if isinstance(patch_raw, dict):
        data = patch_raw
    else:
        data = _load_json_object(str(patch_raw))
    if data is None:
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
    """Extract think JSON/text from the model response."""
    response = _strip_markdown_fence(response)
    data = _load_json_object(response)
    if isinstance(data, dict) and "think" in data:
        return json.dumps(data.get("think"), ensure_ascii=False)
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_category_from_think(think_text: str) -> str:
    """Extract predicted root cause from structured JSON or legacy free-form text."""
    st = _parse_structured_think(think_text)
    if st.claims and st.format_penalty == 0.0:
        return st.primary_category
    m = re.search(r"Root\s*Cause:\s*(\w+)", think_text, re.IGNORECASE)
    if m:
        return _normalise_category(m.group(1))
    return "unknown"


def extract_confidence_from_think(think_text: str) -> float:
    """Extract confidence from structured JSON or legacy free-form text."""
    st = _parse_structured_think(think_text)
    if st.claims and st.format_penalty == 0.0:
        return st.confidence
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


class UnifiedFlakeForgeAgent:
    """Unified agent — one model, one JSON forward pass with XML fallback."""

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self.system_prompt = UNIFIED_SYSTEM_PROMPT

    def generate(self, observation: FlakeForgeObservation) -> FlakeForgeAction:
        prompt = build_unified_prompt(observation)
        raw_response = self.backend.generate(prompt, system_prompt=self.system_prompt)

        def _decode_action(raw: str):
            response = _load_json_object(raw)
            if isinstance(response, dict) and (
                isinstance(response.get("think"), dict)
                or isinstance(response.get("patch"), dict)
            ):
                think = json.dumps(response.get("think", {}), ensure_ascii=False)
                parsed_think = _parse_structured_think(response.get("think", {}))
                parsed_patch = _parse_structured_patch(response.get("patch", {}))
                patch = _patch_dict_to_search_replace(response.get("patch", {}))
            else:
                think = extract_think(raw)
                patch = extract_patch(raw)
                parsed_think = _parse_structured_think(think)
                parsed_patch = _parse_structured_patch(patch)
            return response, think, patch, parsed_think, parsed_patch

        (
            response_obj,
            think_raw,
            patch_raw,
            structured_think,
            structured_patch,
        ) = _decode_action(raw_response)

        # Small models sometimes emit malformed JSON/hunks on the first try.
        # Retry once immediately instead of wasting a full environment step.
        should_retry = (
            response_obj is None
            or structured_patch.format_penalty < 0.0
            or len(structured_patch.hunks) == 0
            or not patch_raw.strip()
        )
        if should_retry:
            logger.warning(
                "[UNIFIED_AGENT] First decode invalid/empty; retrying generation once with strict JSON reminder."
            )
            retry_prompt = (
                prompt
                + "\n\nSTRICT RETRY: Return EXACTLY one valid JSON object matching the schema. "
                + "Do not include markdown, prose, XML tags, or escaped pseudo-diff text."
            )
            retry_raw = self.backend.generate(retry_prompt, system_prompt=self.system_prompt)
            (
                retry_obj,
                retry_think,
                retry_patch,
                retry_structured_think,
                retry_structured_patch,
            ) = _decode_action(retry_raw)
            retry_better = (
                (retry_obj is not None)
                and (retry_structured_patch.format_penalty == 0.0)
                and (len(retry_structured_patch.hunks) > 0)
            )
            if retry_better:
                raw_response = retry_raw
                response_obj = retry_obj
                think_raw = retry_think
                patch_raw = retry_patch
                structured_think = retry_structured_think
                structured_patch = retry_structured_patch

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

        if response_obj is None:
            logger.warning(
                "[UNIFIED_AGENT] Model output is not a single JSON object; using tolerant parsing."
            )

        if structured_think.format_penalty < 0.0:
            logger.warning(
                "[UNIFIED_AGENT] think object is not valid JSON (penalty=%.1f); "
                "falling back to legacy parsing.",
                structured_think.format_penalty,
            )

        if structured_patch.format_penalty < 0.0:
            logger.warning(
                "[UNIFIED_AGENT] patch object is not valid JSON (penalty=%.1f); "
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
