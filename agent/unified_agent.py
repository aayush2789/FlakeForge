"""Unified FlakeForge Agent — single model, single forward pass producing JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path
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

# Fuzzy/approximate "search" is OK — applier can align indentation; line_number mode is allowed.


def _target_line_snippets(source: str, limit: int = 6) -> List[str]:
    """Short lines the model is likely to edit (timeouts, async, etc.)."""
    if not source:
        return []
    out: List[str] = []
    keys = (
        "timeout", "wait_for", "asyncio", "await ", "async def", "sleep(",
        "thread", "lock", "random", "time.", "patch(", "monkeypatch",
    )
    for line in source.splitlines():
        low = line.lower()
        if any(k in low for k in keys):
            s = line.rstrip()[:220]
            if s.strip() and s not in out:
                out.append(s)
        if len(out) >= limit:
            break
    return out


def _primary_target_line(source: str) -> str:
    """One high-salience line to copy exactly (or use line_number + replace)."""
    if not source:
        return ""
    for pat in (r"timeout", r"asyncio\.", r"async def", r"await ", r"wait_for", r"sleep\("):
        for line in source.splitlines():
            if re.search(pat, line, re.I) and line.strip():
                return line.rstrip()[:500]
    for line in source.splitlines():
        if line.strip():
            return line.rstrip()[:500]
    return ""


class ModelBackend(Protocol):
    def generate(self, prompt: str, *, system_prompt: str) -> str: ...


UNIFIED_SYSTEM_PROMPT = """\
You are FlakeForge, a debugging agent that fixes flaky Python tests.

OUTPUT: exactly one JSON object — no text before or after, no markdown fences, no XML tags.

Do NOT include a "file" field in patch hunks — the run assigns PATCH TARGET FILE for you.

{
  "think": {
    "claims": [
      {
        "claim_id": "c1",
        "category": "<category>",
        "entity": "function_or_class_name",
        "location": "path/to/file.py::Class.method",
        "polarity": "present",
        "reason": "one sentence why it causes flakiness"
      }
    ],
    "confidence": 0.8
  },
  "patch": {
    "hunks": [
      {
        "hunk_id": "h1",
        "search": "a key line or small block from TARGET LINE / SOURCE (approximate is OK)",
        "replace": "the fixed line(s), same indent as the original",
        "addresses_claim": "c1"
      }
    ]
  }
}

OR use line mode (no search needed; give 1-based line number and full new line):
  "hunks": [ { "hunk_id": "h1", "line_number": 42, "replace": "    x = 1" } ]

CATEGORIES — choose based on what the test logs and source show:
  async_wait           logs: TimeoutError, asyncio.TimeoutError, "timed out"
                       fix:  increase timeout, add await, use asyncio.wait_for
  concurrency          logs: race condition, assertion changes between runs, sporadic failure
                       fix:  add threading.Lock, fix shared mutable access
  test_order_dependency  logs: passes alone (-k), fails in full suite
                       fix:  add teardown/setUp to isolate state between tests
  resource_leak        logs: "Address already in use", file descriptor exhaustion
                       fix:  add close/teardown, use context manager
  shared_state         logs: global or class-level variable mutated in one test breaks another
                       fix:  reset state in fixture teardown or setUp
  network              logs: ConnectionRefusedError, DNS failure, slow remote call
                       fix:  mock the network call
  nondeterminism       logs: float comparison fails, dict ordering differs, random output
                       fix:  seed random, use assertAlmostEqual, sort output
  import_side_effect   logs: module-level code executes on import and changes global state
                       fix:  move initialization inside a function
  module_cache_pollution  source: @lru_cache, sys.modules mutation, module-level globals
                       fix:  call cache_clear() in fixture teardown
  fixture_scope_leak   source: session/module fixture returns mutable object without yield
                       fix:  add yield and cleanup after yield
  mock_residue         source: mock.patch() called without with-block or .stop()
                       fix:  use "with mock.patch(...)" or call patcher.stop() in teardown
  unknown              use when evidence is genuinely unclear

PATCH RULES:
- Prefer the TARGET LINE the prompt gives you: copy that line for "search" when possible, or set "line_number" + "replace".
- "search" can be a key line or short block (applier can fuzzy/match; close is OK).
- "replace" must use the same indentation as the line(s) you change.
- Prefer patching the source-under-test over test code unless the test/fixture is the bug.
- Keep hunks small. "replace" may be "" to delete a small block.
- Do NOT add sleep(), @pytest.mark.skip, or broad refactors.
- For async / timeout: increase timeout= values or use asyncio.wait_for with a higher timeout; keep call shape.

INVALID JSON = zero reward. Output only the JSON object."""


def build_unified_prompt(observation: FlakeForgeObservation, *, retry_hint: Optional[str] = None) -> str:
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

    if getattr(observation, "source_file", None):
        parts.append("=== PATCH TARGET FILE (use this; do not choose another file in JSON) ===")
        parts.append(str(observation.source_file))
        parts.append("")

    sut = observation.source_under_test or ""
    if sut:
        ptl = _primary_target_line(sut)
        if ptl:
            parts.append("=== TARGET LINE (copy this line for search, or set line_number to its line) ===")
            parts.append(ptl)
            parts.append("")
        tsnip = _target_line_snippets(sut, limit=5)
        if tsnip:
            parts.append("=== CANDIDATE LINES (relevant to timeouts / async / nondeterminism) ===")
            for line in tsnip:
                parts.append(f"  • {line}")
            parts.append("")

    stack_and_sut = f"{getattr(observation, 'failing_stack_trace', '') or ''}\n{sut}"
    if re.search(r"timeout|Timeout|asyncio|async def|await |Cancelled", stack_and_sut, re.I):
        parts.append(
            "=== TEMPLATE: async / timeout (async_wait) ===\n"
            "Find timeout=, wait_for(..., timeout=), or time limits and INCREASE the numeric value; "
            "or ensure awaits complete. Do not change unrelated logic."
        )
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

    if retry_hint:
        parts.append(f"\n=== RETRY INSTRUCTION ===\n{retry_hint}")

    parts.append("\n=== YOUR TURN ===")

    # Determine the right recovery instruction based on what happened last.
    last_patch = observation.patches_applied[-1] if observation.patches_applied else None
    last_regression = (
        last_patch is not None
        and last_patch.applied_successfully
        and last_patch.pass_rate_after < observation.baseline_pass_rate - 0.05
    )
    last_patch_failed = last_patch is not None and not last_patch.applied_successfully

    if retry_hint:
        # Per-attempt hint takes precedence — it already contains the action to take.
        turn_instruction = "Fix the issue described in RETRY INSTRUCTION above, then output one JSON object."
    elif last_regression:
        turn_instruction = (
            f"REGRESSION DETECTED: last patch lowered pass_rate to {last_patch.pass_rate_after:.2f}. "
            "Keep the SAME root cause hypothesis and produce a more conservative or exact fix. "
            "Do NOT switch categories. Output one JSON object."
        )
    elif last_patch_failed:
        turn_instruction = (
            "PATCH FAILED: the search text was not found in the source. "
            "Copy the search block EXACTLY (same whitespace) from the source shown above. "
            "Output one JSON object."
        )
    elif observation.think_history:
        stale_cats = {
            h["categories"][0]
            for h in observation.think_history
            if h.get("categories")
        }
        attempted = ", ".join(sorted(stale_cats))
        turn_instruction = (
            f"IMPORTANT: You already tried [{attempted}] with no improvement. "
            "Propose a DIFFERENT root cause category supported by the evidence above. "
            "Output one JSON object."
        )
    else:
        turn_instruction = (
            "Output one JSON object with top-level keys think and patch. "
            "No XML tags, no markdown fences, no text outside the JSON."
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


def _patch_dict_to_search_replace(
    patch_data: Any, *, default_file: str = "",
) -> str:
    """Convert JSON patch hunks to legacy search/replace patch text (with --- file header)."""
    if not isinstance(patch_data, dict):
        return ""
    hunks = patch_data.get("hunks", [])
    if not isinstance(hunks, list):
        return ""

    blocks: List[str] = []
    for hunk in hunks:
        if not isinstance(hunk, dict):
            continue
        file_path = str(hunk.get("file") or default_file or "").strip()
        search = hunk.get("search")
        replace = hunk.get("replace")
        ln = hunk.get("line_number")
        if not isinstance(replace, str):
            continue
        if not file_path:
            continue
        # Line-only hunks: search filled later by finalize_for_inference; skip until then.
        if (ln is not None and int(ln) > 0) and not (isinstance(search, str) and search.strip()):
            continue
        if not isinstance(search, str) or (not str(search).strip() and (ln is None or int(ln) <= 0)):
            continue
        blocks.append(
            "\n".join([
                f"--- {file_path}",
                "<<<<<<< SEARCH",
                (search or "").rstrip("\n"),
                "=======",
                replace.rstrip("\n"),
                ">>>>>>> REPLACE",
            ])
        )
    return "\n\n".join(blocks)


def finalize_for_inference(
    action: "FlakeForgeAction",
    observation: "FlakeForgeObservation",
    repo_root: Path,
) -> "FlakeForgeAction":
    """Assign PATCH TARGET file from the observation, expand line_number→search, rebuild patch_text.

    Use this in the inference loop only; the model must not pick ``file`` itself.
    """
    root = Path(str(repo_root))
    if not action.structured_patch or not action.structured_patch.hunks:
        return action
    tgt = (getattr(observation, "source_file", None) or "").replace("\\", "/").strip()
    if not tgt:
        return action

    new_hunks: List[PatchHunk] = []
    for h in action.structured_patch.hunks:
        nh = h.model_copy()
        nh.file = tgt
        if nh.line_number is not None and int(nh.line_number) > 0:
            path = root / tgt
            if path.is_file():
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    lines = text.splitlines()
                    idx = int(nh.line_number) - 1
                    if 0 <= idx < len(lines):
                        nh.search = lines[idx]
                except OSError:
                    pass
        new_hunks.append(nh)

    new_sp = action.structured_patch.model_copy(update={"hunks": new_hunks})
    patch_dict = {
        "hunks": [h.model_dump(exclude_unset=False) for h in new_hunks],
    }
    new_pt = _patch_dict_to_search_replace(patch_dict, default_file=tgt)
    return action.model_copy(
        update={"structured_patch": new_sp, "patch_text": new_pt}
    )


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
    "hunk_id", "file", "search", "replace", "rationale", "addresses_claim", "line_number",
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
        clean.setdefault("file", "")
        clean.setdefault("search", "")
        if "replace" not in clean or not isinstance(clean.get("replace"), str):
            partial_penalty = min(partial_penalty, -0.3)
            continue
        # line mode OR search mode
        ln: Optional[int] = None
        if "line_number" in clean and clean["line_number"] is not None:
            try:
                ln = int(clean["line_number"])
            except (TypeError, ValueError):
                ln = None
            clean["line_number"] = ln
        if ln is not None and ln > 0:
            if not (clean.get("search") or "").strip():
                # ok — finalize will set search
                pass
        elif not (clean.get("search") or "").strip():
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

    def generate(self, observation: FlakeForgeObservation, *, retry_hint: Optional[str] = None) -> FlakeForgeAction:
        prompt = build_unified_prompt(observation, retry_hint=retry_hint)
        raw_response = self.backend.generate(prompt, system_prompt=self.system_prompt)

        response_obj = _load_json_object(raw_response)
        if isinstance(response_obj, dict) and (
            isinstance(response_obj.get("think"), dict)
            or isinstance(response_obj.get("patch"), dict)
        ):
            think_raw = json.dumps(response_obj.get("think", {}), ensure_ascii=False)
            structured_think = _parse_structured_think(response_obj.get("think", {}))
            structured_patch = _parse_structured_patch(response_obj.get("patch", {}))
            patch_raw = _patch_dict_to_search_replace(
                response_obj.get("patch", {}),
                default_file=getattr(observation, "source_file", None) or "",
            )
        else:
            think_raw = extract_think(raw_response)
            patch_raw = extract_patch(raw_response)
            structured_think = _parse_structured_think(think_raw)
            structured_patch = _parse_structured_patch(patch_raw)

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
