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


UNIFIED_SYSTEM_PROMPT = """You are FlakeForge, a debugging agent that fixes flaky Python tests.

Reply with ONE JSON object. No markdown, no XML, no commentary.

Shape (single-line example, no embedded newlines):
{"think":{"claims":[{"category":"async_wait","entity":"asyncio.wait_for","location":"source.py::fetch","polarity":"present","reason":"timeout=0.5 too small for slow runs"}],"confidence":0.85},"patch":{"hunks":[{"file":"source.py","search":"timeout=0.5","replace":"timeout=5.0"}]}}

CATEGORIES (pick exactly ONE per claim):
async_wait, concurrency, test_order_dependency, resource_leak, shared_state,
network, platform_dependency, nondeterminism, import_side_effect,
module_cache_pollution, fixture_scope_leak, mock_residue, unknown.

PATCH RULES (read carefully — failing these breaks the run):
1. "search" must be ONE single line copied VERBATIM from SOURCE UNDER TEST.
   - Same leading spaces. No tabs. No trailing spaces.
   - Pick a line that is unique in the file (a distinctive expression or assignment).
2. Do NOT put line breaks inside "search" or "replace". No "\\n" sequences.
   If you need a multi-line edit, pick the SHORTEST single anchor line and
   express the whole new block in "replace" using JSON's standard "\\n" escape
   ONLY when truly necessary (prefer single-line replacements).
3. "replace" must keep the SAME leading indentation as the search line.
4. Inside JSON strings escape only what JSON requires:
     " becomes \\"
     \\ becomes \\\\
   Never invent extra backslashes. Never add stray commas after backslashes.
5. Prefer the smallest fix. Never add sleep(), retry decorators, or @pytest.mark.skip.
6. Patch source.py, not the test file, unless the bug is in the test itself.
7. If you cannot find a clear fix, return:
   {"think":{"claims":[{"category":"unknown","entity":"","location":"","polarity":"present","reason":"need more evidence"}],"confidence":0.2},"patch":{"hunks":[]}}

NEVER:
- Wrap the JSON in ```json ... ``` fences.
- Emit <think> or <patch> tags.
- Add prose before or after the JSON.
- Repeat a category that already failed twice in the run history.
"""


_CATEGORY_CHEATSHEET = (
    "=== HOW TO PICK A CATEGORY (read RECENT RUNS / LAST FAILURE first) ===\n"
    "- TimeoutError, hang, slow run, asyncio.wait_for / await        -> async_wait\n"
    "- random.random(), time.time(), datetime.now(), shuffle         -> nondeterminism\n"
    "- threads, Lock/Event/Queue, race messages, queue_full           -> concurrency\n"
    "- globals reused across tests, leak symptoms after second run   -> shared_state\n"
    "- requests/httpx/socket flakes, ConnectionError                  -> network\n"
    "- @pytest.fixture(scope=session/module) with mutable state       -> fixture_scope_leak\n"
    "- @lru_cache, sys.modules reuse, cache decorators                -> module_cache_pollution\n"
    "- top-level side effects at import time                          -> import_side_effect\n"
    "- patch() / monkeypatch not torn down                            -> mock_residue\n"
    "- only fails when run after another test                         -> test_order_dependency\n"
    "- Linux-only / Windows-only / freeing socket ports               -> platform_dependency\n"
    "- file/socket/handle not closed                                  -> resource_leak\n"
    "- nothing matches confidently                                    -> unknown"
)


def _last_attempt_diagnosis(observation: FlakeForgeObservation) -> List[str]:
    """Summarise the most recent step's outcome in one short block.

    The 7B model needs an explicit verdict so it can decide whether to keep
    the same hypothesis (clean failure / unapplied patch) or switch to a new
    one (regression / repeated dead-end).
    """
    history = observation.think_history or []
    patches = observation.patches_applied or []

    if not history and not patches:
        return []

    last_think = history[-1] if history else {}
    last_patch = patches[-1] if patches else None

    cats = last_think.get("categories") or []
    primary_cat = cats[0] if cats else "unknown"
    entities = last_think.get("entities") or []
    pr_after = last_think.get("pass_rate_after", observation.current_pass_rate)
    pr_before = observation.baseline_pass_rate
    reward = last_think.get("reward", observation.last_reward)

    applied = bool(last_patch and last_patch.applied_successfully)
    files = ", ".join(last_patch.target_files) if last_patch else "—"

    if not applied:
        outcome = "PATCH DID NOT APPLY (search text not found in file)"
        verdict = (
            "Fix this turn: keep the SAME root cause but rewrite \"search\" as ONE shorter, "
            "more distinctive line copied verbatim from SOURCE UNDER TEST."
        )
    elif pr_after < pr_before - 0.05:
        outcome = f"REGRESSION applied {pr_before:.2f} -> {pr_after:.2f}"
        verdict = (
            "The previous category was wrong. Pick a DIFFERENT category this turn "
            "and patch a different line."
        )
    elif pr_after > pr_before + 0.05:
        outcome = f"IMPROVED {pr_before:.2f} -> {pr_after:.2f}"
        verdict = "Keep refining the same category; tighten the patch."
    else:
        outcome = f"NO EFFECT {pr_before:.2f} -> {pr_after:.2f}"
        verdict = (
            "Same category may still be right but the line you patched was not the cause. "
            "Pick a DIFFERENT line, or switch to a related category."
        )

    lines = [
        f"Hypothesis: {primary_cat}",
        f"Entities:   {', '.join(entities[:3]) if entities else '—'}",
        f"Files:      {files}",
        f"Outcome:    {outcome}",
        f"Reward:     {reward:+.3f}",
        f"Verdict:    {verdict}",
    ]
    return lines


def _scenario_hints(observation: FlakeForgeObservation) -> List[str]:
    """Per-state guidance the model can act on without re-reading everything."""
    history = observation.think_history or []
    if not history:
        return [
            "- This is the FIRST step. Pick the most likely category from RECENT RUNS.",
            "- Patch source.py (not tests/) unless the test itself is buggy.",
            "- Keep \"search\" to ONE distinctive single line from SOURCE UNDER TEST.",
        ]

    cat_counts: Dict[str, int] = {}
    for h in history:
        cats = h.get("categories") or ["unknown"]
        cat_counts[cats[0]] = cat_counts.get(cats[0], 0) + 1
    repeated = [c for c, n in cat_counts.items() if n >= 2]

    hints: List[str] = []
    if repeated:
        hints.append(
            f"- You already tried {', '.join(sorted(repeated))} more than once with no success. "
            "Do NOT pick those categories again unless you have new evidence."
        )
    hints.extend([
        "- If your previous patch did not apply: shorten \"search\" to ONE distinctive line.",
        "- If pass_rate dropped: the last category was wrong; switch.",
        "- If pass_rate did not move: try a different LINE (still in source.py) or a related category.",
    ])
    return hints


def build_unified_prompt(observation: FlakeForgeObservation) -> str:
    """Compact prompt designed for small (≤7B) models.

    Layout (each section is short and bounded):
      1. TASK header         — what test, what step, what pass-rate goal.
      2. SOURCE UNDER TEST   — the file the model should patch (≤1800 chars).
      3. TEST FUNCTION       — the test that is flaky (≤1000 chars).
      4. LAST FAILURE        — tail of the most recent failing stack (≤900 chars).
      5. RECENT RUNS         — at most 5 most recent run outcomes.
      6. DEEP SIGNALS        — single-line summary of static / dynamic alerts.
      7. TARGETING HINTS     — top causal hints (≤4).
      8. LAST ATTEMPT        — verdict on the previous step (only if any).
      9. SCENARIO GUIDE      — what to do this turn given the state above.
     10. CATEGORY HINTS      — short cheatsheet to map symptoms -> category.
     11. YOUR TURN           — one-line schema reminder, single-line example.
    """
    parts: List[str] = []

    parts.append("=== TASK ===")
    parts.append(f"Test:       {observation.test_identifier}")
    parts.append(f"Step:       {observation.step + 1} of {observation.step + observation.steps_remaining}")
    parts.append(
        f"Pass rate:  baseline={observation.baseline_pass_rate:.2f}  "
        f"current={observation.current_pass_rate:.2f}  goal=1.00"
    )
    parts.append("")

    if observation.source_under_test:
        parts.append("=== SOURCE UNDER TEST  (PATCH THIS FILE: source.py) ===")
        parts.append(observation.source_under_test[:1800])
        parts.append("")

    if observation.test_function_source:
        parts.append("=== TEST FUNCTION ===")
        parts.append(observation.test_function_source[:1000])
        parts.append("")

    if observation.failing_stack_trace:
        parts.append("=== LAST FAILURE (tail) ===")
        parts.append(observation.failing_stack_trace[-900:])
        parts.append("")

    if observation.run_history:
        parts.append("=== RECENT RUNS (last 5) ===")
        for r in observation.run_history[-5:]:
            verdict = "PASS" if r.passed else f"FAIL({r.error_type or 'err'})"
            line = f"- {verdict}  {r.duration_ms}ms"
            if not r.passed and r.error_message:
                line += f"  msg={r.error_message[:80]}"
            parts.append(line)
        parts.append("")

    deep: List[str] = []
    if observation.async_contamination_alive:
        deep.append("async tasks/threads survived past test boundary")
    if observation.module_cache_violations:
        deep.append(f"module-cache hot spots: {', '.join(observation.module_cache_violations[:3])}")
    if observation.fixture_scope_risks:
        deep.append(f"fixture-scope risks: {', '.join(observation.fixture_scope_risks[:3])}")
    if observation.mock_residue_sites:
        deep.append(f"mock residue: {', '.join(observation.mock_residue_sites[:3])}")
    if observation.import_side_effect_files:
        deep.append(f"import side effects: {', '.join(observation.import_side_effect_files[:3])}")
    if observation.order_dependency_detected:
        deep.append("test order matters (fails when reordered)")
    if observation.infrastructure_sensitive:
        deep.append("infrastructure-sensitive (fails under resource pressure)")
    if deep:
        parts.append("=== DEEP SIGNALS ===")
        parts.extend(f"- {s}" for s in deep)
        parts.append("")

    if observation.causal_hints:
        parts.append("=== TARGETING HINTS ===")
        parts.extend(f"- {h}" for h in observation.causal_hints[:4])
        parts.append("")

    diag = _last_attempt_diagnosis(observation)
    if diag:
        parts.append("=== LAST ATTEMPT ===")
        parts.extend(diag)
        parts.append("")

    parts.append("=== SCENARIO GUIDE (do this now) ===")
    parts.extend(_scenario_hints(observation))
    parts.append("")

    parts.append(_CATEGORY_CHEATSHEET)
    parts.append("")

    parts.append("=== YOUR TURN ===")
    parts.append("Reply with ONE JSON object. No markdown, no XML, no commentary.")
    parts.append("Single-line example shape (do NOT copy values, just the structure):")
    parts.append(
        '{"think":{"claims":[{"category":"<one>","entity":"<symbol>",'
        '"location":"source.py::<func>","polarity":"present","reason":"<short>"}],'
        '"confidence":0.0},"patch":{"hunks":[{"file":"source.py",'
        '"search":"<one distinctive line copied from SOURCE UNDER TEST>",'
        '"replace":"<replacement line>"}]}}'
    )
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
                + "\n\n=== STRICT RETRY ===\n"
                + "Your previous reply was not valid JSON or had no usable hunks.\n"
                + "Common mistakes to AVOID this time:\n"
                + "- Multi-line \"search\" (use ONE single line copied from SOURCE UNDER TEST).\n"
                + "- Stray backslashes (\\\\\\\\, \\,, trailing \\). Inside JSON only \\\" and \\\\ are needed.\n"
                + "- Markdown fences, XML tags, or any text outside the JSON object.\n"
                + "Reply with EXACTLY one JSON object that matches the shape shown above."
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
