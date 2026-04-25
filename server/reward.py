"""V3 Reward Architecture — Six-signal verifiable reward, no LLM judge.

Each signal is deterministic and derived from execution outcomes:
1. Format Compliance: Valid <think>+<patch> structure
2. Compile/Syntax: Modified file parses without errors
3. Stability Delta: Pass-rate improvement over baseline
4. Causal Proximity: Patch touches causal frontier, not distant code
5. Failure Entropy Reduction: Fewer distinct error modes after fix
6. Anti-Hack Penalty: Catch test deletion, sleep injection, broad try/except

Research basis:
- DeepSeek R1: Format+correctness reward (no human prefs)
- RLEF: Execution-verified compilability
- Reflexion NeurIPS 2023: Self-consistency check for reasoning
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

try:
    from models import (
        FlakeForgeAction,
        FlakeForgeObservation,
        RewardBreakdown,
        ROOT_CAUSE_TYPES,
        RELATED_CATEGORIES,
        failure_mode_entropy,
    )
except ImportError:
    from ..models import (
        FlakeForgeAction,
        FlakeForgeObservation,
        RewardBreakdown,
        ROOT_CAUSE_TYPES,
        RELATED_CATEGORIES,
        failure_mode_entropy,
    )


# ── Signal 1: Format Compliance ──────────────────────────────────────────────

def compute_format_reward(action: FlakeForgeAction) -> float:
    """Check that the model produced valid <think> and <patch> blocks.

    Returns:
        1.0 if both blocks present with required fields
        0.5 if only one block present
        0.0 if neither
    """
    score = 0.0

    # Check think block
    if action.think_text:
        think = action.think_text.lower()
        has_root_cause = "root cause:" in think or "root_cause:" in think
        has_confidence = "confidence:" in think
        has_evidence = any(word in think for word in ["evidence:", "because", "the ", "causing", "due to"])

        if has_root_cause and has_confidence:
            score += 0.5
        elif has_root_cause or has_confidence:
            score += 0.25

    # Check patch block
    if action.patch_text:
        has_search = "<<<<<<< SEARCH" in action.patch_text or "<<<<<<<" in action.patch_text
        has_replace = ">>>>>>> REPLACE" in action.patch_text or ">>>>>>>" in action.patch_text
        has_separator = "=======" in action.patch_text

        if has_search and has_replace and has_separator:
            score += 0.5
        elif has_search or has_replace:
            score += 0.25

    return round(score, 2)


# ── Signal 2: Compile/Syntax ────────────────────────────────────────────────

def compute_compile_reward(
    patch_applied_successfully: bool,
    syntax_error: Optional[str] = None,
) -> float:
    """Check that the patched file compiles without errors.

    Returns:
        1.0 if patch applied + no syntax errors
        0.5 if patch applied but has warnings
       -1.0 if syntax error
    """
    if not patch_applied_successfully:
        return -1.0
    if syntax_error:
        return -0.5
    return 1.0


# ── Signal 3: Stability Delta ───────────────────────────────────────────────

def compute_stability_reward(
    baseline_pass_rate: float,
    current_pass_rate: float,
) -> float:
    """Reward proportional to pass-rate improvement.

    Uses a clipped linear reward:
    - Positive if pass rate improved
    - Negative if pass rate regressed
    - Bonus for reaching 100%

    Returns:
        Float in [-1.0, 2.0]
    """
    delta = current_pass_rate - baseline_pass_rate

    if current_pass_rate >= 1.0:
        return 2.0  # Full stability achieved — terminal bonus
    elif delta > 0:
        return round(min(delta * 3.0, 1.5), 4)  # Scale positives
    elif delta < 0:
        return round(max(delta * 5.0, -1.0), 4)  # Harsher penalty for regression
    else:
        return -0.1  # No change — slight penalty for wasted step


# ── Signal 4: Causal Proximity ──────────────────────────────────────────────

def compute_causal_proximity_reward(
    patch_files: List[str],
    failure_frontier: str,
    call_chain: List[str],
    boundary_crossings: List[str],
) -> float:
    """Reward patches that touch code near the failure site.

    Patches to files far from the causal frontier are likely
    noise or workarounds, not root-cause fixes.

    Returns:
        Float in [-0.5, 1.0]
    """
    if not patch_files or not failure_frontier:
        return 0.0

    score = 0.0
    frontier_file = failure_frontier.split(":")[0] if ":" in failure_frontier else failure_frontier

    # Direct hit — same file as failure frontier
    for pf in patch_files:
        pf_name = pf.split("/")[-1] if "/" in pf else pf
        frontier_name = frontier_file.split("/")[-1] if "/" in frontier_file else frontier_file
        if pf_name == frontier_name:
            score = 1.0
            break

    if score == 0.0:
        # Check if patch file is in the call chain
        for pf in patch_files:
            pf_name = pf.split("/")[-1].replace(".py", "")
            for frame in call_chain:
                if pf_name in frame:
                    score = max(score, 0.5)
                    break

    if score == 0.0:
        # Patch is far from causal frontier — penalize
        score = -0.3

    # Bonus if patch crosses a known boundary (meaningful infrastructure fix)
    if boundary_crossings and score > 0:
        score = min(score + 0.2, 1.0)

    return round(score, 2)


# ── Signal 5: Failure Entropy Reduction ─────────────────────────────────────

def compute_entropy_reward(
    pre_entropy: float,
    post_entropy: float,
) -> float:
    """Reward reduction in failure mode entropy.

    Lower entropy = more deterministic = closer to fixing root cause.
    If all errors converge to one type (or zero), the fix is targeted.

    Returns:
        Float in [-0.5, 1.0]
    """
    if pre_entropy <= 0 and post_entropy <= 0:
        return 0.5  # No errors at all — good

    delta = pre_entropy - post_entropy
    if delta > 0:
        return round(min(delta * 2.0, 1.0), 4)
    elif delta < 0:
        return round(max(delta * 1.5, -0.5), 4)
    return 0.0


# ── Signal 6: Anti-Hack Penalty ─────────────────────────────────────────────

def compute_anti_hack_penalty(
    patch_text: str,
    files_modified: List[str],
    lines_changed: int,
) -> float:
    """Penalize common reward-hacking patterns.

    Catches:
    - Test deletion (removing assertions or entire test)
    - Sleep injection (masking timing issues)
    - Broad try/except (masking errors)
    - Excessive changes (> 50 lines = shotgun fix)
    - Skip decorators (@pytest.mark.skip)

    Returns:
        Float in [-2.0, 0.0] (always non-positive)
    """
    penalty = 0.0
    patch_lower = patch_text.lower()

    # Test deletion — deleting assertions
    if "assert" in patch_lower:
        # Check if assertions are being removed (in SEARCH but not in REPLACE)
        search_blocks = re.findall(r"<<<<<<< SEARCH\n(.*?)=======", patch_text, re.DOTALL)
        replace_blocks = re.findall(r"=======\n(.*?)>>>>>>> REPLACE", patch_text, re.DOTALL)

        for search, replace in zip(search_blocks, replace_blocks):
            search_asserts = search.lower().count("assert")
            replace_asserts = replace.lower().count("assert")
            if search_asserts > replace_asserts:
                penalty -= 0.5 * (search_asserts - replace_asserts)

    # Sleep injection
    sleep_patterns = ["time.sleep(", "await asyncio.sleep(", "sleep("]
    for pattern in sleep_patterns:
        if pattern in patch_lower:
            # Only penalize if sleep is being ADDED (in replace but not search)
            replace_count = sum(
                1 for b in re.findall(r"=======\n(.*?)>>>>>>> REPLACE", patch_text, re.DOTALL)
                if pattern in b.lower()
            )
            search_count = sum(
                1 for b in re.findall(r"<<<<<<< SEARCH\n(.*?)=======", patch_text, re.DOTALL)
                if pattern in b.lower()
            )
            if replace_count > search_count:
                penalty -= 0.3 * (replace_count - search_count)

    # Broad try/except
    if "except:" in patch_lower or "except Exception:" in patch_lower:
        replace_blocks = re.findall(r"=======\n(.*?)>>>>>>> REPLACE", patch_text, re.DOTALL)
        for block in replace_blocks:
            if "except:" in block.lower() or "except exception:" in block.lower():
                penalty -= 0.5

    # Skip decorator injection
    skip_patterns = ["@pytest.mark.skip", "@unittest.skip", "pytest.skip("]
    for pattern in skip_patterns:
        if pattern in patch_lower:
            penalty -= 1.0

    # Excessive changes
    if lines_changed > 50:
        penalty -= min((lines_changed - 50) / 100.0, 0.5)

    return round(max(penalty, -2.0), 2)


# ── Reasoning Consistency (Bonus Signal) ────────────────────────────────────

def compute_reasoning_consistency(
    predicted_category: str,
    inferred_category_from_patch: str,
    think_text: str,
    patch_text: str,
) -> float:
    """Check that the reasoning in <think> is consistent with the <patch>.

    Verifies that the diagnosed root cause category matches what the
    patch actually modifies. Prevents "vibe-coding" (reasoning that
    sounds good but doesn't match the actual fix).

    Returns:
        Float in [-0.5, 0.5]
    """
    if not predicted_category or not inferred_category_from_patch:
        return 0.0

    # Exact match
    if predicted_category == inferred_category_from_patch:
        return 0.5

    # Related category match (e.g., concurrency ↔ async_wait)
    related = RELATED_CATEGORIES.get(predicted_category, set())
    if inferred_category_from_patch in related:
        return 0.25

    # Mismatch — reasoning doesn't match what was actually changed
    return -0.5


# ── Composite Reward ────────────────────────────────────────────────────────

def compute_verifiable_reward(
    action: FlakeForgeAction,
    observation: FlakeForgeObservation,
    patch_result: Dict[str, Any],
    post_run_results: List[Dict[str, Any]],
    baseline_pass_rate: float,
    pre_entropy: float,
) -> RewardBreakdown:
    """Compute the full six-signal verifiable reward.

    All signals are deterministic and derived from execution outcomes.
    No LLM judge required.
    """
    # Patch metadata
    patch_applied = patch_result.get("success", False)
    syntax_error = patch_result.get("error") if not patch_applied else None
    files_modified = patch_result.get("files_modified", [])
    lines_changed = patch_result.get("lines_changed", 0)

    # Post-run results
    post_pass_count = sum(1 for r in post_run_results if r.get("passed", False))
    post_pass_rate = post_pass_count / max(len(post_run_results), 1)

    # Post entropy
    post_errors = [r.get("error_type", "") for r in post_run_results if not r.get("passed", True)]
    post_counter = Counter(e for e in post_errors if e)
    if post_counter:
        total = sum(post_counter.values())
        post_entropy = -sum((c / total) * math.log2(c / total) for c in post_counter.values())
        max_e = math.log2(len(post_counter)) if len(post_counter) > 1 else 1.0
        post_entropy = post_entropy / max_e if max_e > 0 else 0.0
    else:
        post_entropy = 0.0

    # Inferred category from actual patch changes
    from agent.unified_agent import infer_category_from_patch
    inferred_cat = infer_category_from_patch(action.patch_text)

    # Compute signals
    breakdown = RewardBreakdown()
    breakdown.format_reward = compute_format_reward(action)
    breakdown.compile_reward = compute_compile_reward(patch_applied, syntax_error)
    breakdown.stability_reward = compute_stability_reward(baseline_pass_rate, post_pass_rate)
    breakdown.causal_proximity_reward = compute_causal_proximity_reward(
        files_modified,
        observation.failure_frontier,
        observation.call_chain_to_frontier,
        observation.boundary_crossings,
    )
    breakdown.failure_entropy_reward = compute_entropy_reward(pre_entropy, post_entropy)
    breakdown.anti_hack_penalty = compute_anti_hack_penalty(
        action.patch_text,
        files_modified,
        lines_changed,
    )
    breakdown.reasoning_consistency_reward = compute_reasoning_consistency(
        action.predicted_category,
        inferred_cat,
        action.think_text,
        action.patch_text,
    )

    # Terminal bonus for full stability
    if patch_applied and post_run_results and post_pass_rate >= 1.0:
        breakdown.terminal_bonus = 2.0
    elif patch_applied and post_run_results and post_pass_rate > baseline_pass_rate + 0.3:
        breakdown.terminal_bonus = 1.0

    # Weighted total
    breakdown.total_reward = round(
        breakdown.format_reward * 0.5
        + breakdown.compile_reward * 1.0
        + breakdown.stability_reward * 2.0
        + breakdown.causal_proximity_reward * 0.5
        + breakdown.failure_entropy_reward * 0.5
        + breakdown.anti_hack_penalty * 1.5
        + breakdown.reasoning_consistency_reward * 0.5
        + breakdown.terminal_bonus * 1.0,
        4,
    )

    return breakdown
