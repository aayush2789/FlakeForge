"""Reward architecture — multi-signal verifiable reward, no LLM judge."""

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
        ThinkClaim,
        failure_mode_entropy,
    )
except ImportError:
    from ..models import (
        FlakeForgeAction,
        FlakeForgeObservation,
        RewardBreakdown,
        ROOT_CAUSE_TYPES,
        RELATED_CATEGORIES,
        ThinkClaim,
        failure_mode_entropy,
    )

try:
    from agent.unified_agent import infer_category_from_patch
except ImportError:
    try:
        from ..agent.unified_agent import infer_category_from_patch
    except ImportError:
        def infer_category_from_patch(patch_text: str) -> str:
            return "unknown"


def _claim_has_required_format(claim: Any) -> bool:
    return (
        getattr(claim, "category", "") in ROOT_CAUSE_TYPES
        and bool(str(getattr(claim, "entity", "")).strip())
        and bool(str(getattr(claim, "location", "")).strip())
        and getattr(claim, "polarity", "") in ("present", "absent")
        and bool(str(getattr(claim, "reason", "")).strip())
    )


def _hunk_has_required_format(hunk: Any) -> bool:
    return (
        bool(str(getattr(hunk, "file", "")).strip())
        and bool(str(getattr(hunk, "search", "")).strip())
        and isinstance(getattr(hunk, "replace", None), str)
    )


def compute_format_reward(action: FlakeForgeAction) -> float:
    """Score current JSON-first action format. Returns 0..1.

    The model-facing schema now requires only:
    - claim: category, entity, location, polarity, reason
    - hunk: file, search, replace
    Optional metadata such as claim_id, ast_node_type, hunk_id, rationale, and
    addresses_claim should not affect format reward.
    """
    score = 0.0

    think_score = 0.0
    st = action.structured_think
    if st is not None:
        valid_claims = [claim for claim in st.claims if _claim_has_required_format(claim)]
        if valid_claims and st.format_penalty == 0.0:
            think_score = 0.5
        elif valid_claims and st.format_penalty > -0.5:
            think_score = 0.4
        elif st.claims:
            think_score = 0.25

    if think_score == 0.0 and action.think_text:
        think = action.think_text.lower()
        if ("root cause:" in think or '"category"' in think) and (
            "confidence" in think or '"reason"' in think
        ):
            think_score = 0.5
    score += think_score

    patch_score = 0.0
    sp = action.structured_patch
    if sp is not None:
        valid_hunks = [hunk for hunk in sp.hunks if _hunk_has_required_format(hunk)]
        if valid_hunks and sp.format_penalty == 0.0:
            patch_score = 0.5
        elif valid_hunks and sp.format_penalty > -0.5:
            patch_score = 0.4
        elif sp.hunks:
            patch_score = 0.25

    if patch_score == 0.0 and action.patch_text:
        has_search = "<<<<<<< SEARCH" in action.patch_text or "<<<<<<<" in action.patch_text
        has_replace = ">>>>>>> REPLACE" in action.patch_text or ">>>>>>>" in action.patch_text
        has_sep = "=======" in action.patch_text
        if has_search and has_replace and has_sep:
            patch_score = 0.5
        elif has_search or has_replace:
            patch_score = 0.25
    score += patch_score

    return round(min(score, 1.0), 2)


def compute_compile_reward(
    patch_applied_successfully: bool,
    syntax_error: Optional[str] = None,
    *,
    rejected_by_validator: bool = False,
    rolled_back: bool = False,
) -> float:
    """Score patch applicability and syntax correctness. Returns -1.0..1.0."""
    if rejected_by_validator:
        return -0.5
    if rolled_back:
        return 0.0
    if not patch_applied_successfully:
        return -1.0
    if syntax_error:
        return -0.5
    return 1.0


def _potential(pass_rate: float) -> float:
    """Potential function Φ(pass_rate) for potential-based reward shaping.

    Uses Φ = pass_rate^2 so improvements near 1.0 are worth more than
    improvements near 0.0. Theory (Ng, Harada, Russell 1999) guarantees
    that Φ(s') - Φ(s) preserves the optimal policy.
    """
    return pass_rate ** 2


def compute_stability_reward(
    baseline_pass_rate: float,
    current_pass_rate: float,
) -> float:
    """Potential-based stability reward: Φ(current) - Φ(baseline). Returns ~-1.0..3.0."""
    shaped = _potential(current_pass_rate) - _potential(baseline_pass_rate)

    if current_pass_rate >= 1.0:
        return max(2.0, round(shaped * 3.0, 4))
    elif shaped > 0:
        return round(min(shaped * 3.0, 1.5), 4)
    elif shaped < 0:
        return round(max(shaped * 5.0, -1.0), 4)
    else:
        return -0.1


def compute_causal_proximity_reward(
    patch_files: List[str],
    failure_frontier: str,
    call_chain: List[str],
    boundary_crossings: List[str],
) -> float:
    """Reward patches that touch code near the failure site. Returns -0.3..1.0."""
    if not patch_files or not failure_frontier:
        return 0.0

    score = 0.0
    frontier_file = failure_frontier.split(":")[0] if ":" in failure_frontier else failure_frontier

    for pf in patch_files:
        pf_name = pf.split("/")[-1] if "/" in pf else pf
        frontier_name = frontier_file.split("/")[-1] if "/" in frontier_file else frontier_file
        if pf_name == frontier_name:
            score = 1.0
            break

    if score == 0.0:
        for pf in patch_files:
            pf_name = pf.split("/")[-1].replace(".py", "")
            for frame in call_chain:
                if pf_name in frame:
                    score = max(score, 0.5)
                    break

    if score == 0.0:
        score = -0.3

    if boundary_crossings and score > 0:
        score = min(score + 0.2, 1.0)

    return round(score, 2)


def compute_entropy_reward(
    pre_entropy: float,
    post_entropy: float,
) -> float:
    """Reward reduction in failure mode entropy. Returns -0.5..1.0."""
    if pre_entropy <= 0 and post_entropy <= 0:
        return 0.5

    delta = pre_entropy - post_entropy
    if delta > 0:
        return round(min(delta * 2.0, 1.0), 4)
    elif delta < 0:
        return round(max(delta * 1.5, -0.5), 4)
    return 0.0


def compute_anti_hack_penalty(
    patch_text: str,
    files_modified: List[str],
    lines_changed: int,
) -> float:
    """Penalise common reward-hacking patterns. Returns -2.0..0.0."""
    penalty = 0.0
    patch_lower = patch_text.lower()

    if re.search(r"\bassert\b", patch_lower):
        search_blocks = re.findall(r"<<<<<<< SEARCH\n(.*?)=======", patch_text, re.DOTALL)
        replace_blocks = re.findall(r"=======\n(.*?)>>>>>>> REPLACE", patch_text, re.DOTALL)

        for search, replace in zip(search_blocks, replace_blocks):
            search_asserts = len(re.findall(r"\bassert\b", search.lower()))
            replace_asserts = len(re.findall(r"\bassert\b", replace.lower()))
            if search_asserts > replace_asserts:
                penalty -= 0.5 * (search_asserts - replace_asserts)

    sleep_patterns = ["time.sleep(", "await asyncio.sleep(", "sleep("]
    for pattern in sleep_patterns:
        if pattern in patch_lower:
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

    if "except:" in patch_lower or "except Exception:" in patch_lower:
        replace_blocks = re.findall(r"=======\n(.*?)>>>>>>> REPLACE", patch_text, re.DOTALL)
        for block in replace_blocks:
            if "except:" in block.lower() or "except exception:" in block.lower():
                penalty -= 0.5

    skip_patterns = ["@pytest.mark.skip", "@unittest.skip", "pytest.skip("]
    for pattern in skip_patterns:
        if pattern in patch_lower:
            penalty -= 1.0

    if lines_changed > 50:
        penalty -= min((lines_changed - 50) / 100.0, 0.5)

    return round(max(penalty, -2.0), 2)


def compute_reasoning_consistency(
    predicted_category: str,
    inferred_category_from_patch: str,
    think_text: str,
    patch_text: str,
) -> float:
    """Category match between predicted and patch-inferred root cause. Returns -0.5..0.5."""
    if not predicted_category or not inferred_category_from_patch:
        return 0.0
    if predicted_category == inferred_category_from_patch:
        return 0.5
    related = RELATED_CATEGORIES.get(predicted_category, set())
    if inferred_category_from_patch in related:
        return 0.25
    return -0.5


def compute_think_history_penalty(
    action: FlakeForgeAction,
    think_history: Optional[List[Dict[str, Any]]],
) -> float:
    """Mildly penalize repeating the same diagnosis without new evidence."""
    if not think_history:
        return 0.0

    categories: set[str] = set()
    entities: set[str] = set()
    reasons: set[str] = set()
    if action.structured_think and action.structured_think.claims:
        for claim in action.structured_think.claims:
            if claim.category:
                categories.add(claim.category)
            if claim.entity:
                entities.add(claim.entity)
            if claim.reason:
                reasons.add(claim.reason[:35].lower().strip())
    elif action.predicted_category:
        categories.add(action.predicted_category)

    if not categories and not entities and not reasons:
        return 0.0

    previous = think_history[-1]
    repeated_category = bool(categories & set(previous.get("categories", [])))
    repeated_entity = bool(entities & set(previous.get("entities", [])))
    repeated_reason = bool(reasons & set(previous.get("reason_signatures", [])))

    if repeated_category and (repeated_entity or repeated_reason):
        return -0.25
    return 0.0


def compute_verifiable_reward(
    action: FlakeForgeAction,
    observation: FlakeForgeObservation,
    patch_result: Dict[str, Any],
    post_run_results: List[Dict[str, Any]],
    baseline_pass_rate: float,
    pre_entropy: float,
    oracle_score: Optional[float] = None,
    regression_detected: bool = False,
    think_history: Optional[List[Dict[str, Any]]] = None,
) -> RewardBreakdown:
    """Compute the full multi-signal verifiable reward."""
    patch_applied = patch_result.get("success", False)
    rejected_by_validator = bool(patch_result.get("rejected_by_validator", False))
    rolled_back = bool(patch_result.get("rolled_back", False))
    syntax_error = patch_result.get("syntax_error") if patch_applied else None
    files_modified = patch_result.get("files_modified", [])
    lines_changed = patch_result.get("lines_changed", 0)
    noop_patch = bool(patch_result.get("noop", False))
    regression_detected = bool(regression_detected or patch_result.get("regression_detected", False))

    post_pass_count = sum(1 for r in post_run_results if r.get("passed", False))
    if post_run_results:
        post_pass_rate = post_pass_count / len(post_run_results)
        post_errors = [r.get("error_type", "") for r in post_run_results if not r.get("passed", True)]
        post_counter = Counter(e for e in post_errors if e)
        if post_counter:
            total = sum(post_counter.values())
            post_entropy = -sum((c / total) * math.log2(c / total) for c in post_counter.values())
            max_e = math.log2(len(post_counter)) if len(post_counter) > 1 else 1.0
            post_entropy = post_entropy / max_e if max_e > 0 else 0.0
        else:
            post_entropy = 0.0
    else:
        # If no tests were run, assume pass rate and entropy haven't changed
        post_pass_rate = observation.current_pass_rate
        post_entropy = pre_entropy

    inferred_cat = infer_category_from_patch(action.patch_text)

    breakdown = RewardBreakdown()

    # ── Hard Gates Evaluation ─────────────────────────────────────────────────
    breakdown.format_reward = compute_format_reward(action)
    breakdown.compile_reward = compute_compile_reward(
        patch_applied,
        syntax_error,
        rejected_by_validator=rejected_by_validator,
        rolled_back=rolled_back,
    )
    breakdown.regression_penalty = -1.5 if regression_detected else 0.0
    breakdown.anti_hack_penalty = compute_anti_hack_penalty(
        action.patch_text, files_modified, lines_changed,
    )

    # Fast-fail short circuits if hard gates are violated (Requires >= 0.75 format reward)
    if breakdown.format_reward < 0.75 or breakdown.compile_reward < 1.0 or breakdown.anti_hack_penalty < 0.0:
        breakdown.total_reward = -2.0 if breakdown.anti_hack_penalty < 0.0 else -1.0
        return breakdown

    # PatchValidator positive shaping
    if patch_applied and patch_result.get("validation_score") is not None:
        try:
            vs = float(patch_result["validation_score"])
            breakdown.patch_validation_signal = round(0.2 * max(0.0, min(1.0, vs)), 4)
        except (TypeError, ValueError):
            breakdown.patch_validation_signal = 0.0
    else:
        breakdown.patch_validation_signal = 0.0

    breakdown.stability_reward = compute_stability_reward(baseline_pass_rate, post_pass_rate)
    breakdown.causal_proximity_reward = compute_causal_proximity_reward(
        files_modified,
        observation.failure_frontier,
        observation.call_chain_to_frontier,
        observation.boundary_crossings,
    )
    breakdown.failure_entropy_reward = compute_entropy_reward(pre_entropy, post_entropy)
    breakdown.noop_patch_penalty = -0.5 if patch_applied and noop_patch else 0.0
    breakdown.think_history_penalty = compute_think_history_penalty(action, think_history)

    # ── Oracle (mild reasoning reward) ──────────────
    if oracle_score is not None:
        breakdown.oracle_reasoning_reward = round(float(oracle_score), 4)
        breakdown.reasoning_consistency_reward = 0.0
    else:
        breakdown.reasoning_consistency_reward = compute_reasoning_consistency(
            action.predicted_category, inferred_cat, action.think_text, action.patch_text,
        )
        breakdown.oracle_reasoning_reward = 0.0

    # ── Terminal bonus (scaled down to 4.0) ───────────────────────────────────
    if patch_applied and post_run_results:
        if post_pass_rate >= 1.0:
            breakdown.terminal_bonus = 4.0   # Solving it = massive positive signal
        elif post_pass_rate > baseline_pass_rate + 0.5:
            breakdown.terminal_bonus = 2.0   # Major improvement
        elif post_pass_rate > baseline_pass_rate + 0.3:
            breakdown.terminal_bonus = 1.0   # Meaningful progress

    # ── Total reward ──────────────────────────────────────────────────────────
    oracle_component = (
        breakdown.oracle_reasoning_reward * 1.0      # reduced weight 1.0 (was 2.5)
        if oracle_score is not None
        else breakdown.reasoning_consistency_reward * 0.5
    )

    breakdown.total_reward = round(
        breakdown.format_reward           * 0.5
        + breakdown.compile_reward        * 1.0
        + breakdown.stability_reward      * 3.0      # increased weight 3.0 (was 2.0)
        + breakdown.causal_proximity_reward * 0.5
        + breakdown.failure_entropy_reward * 0.5
        + breakdown.anti_hack_penalty     * 1.5
        + breakdown.regression_penalty    * 1.0
        + oracle_component
        + breakdown.patch_validation_signal * 1.0
        + breakdown.noop_patch_penalty    * 1.0
        + breakdown.think_history_penalty * 1.0
        + breakdown.terminal_bonus        * 1.0,
        4,
    )

    return breakdown
