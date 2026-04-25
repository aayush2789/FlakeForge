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


def compute_format_reward(action: FlakeForgeAction) -> float:
    """Score format compliance of think JSON and patch structure. Returns 0..1."""
    score = 0.0

    st = action.structured_think
    if st is not None:
        if st.claims and st.format_penalty == 0.0:
            score += 0.5
        elif st.claims and st.format_penalty > -0.5:
            score += 0.25
    else:
        if action.think_text:
            think = action.think_text.lower()
            if ("root cause:" in think or '"category"' in think) and "confidence" in think:
                score += 0.25

    if action.patch_text:
        has_search = "<<<<<<< SEARCH" in action.patch_text or "<<<<<<<" in action.patch_text
        has_replace = ">>>>>>> REPLACE" in action.patch_text or ">>>>>>>" in action.patch_text
        has_sep = "=======" in action.patch_text
        if has_search and has_replace and has_sep:
            score += 0.5
        elif has_search or has_replace:
            score += 0.25

    return round(score, 2)


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
    """Potential-based stability reward: Φ(current) - Φ(baseline). Returns ~-1.0..2.0."""
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

    if "assert" in patch_lower:
        search_blocks = re.findall(r"<<<<<<< SEARCH\n(.*?)=======", patch_text, re.DOTALL)
        replace_blocks = re.findall(r"=======\n(.*?)>>>>>>> REPLACE", patch_text, re.DOTALL)

        for search, replace in zip(search_blocks, replace_blocks):
            search_asserts = search.lower().count("assert")
            replace_asserts = replace.lower().count("assert")
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


def compute_diversity_penalty(
    current_categories: List[str],
    think_history: List[Dict[str, Any]],
) -> float:
    """Penalise repeating the same root-cause category across steps. Returns -1.8..0.0."""
    if not think_history or not current_categories:
        return 0.0

    primary_cat = current_categories[0]
    repeat_count = sum(
        1 for h in think_history
        if h.get("categories") and h["categories"][0] == primary_cat
    )

    if repeat_count == 0:
        return 0.0
    if repeat_count == 1:
        return -0.4
    if repeat_count == 2:
        return -0.9
    return -1.8


def compute_claim_novelty_score(
    current_claims: List[ThinkClaim],
    think_history: List[Dict[str, Any]],
) -> float:
    """Reward genuinely new entities or reasoning. Returns -0.5..0.5."""
    if not current_claims:
        return 0.0

    if not think_history:
        return 0.3  # First step — reward the initial hypothesis attempt

    prev_entities: set = set()
    prev_reason_sigs: set = set()
    for h in think_history:
        for ent in h.get("entities", []):
            prev_entities.add(ent.lower().strip())
        for sig in h.get("reason_signatures", []):
            prev_reason_sigs.add(sig)

    current_entities = {c.entity.lower().strip() for c in current_claims if c.entity}
    current_sigs = {c.reason[:35].lower().strip() for c in current_claims if c.reason}

    novel_ents = current_entities - prev_entities
    repeated_ents = current_entities & prev_entities
    novel_sigs = current_sigs - prev_reason_sigs
    repeated_sigs = current_sigs & prev_reason_sigs

    score = 0.0

    if novel_ents:
        score += min(len(novel_ents) * 0.15, 0.3)
    if novel_sigs:
        score += min(len(novel_sigs) * 0.10, 0.2)

    if repeated_ents and not novel_ents:
        score -= min(len(repeated_ents) * 0.20, 0.4)
    if repeated_sigs and not novel_sigs:
        score -= 0.15

    return round(max(-0.5, min(0.5, score)), 3)


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
    history: List[Dict[str, Any]] = think_history or []

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

    from agent.unified_agent import infer_category_from_patch
    inferred_cat = infer_category_from_patch(action.patch_text)

    breakdown = RewardBreakdown()
    breakdown.format_reward = compute_format_reward(action)
    breakdown.compile_reward = compute_compile_reward(
        patch_applied,
        syntax_error,
        rejected_by_validator=rejected_by_validator,
        rolled_back=rolled_back,
    )

    if rejected_by_validator:
        breakdown.patch_validation_signal = -0.8
    elif patch_applied and patch_result.get("validation_score") is not None:
        try:
            vs = float(patch_result["validation_score"])
            breakdown.patch_validation_signal = round(0.2 * max(0.0, min(1.0, vs)), 4)
        except (TypeError, ValueError):
            breakdown.patch_validation_signal = 0.0
    elif patch_applied:
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
    breakdown.anti_hack_penalty = compute_anti_hack_penalty(
        action.patch_text, files_modified, lines_changed,
    )
    breakdown.noop_patch_penalty = -0.5 if patch_applied and noop_patch else 0.0


    if regression_detected:
        regression_magnitude = max(0.0, baseline_pass_rate - post_pass_rate)
        severity_fraction = min(
            regression_magnitude / max(baseline_pass_rate, 0.05), 1.0
        )
        breakdown.regression_penalty = round(-1.0 - severity_fraction * 1.5, 2)
    else:
        breakdown.regression_penalty = 0.0

    if oracle_score is not None:
        oracle_gate = 0.0
        if oracle_score < -0.3:
            oracle_gate = round(-1.0 * abs(oracle_score + 0.3) * 2.0, 3)
        breakdown.oracle_reasoning_reward = round(float(oracle_score) + oracle_gate, 4)
        breakdown.reasoning_consistency_reward = 0.0
    else:
        breakdown.reasoning_consistency_reward = compute_reasoning_consistency(
            action.predicted_category, inferred_cat, action.think_text, action.patch_text,
        )
        breakdown.oracle_reasoning_reward = 0.0

    current_cats: List[str] = []
    if action.structured_think and action.structured_think.claims:
        current_cats = [c.category for c in action.structured_think.claims]
    elif action.predicted_category:
        current_cats = [action.predicted_category]
    breakdown.diversity_penalty = compute_diversity_penalty(current_cats, history)

    current_claims: List[ThinkClaim] = (
        action.structured_think.claims
        if action.structured_think and action.structured_think.claims
        else []
    )
    breakdown.claim_novelty_reward = compute_claim_novelty_score(current_claims, history)

    if not regression_detected and patch_applied and post_run_results:
        step = observation.step
        max_steps = step + observation.steps_remaining
        early_factor = 1.0 + 0.5 * max(0.0, 1.0 - step / max(max_steps, 1))

        if post_pass_rate >= 1.0:
            breakdown.terminal_bonus = round(5.0 * early_factor, 2)
        elif post_pass_rate > baseline_pass_rate + 0.5:
            breakdown.terminal_bonus = round(2.0 * early_factor, 2)
        elif post_pass_rate > baseline_pass_rate + 0.3:
            breakdown.terminal_bonus = round(1.0 * early_factor, 2)

    oracle_component = (
        breakdown.oracle_reasoning_reward * 2.5
        if oracle_score is not None
        else breakdown.reasoning_consistency_reward * 0.5
    )

    breakdown.total_reward = round(
        breakdown.format_reward            * 0.5
        + breakdown.compile_reward         * 1.0
        + breakdown.stability_reward       * 2.0
        + breakdown.causal_proximity_reward * 0.5
        + breakdown.failure_entropy_reward * 0.5
        + breakdown.anti_hack_penalty      * 1.5
        + oracle_component
        + breakdown.diversity_penalty      * 1.0
        + breakdown.claim_novelty_reward   * 1.0
        + breakdown.patch_validation_signal * 1.0
        + breakdown.noop_patch_penalty     * 1.0
        + breakdown.regression_penalty     * 1.5
        + breakdown.terminal_bonus         * 1.0,
        4,
    )

    return breakdown
