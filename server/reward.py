"""V3.1 Reward Architecture — ten-signal verifiable reward, no LLM judge.

Signals
-------
1.  Format Compliance:       valid JSON <think> + valid <patch> structure
2.  Compile/Syntax:          modified file parses without errors
3.  Stability Delta:         pass-rate improvement over baseline            (weight 2.0)
4.  Causal Proximity:        patch touches causal frontier
5.  Failure Entropy:         fewer distinct error modes
6.  Anti-Hack Penalty:       sleep injection, test deletion, skip decorators
7.  Oracle Reasoning:        structured-claim differential verification     (weight 2.5)
8.  Oracle Gate Penalty:     hard amplification when oracle strongly refutes
9.  Diversity Penalty:       penalise repeating same root-cause category    (NEW)
10. Claim Novelty Reward:    bonus for genuinely new entities / reasoning   (NEW)

Key fixes vs V3.0
-----------------
- Oracle weight 1.0 → 2.5  (oracle now dominates over stability on short episodes)
- Oracle gate: if oracle_score < -0.3, additional -1.0 applied after weighting
- Regression penalty: was fixed -3.0 × 2.0 = -6.0; now scaled by magnitude (-0.75 → -3.75)
- Terminal bonus: 2.0 → 5.0 for pass_rate == 1.0 (agent MUST learn success = big reward)
- Diversity penalty: forces exploration when same category is repeated across steps
- Claim novelty: rewards the agent for introducing fresh entities and reasoning

Research basis
--------------
- DeepSeek R1: format+correctness reward (no human prefs)
- RLEF: execution-verified compilability
- Reflexion NeurIPS 2023: self-consistency for reasoning
- DIVER (2025): diversity-incentivised exploration via intrinsic diversity reward
- UCAS (2025): uncertainty-aware advantage shaping, entropy-collapse prevention
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


# ── Signal 1: Format Compliance ───────────────────────────────────────────────

def compute_format_reward(action: FlakeForgeAction) -> float:
    """Check that the model produced a valid <think> JSON block and a <patch> block.

    JSON think:  +0.5 if StructuredThink parsed without penalty.
                 +0.25 if parsed with minor penalty.
                  0.0 if completely malformed.
    Patch block: +0.5 if valid search/replace hunk present.
    """
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


# ── Signal 2: Compile/Syntax ────────────────────────────────────────────────

def compute_compile_reward(
    patch_applied_successfully: bool,
    syntax_error: Optional[str] = None,
    *,
    rejected_by_validator: bool = False,
    rolled_back: bool = False,
) -> float:
    """Check that the patched file compiles without errors.

    Returns:
        1.0 if patch applied + no syntax errors
        0.5 if patch applied but has warnings
       -1.0 if patch failed for generic reasons (not validator rejection)
       -0.5 if patch was rejected by PatchValidator (no filesystem mutation)
        0.0 if apply was rolled back after a sanity syntax check (disk restored)
       -0.5 if syntax error after successful apply
    """
    if rejected_by_validator:
        return -0.5
    if rolled_back:
        return 0.0
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

    Returns:
        Float in [-1.0, 2.0]
    """
    delta = current_pass_rate - baseline_pass_rate

    if current_pass_rate >= 1.0:
        return 2.0
    elif delta > 0:
        return round(min(delta * 3.0, 1.5), 4)
    elif delta < 0:
        return round(max(delta * 5.0, -1.0), 4)
    else:
        return -0.1


# ── Signal 4: Causal Proximity ──────────────────────────────────────────────

def compute_causal_proximity_reward(
    patch_files: List[str],
    failure_frontier: str,
    call_chain: List[str],
    boundary_crossings: List[str],
) -> float:
    """Reward patches that touch code near the failure site.

    Returns:
        Float in [-0.5, 1.0]
    """
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


# ── Signal 5: Failure Entropy Reduction ─────────────────────────────────────

def compute_entropy_reward(
    pre_entropy: float,
    post_entropy: float,
) -> float:
    """Reward reduction in failure mode entropy.

    Returns:
        Float in [-0.5, 1.0]
    """
    if pre_entropy <= 0 and post_entropy <= 0:
        return 0.5

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

    Returns:
        Float in [-2.0, 0.0]
    """
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


# ── Signal 9: Diversity Penalty ──────────────────────────────────────────────

def compute_diversity_penalty(
    current_categories: List[str],
    think_history: List[Dict[str, Any]],
) -> float:
    """Penalise repeating the same root-cause category across episode steps.

    Inspired by DIVER (2025): diversity-incentivised exploration prevents the
    agent from getting stuck in a local search loop over one category.

    Penalty schedule:
      0 prior uses  →  0.0    (fresh exploration — no penalty)
      1 prior use   → -0.4    (mild nudge)
      2 prior uses  → -0.9    (strong nudge)
      3+ prior uses → -1.8    (force branch — agent should try something new)

    Returns: float in [-1.8, 0.0]
    """
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


# ── Signal 10: Claim Novelty Reward ──────────────────────────────────────────

def compute_claim_novelty_score(
    current_claims: List[ThinkClaim],
    think_history: List[Dict[str, Any]],
) -> float:
    """Reward claims that introduce genuinely new entities or reasoning.

    Prevents the agent from copy-pasting its previous think block with only
    minor tweaks (e.g. timeout = 0.2 → 0.25).  Novel entity names and
    distinct causal reasoning earn a positive score; repeated ones earn a
    penalty.

    Returns: float in [-0.5, 0.5]
    """
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


# ── Reasoning Consistency (legacy fallback) ───────────────────────────────────

def compute_reasoning_consistency(
    predicted_category: str,
    inferred_category_from_patch: str,
    think_text: str,
    patch_text: str,
) -> float:
    """Category match between predicted and patch-inferred root cause.

    Used only when the oracle has not been run (no structured_think).
    Returns: float in [-0.5, 0.5]
    """
    if not predicted_category or not inferred_category_from_patch:
        return 0.0
    if predicted_category == inferred_category_from_patch:
        return 0.5
    related = RELATED_CATEGORIES.get(predicted_category, set())
    if inferred_category_from_patch in related:
        return 0.25
    return -0.5


# ── Composite Reward ──────────────────────────────────────────────────────────

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
    """Compute the full ten-signal verifiable reward.

    Weight allocation (oracle path):
      stability_reward        × 2.0   (~0.45 of practical range)
      oracle_reasoning_reward × 2.5   (~0.35)  ← amplified from 1.0
      oracle_gate_penalty     × 1.0   (hard gate, fires if oracle < -0.3)
      compile_reward          × 1.0
      anti_hack_penalty       × 1.5
      causal_proximity        × 0.5
      failure_entropy         × 0.5
      format_reward           × 0.5
      diversity_penalty       × 1.0   (NEW)
      claim_novelty_reward    × 1.0   (NEW)
      terminal_bonus          × 1.0   (5.0 for full solve, was 2.0)
      regression_penalty      × 1.5   (scaled, was hard -3.0 × 2.0)

    When oracle_score is None the legacy reasoning_consistency signal runs.
    """
    history: List[Dict[str, Any]] = think_history or []

    patch_applied = patch_result.get("success", False)
    rejected_by_validator = bool(patch_result.get("rejected_by_validator", False))
    rolled_back = bool(patch_result.get("rolled_back", False))
    syntax_error = patch_result.get("syntax_error") if patch_applied else None
    files_modified = patch_result.get("files_modified", [])
    lines_changed = patch_result.get("lines_changed", 0)
    noop_patch = bool(patch_result.get("noop", False))
    protected_file = bool(patch_result.get("protected_file", False))
    regression_detected = bool(regression_detected or patch_result.get("regression_detected", False))

    post_pass_count = sum(1 for r in post_run_results if r.get("passed", False))
    post_pass_rate = post_pass_count / max(len(post_run_results), 1)

    post_errors = [r.get("error_type", "") for r in post_run_results if not r.get("passed", True)]
    post_counter = Counter(e for e in post_errors if e)
    if post_counter:
        total = sum(post_counter.values())
        post_entropy = -sum((c / total) * math.log2(c / total) for c in post_counter.values())
        max_e = math.log2(len(post_counter)) if len(post_counter) > 1 else 1.0
        post_entropy = post_entropy / max_e if max_e > 0 else 0.0
    else:
        post_entropy = 0.0

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

    # PatchValidator shaping: invalid patch → -0.3; valid → +0.2 × validation score
    if rejected_by_validator:
        breakdown.patch_validation_signal = -0.3
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
    breakdown.protected_file_penalty = -2.0 if protected_file else 0.0

    # ── Scaled regression penalty (was hard -3.0 × 2.0 = -6.0) ──────────────
    if regression_detected:
        regression_magnitude = max(0.0, baseline_pass_rate - post_pass_rate)
        severity_fraction = min(
            regression_magnitude / max(baseline_pass_rate, 0.05), 1.0
        )
        # Ranges from -1.0 (tiny regression) to -2.5 (catastrophic)
        breakdown.regression_penalty = round(-1.0 - severity_fraction * 1.5, 2)
    else:
        breakdown.regression_penalty = 0.0

    # ── Oracle (amplified weight + hard gate) ─────────────────────────────────
    if oracle_score is not None:
        breakdown.oracle_reasoning_reward = round(float(oracle_score), 4)
        breakdown.reasoning_consistency_reward = 0.0

        # Hard gate: if oracle strongly disagrees (< -0.3), apply extra penalty.
        # This ensures the oracle can't be drowned out by a good pass-rate.
        if oracle_score < -0.3:
            breakdown.oracle_gate_penalty = round(-1.0 * abs(oracle_score + 0.3) * 2.0, 3)
    else:
        breakdown.reasoning_consistency_reward = compute_reasoning_consistency(
            action.predicted_category, inferred_cat, action.think_text, action.patch_text,
        )
        breakdown.oracle_reasoning_reward = 0.0
        breakdown.oracle_gate_penalty = 0.0

    # ── Diversity penalty — repeat-category suppression ───────────────────────
    current_cats: List[str] = []
    if action.structured_think and action.structured_think.claims:
        current_cats = [c.category for c in action.structured_think.claims]
    elif action.predicted_category:
        current_cats = [action.predicted_category]
    breakdown.diversity_penalty = compute_diversity_penalty(current_cats, history)

    # ── Claim novelty reward — encourage fresh reasoning ──────────────────────
    current_claims: List[ThinkClaim] = (
        action.structured_think.claims
        if action.structured_think and action.structured_think.claims
        else []
    )
    breakdown.claim_novelty_reward = compute_claim_novelty_score(current_claims, history)

    # ── Terminal bonus (greatly amplified — success MUST be strongly rewarded) ─
    if not regression_detected and patch_applied and post_run_results:
        if post_pass_rate >= 1.0:
            breakdown.terminal_bonus = 5.0   # Solving it = massive positive signal
        elif post_pass_rate > baseline_pass_rate + 0.5:
            breakdown.terminal_bonus = 2.0   # Major improvement
        elif post_pass_rate > baseline_pass_rate + 0.3:
            breakdown.terminal_bonus = 1.0   # Meaningful progress

    # ── Total reward ──────────────────────────────────────────────────────────
    oracle_component = (
        breakdown.oracle_reasoning_reward * 2.5      # weight × 2.5 (was 1.0)
        + breakdown.oracle_gate_penalty * 1.0
        if oracle_score is not None
        else breakdown.reasoning_consistency_reward * 0.5
    )

    breakdown.total_reward = round(
        breakdown.format_reward           * 0.5
        + breakdown.compile_reward        * 1.0
        + breakdown.stability_reward      * 2.0
        + breakdown.causal_proximity_reward * 0.5
        + breakdown.failure_entropy_reward * 0.5
        + breakdown.anti_hack_penalty     * 1.5
        + oracle_component
        + breakdown.diversity_penalty     * 1.0
        + breakdown.claim_novelty_reward  * 1.0
        + breakdown.patch_validation_signal * 1.0
        + breakdown.noop_patch_penalty    * 1.0
        + breakdown.protected_file_penalty * 1.0
        + breakdown.regression_penalty    * 1.5     # scaled, not fixed -6
        + breakdown.terminal_bonus        * 1.0,
        4,
    )

    return breakdown
