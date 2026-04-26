"""V3 Observation Utilities — builds observations from deep flakiness signals.

Changes from V2:
- No hypothesis fields
- Deep flakiness signals integrated
- Causal frontier extraction
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from models import FlakeForgeObservation, RunRecord, PatchRecord
except ImportError:
    from ..models import FlakeForgeObservation, RunRecord, PatchRecord


def build_observation_from_state(state: Any) -> FlakeForgeObservation:
    """Build a V3 observation from an EpisodeState object.

    This utility bridges the state → observation gap when the state
    object is from a different context (e.g., deserialized from JSON).
    """
    import math

    durations = [r.duration_ms for r in (state.run_history or [])[-20:]]
    dur_mean = sum(durations) / max(len(durations), 1) if durations else 0
    dur_std = (
        math.sqrt(sum((d - dur_mean) ** 2 for d in durations) / max(len(durations), 1))
        if durations else 0
    )

    return FlakeForgeObservation(
        episode_id=state.episode_id,
        test_identifier=state.test_identifier,
        step=state.step_count,
        steps_remaining=state.steps_remaining,
        test_function_source=state.current_test_source,
        source_under_test=state.current_source_under_test,
        source_file=getattr(state, "source_file", None) or "",
        run_history=state.run_history[-20:] if state.run_history else [],
        current_pass_rate=state.current_pass_rate,
        baseline_pass_rate=state.baseline_pass_rate,
        patches_applied=state.patches_applied or [],
        total_diff_lines=state.total_diff_lines,
        # Deep signals
        module_cache_violations=state.module_cache_violations or [],
        fixture_scope_risks=state.fixture_scope_risks or [],
        mock_residue_sites=state.mock_residue_sites or [],
        import_side_effect_files=state.import_side_effect_files or [],
        async_contamination_alive=state.async_contamination_alive or False,
        # Causal
        failure_frontier=state.failure_frontier or "",
        call_chain_to_frontier=state.call_chain_to_frontier or [],
        boundary_crossings=state.boundary_crossings or [],
        # iDFlakies
        order_dependency_detected=state.order_dependency_detected or False,
        infrastructure_sensitive=state.infrastructure_sensitive or False,
        # Graph
        causal_graph=state.causal_graph,
        causal_hints=state.causal_hints or [],
        # Failure
        failing_stack_trace=state.failing_stack_trace or "",
        duration_fingerprint={"mean": dur_mean, "std": dur_std},
        # Context
        last_think_text=state.last_think_text or "",
        last_patch_text=state.last_patch_text or "",
        last_reward=state.last_reward or 0.0,
        file_tree=state.file_tree or [],
    )


def summarize_observation(obs: FlakeForgeObservation) -> Dict[str, Any]:
    """Create a compact summary of an observation for logging/debugging."""
    return {
        "test": obs.test_identifier,
        "step": f"{obs.step}/{obs.step + obs.steps_remaining}",
        "pass_rate": f"{obs.current_pass_rate:.2f}",
        "baseline": f"{obs.baseline_pass_rate:.2f}",
        "patches_applied": len(obs.patches_applied),
        "deep_signals": {
            "cache_violations": len(obs.module_cache_violations),
            "fixture_risks": len(obs.fixture_scope_risks),
            "mock_residue": len(obs.mock_residue_sites),
            "import_effects": len(obs.import_side_effect_files),
            "async_alive": obs.async_contamination_alive,
        },
        "causal": {
            "frontier": obs.failure_frontier[:50] if obs.failure_frontier else "",
            "chain_depth": len(obs.call_chain_to_frontier),
            "boundaries": len(obs.boundary_crossings),
        },
        "order_dep": obs.order_dependency_detected,
        "infra_sensitive": obs.infrastructure_sensitive,
    }
