from typing import Any, Dict

try:
    from ..models import FlakeForgeObservation, Hypothesis
except ImportError:
    from models import FlakeForgeObservation, Hypothesis

try:
    from ..server.hypothesis_engine import compute_duration_fingerprint
except ImportError:
    from server.hypothesis_engine import compute_duration_fingerprint

def _hypothesis_payload(hypothesis: Hypothesis) -> Dict[str, Any]:
    return {
        "root_cause_category": hypothesis.root_cause_category,
        "confidence": hypothesis.confidence,
        "evidence": list(hypothesis.evidence),
        "suggested_action": hypothesis.suggested_action,
        "reasoning_steps": list(hypothesis.reasoning_steps),
        "uncertainty": hypothesis.uncertainty,
        "next_best_action": hypothesis.next_best_action,
        "predicted_effect": hypothesis.predicted_effect,
    }

def _first_lines(text: str, line_count: int) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[:line_count])

def build_compact_observation(observation: FlakeForgeObservation, include_sources: bool = False, for_judge: bool = False) -> Dict[str, Any]:
    """Unifies observation payload construction for Fixer, Analyzer, and Judge."""
    run_history = [
        {
            "passed": r.passed,
            "error_type": r.error_type,
            "duration_ms": r.duration_ms,
        }
        for r in observation.run_history[-5:]
    ]

    # Shared common attributes between judge and agent payload
    payload: Dict[str, Any] = {
        "test_identifier": observation.test_identifier,
        "step": observation.step,
        "current_pass_rate": observation.current_pass_rate,
        "baseline_pass_rate": observation.baseline_pass_rate,
        "run_history": run_history,
        "current_hypothesis": _hypothesis_payload(observation.current_hypothesis) if observation.current_hypothesis else None,
        "secondary_hypothesis": {
            "root_cause_category": observation.secondary_hypothesis.root_cause_category,
            "confidence": observation.secondary_hypothesis.confidence,
            "suggested_action": getattr(observation.secondary_hypothesis, "suggested_action", ""),
        } if observation.secondary_hypothesis else None,
    }

    if for_judge:
        # Extra parts needed solely by Judge
        fp = observation.duration_fingerprint or compute_duration_fingerprint(observation.run_history)
        causal = observation.causal_graph or {}
        payload.update({
            "infrastructure_sensitive": observation.infrastructure_sensitive,
            "duration_fingerprint": fp,
            "boundary_warnings": list(causal.get("boundary_warnings", []))[:5],
            "boundary_nodes": list(causal.get("boundary_nodes", []))[:5],
            "test_function_source": _first_lines(observation.test_function_source, 50),
        })
    else:
        # Extra parts needed solely by Agent
        payload.update({
            "episode_id": observation.episode_id,
            "steps_remaining": observation.steps_remaining,
            "async_markers": list(observation.async_markers)[:20],
            "log_snippets": list(observation.log_snippets)[-3:],
            "duration_fingerprint": observation.duration_fingerprint,
            "last_actions": list(observation.last_actions)[-3:],
            "last_outcomes": list(observation.last_outcomes)[-3:],
            "prediction_error_history": list(observation.prediction_error_history)[-5:],
            "failure_pattern_summary": observation.failure_pattern_summary,
            "causal_hints": list(observation.causal_hints)[-5:],
            "reflection": observation.reflection,
            "actions_tried": list(set(observation.last_actions)),
        })
        if include_sources:
            payload["test_function_source"] = _first_lines(observation.test_function_source, 50)
            payload["source_under_test"] = _first_lines(observation.source_under_test, 50)

    return payload
