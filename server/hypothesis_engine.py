import statistics
from typing import Any, Dict, List, Optional

try:
    from ..models import Hypothesis
except ImportError:
    from models import Hypothesis

from .state import EpisodeState

def pass_rate(runs: List[Any]) -> float:
    """Calculate pass rate from a list of runs (RunRecord objects with .passed)."""
    if not runs:
        return 0.0
    passed = sum(1 for r in runs if getattr(r, "passed", False))
    return passed / len(runs)

def compute_duration_fingerprint(runs: List[Any]) -> Dict[str, float]:
    """Compute timing statistics that drive confidence boosting."""
    durations = [r.duration_ms for r in runs if hasattr(r, "duration_ms") and r.duration_ms is not None]
    if not durations:
        return {"mean_ms": 0.0, "std_ms": 0.0, "cv": 0.0, "flakiness_score": 0.0}
    mean_ms = statistics.mean(durations)
    std_ms = statistics.stdev(durations) if len(durations) > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0
    pr = pass_rate(runs)
    flakiness_score = round((1.0 - pr) * 0.6 + min(cv, 1.0) * 0.4, 4)
    return {
        "mean_ms": round(mean_ms, 1),
        "std_ms": round(std_ms, 1),
        "cv": round(cv, 4),
        "flakiness_score": flakiness_score,
    }

def make_secondary_hypothesis(
    primary_category: str,
    top_error: str,
    pr: float,
) -> Optional[Hypothesis]:
    """Generate a runner-up hypothesis when primary confidence is low."""
    FALLBACKS: Dict[str, tuple] = {
        "NONDETERMINISM":           ("SHARED_STATE",        "RESET_STATE"),
        "SHARED_STATE":             ("TIMING_RACE",         "ADD_TIMING_GUARD"),
        "TIMING_RACE":              ("ASYNC_DEADLOCK",      "EXTRACT_ASYNC_SCOPE"),
        "ASYNC_DEADLOCK":           ("TIMING_RACE",         "ADD_TIMING_GUARD"),
        "ORDER_DEPENDENCY":         ("SHARED_STATE",        "RESET_STATE"),
        "EXTERNAL_DEPENDENCY":      ("TIMING_RACE",         "ADD_TIMING_GUARD"),
        "RESOURCE_LEAK":            ("SHARED_STATE",        "RESET_STATE"),
        "INFRASTRUCTURE_SENSITIVE": ("EXTERNAL_DEPENDENCY", "ISOLATE_BOUNDARY"),
    }
    fallback = FALLBACKS.get(primary_category)
    if not fallback:
        return None
    secondary_category, secondary_action = fallback
    secondary_confidence = max(0.1, min(0.4, (1.0 - pr) * 0.4))
    try:
        return Hypothesis(
            root_cause_category=secondary_category,
            confidence=secondary_confidence,
            evidence=[top_error or "intermittent_failure"],
            suggested_action=secondary_action,
        )
    except Exception:
        return None

def infer_hypothesis(episode: EpisodeState, runs: List[Any], test_id: str) -> Hypothesis:
    """
    V2 Enhanced: 5-level priority stack for root cause inference.
    """
    error_types = [getattr(r, "error_type", "") or "" for r in runs if not getattr(r, "passed", False)]
    top_error = error_types[0] if error_types else ""
    pr = pass_rate(runs)
    confidence = max(0.3, min(0.95, 1.0 - abs(pr - 0.5)))
    
    # ── Level 1: Infrastructure Sensitivity ─────────────────────────────────
    if getattr(episode, "infrastructure_sensitive", False):
        evidence = [
            "Infrastructure-sensitive flakiness detected via chaos probe",
            top_error or "intermittent_failure",
        ]
        return Hypothesis(
            root_cause_category="INFRASTRUCTURE_SENSITIVE",
            confidence=min(0.95, confidence + 0.2),
            evidence=evidence,
            suggested_action="ISOLATE_BOUNDARY" if top_error else "GATHER_EVIDENCE",
        )
    
    # ── Level 2: Causal Graph Warnings (ASYNC_DEADLOCK) ───────────────────
    causal_graph_dict = getattr(episode, "causal_graph_dict", None)
    if causal_graph_dict:
        boundary_warnings = causal_graph_dict.get("boundary_warnings", [])
        for warning in boundary_warnings:
            if "threading.Lock" in warning and "async" in warning.lower():
                return Hypothesis(
                    root_cause_category="ASYNC_DEADLOCK",
                    confidence=min(0.95, confidence + 0.15),
                    evidence=[f"Causal graph detected: {warning}", top_error or "intermittent_failure"],
                    suggested_action="EXTRACT_ASYNC_SCOPE",
                )
            if "blocking" in warning.lower() and "async" in warning.lower():
                return Hypothesis(
                    root_cause_category="ASYNC_DEADLOCK",
                    confidence=min(0.95, confidence + 0.15),
                    evidence=[f"Causal graph detected: {warning}", top_error or "intermittent_failure"],
                    suggested_action="EXTRACT_ASYNC_SCOPE",
                )
    
    # ── Level 3: Boundary Nodes (External Dependencies) ────────────────────
    if causal_graph_dict:
        boundary_nodes = causal_graph_dict.get("boundary_nodes", [])
        nodes_by_type = {}
        for node_id in boundary_nodes:
            for node in causal_graph_dict.get("nodes", []):
                if node.get("id") == node_id:
                    boundary_type = node.get("boundary")
                    if boundary_type:
                        nodes_by_type.setdefault(boundary_type, []).append(node_id)
                    break
        
        if "db" in nodes_by_type or "queue" in nodes_by_type:
            return Hypothesis(
                root_cause_category="EXTERNAL_DEPENDENCY",
                confidence=min(0.95, confidence + 0.1),
                evidence=[f"Boundary nodes detected: {list(nodes_by_type.keys())}", top_error or "intermittent_failure"],
                suggested_action="MOCK_DEPENDENCY",
            )
        if "http" in nodes_by_type or "grpc" in nodes_by_type:
            return Hypothesis(
                root_cause_category="EXTERNAL_DEPENDENCY",
                confidence=min(0.95, confidence + 0.1),
                evidence=[f"HTTP/gRPC boundary detected: {list(nodes_by_type.keys())}", top_error or "intermittent_failure"],
                suggested_action="ISOLATE_BOUNDARY",
            )
    
    # ── Level 4: Error String Analysis (V1 pattern matching) ───────────────
    category = "NONDETERMINISM"
    suggested_action = "GATHER_EVIDENCE" if confidence < 0.5 else "ADD_TIMING_GUARD"

    if "Timeout" in top_error or "TimeoutError" in top_error:
        category = "TIMING_RACE"
        suggested_action = "ADD_TIMING_GUARD"
    elif "Connection" in top_error or "connection" in top_error.lower():
        category = "EXTERNAL_DEPENDENCY"
        suggested_action = "MOCK_DEPENDENCY"
    elif "Assertion" in top_error:
        category = "ORDER_DEPENDENCY"
        suggested_action = "ADD_SYNCHRONIZATION"
    elif "state" in top_error.lower() or "shared" in top_error.lower():
        category = "SHARED_STATE"
        suggested_action = "RESET_STATE"
    elif "resource" in top_error.lower() or "leak" in top_error.lower():
        category = "RESOURCE_LEAK"
        suggested_action = "RESET_STATE"

    # ── Improvement 4: Duration fingerprint confidence boosting ────────────
    fp = episode.duration_fingerprint
    if fp and fp.get("cv", 0) > 0.3:
        timing_boost = min(0.2, fp["cv"] * 0.5)
        confidence = min(0.95, confidence + timing_boost)
        if category == "NONDETERMINISM":
            category = "TIMING_RACE"
            suggested_action = "ADD_TIMING_GUARD"

    evidence = [
        test_id.split("::")[-1],
        top_error or "intermittent_failure",
    ]

    primary = Hypothesis(
        root_cause_category=category,
        confidence=confidence,
        evidence=evidence,
        suggested_action=suggested_action,
    )

    # ── Improvement 5: Top-2 hypothesis tracking ───────────────────────────
    if confidence <= 0.5:
        episode.secondary_hypothesis = make_secondary_hypothesis(
            primary_category=category,
            top_error=top_error,
            pr=pr,
        )
    else:
        episode.secondary_hypothesis = None

    return primary
