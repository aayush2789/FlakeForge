import subprocess
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..models import FlakeForgeAction, Hypothesis, PatchRecord
except ImportError:
    from models import FlakeForgeAction, Hypothesis, PatchRecord

def canonical_action(action_type: str) -> str:
    return {
        "detect_flakiness": "GATHER_EVIDENCE",
        "analyze_logs": "GATHER_EVIDENCE",
        "add_sleep": "ADD_TIMING_GUARD",
        "add_lock": "ADD_SYNCHRONIZATION",
        "mock_dependency": "MOCK_DEPENDENCY",
        "isolate_state": "RESET_STATE",
        "reorder_execution": "RESET_STATE",
        "retry_test": "ADD_RETRY",
    }.get(action_type, action_type)

def injection_points_from_hypothesis(hypothesis: Optional[Hypothesis], resolved_target: Dict[str, Any]) -> List[Dict[str, str]]:
    if resolved_target.get("type") == "function":
        return [{"function_name": resolved_target.get("identifier", "test_flaky_case"), "position": "entry"}]
    if not hypothesis:
        return [{"function_name": "test_flaky_case", "position": "entry"}]
    points: List[Dict[str, str]] = []
    for evidence in hypothesis.evidence:
        fn = evidence.split(":", 1)[0].strip() if ":" in evidence else evidence.strip()
        if fn:
            points.append({"function_name": fn, "position": "entry"})
    return points or [{"function_name": "test_flaky_case", "position": "entry"}]

def extract_log_snippets(runs: List[Any]) -> List[str]:
    snippets: List[str] = []
    for run in runs:
        if getattr(run, "stderr_excerpt", None):
            for line in run.stderr_excerpt.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    snippets.append(line)
        payload = {
            "passed": getattr(run, "passed", False),
            "duration_ms": getattr(run, "duration_ms", 0),
            "error_type": getattr(run, "error_type", ""),
            "error_message": getattr(run, "error_message", ""),
        }
        snippets.append(json.dumps(payload))
    return snippets[:3]

def build_patch_spec(action: FlakeForgeAction, resolved_target: Dict[str, Any]) -> Dict[str, Any]:
    c_action = canonical_action(action.action_type)
    if c_action == "ADD_TIMING_GUARD":
        return {
            "operation": "insert_before",
            "target": {"type": "call", "identifier": resolved_target.get("identifier", "await")},
            "code_template": "await asyncio.sleep({delay_ms} / 1000)",
            "parameters": {"delay_ms": action.parameters.get("delay_ms", 100)},
        }
    if c_action == "ADD_SYNCHRONIZATION":
        return {
            "operation": "wrap_with",
            "target": {"type": "function", "identifier": resolved_target.get("identifier", "test")},
            "code_template": "with {lock_var}:\\n    {body}",
            "parameters": {"lock_var": "_flakeforge_lock", "primitive": action.parameters.get("primitive", "threading.Lock")},
        }
    if c_action == "MOCK_DEPENDENCY":
        return {
            "operation": "add_decorator",
            "target": {"type": "function", "identifier": resolved_target.get("identifier", "test")},
            "code_template": "@unittest.mock.patch('{target}')",
            "parameters": {"target": action.parameters.get("target", "builtins.open")},
        }
    if c_action == "RESET_STATE":
        return {
            "operation": "ensure_reset_fixture",
            "target": {"scope": action.parameters.get("scope", "function")},
            "code_template": "",
            "parameters": {},
        }
    if c_action == "ADD_RETRY":
        return {
            "operation": "ensure_retry_wrapper",
            "target": {
                "function_name": resolved_target.get("identifier", "test"),
                "max_attempts": action.parameters.get("max_attempts", 3),
                "backoff_ms": action.parameters.get("backoff_ms", 0),
            },
            "code_template": "",
            "parameters": {},
        }
    if c_action == "SEED_RANDOMNESS":
        return {
            "operation": "ensure_seed_call",
            "target": {
                "function_name": resolved_target.get("identifier", "test"),
                "library": action.parameters.get("library", "random"),
            },
            "code_template": "",
            "parameters": {},
        }
    # V2 Deep-Action Handlers
    if c_action == "REFACTOR_CONCURRENCY":
        return {
            "operation": "refactor_concurrency_primitive",
            "target": {
                "function_name": action.parameters.get("target_function", ""),
                "from_primitive": action.parameters.get("from_primitive", ""),
                "to_primitive": action.parameters.get("to_primitive", ""),
            },
            "code_template": "",
            "parameters": dict(action.parameters),
        }
    if c_action == "ISOLATE_BOUNDARY":
        return {
            "operation": "isolate_boundary_call",
            "target": {
                "boundary_call": action.parameters.get("boundary_call", ""),
                "pattern": action.parameters.get("pattern", ""),
            },
            "code_template": "",
            "parameters": dict(action.parameters),
        }
    if c_action == "EXTRACT_ASYNC_SCOPE":
        return {
            "operation": "extract_async_scope",
            "target": {
                "function_name": action.parameters.get("target_function", ""),
                "direction": action.parameters.get("direction", ""),
            },
            "code_template": "",
            "parameters": dict(action.parameters),
        }
    if c_action == "HARDEN_IDEMPOTENCY":
        return {
            "operation": "harden_idempotency",
            "target": {
                "state_target": action.parameters.get("state_target", ""),
                "key_strategy": action.parameters.get("key_strategy", ""),
            },
            "code_template": "",
            "parameters": dict(action.parameters),
        }
    raise ValueError(f"Unsupported action type for patching: {action.action_type}")
