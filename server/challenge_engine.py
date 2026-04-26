"""Challenge engine — analyzes user-submitted flaky tests via pattern matching and optional LLM."""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .api_models import ChallengeAnalysis

logger = logging.getLogger(__name__)

# Pattern detectors: each returns (category, confidence, explanation, suggested_fix)
_PATTERN_DETECTORS: List = []


def _register(fn):
    _PATTERN_DETECTORS.append(fn)
    return fn


@_register
def _detect_timing_race(code: str, test_code: str) -> Optional[Tuple[str, float, str, str]]:
    indicators = ["threading.Thread", "thread", "Thread(", "Lock(", "global "]
    race_signals = ["global ", "temp =", "counter", "+= 1", "counter ="]
    thread_count = sum(1 for i in indicators if i in code or i in test_code)
    race_count = sum(1 for s in race_signals if s in code)
    if thread_count >= 2 and race_count >= 2:
        return (
            "concurrency",
            0.92,
            "Non-atomic read-modify-write detected in threaded context. "
            "Multiple threads access shared state without synchronization.",
            "Wrap the critical section with threading.Lock() to make the operation atomic.",
        )
    if thread_count >= 1 and race_count >= 1:
        return (
            "concurrency",
            0.75,
            "Shared mutable state accessed from threads without explicit locking.",
            "Add threading.Lock() around shared state access.",
        )
    return None


@_register
def _detect_async_wait(code: str, test_code: str) -> Optional[Tuple[str, float, str, str]]:
    combined = code + test_code
    if "asyncio" in combined and ("timeout" in combined or "wait_for" in combined):
        return (
            "async_wait",
            0.88,
            "Async operation with tight timeout detected. Under load, the event loop "
            "may not schedule the coroutine in time.",
            "Increase timeout or use asyncio.Lock() for proper async synchronization.",
        )
    if "await" in combined and ("gather" in combined or "create_task" in combined):
        if "session" in combined.lower() or "lock" in combined.lower():
            return (
                "async_wait",
                0.82,
                "Concurrent async tasks sharing a session or resource without async locking.",
                "Use asyncio.Lock() to serialize access, or create separate sessions per task.",
            )
    return None


@_register
def _detect_db_commit(code: str, test_code: str) -> Optional[Tuple[str, float, str, str]]:
    combined = code + test_code
    has_db = any(k in combined for k in ["sqlite3", "connect(", "execute(", "cursor"])
    has_insert = "INSERT" in combined or "insert" in combined
    missing_commit = "commit()" not in combined
    if has_db and has_insert and missing_commit:
        return (
            "resource_leak",
            0.95,
            "Database write without explicit commit(). Transaction may not be flushed "
            "before the read query, causing intermittent data loss.",
            "Add conn.commit() after the INSERT to ensure data is persisted before reading.",
        )
    return None


@_register
def _detect_external_dep(code: str, test_code: str) -> Optional[Tuple[str, float, str, str]]:
    combined = code + test_code
    external_signals = ["requests.post", "requests.get", "httpx", "urllib", "sandbox", "api."]
    hits = sum(1 for s in external_signals if s in combined)
    if hits >= 1:
        return (
            "network",
            0.85,
            "Test depends on an external HTTP endpoint. Network latency, DNS resolution, "
            "and endpoint availability introduce non-determinism.",
            "Mock the external dependency using unittest.mock.patch or responses library.",
        )
    return None


@_register
def _detect_shared_state(code: str, test_code: str) -> Optional[Tuple[str, float, str, str]]:
    combined = code + test_code
    global_count = combined.count("global ")
    class_var_pattern = re.findall(r"class\s+\w+.*?:\s*\n\s+\w+\s*=", combined, re.DOTALL)
    if global_count >= 2 or len(class_var_pattern) >= 1:
        if "clear()" not in combined and "reset" not in combined.lower():
            return (
                "shared_state",
                0.78,
                "Mutable global or class-level state is shared across test runs without cleanup.",
                "Reset shared state in a fixture teardown or use test-local copies.",
            )
    return None


@_register
def _detect_nondeterminism(code: str, test_code: str) -> Optional[Tuple[str, float, str, str]]:
    combined = code + test_code
    nd_signals = ["random.", "time.time()", "datetime.now()", "uuid.", "shuffle("]
    hits = sum(1 for s in nd_signals if s in combined)
    if hits >= 1:
        return (
            "nondeterminism",
            0.80,
            "Test relies on non-deterministic values (random, time, UUID) without seeding.",
            "Seed random generators or mock time/uuid to produce deterministic results.",
        )
    return None


@_register
def _detect_fixture_scope(code: str, test_code: str) -> Optional[Tuple[str, float, str, str]]:
    combined = code + test_code
    if "scope=" in combined and ("session" in combined or "module" in combined):
        if "yield" in combined:
            return (
                "fixture_scope_leak",
                0.82,
                "Session/module-scoped fixture yields mutable state that may leak across tests.",
                "Use function-scoped fixtures or deep-copy the yielded value.",
            )
    return None


@_register
def _detect_mock_residue(code: str, test_code: str) -> Optional[Tuple[str, float, str, str]]:
    combined = code + test_code
    has_patch = "patch(" in combined or "monkeypatch" in combined
    has_cleanup = "stop()" in combined or "with " in combined
    if has_patch and not has_cleanup:
        return (
            "mock_residue",
            0.80,
            "Mock/monkeypatch applied without proper teardown. Patched state leaks to subsequent tests.",
            "Use context manager (with patch(...)) or ensure .stop() is called in teardown.",
        )
    return None


def _extract_function_name(code: str) -> str:
    """Extract the first function name from code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except Exception:
        pass
    m = re.search(r"def\s+(\w+)", code)
    return m.group(1) if m else ""


def _extract_file_hint(code: str) -> str:
    """Guess a filename from imports or class names."""
    m = re.search(r"from\s+([\w.]+)\s+import", code)
    if m:
        parts = m.group(1).split(".")
        return parts[-1] + ".py"
    m = re.search(r"import\s+([\w.]+)", code)
    if m:
        parts = m.group(1).split(".")
        return parts[-1] + ".py"
    return "source.py"


def _build_causal_chain(code: str, category: str) -> List[str]:
    """Build a simple causal chain from the code and detected category."""
    chain = []
    func = _extract_function_name(code)
    file_hint = _extract_file_hint(code)

    chain.append(f"test entry -> {func or 'target function'}")

    if category == "concurrency":
        chain.append(f"{func} -> Thread/Process spawn")
        chain.append("Thread -> shared state access (non-atomic)")
        chain.append("shared state -> RACE CONDITION")
    elif category == "async_wait":
        chain.append(f"{func} -> async task/gather")
        chain.append("async task -> shared resource contention")
        chain.append("contention -> TIMEOUT/DEADLOCK")
    elif category == "resource_leak":
        chain.append(f"{func} -> resource open (db/file/socket)")
        chain.append("resource -> missing cleanup/commit")
        chain.append("missing cleanup -> STALE STATE")
    elif category == "network":
        chain.append(f"{func} -> external HTTP call")
        chain.append("HTTP call -> network/endpoint variability")
        chain.append("variability -> NON-DETERMINISTIC RESPONSE")
    elif category == "shared_state":
        chain.append(f"{func} -> global/class state mutation")
        chain.append("mutation -> cross-test contamination")
        chain.append("contamination -> ORDER-DEPENDENT FAILURE")
    else:
        chain.append(f"{func} -> non-determinism source")
        chain.append("non-determinism -> FLAKY OUTCOME")

    return chain


def _generate_patch_diff(code: str, category: str, func_name: str) -> str:
    """Generate a representative patch diff for the detected issue."""
    if category == "concurrency":
        return (
            f"--- {_extract_file_hint(code)}\n"
            "<<<<<<< SEARCH\n"
            f"  global counter\n"
            "=======\n"
            "  _lock = threading.Lock()\n"
            "  with _lock:\n"
            ">>>>>>> REPLACE"
        )
    elif category == "async_wait":
        return (
            f"--- {_extract_file_hint(code)}\n"
            "<<<<<<< SEARCH\n"
            "  timeout=0.5\n"
            "=======\n"
            "  timeout=5.0\n"
            ">>>>>>> REPLACE"
        )
    elif category == "resource_leak":
        return (
            f"--- {_extract_file_hint(code)}\n"
            "<<<<<<< SEARCH\n"
            "  conn.execute('INSERT INTO t VALUES (42)')\n"
            "=======\n"
            "  conn.execute('INSERT INTO t VALUES (42)')\n"
            "  conn.commit()\n"
            ">>>>>>> REPLACE"
        )
    elif category == "network":
        return (
            f"--- {_extract_file_hint(code)}\n"
            "<<<<<<< SEARCH\n"
            f"  r = requests.post(url, json=payload)\n"
            "=======\n"
            "  from unittest.mock import patch, MagicMock\n"
            "  mock_resp = MagicMock(status_code=200)\n"
            "  with patch('requests.post', return_value=mock_resp):\n"
            f"      r = requests.post(url, json=payload)\n"
            ">>>>>>> REPLACE"
        )
    return ""


def analyze_challenge(code: str, test_code: str = "", preset: str = "") -> ChallengeAnalysis:
    """Analyze a user-submitted flaky test and return structured diagnosis."""
    if not code.strip() and not test_code.strip():
        return ChallengeAnalysis(
            detected_category="unknown",
            confidence=0.0,
            explanation="No code provided for analysis.",
        )

    combined_code = code
    combined_test = test_code

    best_match: Optional[Tuple[str, float, str, str]] = None
    for detector in _PATTERN_DETECTORS:
        result = detector(combined_code, combined_test)
        if result is not None:
            if best_match is None or result[1] > best_match[1]:
                best_match = result

    if best_match is None:
        return ChallengeAnalysis(
            detected_category="unknown",
            confidence=0.3,
            explanation="No strong flakiness pattern detected. The code may have subtle "
                        "non-determinism not covered by static analysis. Consider running "
                        "the full FlakeForge episode with chaos probes for deeper analysis.",
        )

    category, confidence, explanation, suggested_fix = best_match
    func_name = _extract_function_name(code)
    file_hint = _extract_file_hint(code)
    causal_chain = _build_causal_chain(code, category)
    patch_diff = _generate_patch_diff(code, category, func_name)

    infra_sensitive = category in ("concurrency", "async_wait", "network")
    estimated_reward = round(confidence * 7.5, 1)

    return ChallengeAnalysis(
        detected_category=category,
        confidence=round(confidence, 2),
        root_cause_file=file_hint,
        root_cause_function=func_name,
        causal_chain=causal_chain,
        infrastructure_sensitive=infra_sensitive,
        suggested_fix=suggested_fix,
        patch_diff=patch_diff,
        explanation=explanation,
        estimated_reward=estimated_reward,
    )
