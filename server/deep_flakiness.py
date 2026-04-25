"""V3 Deep Flakiness Detection Layer.

Implements the five deep flakiness patterns from SE literature:
1. Module Cache Pollution — @lru_cache, mutable defaults, global mutations
2. Fixture Scope Contamination — session/module fixtures without yield teardown
3. Mock Residue — patch() without with-context or .stop()
4. Import Side-Effects — top-level module code with non-constant expressions
5. Async/Thread Contamination — tasks/threads surviving past test boundaries

Research sources:
- Kraken Engineering: Module cache pollution patterns
- Luo FSE 2014: Flaky test root cause taxonomy
- pytest internals: Fixture scope contamination
- Parry survey: Import side-effect patterns
- FlakyLens OOPSLA 2025: Async Wait vs Concurrency discrimination
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def detect_module_cache_pollution(repo_path: Path) -> List[str]:
    """Detect @lru_cache, mutable defaults, module-level global mutations.

    These cause cross-test pollution when cached values carry state
    between test invocations.
    """
    violations: List[str] = []

    for pyfile in repo_path.rglob("*.py"):
        if _should_skip(pyfile):
            continue

        try:
            source = pyfile.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except Exception:
            continue

        relative = str(pyfile.relative_to(repo_path)).replace("\\", "/")
        reasons: List[str] = []

        for node in ast.walk(tree):
            # @lru_cache, @cache, @cached decorators
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for deco in node.decorator_list:
                    deco_name = _get_decorator_name(deco)
                    if deco_name in ("lru_cache", "cache", "cached", "functools.lru_cache", "functools.cache"):
                        reasons.append(f"@{deco_name} on {node.name}")

                # Mutable default arguments
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        reasons.append(f"mutable_default in {node.name}")

            # Module-level global statement
            if isinstance(node, ast.Global):
                reasons.append(f"global {', '.join(node.names)}")

        if reasons:
            violations.append(f"{relative}: {'; '.join(reasons[:3])}")

    return violations


def detect_fixture_scope_leaks(repo_path: Path) -> List[str]:
    """Detect session/module scoped fixtures returning mutable objects without yield.

    Session/module fixtures that return mutable objects (lists, dicts, sets)
    without using yield for teardown can leak state between tests.
    """
    risks: List[str] = []

    for conftest in repo_path.rglob("conftest.py"):
        if _should_skip(conftest):
            continue

        try:
            source = conftest.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except Exception:
            continue

        relative = str(conftest.relative_to(repo_path)).replace("\\", "/")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Check if it's a pytest fixture with session/module scope
            scope = _get_fixture_scope(node)
            if scope not in ("session", "module"):
                continue

            # Check if it uses yield (proper teardown)
            has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))

            # Check if it returns mutable objects
            returns_mutable = _returns_mutable_type(node)

            if returns_mutable and not has_yield:
                risks.append(f"{relative}:{node.name} scope={scope} returns_mutable=True no_yield=True")

    # Also scan test files for session/module fixtures
    for test_file in repo_path.rglob("test_*.py"):
        if _should_skip(test_file):
            continue

        try:
            source = test_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except Exception:
            continue

        relative = str(test_file.relative_to(repo_path)).replace("\\", "/")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            scope = _get_fixture_scope(node)
            if scope not in ("session", "module"):
                continue

            has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))
            if not has_yield:
                risks.append(f"{relative}:{node.name} scope={scope} no_yield=True")

    return risks


def detect_monkeypatch_residue(repo_path: Path) -> List[str]:
    """Detect patch() calls without proper cleanup (with-context or .stop()).

    Uncleaned patches leak mock objects into subsequent tests,
    causing hard-to-debug flakiness.
    """
    violations: List[str] = []

    for pyfile in repo_path.rglob("*.py"):
        if _should_skip(pyfile):
            continue

        try:
            source = pyfile.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        relative = str(pyfile.relative_to(repo_path)).replace("\\", "/")

        # Find patch() calls not inside with-blocks and without .stop()
        if "patch(" in source:
            lines = source.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Skip comment lines
                if stripped.startswith("#"):
                    continue

                if "patch(" in stripped:
                    # Check if it's inside a with block
                    if stripped.startswith("with ") or "with mock.patch" in stripped:
                        continue
                    # Check if it's a decorator
                    if stripped.startswith("@"):
                        continue
                    # Check if .stop() is called nearby
                    window = "\n".join(lines[max(0, i - 1):min(len(lines), i + 10)])
                    if ".stop()" in window:
                        continue

                    violations.append(f"{relative}:{i} — patch() without with-context or .stop()")

    return violations


def detect_import_side_effects(repo_path: Path) -> List[str]:
    """Detect top-level module code with non-constant expressions.

    Top-level code that performs I/O, makes network calls, or reads
    configuration executes on import, causing non-deterministic behavior.
    """
    violations: List[str] = []

    for pyfile in repo_path.rglob("*.py"):
        if _should_skip(pyfile):
            continue

        try:
            source = pyfile.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except Exception:
            continue

        relative = str(pyfile.relative_to(repo_path)).replace("\\", "/")
        reasons: List[str] = []

        for node in tree.body:
            # Skip harmless top-level constructs
            if isinstance(node, (
                ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                ast.ClassDef, ast.If, ast.Pass, ast.Expr,
            )):
                # Check if it's a bare expression with side effects
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call_name = _get_call_name(node.value)
                    # Harmless calls
                    if call_name not in ("print", "logging.getLogger", "warnings.warn"):
                        reasons.append(f"top-level call: {call_name}()")
                continue

            # Top-level assignments to mutable objects from function calls
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Call):
                            call_name = _get_call_name(node.value)
                            if call_name and "." in call_name:
                                reasons.append(f"{target.id} = {call_name}()")

        if reasons:
            violations.append(f"{relative}: {'; '.join(reasons[:3])}")

    return violations


def detect_async_contamination(repo_path: Path) -> bool:
    """Detect patterns that indicate async tasks/threads may survive past test boundaries.

    Looks for: fire-and-forget asyncio.create_task(), threading.Thread().start()
    without corresponding join/await, daemon threads, etc.
    """
    for pyfile in repo_path.rglob("*.py"):
        if _should_skip(pyfile):
            continue

        try:
            source = pyfile.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Fire-and-forget patterns
        if re.search(r"asyncio\.create_task\(", source):
            if "await" not in source or "gather" not in source:
                return True

        if re.search(r"\.start\(\)", source) and "threading" in source:
            if ".join()" not in source:
                return True

    return False


def build_deep_observation_signals(repo_path: Path) -> Dict[str, Any]:
    """Build all deep flakiness signals for the observation.

    This is designed to run in <5ms via AST scanning only.
    """
    return {
        "module_cache_violations": detect_module_cache_pollution(repo_path),
        "fixture_scope_risks": detect_fixture_scope_leaks(repo_path),
        "mock_residue_sites": detect_monkeypatch_residue(repo_path),
        "import_side_effect_files": detect_import_side_effects(repo_path),
        "async_contamination_alive": detect_async_contamination(repo_path),
    }


def extract_failure_frontier(
    failing_trace: str,
    repo_path: Path,
) -> Tuple[str, List[str], List[str]]:
    """Extract the deepest user-code frame from a stack trace.

    Returns:
        (failure_frontier, call_chain, boundary_crossings)
    """
    if not failing_trace:
        return "", [], []

    # Parse stack trace lines for file:line references
    frame_pattern = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')
    frames: List[Dict[str, str]] = []

    for match in frame_pattern.finditer(failing_trace):
        file_path, line_num, func_name = match.groups()
        frames.append({
            "file": file_path,
            "line": line_num,
            "function": func_name,
        })

    if not frames:
        return "", [], []

    # Find deepest user-code frame (not stdlib/site-packages)
    user_frames = [
        f for f in frames
        if "site-packages" not in f["file"]
        and "lib/python" not in f["file"]
        and not f["file"].startswith("<")
    ]

    if not user_frames:
        user_frames = frames

    # Build call chain
    call_chain = [f"{f['function']}" for f in user_frames]
    failure_frontier = f"{user_frames[-1]['file']}:{user_frames[-1]['line']}:{user_frames[-1]['function']}"

    # Detect boundary crossings
    boundary_crossings: List[str] = []
    for f in user_frames:
        func = f["function"].lower()
        if any(kw in func for kw in ("request", "http", "fetch", "get", "post")):
            boundary_crossings.append(f"http:{f['function']}")
        elif any(kw in func for kw in ("query", "execute", "commit", "cursor")):
            boundary_crossings.append(f"db:{f['function']}")
        elif any(kw in func for kw in ("send", "publish", "enqueue")):
            boundary_crossings.append(f"queue:{f['function']}")

    return failure_frontier, call_chain, list(set(boundary_crossings))


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _should_skip(path: Path) -> bool:
    """Skip non-project files."""
    skip_dirs = {"__pycache__", ".git", "node_modules", "venv", ".venv", ".pytest_cache"}
    return any(part in skip_dirs for part in path.parts)


def _get_decorator_name(deco: ast.expr) -> str:
    """Extract decorator name from AST node."""
    if isinstance(deco, ast.Name):
        return deco.id
    if isinstance(deco, ast.Attribute):
        parts = []
        current = deco
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    if isinstance(deco, ast.Call):
        return _get_decorator_name(deco.func)
    return ""


def _get_fixture_scope(node: ast.FunctionDef) -> str:
    """Get the scope of a pytest fixture from its decorators."""
    for deco in node.decorator_list:
        if isinstance(deco, ast.Call):
            deco_name = _get_decorator_name(deco.func)
            if deco_name in ("pytest.fixture", "fixture"):
                for keyword in deco.keywords:
                    if keyword.arg == "scope" and isinstance(keyword.value, ast.Constant):
                        return str(keyword.value.value)
    return "function"


def _returns_mutable_type(node: ast.FunctionDef) -> bool:
    """Check if a function returns mutable types (list, dict, set)."""
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            if isinstance(child.value, (ast.List, ast.Dict, ast.Set)):
                return True
            if isinstance(child.value, ast.Call):
                call_name = _get_call_name(child.value)
                if call_name in ("dict", "list", "set", "defaultdict", "OrderedDict"):
                    return True
    return False


def _get_call_name(call: ast.Call) -> str:
    """Get the name of a function call."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        parts = []
        current = call.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""
