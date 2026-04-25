"""Cross-Repository Causal Graph Engine — traces call chains and detects boundaries.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── External boundary signatures ──────────────────────────────────────────────
_HTTP_SIGNATURES: Set[str] = {
    "requests.get", "requests.post", "requests.put", "requests.delete", "requests.patch",
    "httpx.get", "httpx.post", "httpx.AsyncClient", "aiohttp.ClientSession",
    "urllib.request.urlopen", "urllib.request.urlretrieve",
}
_DB_SIGNATURES: Set[str] = {
    "session.commit", "session.execute", "session.add", "session.flush",
    "cursor.execute", "cursor.executemany",
    "db.commit", "db.execute", "db.session.commit", "db.session.execute",
    "engine.connect", "engine.execute",
    "collection.find", "collection.insert_one", "collection.update_one",
    "redis.set", "redis.get", "redis.hset",
}
_QUEUE_SIGNATURES: Set[str] = {
    "producer.send", "channel.basic_publish", "queue.put", "queue.put_nowait",
    "celery.send_task", "task.delay", "task.apply_async",
}
_GRPC_SIGNATURES: Set[str] = {"stub.", "channel.unary_unary", "channel.stream_unary"}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CausalNode:
    node_id: str                     # e.g. "billing.charge"
    module_path: str                 # dotted import path
    source_file: str                 # absolute path to the file
    source_excerpt: str              # first 60 lines of the function body
    is_external_boundary: bool = False
    boundary_type: Optional[str] = None   # "http" | "db" | "queue" | "grpc"
    is_async: bool = False
    depth: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class CausalEdge:
    caller_id: str
    callee_id: str
    call_site_line: int
    call_type: str = "direct"        # "direct" | "async_await" | "thread" | "subprocess"


@dataclass
class CausalGraph:
    nodes: List[CausalNode]
    edges: List[CausalEdge]
    boundary_warnings: List[str]
    entry_node_id: str
    max_depth_reached: int
    unresolved_imports: List[str] = field(default_factory=list)  # chains the tracer couldn't follow

    def to_observation_dict(self) -> Dict[str, Any]:
        """Render a compact, token-efficient summary for the LLM."""
        return {
            "entry": self.entry_node_id,
            "max_depth": self.max_depth_reached,
            "nodes": [
                {
                    "id": n.node_id,
                    "async": n.is_async,
                    "depth": n.depth,
                    "boundary": n.boundary_type,
                    "file": Path(n.source_file).name,
                    "excerpt_lines": n.source_excerpt.count("\n") + 1,
                    "warnings": n.warnings,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "from": e.caller_id,
                    "to": e.callee_id,
                    "line": e.call_site_line,
                    "type": e.call_type,
                }
                for e in self.edges
            ],
            "boundary_warnings": self.boundary_warnings,
            "boundary_nodes": [n.node_id for n in self.nodes if n.is_external_boundary],
            # Tells the agent which import chains were cut off (can't be traced on disk)
            "unresolved_imports": self.unresolved_imports,
        }


@dataclass
class EpisodeCausalTrace:
    """Tracks symptoms, hypotheses and actions over one episode."""

    symptoms: List[str] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    final_cause: str = ""
    fix_applied: str = ""
    outcome: str = "failure"

    def add_symptom(self, symptom: str) -> None:
        if symptom and symptom not in self.symptoms:
            self.symptoms.append(symptom)

    def add_hypothesis(self, step: int, hypothesis: Dict[str, Any]) -> None:
        payload = {"step": step, **hypothesis}
        self.hypotheses.append(payload)

    def add_action(self, step: int, action: Dict[str, Any]) -> None:
        payload = {"step": step, **action}
        self.actions_taken.append(payload)

    def finalize(self, final_cause: str, fix_applied: str, success: bool) -> None:
        self.final_cause = final_cause
        self.fix_applied = fix_applied
        self.outcome = "success" if success else "failure"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symptoms": list(self.symptoms),
            "hypotheses": list(self.hypotheses),
            "actions_taken": list(self.actions_taken),
            "final_cause": self.final_cause,
            "fix_applied": self.fix_applied,
            "outcome": self.outcome,
        }


# ── Main builder class ─────────────────────────────────────────────────────────

class CrossRepoGraphBuilder:
    """
    Builds a CausalGraph by walking the AST of every reachable function
    starting from an entry point, following imports and call sites up to
    ``max_depth`` hops.
    """

    # Common alternative source layout roots to search when resolving imports.
    # Covers: flat layout, src-layout, app-layout (FastAPI/Django conventions).
    _LAYOUT_ROOTS = [".", "src", "app", "lib"]

    def __init__(self, repo_root: str, max_depth: int = 3) -> None:
        self.repo_root = Path(repo_root)
        self.max_depth = max_depth
        self._visited: Set[str] = set()
        self._nodes: List[CausalNode] = []
        self._edges: List[CausalEdge] = []
        self._boundary_warnings: List[str] = []
        self._unresolved_imports: Set[str] = set()

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self, entry_file: str, entry_function: str) -> CausalGraph:
        """Build and return the full causal graph from the entry point."""
        self._visited.clear()
        self._nodes.clear()
        self._edges.clear()
        self._boundary_warnings.clear()

        entry_id = f"{Path(entry_file).stem}.{entry_function}"
        self._walk(
            file_path=entry_file,
            function_name=entry_function,
            node_id=entry_id,
            depth=0,
            parent_id=None,
            call_site_line=0,
            call_type="direct",
        )

        return CausalGraph(
            nodes=self._nodes,
            edges=self._edges,
            boundary_warnings=self._boundary_warnings,
            entry_node_id=entry_id,
            max_depth_reached=max((n.depth for n in self._nodes), default=0),
            unresolved_imports=list(self._unresolved_imports),
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _walk(
        self,
        file_path: str,
        function_name: str,
        node_id: str,
        depth: int,
        parent_id: Optional[str],
        call_site_line: int,
        call_type: str,
    ) -> None:
        if node_id in self._visited or depth > self.max_depth:
            return
        self._visited.add(node_id)

        src_path = Path(file_path)
        if not src_path.exists():
            return

        try:
            source = src_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception as exc:
            logger.debug("causal_graph: failed to parse %s: %s", file_path, exc)
            return

        func_node = self._find_function(tree, function_name)
        if func_node is None:
            return

        excerpt = self._extract_excerpt(source, func_node)
        is_async = isinstance(func_node, ast.AsyncFunctionDef)
        boundary_type, boundary_warnings = self._detect_boundaries(func_node, node_id, is_async)

        node = CausalNode(
            node_id=node_id,
            module_path=f"{src_path.stem}.{function_name}",
            source_file=file_path,
            source_excerpt=excerpt,
            is_external_boundary=boundary_type is not None,
            boundary_type=boundary_type,
            is_async=is_async,
            depth=depth,
            warnings=boundary_warnings,
        )
        self._nodes.append(node)
        self._boundary_warnings.extend(boundary_warnings)

        if parent_id is not None:
            self._edges.append(CausalEdge(
                caller_id=parent_id,
                callee_id=node_id,
                call_site_line=call_site_line,
                call_type=call_type,
            ))

        if depth < self.max_depth:
            self._follow_calls(func_node, file_path, node_id, depth, tree)

    @staticmethod
    def _build_parent_map(
        func_node: "ast.FunctionDef | ast.AsyncFunctionDef",
    ) -> Dict[int, ast.AST]:
        """
        Build a {id(child): parent} mapping for every node inside func_node.

        ast.walk() is a flat BFS iterator with no parent information.
        The only reliable way to know a node's parent is a dedicated pre-pass.
        This is the standard pattern used by astroid, pyflakes, and mypy.
        """
        parent_map: Dict[int, ast.AST] = {}
        for parent in ast.walk(func_node):
            for child in ast.iter_child_nodes(parent):
                parent_map[id(child)] = parent
        return parent_map

    def _follow_calls(
        self,
        func_node: "ast.FunctionDef | ast.AsyncFunctionDef",
        current_file: str,
        current_id: str,
        depth: int,
        tree: ast.Module,
    ) -> None:
        """Walk the body of a function to find outgoing calls and follow them."""
        import_map = self._build_import_map(tree, current_file)

        # Pre-pass: build parent map so we can correctly classify each call's context.
        # Without this, ast.walk() gives no parent info and call_type is always 'direct'.
        parent_map = self._build_parent_map(func_node)

        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue

            call_name = self._call_name(node)
            if not call_name:
                continue

            # Correctly determine call type using the parent map.
            call_type = "direct"
            parent = parent_map.get(id(node))
            if isinstance(parent, ast.Await):
                call_type = "async_await"
            elif isinstance(parent, ast.Call):
                parent_name = self._call_name(parent) or ""
                if parent_name in {
                    "threading.Thread",
                    "concurrent.futures.ThreadPoolExecutor",
                    "asyncio.to_thread",
                    "loop.run_in_executor",
                }:
                    call_type = "thread"

            # Resolve the file from import map
            target_file = import_map.get(call_name.split(".")[0])
            if target_file is None:
                continue

            callee_func = call_name.split(".")[-1]
            callee_id = f"{Path(target_file).stem}.{callee_func}"
            call_site_line = getattr(node, "lineno", 0)

            self._walk(
                file_path=target_file,
                function_name=callee_func,
                node_id=callee_id,
                depth=depth + 1,
                parent_id=current_id,
                call_site_line=call_site_line,
                call_type=call_type,
            )

    def _resolve_module_path(self, module_dotted: str) -> Optional[Path]:
        """
        Resolve a dotted module name to a physical file path.

        Searches multiple layout roots so we handle:
          - Flat layout:   repo_root/billing.py
          - src-layout:    repo_root/src/mypackage/billing.py
          - app-layout:    repo_root/app/billing.py
          - Namespace pkg: repo_root/billing/ (no __init__.py, Python 3.3+)

        If a module resolves to an __init__.py, we also scan its body for
        re-exported names (``from .billing import charge``) so the tracer
        can follow through package facades.
        """
        module_rel = Path(module_dotted.replace(".", "/"))
        for layout_root in self._LAYOUT_ROOTS:
            base = self.repo_root / layout_root
            candidates = [
                base / f"{module_rel}.py",
                base / module_rel / "__init__.py",
                # Namespace packages — directory with no __init__.py
                base / module_rel,
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
                if candidate.is_dir():
                    # Treat directory as namespace package — return None
                    # (caller will handle individual name resolution)
                    return None
        return None

    def _resolve_reexported_name(
        self, init_path: Path, name: str
    ) -> Optional[Path]:
        """
        If `init_path` is an __init__.py that re-exports `name` via
        ``from .submodule import name``, resolve to the submodule's file.
        """
        try:
            source = init_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception:
            return None

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0:  # absolute import — not a re-export
                continue
            for alias in (node.names or []):
                exported_name = alias.asname or alias.name
                if exported_name == name and node.module:
                    # Resolve the relative sub-module
                    sub_rel = node.module.replace(".", "/")
                    sub_candidate = init_path.parent / f"{sub_rel}.py"
                    if sub_candidate.exists():
                        return sub_candidate
        return None

    def _build_import_map(self, tree: ast.Module, current_file: str) -> Dict[str, str]:
        """
        Maps every imported name to its resolved file path within the repo.

        Handles:
          * ``from package import name`` (including re-exports via __init__.py)
          * ``import module`` / ``import module as alias``
          * src-layout, app-layout, flat layout
          * Namespace packages

        Names that cannot be resolved are logged to ``_unresolved_imports``
        so the observation dict tells the agent where the trace was cut off.
        """
        mapping: Dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                resolved = self._resolve_module_path(node.module)
                for alias in (node.names or []):
                    exported = alias.asname or alias.name
                    if resolved is not None and resolved.suffix == ".py":
                        # Direct module file resolved
                        mapping[exported] = str(resolved)
                    elif resolved is not None and resolved.name == "__init__.py":
                        # Package __init__.py — check for re-exports
                        reexported = self._resolve_reexported_name(resolved, alias.name)
                        mapping[exported] = str(reexported if reexported else resolved)
                    else:
                        # Could not resolve — record for the agent's observation
                        self._unresolved_imports.add(f"{node.module}.{alias.name}")
                        logger.debug(
                            "causal_graph: unresolved import '%s.%s' — "
                            "chain truncated here.",
                            node.module,
                            alias.name,
                        )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve_module_path(alias.name)
                    imported_as = alias.asname or alias.name.split(".")[0]
                    if resolved is not None:
                        mapping[imported_as] = str(resolved)
                    else:
                        self._unresolved_imports.add(alias.name)
                        logger.debug(
                            "causal_graph: unresolved import '%s' — "
                            "chain truncated here.",
                            alias.name,
                        )

        return mapping

    @staticmethod
    def _find_function(
        tree: ast.Module, name: str
    ) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        return None

    @staticmethod
    def _extract_excerpt(
        source: str,
        func_node: "ast.FunctionDef | ast.AsyncFunctionDef",
        max_lines: int = 60,
    ) -> str:
        lines = source.splitlines()
        start = func_node.lineno - 1
        end = min(start + max_lines, len(lines))
        return "\n".join(lines[start:end])

    @staticmethod
    def _call_name(call_node: ast.Call) -> Optional[str]:
        """Reconstruct a dotted call name from an AST Call node."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            current: ast.expr = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None

    def _detect_boundaries(
        self,
        func_node: "ast.FunctionDef | ast.AsyncFunctionDef",
        node_id: str,
        is_async: bool,
    ) -> tuple[Optional[str], List[str]]:
        """
        Detects external boundary calls and emits warnings for dangerous patterns
        (e.g. blocking call inside async function, thread lock in async context).
        """
        warnings: List[str] = []
        found_boundary: Optional[str] = None
        has_threading_lock = False
        has_blocking_call = False

        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue
            call_name = self._call_name(node) or ""

            # Detect boundary types
            if any(sig in call_name for sig in _HTTP_SIGNATURES):
                found_boundary = "http"
            elif any(sig in call_name for sig in _DB_SIGNATURES):
                found_boundary = "db"
            elif any(sig in call_name for sig in _QUEUE_SIGNATURES):
                found_boundary = "queue"
            elif any(sig.split(".")[0] in call_name for sig in _GRPC_SIGNATURES):
                found_boundary = "grpc"

            # Detect dangerous patterns
            if "threading.Lock" in call_name or "threading.RLock" in call_name:
                has_threading_lock = True
            if call_name in {"time.sleep", "open", "socket.recv", "socket.accept"}:
                has_blocking_call = True

        # Emit warnings for dangerous combinations
        if is_async and has_threading_lock:
            msg = (
                f"[{node_id}] threading.Lock() used inside async function — "
                "blocks the event loop. Use asyncio.Lock() instead. "
                "Likely cause: ASYNC_DEADLOCK"
            )
            warnings.append(msg)

        if is_async and has_blocking_call:
            msg = (
                f"[{node_id}] blocking I/O call detected inside async function — "
                "offload with loop.run_in_executor() or use async alternative. "
                "Likely cause: ASYNC_DEADLOCK / TIMING_RACE"
            )
            warnings.append(msg)

        return found_boundary, warnings
