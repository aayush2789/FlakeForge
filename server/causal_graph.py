# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pillar 1 — Cross-Repository Causal Graph Engine.

Traces function call chains across files and packages up to a configurable
depth, detects external boundaries (HTTP / gRPC / DB / queue), and returns
a compact, token-efficient graph summary for the LLM observation.

Design inspired by AMER-RCL (ICSE 2026): recursiveness, multi-dimensional
expansion, and cross-modal reasoning are the three pillars of real SRE RCA.
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

    def __init__(self, repo_root: str, max_depth: int = 3) -> None:
        self.repo_root = Path(repo_root)
        self.max_depth = max_depth
        self._visited: Set[str] = set()
        self._nodes: List[CausalNode] = []
        self._edges: List[CausalEdge] = []
        self._boundary_warnings: List[str] = []

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

        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue

            call_name = self._call_name(node)
            if not call_name:
                continue

            # Determine call type (async_await / thread / direct)
            call_type = "direct"
            for parent in ast.walk(func_node):
                if isinstance(parent, ast.Await) and parent.value is node:
                    call_type = "async_await"
                    break
                if isinstance(parent, ast.Call):
                    pname = self._call_name(parent)
                    if pname in {"threading.Thread", "concurrent.futures.ThreadPoolExecutor"}:
                        call_type = "thread"
                        break

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

    def _build_import_map(self, tree: ast.Module, current_file: str) -> Dict[str, str]:
        """Maps imported names to their resolved file paths within the repo."""
        mapping: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module_rel = node.module.replace(".", "/")
                candidates = [
                    self.repo_root / f"{module_rel}.py",
                    self.repo_root / module_rel / "__init__.py",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        for alias in (node.names or []):
                            mapping[alias.asname or alias.name] = str(candidate)
                        break
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_rel = alias.name.replace(".", "/")
                    candidates = [
                        self.repo_root / f"{module_rel}.py",
                        self.repo_root / module_rel / "__init__.py",
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            mapping[alias.asname or alias.name] = str(candidate)
                            break
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
