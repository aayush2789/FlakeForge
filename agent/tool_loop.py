"""Tool registry, execution, caching, and budgets for the agent inner loop."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from models import FlakeForgeObservation, RunRecord
except ImportError:
    from ..models import FlakeForgeObservation, RunRecord


def _truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12] + "\n[truncated]"


def _repo_mtime_fingerprint(repo_root: str, *, max_files: int = 500) -> str:
    root = Path(repo_root)
    if not root.is_dir():
        return "0"
    mtimes: List[int] = []
    for i, p in enumerate(root.rglob("*.py")):
        if i >= max_files:
            break
        if any(x in p.parts for x in ("__pycache__", ".git", "venv", ".venv", ".pytest_cache")):
            continue
        try:
            mtimes.append(int(p.stat().st_mtime_ns))
        except OSError:
            continue
    if not mtimes:
        return "0"
    return f"{len(mtimes)}:{max(mtimes)}"


def _deep_signals_from_observation(obs: FlakeForgeObservation) -> Dict[str, Any]:
    return {
        "module_cache_violations": list(obs.module_cache_violations or []),
        "fixture_scope_risks": list(obs.fixture_scope_risks or []),
        "mock_residue_sites": list(obs.mock_residue_sites or []),
        "import_side_effect_files": list(obs.import_side_effect_files or []),
        "async_contamination_alive": bool(obs.async_contamination_alive),
    }


def summarize_deep_signals(data: Dict[str, Any], *, max_items: int = 5) -> str:
    lines: List[str] = []
    for key in (
        "module_cache_violations",
        "fixture_scope_risks",
        "mock_residue_sites",
        "import_side_effect_files",
    ):
        vals = data.get(key) or []
        if not isinstance(vals, list) or not vals:
            continue
        short = vals[:max_items]
        more = len(vals) - len(short)
        suffix = f" (+{more} more)" if more > 0 else ""
        lines.append(f"{key}: {len(vals)} — {short}{suffix}")
    if data.get("async_contamination_alive"):
        lines.append("async_contamination_alive: true (possible fire-and-forget tasks/threads)")
    if not lines:
        return "No deep static flakiness hits in scan."
    return "\n".join(lines)


def summarize_causal_graph_dict(graph: Optional[Dict[str, Any]], *, max_nodes: int = 8) -> str:
    if not graph:
        return "No causal graph available."
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    entry = graph.get("entry", "")
    warnings = graph.get("boundary_warnings") or []
    lines = [
        f"entry={entry} nodes={len(nodes)} edges={len(edges)} max_depth={graph.get('max_depth')}",
    ]
    if warnings:
        lines.append("boundary_warnings: " + "; ".join(str(w) for w in warnings[:5]))
    for n in nodes[:max_nodes]:
        if not isinstance(n, dict):
            continue
        nid = n.get("id", "")
        fn = n.get("file", "")
        depth = n.get("depth", "")
        boundary = n.get("boundary", "")
        lines.append(f"  node {nid} file={fn} depth={depth} boundary={boundary}")
    if len(nodes) > max_nodes:
        lines.append(f"  ... {len(nodes) - max_nodes} more nodes omitted")
    return "\n".join(lines)


@dataclass
class ToolContext:
    """Everything needed to run analysis tools against the current episode."""

    repo_root: str
    observation: FlakeForgeObservation
    env: Any = None


@dataclass
class ToolTraceEntry:
    """One tool invocation result summarized for the prompt."""

    turn: int
    tool: str
    ok: bool
    summary: str


def format_tool_trace_for_prompt(entries: List[ToolTraceEntry], *, max_entries: int = 12) -> str:
    if not entries:
        return ""
    lines = ["=== TOOL OBSERVATIONS ==="]
    for e in entries[-max_entries:]:
        status = "ok" if e.ok else "error"
        lines.append(f"{e.turn}. [{e.tool}] {status}: {_truncate_text(e.summary, 900)}")
    return "\n".join(lines)


def summarize_run_history(records: List[RunRecord], *, max_sample: int = 8) -> str:
    if not records:
        return "No run history in observation."
    n = len(records)
    passed = sum(1 for r in records if r.passed)
    errs = [r.error_type for r in records if not r.passed and r.error_type]
    top_err = Counter(errs).most_common(1)
    top_err_s = top_err[0][0] if top_err else "n/a"
    durs = [r.duration_ms for r in records]
    mean_d = sum(durs) / max(len(durs), 1)
    lines = [
        f"runs_in_view={n} pass_rate={passed / max(n, 1):.2f} dominant_error={top_err_s} mean_duration_ms={mean_d:.0f}",
        "recent:",
    ]
    for r in records[-max_sample:]:
        st = "PASS" if r.passed else f"FAIL({r.error_type or 'err'})"
        msg = (r.error_message or "")[:60]
        lines.append(f"  - {st} {r.duration_ms}ms {msg}")
    return "\n".join(lines)


ToolFn = Callable[[ToolContext, Dict[str, Any]], str]


class ToolExecutor:
    """Execute named tools with caching, per-tool limits, and error handling."""

    def __init__(
        self,
        *,
        max_calls_total: int = 16,
        per_tool_max: Optional[Dict[str, int]] = None,
    ) -> None:
        self.max_calls_total = max_calls_total
        self.per_tool_max = per_tool_max or {
            "deep_flakiness_scan": 3,
            "causal_graph_summary": 2,
            "targeting_hints": 3,
            "list_repo": 4,
            "read_file": 8,
            "ast_summary": 6,
            "run_history_summary": 4,
        }
        self._total_used = 0
        self._per_tool_used: Dict[str, int] = {}
        self._cache: Dict[Tuple[str, str, str, str], str] = {}
        self._handler_map: Dict[str, ToolFn] = {}

    def reset_budgets(self) -> None:
        self._total_used = 0
        self._per_tool_used.clear()
        self._cache.clear()

    def _cache_key(self, tool: str, args: Dict[str, Any], repo_root: str) -> Tuple[str, str, str, str]:
        fp = _repo_mtime_fingerprint(repo_root)
        canon = json.dumps(args or {}, sort_keys=True, default=str)
        return tool, canon, repo_root, fp

    def set_handlers(self, mapping: Dict[str, ToolFn]) -> None:
        self._handler_map = dict(mapping)

    def calls_used(self) -> int:
        return self._total_used

    def budget_remaining(self) -> int:
        return max(0, self.max_calls_total - self._total_used)

    def execute(self, ctx: ToolContext, tool: str, args: Dict[str, Any]) -> ToolTraceEntry:
        turn = self._total_used + 1
        if self._total_used >= self.max_calls_total:
            return ToolTraceEntry(
                turn=turn,
                tool=tool,
                ok=False,
                summary="Tool budget exhausted; emit action patch with your best fix or empty hunks.",
            )

        cap = self.per_tool_max.get(tool, 8)
        used = self._per_tool_used.get(tool, 0)
        if used >= cap:
            return ToolTraceEntry(
                turn=turn,
                tool=tool,
                ok=False,
                summary=f"Per-tool limit reached for {tool} ({cap}); choose a different tool or patch.",
            )

        if tool not in self._handler_map:
            return ToolTraceEntry(turn=turn, tool=tool, ok=False, summary=f"Unknown tool: {tool}")

        key = self._cache_key(tool, args, ctx.repo_root)
        if key in self._cache:
            self._total_used += 1
            self._per_tool_used[tool] = used + 1
            return ToolTraceEntry(turn=turn, tool=tool, ok=True, summary=self._cache[key])

        try:
            out = self._handler_map[tool](ctx, dict(args or {}))
            out = _truncate_text(str(out), 4000)
        except Exception as exc:
            self._total_used += 1
            self._per_tool_used[tool] = used + 1
            return ToolTraceEntry(turn=turn, tool=tool, ok=False, summary=f"Tool error: {exc}")

        self._cache[key] = out
        self._total_used += 1
        self._per_tool_used[tool] = used + 1
        return ToolTraceEntry(turn=turn, tool=tool, ok=True, summary=out)


def _resolve_repo_path(ctx: ToolContext, rel: str) -> Path:
    root = Path(ctx.repo_root).resolve()
    p = Path(rel)
    path = p if p.is_absolute() else (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("path escapes repo_root") from None
    return path


def _tool_deep_flakiness_scan(ctx: ToolContext, args: Dict[str, Any]) -> str:
    del args
    if ctx.env is not None and getattr(ctx.env, "_episode_state", None) is not None:
        st = ctx.env._episode_state
        data = {
            "module_cache_violations": list(st.module_cache_violations or []),
            "fixture_scope_risks": list(st.fixture_scope_risks or []),
            "mock_residue_sites": list(st.mock_residue_sites or []),
            "import_side_effect_files": list(st.import_side_effect_files or []),
            "async_contamination_alive": bool(st.async_contamination_alive),
        }
    else:
        try:
            from server.deep_flakiness import build_deep_observation_signals
        except ImportError:
            from FlakeForge.server.deep_flakiness import build_deep_observation_signals  # type: ignore
        data = build_deep_observation_signals(Path(ctx.repo_root))
    return summarize_deep_signals(data)


def _tool_causal_graph_summary(ctx: ToolContext, args: Dict[str, Any]) -> str:
    del args
    graph = None
    if ctx.env is not None and getattr(ctx.env, "_episode_state", None) is not None:
        graph = ctx.env._episode_state.causal_graph
    if graph is None:
        try:
            try:
                from server.causal_graph import CrossRepoGraphBuilder
            except ImportError:
                from FlakeForge.server.causal_graph import CrossRepoGraphBuilder  # type: ignore
            tid = ctx.observation.test_identifier
            test_file, _, test_func = tid.partition("::")
            entry_file = str(Path(ctx.repo_root) / test_file)
            if not test_func.strip():
                return "Cannot build causal graph: missing test function in test_identifier."
            builder = CrossRepoGraphBuilder(str(ctx.repo_root), max_depth=3)
            graph_obj = builder.build(entry_file=entry_file, entry_function=test_func)
            graph = graph_obj.to_observation_dict()
        except Exception as exc:
            return f"Causal graph build failed: {exc}"
    return summarize_causal_graph_dict(graph)


def _tool_targeting_hints(ctx: ToolContext, args: Dict[str, Any]) -> str:
    try:
        try:
            from server.tools import build_agent_targeting_hints
        except ImportError:
            from FlakeForge.server.tools import build_agent_targeting_hints  # type: ignore
    except ImportError:
        from ..server.tools import build_agent_targeting_hints  # type: ignore

    deep = _deep_signals_from_observation(ctx.observation)
    if ctx.env is not None and getattr(ctx.env, "_episode_state", None) is not None:
        st = ctx.env._episode_state
        deep = {
            "module_cache_violations": list(st.module_cache_violations or []),
            "fixture_scope_risks": list(st.fixture_scope_risks or []),
            "mock_residue_sites": list(st.mock_residue_sites or []),
            "import_side_effect_files": list(st.import_side_effect_files or []),
            "async_contamination_alive": bool(st.async_contamination_alive),
        }

    max_hints = 8
    if args and args.get("max_hints") is not None:
        try:
            max_hints = int(args["max_hints"])
        except (TypeError, ValueError):
            max_hints = 8
    hints = build_agent_targeting_hints(
        repo_path=str(ctx.repo_root),
        test_identifier=ctx.observation.test_identifier,
        failing_stack_trace=ctx.observation.failing_stack_trace or "",
        source_under_test=ctx.observation.source_under_test or "",
        causal_frontier=ctx.observation.failure_frontier or "",
        deep_signals=deep,
        max_hints=max_hints,
    )
    if not hints:
        return "No targeting hints produced."
    return "\n".join(f"- {h}" for h in hints[:10])


def _tool_list_repo(ctx: ToolContext, args: Dict[str, Any]) -> str:
    try:
        try:
            from server.tools import list_repo_structure
        except ImportError:
            from FlakeForge.server.tools import list_repo_structure  # type: ignore
    except ImportError:
        from ..server.tools import list_repo_structure  # type: ignore
    limit = int(args.get("limit", 40))
    entries = list_repo_structure(ctx.repo_root)[:limit]
    lines = [f"{e.get('path')} test={e.get('is_test')} async={e.get('has_async')}" for e in entries]
    return "\n".join(lines) if lines else "(empty)"


def _tool_read_file(ctx: ToolContext, args: Dict[str, Any]) -> str:
    rel = str(args.get("path", "")).strip()
    start = int(args.get("start_line", 1))
    end = int(args.get("end_line", start + 40))
    try:
        try:
            from server.tools import read_file_excerpt
        except ImportError:
            from FlakeForge.server.tools import read_file_excerpt  # type: ignore
    except ImportError:
        from ..server.tools import read_file_excerpt  # type: ignore
    try:
        path = _resolve_repo_path(ctx, rel)
    except ValueError:
        return "Path escapes repo_root; rejected."
    return read_file_excerpt(str(path), start, end)


def _tool_ast_summary(ctx: ToolContext, args: Dict[str, Any]) -> str:
    rel = str(args.get("path", "")).strip()
    try:
        try:
            from server.tools import parse_ast_summary
        except ImportError:
            from FlakeForge.server.tools import parse_ast_summary  # type: ignore
    except ImportError:
        from ..server.tools import parse_ast_summary  # type: ignore
    try:
        path = _resolve_repo_path(ctx, rel)
    except ValueError:
        return "Path escapes repo_root; rejected."
    summary = parse_ast_summary(str(path))
    data = {
        "functions": (summary.functions or [])[:15],
        "imports": (summary.imports or [])[:20],
        "threading_primitives": summary.threading_primitives or [],
        "external_calls": (summary.external_calls or [])[:15],
    }
    return json.dumps(data, indent=2, default=str)[:3500]


def _tool_run_history_summary(ctx: ToolContext, args: Dict[str, Any]) -> str:
    del args
    return summarize_run_history(list(ctx.observation.run_history or []))


def build_default_tool_executor(**kwargs: Any) -> ToolExecutor:
    ex = ToolExecutor(**kwargs)
    ex.set_handlers(
        {
            "deep_flakiness_scan": _tool_deep_flakiness_scan,
            "causal_graph_summary": _tool_causal_graph_summary,
            "targeting_hints": _tool_targeting_hints,
            "list_repo": _tool_list_repo,
            "read_file": _tool_read_file,
            "ast_summary": _tool_ast_summary,
            "run_history_summary": _tool_run_history_summary,
        }
    )
    return ex


TOOL_MANIFEST_TEXT = """
Available tools (call with action \"tool_call\"):
- deep_flakiness_scan — static repo scan: caches, fixtures, mocks, imports, async contamination (args {})
- causal_graph_summary — compact call graph from test entry (args {})
- targeting_hints — ranked file/hypothesis hints from stack + signals (args {max_hints?: number})
- list_repo — python files with flags (args {limit?: number})
- read_file — excerpt (args {path, start_line, end_line})
- ast_summary — functions/imports/calls for one file (args {path})
- run_history_summary — summarize observation.run_history (args {})
""".strip()
