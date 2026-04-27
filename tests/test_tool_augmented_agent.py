"""Tests for tool-augmented agent JSON parsing, tool executor, and inner loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models import FlakeForgeObservation, RunRecord
from agent.action_schema import PatchActionModel, ToolCallActionModel, parse_agent_step_json
from agent.tool_loop import (
    ToolContext,
    ToolExecutor,
    ToolTraceEntry,
    build_default_tool_executor,
    format_tool_trace_for_prompt,
    summarize_deep_signals,
)
from agent.unified_agent import ToolAugmentedFlakeForgeAgent, build_minimal_agent_prompt


def _minimal_obs(**kwargs) -> FlakeForgeObservation:
    base = dict(
        episode_id="e1",
        test_identifier="tests/test_x.py::test_foo",
        step=0,
        steps_remaining=3,
        repo_root="",
        test_function_source="def test_foo():\n    assert True\n",
        source_under_test="def foo():\n    return 1\n",
        failing_stack_trace="Error\n  File \"src/x.py\", line 2, in foo\n",
        run_history=[
            RunRecord(passed=False, duration_ms=10, error_type="AssertionError", error_message="x"),
        ],
    )
    base.update(kwargs)
    return FlakeForgeObservation(**base)


def test_parse_agent_step_tool_call() -> None:
    data = json.loads('{"action":"tool_call","tool":"list_repo","args":{"limit":3}}')
    m = parse_agent_step_json(data)
    assert isinstance(m, ToolCallActionModel)
    assert m.tool == "list_repo"
    assert m.args == {"limit": 3}


def test_parse_agent_step_patch() -> None:
    data = json.loads(
        '{"action":"patch","think":{"claims":[],"confidence":0.5},"patch":{"hunks":[]}}'
    )
    m = parse_agent_step_json(data)
    assert isinstance(m, PatchActionModel)


def test_parse_legacy_think_patch() -> None:
    data = json.loads('{"think":{"claims":[],"confidence":0.3},"patch":{"hunks":[]}}')
    m = parse_agent_step_json(data)
    assert isinstance(m, PatchActionModel)
    assert m.action == "patch"


def test_tool_executor_budget_and_cache() -> None:
    ex = ToolExecutor(max_calls_total=2, per_tool_max={"t1": 5})
    calls = []

    def h(ctx: ToolContext, args: dict) -> str:
        calls.append(1)
        return "ok"

    ex.set_handlers({"t1": h})
    ctx = ToolContext(repo_root="/tmp", observation=_minimal_obs())
    # Distinct args so the second call is not a cache hit (budget test).
    e1 = ex.execute(ctx, "t1", {"i": 1})
    e2 = ex.execute(ctx, "t1", {"i": 2})
    e3 = ex.execute(ctx, "t1", {"i": 3})
    assert e1.ok and e2.ok
    assert not e3.ok
    assert "budget" in e3.summary.lower()
    assert sum(calls) == 2

    ex2 = build_default_tool_executor(max_calls_total=10)
    ctx2 = ToolContext(repo_root=str(Path(__file__).resolve().parent), observation=_minimal_obs())
    a = ex2.execute(ctx2, "run_history_summary", {})
    b = ex2.execute(ctx2, "run_history_summary", {})
    assert a.ok and b.ok
    assert a.summary == b.summary


def test_tool_executor_unknown_tool() -> None:
    ex = build_default_tool_executor(max_calls_total=3)
    ctx = ToolContext(repo_root=".", observation=_minimal_obs())
    e = ex.execute(ctx, "not_a_real_tool", {})
    assert not e.ok


def test_format_tool_trace() -> None:
    t = [
        ToolTraceEntry(turn=1, tool="x", ok=True, summary="hello " * 200),
    ]
    s = format_tool_trace_for_prompt(t)
    assert "TOOL OBSERVATIONS" in s
    assert "truncated" in s or len(s) < 2500


def test_build_minimal_prompt_no_bloat() -> None:
    obs = _minimal_obs(
        repo_root="/repo",
        module_cache_violations=["a.py: lru_cache"],
        causal_hints=["hint1", "hint2"],
    )
    p = build_minimal_agent_prompt(obs, [])
    assert "/repo" in p
    assert "SOURCE UNDER TEST" in p
    assert "DEEP SIGNALS" not in p
    assert "TARGETING HINTS" not in p


def test_tool_augmented_inner_loop_two_llm_turns(tmp_path: Path) -> None:
    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / "lib.py").write_text("X = 1\n", encoding="utf-8")

    obs = _minimal_obs(repo_root=str(repo))
    ctx = ToolContext(repo_root=str(repo), observation=obs, env=None)

    responses = [
        '{"action":"tool_call","tool":"list_repo","args":{"limit":5}}',
        (
            '{"action":"patch","think":{"claims":[{"claim_id":"c1","category":"unknown",'
            '"entity":"","location":"","polarity":"present","reason":"done"}],'
            '"confidence":0.4},"patch":{"hunks":[{"hunk_id":"h1","file":"lib.py",'
            '"search":"X = 1","replace":"X = 2","rationale":"","addresses_claim":""}]}}'
        ),
    ]
    backend = MagicMock()
    backend.generate.side_effect = responses

    agent = ToolAugmentedFlakeForgeAgent(backend, max_tool_calls=8, max_llm_rounds=8)
    action = agent.generate(obs, tool_context=ctx)

    assert backend.generate.call_count == 2
    assert "lib.py" in action.patch_text
    assert action.structured_patch is not None
    assert len(action.structured_patch.hunks) >= 1


def test_summarize_deep_signals_empty() -> None:
    s = summarize_deep_signals({})
    assert "No deep" in s or "scan" in s
