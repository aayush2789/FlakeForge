"""Visualize exactly how causal graph output is used by FlakeForge.

This script shows five layers side-by-side:
1) Raw causal graph dictionary (nodes/edges/boundaries)
2) Raw causal hints from CrossRepoGraphBuilder (boundary warnings)
3) Tools-based targeting hints (stack trace/import/deep signals)
4) Merged hints stored in observation.causal_hints
5) Final TARGETING HINTS section that the agent sees in its prompt

Usage:
  c:/CodingNest/FlakeForge/venv/Scripts/python.exe tests/visualize_causal_graph_usage.py
  c:/CodingNest/FlakeForge/venv/Scripts/python.exe tests/visualize_causal_graph_usage.py --repo-path test_repos/moderate_load_jitter_flaky --test-id tests/test_flaky.py::test_request_processing_should_succeed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.unified_agent import build_unified_prompt
from models import RunRecord
from server.FlakeForge_environment import FlakeForgeEnvironment
from server.tools import build_agent_targeting_hints


class DemoRunner:
    """Alternating pass/fail runner so reset builds realistic failure context."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        self.counter = 0

    def run_test(self, test_identifier: str) -> RunRecord:
        self.counter += 1
        test_file = test_identifier.split("::", 1)[0]
        src = self._best_source_candidate()

        if self.counter % 2 == 0:
            return RunRecord(
                passed=True,
                duration_ms=35 + (self.counter % 9),
                error_type=None,
                error_message=None,
                stderr_excerpt=None,
            )

        trace = (
            "Traceback (most recent call last):\n"
            f"  File \"{(self.repo_path / test_file).as_posix()}\", line 28, in target_test\n"
            f"  File \"{(self.repo_path / src).as_posix()}\", line 17, in process_request\n"
            "AssertionError: intermittent behavior\n"
        )
        return RunRecord(
            passed=False,
            duration_ms=80 + (self.counter % 11),
            error_type="AssertionError",
            error_message="intermittent behavior",
            stderr_excerpt=trace,
        )

    def _best_source_candidate(self) -> str:
        if (self.repo_path / "source.py").exists():
            return "source.py"
        for path in self.repo_path.rglob("*.py"):
            rel = path.relative_to(self.repo_path).as_posix()
            if not rel.startswith("tests/"):
                return rel
        return "tests/test_flaky.py"


def _extract_section(prompt: str, header: str) -> str:
    lines = prompt.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
    if start is None:
        return f"({header} section not found)"

    out: List[str] = []
    for line in lines[start:]:
        if line.startswith("=== ") and line.strip() != header and out:
            break
        out.append(line)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize causal graph usage in FlakeForge")
    parser.add_argument(
        "--repo-path",
        default=os.environ.get("FF_REPO_PATH", "test_repos/moderate_load_jitter_flaky"),
        help="Repo path",
    )
    parser.add_argument(
        "--test-id",
        default=os.environ.get("FF_TEST_ID", "tests/test_flaky.py::test_request_processing_should_succeed"),
        help="Test identifier",
    )
    parser.add_argument("--quick-runs", type=int, default=4, help="Preflight quick runs")
    parser.add_argument("--confirm-runs", type=int, default=4, help="Preflight confirm runs")
    args = parser.parse_args()

    repo_path = Path(args.repo_path)
    env = FlakeForgeEnvironment(
        repo_path=str(repo_path),
        test_identifier=args.test_id,
        runner=DemoRunner(repo_path),
        max_steps=3,
        num_runs=8,
    )

    obs = env.reset(
        preflight_quick_runs=args.quick_runs,
        preflight_confirm_runs=args.confirm_runs,
        drop_deterministic_bugs=False,
    )

    # Raw causal graph output from reset observation.
    causal_graph = obs.causal_graph or {}
    raw_boundary_hints = list(causal_graph.get("boundary_warnings", []))

    # Call builder directly to show underlying raw hints source.
    direct_graph, direct_hints = env._build_causal_graph(obs.test_function_source)

    tool_hints = build_agent_targeting_hints(
        repo_path=str(repo_path),
        test_identifier=obs.test_identifier,
        failing_stack_trace=obs.failing_stack_trace,
        source_under_test=obs.source_under_test,
        causal_frontier=obs.failure_frontier,
        deep_signals={
            "module_cache_violations": obs.module_cache_violations,
            "fixture_scope_risks": obs.fixture_scope_risks,
            "mock_residue_sites": obs.mock_residue_sites,
            "import_side_effect_files": obs.import_side_effect_files,
        },
        max_hints=8,
    )

    merged_recomputed = list(dict.fromkeys([*direct_hints, *tool_hints]))[:10]

    print("\n=== 1) RAW CAUSAL GRAPH (observation.causal_graph) ===")
    print(json.dumps(causal_graph, indent=2))

    print("\n=== 2) RAW CAUSAL HINTS (from causal graph boundary warnings) ===")
    if raw_boundary_hints:
        for i, hint in enumerate(raw_boundary_hints, 1):
            print(f"{i:02d}. {hint}")
    else:
        print("(none)")

    print("\n=== 3) DIRECT BUILDER OUTPUT (for verification) ===")
    print("Raw hints from env._build_causal_graph():")
    if direct_hints:
        for i, hint in enumerate(direct_hints, 1):
            print(f"{i:02d}. {hint}")
    else:
        print("(none)")
    print("Graph has nodes:", len((direct_graph or {}).get("nodes", [])) if direct_graph else 0)
    print("Graph has edges:", len((direct_graph or {}).get("edges", [])) if direct_graph else 0)

    print("\n=== 4) TOOLS HINTS (non-causal) ===")
    for i, hint in enumerate(tool_hints, 1):
        print(f"{i:02d}. {hint}")

    print("\n=== 5) MERGED HINTS USED BY ENV ===")
    for i, hint in enumerate(obs.causal_hints, 1):
        print(f"{i:02d}. {hint}")

    print("\n=== 6) RECOMPUTED MERGE CHECK ===")
    print("Matches observation.causal_hints:", obs.causal_hints == merged_recomputed)

    prompt = build_unified_prompt(obs)
    print("\n=== 7) PROMPT SECTION FED TO AGENT ===")
    print(_extract_section(prompt, "=== TARGETING HINTS ==="))


if __name__ == "__main__":
    main()
