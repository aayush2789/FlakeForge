"""Visualize exactly how deep flakiness signals are used in FlakeForge.

This script demonstrates the complete path:
1) Raw detector output from server.deep_flakiness.build_deep_observation_signals
2) Deep signal fields stored in observation
3) Influence on tools-based file targeting hints
4) Final prompt sections fed to the agent

Usage:
  c:/CodingNest/FlakeForge/venv/Scripts/python.exe tests/visualize_deep_flakiness_usage.py
  c:/CodingNest/FlakeForge/venv/Scripts/python.exe tests/visualize_deep_flakiness_usage.py --repo-path test_repos/moderate_load_jitter_flaky --test-id tests/test_flaky.py::test_request_processing_should_succeed
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
from server.deep_flakiness import build_deep_observation_signals
from server.tools import build_agent_targeting_hints


class DemoRunner:
    """Alternating pass/fail runner so reset captures failing traces."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        self.counter = 0

    def run_test(self, test_identifier: str) -> RunRecord:
        self.counter += 1
        test_file = test_identifier.split("::", 1)[0]
        source_file = self._source_candidate()

        if self.counter % 2 == 0:
            return RunRecord(
                passed=True,
                duration_ms=30 + (self.counter % 10),
                error_type=None,
                error_message=None,
                stderr_excerpt=None,
            )

        trace = (
            "Traceback (most recent call last):\n"
            f"  File \"{(self.repo_path / test_file).as_posix()}\", line 35, in flaky_test\n"
            f"  File \"{(self.repo_path / source_file).as_posix()}\", line 20, in process_request\n"
            "AssertionError: unstable result\n"
        )
        return RunRecord(
            passed=False,
            duration_ms=90 + (self.counter % 12),
            error_type="AssertionError",
            error_message="unstable result",
            stderr_excerpt=trace,
        )

    def _source_candidate(self) -> str:
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
    parser = argparse.ArgumentParser(description="Visualize deep flakiness signal usage in FlakeForge")
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

    # 1) Raw detector output (direct call)
    raw_deep = build_deep_observation_signals(repo_path)

    # 2) Through environment reset -> observation
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

    obs_deep = {
        "module_cache_violations": obs.module_cache_violations,
        "fixture_scope_risks": obs.fixture_scope_risks,
        "mock_residue_sites": obs.mock_residue_sites,
        "import_side_effect_files": obs.import_side_effect_files,
        "async_contamination_alive": obs.async_contamination_alive,
    }

    # 3) Influence on targeting hints
    hints_with_deep = build_agent_targeting_hints(
        repo_path=str(repo_path),
        test_identifier=obs.test_identifier,
        failing_stack_trace=obs.failing_stack_trace,
        source_under_test=obs.source_under_test,
        causal_frontier=obs.failure_frontier,
        deep_signals=obs_deep,
        max_hints=8,
    )

    hints_without_deep = build_agent_targeting_hints(
        repo_path=str(repo_path),
        test_identifier=obs.test_identifier,
        failing_stack_trace=obs.failing_stack_trace,
        source_under_test=obs.source_under_test,
        causal_frontier=obs.failure_frontier,
        deep_signals={
            "module_cache_violations": [],
            "fixture_scope_risks": [],
            "mock_residue_sites": [],
            "import_side_effect_files": [],
        },
        max_hints=8,
    )

    prompt = build_unified_prompt(obs)

    print("\n=== 1) RAW DEEP DETECTOR OUTPUT ===")
    print(json.dumps(raw_deep, indent=2))

    print("\n=== 2) OBSERVATION DEEP SIGNALS (post-reset) ===")
    print(json.dumps(obs_deep, indent=2))

    print("\n=== 3) TARGETING HINTS WITH DEEP SIGNALS ===")
    for i, hint in enumerate(hints_with_deep, 1):
        print(f"{i:02d}. {hint}")

    print("\n=== 4) TARGETING HINTS WITHOUT DEEP SIGNALS ===")
    for i, hint in enumerate(hints_without_deep, 1):
        print(f"{i:02d}. {hint}")

    added_by_deep = [h for h in hints_with_deep if h not in hints_without_deep]
    print("\n=== 5) HINTS CONTRIBUTED BY DEEP SIGNALS ===")
    if added_by_deep:
        for i, hint in enumerate(added_by_deep, 1):
            print(f"{i:02d}. {hint}")
    else:
        print("(none in this repo snapshot)")

    print("\n=== 6) PROMPT SECTION: DEEP FLAKINESS SIGNALS ===")
    print(_extract_section(prompt, "=== DEEP FLAKINESS SIGNALS ==="))

    print("\n=== 7) PROMPT SECTION: TARGETING HINTS ===")
    print(_extract_section(prompt, "=== TARGETING HINTS ==="))


if __name__ == "__main__":
    main()
