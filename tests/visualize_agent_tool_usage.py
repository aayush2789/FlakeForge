"""Visualize how FlakeForge tool evidence is produced and fed to the agent.

This script does not require Docker. It uses a lightweight demo runner that
alternates pass/fail outcomes and emits traceback-like stderr so the same
pipeline pieces are exercised:
- preflight classification
- deep flakiness signals
- causal frontier extraction
- tools-based file targeting hints
- final prompt section shown to the agent

Usage:
  c:/CodingNest/FlakeForge/venv/Scripts/python.exe tests/visualize_agent_tool_usage.py
  c:/CodingNest/FlakeForge/venv/Scripts/python.exe tests/visualize_agent_tool_usage.py --repo-path test_repos/moderate_load_jitter_flaky --test-id tests/test_flaky.py::test_request_processing_should_succeed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root imports resolve when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.unified_agent import build_unified_prompt
from models import RunRecord
from server.FlakeForge_environment import FlakeForgeEnvironment
from server.tools import build_agent_targeting_hints


class DemoRunner:
    """Deterministic alternating pass/fail runner for visualization only."""

    def __init__(self, repo_path: Path, test_identifier: str) -> None:
        self.repo_path = repo_path
        self.test_identifier = test_identifier
        self.counter = 0

    def run_test(self, test_identifier: str) -> RunRecord:
        self.counter += 1
        test_file = test_identifier.split("::", 1)[0]
        source_file = self._best_source_candidate(test_file)

        # Alternate outcomes so preflight can classify as flaky.
        passed = (self.counter % 2 == 0)
        if passed:
            return RunRecord(
                passed=True,
                duration_ms=40 + (self.counter % 7),
                error_type=None,
                error_message=None,
                stderr_excerpt=None,
            )

        trace = (
            "Traceback (most recent call last):\n"
            f"  File \"{(self.repo_path / test_file).as_posix()}\", line 42, in test_case\n"
            f"  File \"{(self.repo_path / source_file).as_posix()}\", line 21, in process_request\n"
            "AssertionError: transient mismatch\n"
        )
        return RunRecord(
            passed=False,
            duration_ms=95 + (self.counter % 13),
            error_type="AssertionError",
            error_message="transient mismatch",
            stderr_excerpt=trace,
        )

    def _best_source_candidate(self, test_file: str) -> str:
        direct = self.repo_path / "source.py"
        if direct.exists():
            return "source.py"

        # Fallback: any .py file that is not a test file.
        for path in self.repo_path.rglob("*.py"):
            rel = path.relative_to(self.repo_path).as_posix()
            if rel.startswith("tests/"):
                continue
            return rel

        return test_file


def _extract_targeting_section(prompt: str) -> str:
    header = "=== TARGETING HINTS ==="
    if header not in prompt:
        return "(TARGETING HINTS section not found in prompt)"

    lines = prompt.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
    if start is None:
        return "(TARGETING HINTS section not found in prompt)"

    collected = []
    for line in lines[start:]:
        if line.startswith("=== ") and line.strip() != header and collected:
            break
        collected.append(line)
    return "\n".join(collected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize FlakeForge tool-driven file targeting")
    parser.add_argument(
        "--repo-path",
        default=os.environ.get("FF_REPO_PATH", "test_repos/moderate_load_jitter_flaky"),
        help="Repo path used for environment reset",
    )
    parser.add_argument(
        "--test-id",
        default=os.environ.get("FF_TEST_ID", "tests/test_flaky.py::test_request_processing_should_succeed"),
        help="Test identifier",
    )
    parser.add_argument(
        "--quick-runs",
        type=int,
        default=6,
        help="Preflight quick runs",
    )
    parser.add_argument(
        "--confirm-runs",
        type=int,
        default=6,
        help="Preflight confirm runs",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo_path)
    runner = DemoRunner(repo_path=repo_path, test_identifier=args.test_id)

    env = FlakeForgeEnvironment(
        repo_path=str(repo_path),
        test_identifier=args.test_id,
        runner=runner,
        max_steps=3,
        num_runs=8,
    )

    obs = env.reset(
        preflight_quick_runs=args.quick_runs,
        preflight_confirm_runs=args.confirm_runs,
        drop_deterministic_bugs=False,
    )

    print("\n=== PREFLIGHT SUMMARY ===")
    print(json.dumps(obs.preflight_result, indent=2))

    print("\n=== DEEP SIGNAL SNAPSHOT ===")
    deep_counts = {
        "module_cache_violations": len(obs.module_cache_violations),
        "fixture_scope_risks": len(obs.fixture_scope_risks),
        "mock_residue_sites": len(obs.mock_residue_sites),
        "import_side_effect_files": len(obs.import_side_effect_files),
        "async_contamination_alive": bool(obs.async_contamination_alive),
    }
    print(json.dumps(deep_counts, indent=2))

    print("\n=== MERGED TARGETING HINTS IN OBSERVATION ===")
    for idx, hint in enumerate(obs.causal_hints[:10], start=1):
        print(f"{idx:02d}. {hint}")

    extra_hints = build_agent_targeting_hints(
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

    print("\n=== TOOLS-ONLY TARGETING HINTS (DIRECT CALL) ===")
    for idx, hint in enumerate(extra_hints, start=1):
        print(f"{idx:02d}. {hint}")

    prompt = build_unified_prompt(obs)
    print("\n=== PROMPT TARGETING SECTION SEEN BY AGENT ===")
    print(_extract_targeting_section(prompt))


if __name__ == "__main__":
    main()
