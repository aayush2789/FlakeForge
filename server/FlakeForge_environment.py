# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlakeForge reinforcement learning environment implementation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        FlakeForgeAction,
        FlakeForgeObservation,
        FlakeForgeState,
        Hypothesis,
        PatchRecord,
        RunRecord,
    )
    from .docker_runner import DockerTestRunner
    from .reward import compute_reward
    from .tools import apply_ast_patch, inject_logging, list_repo_structure, parse_ast_summary, read_file_excerpt
except ImportError:
    from models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState, Hypothesis, PatchRecord, RunRecord
    from server.docker_runner import DockerTestRunner
    from server.reward import compute_reward
    from server.tools import apply_ast_patch, inject_logging, list_repo_structure, parse_ast_summary, read_file_excerpt


@dataclass
class EpisodeState:
    episode_id: str
    step_count: int = 0
    max_steps: int = 14
    done: bool = False
    test_identifier: str = ""
    current_pass_rate: float = 0.0
    baseline_pass_rate: float = 0.0
    regression_detected: bool = False
    run_history: List[RunRecord] = field(default_factory=list)
    patches_applied: List[PatchRecord] = field(default_factory=list)
    log_snippets: List[str] = field(default_factory=list)
    current_hypothesis: Optional[Hypothesis] = None
    judge_scores: List[Dict[str, Any]] = field(default_factory=list)
    total_diff_lines: int = 0
    actions_taken: List[str] = field(default_factory=list)
    hypothesis_confidence_at_each_step: List[float] = field(default_factory=list)


class FlakeForgeEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, repo_path: str = "/app/seed_repos/timing_race", test_id: str = "tests/test_flaky.py::test_flaky_case", max_steps: int = 14):
        self.repo_path = Path(repo_path)
        self.test_id = test_id
        self.max_steps = max_steps
        self.runner = DockerTestRunner(str(self.repo_path))
        self._state = FlakeForgeState(episode_id=str(uuid4()), step_count=0)
        self._episode = EpisodeState(episode_id=self._state.episode_id, max_steps=max_steps, test_identifier=test_id)

    def reset(self) -> FlakeForgeObservation:
        self._restore_clean_repo()
        self._episode = EpisodeState(
            episode_id=str(uuid4()),
            max_steps=self.max_steps,
            test_identifier=self.test_id,
        )

        baseline_runs = self._run_test_n_times(n=10)
        self._episode.run_history = baseline_runs[-10:]
        self._episode.baseline_pass_rate = self._pass_rate(baseline_runs)
        self._episode.current_pass_rate = self._episode.baseline_pass_rate

        self._state = FlakeForgeState(
            episode_id=self._episode.episode_id,
            step_count=0,
            done=False,
            current_pass_rate=self._episode.current_pass_rate,
            baseline_pass_rate=self._episode.baseline_pass_rate,
            regression_detected=False,
            judge_scores=[],
        )
        return self._build_initial_observation()

    def step(self, action: FlakeForgeAction) -> FlakeForgeObservation:  # type: ignore[override]
        if self._episode.done:
            obs = self._build_observation(reward=0.0, done=True)
            obs.metadata = {"reason": "episode_already_done"}
            return obs

        self._episode.step_count += 1
        self._state.step_count = self._episode.step_count
        self._episode.actions_taken.append(action.action_type)
        self._episode.hypothesis_confidence_at_each_step.append(
            self._episode.current_hypothesis.confidence if self._episode.current_hypothesis else 0.0
        )

        execution = self._execute_action(action)
        if execution.get("no_op"):
            obs = self._build_observation(reward=-0.5, done=False)
            obs.metadata = execution
            return obs

        post_runs = self._run_test_n_times(n=20)
        self._episode.run_history.extend(post_runs)
        self._episode.run_history = self._episode.run_history[-10:]
        self._episode.current_pass_rate = self._pass_rate(post_runs)
        self._episode.regression_detected = self.runner.check_regressions(exclude_test_id=self.test_id, timeout_seconds=30)

        done = self._episode.regression_detected or self._episode.step_count >= self.max_steps or self._episode.current_pass_rate >= 1.0
        self._episode.done = done

        judge_scores = action.parameters.get("judge_scores", {}) if isinstance(action.parameters, dict) else {}
        step_result = {
            "current_pass_rate": self._episode.current_pass_rate,
            "regression_detected": self._episode.regression_detected,
            "action_taken": action.action_type,
            "done": done,
            "timed_out": self._episode.step_count >= self.max_steps and self._episode.current_pass_rate < 0.9,
        }
        reward = compute_reward(self._episode, step_result, judge_scores)

        self._state.current_pass_rate = self._episode.current_pass_rate
        self._state.baseline_pass_rate = self._episode.baseline_pass_rate
        self._state.done = done
        self._state.regression_detected = self._episode.regression_detected

        obs = self._build_observation(reward=reward, done=done)
        obs.metadata = {**execution, "step_result": step_result}
        return obs

    @property
    def state(self) -> FlakeForgeState:
        return self._state

    def _restore_clean_repo(self) -> None:
        if not self.repo_path.exists():
            return
        try:
            subprocess.run(["git", "checkout", "--", "."], cwd=self.repo_path, check=False, capture_output=True, text=True)
        except Exception:
            pass

    def _run_test_n_times(self, n: int) -> List[RunRecord]:
        return self.runner.run_test_n_times(self.test_id, n=n, max_workers=4)

    def _pass_rate(self, records: List[RunRecord]) -> float:
        if not records:
            return 0.0
        passed = sum(1 for r in records if r.passed)
        return passed / len(records)

    def _build_initial_observation(self) -> FlakeForgeObservation:
        return self._build_observation(reward=0.0, done=False)

    def _build_observation(self, reward: float, done: bool) -> FlakeForgeObservation:
        test_file = self.test_id.split("::", 1)[0]
        test_path = self.repo_path / test_file
        source_candidates = [p for p in self.repo_path.rglob("*.py") if "tests" not in p.parts]
        source_path = source_candidates[0] if source_candidates else test_path

        test_ast = parse_ast_summary(str(test_path)) if test_path.exists() else None
        source_ast = parse_ast_summary(str(source_path)) if source_path.exists() else None

        test_src = read_file_excerpt(str(test_path), 1, 200) if test_path.exists() else ""
        src_under_test = read_file_excerpt(str(source_path), 1, 200) if source_path.exists() else ""

        async_markers: List[str] = []
        if test_ast:
            async_markers.extend([f["name"] for f in test_ast.functions if f.get("is_async")])
        if source_ast:
            async_markers.extend([f["name"] for f in source_ast.functions if f.get("is_async")])

        repo_entries = list_repo_structure(str(self.repo_path)) if self.repo_path.exists() else []

        return FlakeForgeObservation(
            episode_id=self._episode.episode_id,
            test_identifier=self.test_id,
            step=self._episode.step_count,
            steps_remaining=max(self.max_steps - self._episode.step_count, 0),
            test_function_source=test_src,
            source_under_test=src_under_test,
            relevant_imports=(test_ast.imports if test_ast else []),
            file_tree=[entry["path"] for entry in repo_entries],
            async_markers=sorted(set(async_markers)),
            run_history=self._episode.run_history[-10:],
            current_hypothesis=self._episode.current_hypothesis,
            patches_applied=self._episode.patches_applied,
            log_snippets=self._episode.log_snippets,
            current_pass_rate=self._episode.current_pass_rate,
            baseline_pass_rate=self._episode.baseline_pass_rate,
            total_diff_lines=self._episode.total_diff_lines,
            reward=reward,
            done=done,
        )

    def _execute_action(self, action: FlakeForgeAction) -> Dict[str, Any]:
        test_file = self.test_id.split("::", 1)[0]
        test_path = self.repo_path / test_file
        target_file = str(test_path)

        if action.action_type == "GATHER_EVIDENCE":
            injection_points = self._injection_points_from_hypothesis()
            patched_source = inject_logging(target_file, injection_points)
            Path(target_file).write_text(patched_source, encoding="utf-8")
            evidence_runs = self._run_test_n_times(n=5)
            self._episode.log_snippets.extend(self._extract_log_snippets(evidence_runs))
            self._episode.log_snippets = self._episode.log_snippets[-3:]
            subprocess.run(["git", "checkout", "--", test_file], cwd=self.repo_path, check=False, capture_output=True, text=True)
            return {"action": action.action_type, "evidence_runs": len(evidence_runs)}

        if action.action_type == "REVERT_LAST_PATCH":
            if not self._episode.patches_applied:
                return {"action": action.action_type, "no_op": True, "reason": "no_patches_to_revert"}

            last_patch = self._episode.patches_applied.pop()
            target = last_patch.target_file
            diff_proc = subprocess.run(
                ["git", "diff", "HEAD", "--", target],
                cwd=self.repo_path,
                check=False,
                capture_output=True,
                text=True,
            )
            if diff_proc.stdout.strip():
                subprocess.run(
                    ["git", "apply", "-R"],
                    cwd=self.repo_path,
                    check=False,
                    input=diff_proc.stdout,
                    text=True,
                    capture_output=True,
                )
            self._episode.total_diff_lines = max(0, self._episode.total_diff_lines - last_patch.lines_changed)
            return {"action": action.action_type, "reverted": target}

        patch_spec = self._build_patch_spec(action)
        result = apply_ast_patch(target_file, patch_spec)
        if not result.get("success"):
            return {"action": action.action_type, "patch_error": result.get("error", "patch_failed")}

        patch_record = PatchRecord(
            action_taken=action.action_type,
            target_file=target_file,
            lines_changed=int(result.get("lines_changed", 0)),
            pass_rate_after=self._episode.current_pass_rate,
            judge_patch_score=0.0,
        )
        self._episode.patches_applied.append(patch_record)
        self._episode.total_diff_lines += patch_record.lines_changed
        return {
            "action": action.action_type,
            "lines_changed": patch_record.lines_changed,
            "diff": result.get("diff", ""),
        }

    def _injection_points_from_hypothesis(self) -> List[Dict[str, str]]:
        if not self._episode.current_hypothesis:
            return [{"function_name": "test_flaky_case", "position": "entry"}]
        points: List[Dict[str, str]] = []
        for evidence in self._episode.current_hypothesis.evidence:
            fn = evidence.split(":", 1)[0].strip() if ":" in evidence else evidence.strip()
            if fn:
                points.append({"function_name": fn, "position": "entry"})
        return points or [{"function_name": "test_flaky_case", "position": "entry"}]

    def _extract_log_snippets(self, runs: List[RunRecord]) -> List[str]:
        snippets = []
        for run in runs:
            payload = {
                "passed": run.passed,
                "duration_ms": run.duration_ms,
                "error_type": run.error_type,
                "error_message": run.error_message,
            }
            snippets.append(json.dumps(payload))
        return snippets[:3]

    def _build_patch_spec(self, action: FlakeForgeAction) -> Dict[str, Any]:
        test_node_identifier = "def test_"
        if action.action_type == "ADD_TIMING_GUARD":
            return {
                "operation": "insert_before",
                "target": {"type": "call", "identifier": "await"},
                "code_template": "await asyncio.sleep({delay_ms} / 1000)",
                "parameters": {"delay_ms": action.parameters["delay_ms"]},
            }
        if action.action_type == "ADD_SYNCHRONIZATION":
            primitive = action.parameters["primitive"]
            return {
                "operation": "insert_before",
                "target": {"type": "line", "identifier": test_node_identifier},
                "code_template": "# synchronization primitive applied: {primitive}",
                "parameters": {"primitive": primitive},
            }
        if action.action_type == "MOCK_DEPENDENCY":
            return {
                "operation": "add_decorator",
                "target": {"type": "function", "identifier": "test"},
                "code_template": "@unittest.mock.patch('{target}')",
                "parameters": {"target": action.parameters["target"]},
            }
        if action.action_type == "RESET_STATE":
            return {
                "operation": "insert_before",
                "target": {"type": "line", "identifier": test_node_identifier},
                "code_template": "# reset state scope: {scope}",
                "parameters": {"scope": action.parameters["scope"]},
            }
        if action.action_type == "ADD_RETRY":
            return {
                "operation": "insert_before",
                "target": {"type": "line", "identifier": test_node_identifier},
                "code_template": "# retry patch max_attempts={max_attempts} backoff_ms={backoff_ms}",
                "parameters": {
                    "max_attempts": action.parameters["max_attempts"],
                    "backoff_ms": action.parameters["backoff_ms"],
                },
            }
        raise ValueError(f"Unsupported action type for patching: {action.action_type}")


# Backward compatible alias for template-generated class name.
FlakeforgeEnvironment = FlakeForgeEnvironment
