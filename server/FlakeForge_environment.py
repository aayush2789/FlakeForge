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
    from .causal_graph import CrossRepoGraphBuilder
    from .chaos_runner import ChaosAmplifiedRunner, ChaosProfile
    from .docker_runner import DockerTestRunner
    from .perf_sentinel import PerformanceSentinel
    from .reward import compute_reward
    from .tools import (
        apply_ast_patch,
        get_failure_pattern,
        inject_logging,
        list_repo_structure,
        parse_ast_summary,
        read_file_excerpt,
        resolve_target_from_evidence,
    )
except ImportError:
    from models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState, Hypothesis, PatchRecord, RunRecord  # type: ignore
    from server.causal_graph import CrossRepoGraphBuilder  # type: ignore
    from server.chaos_runner import ChaosAmplifiedRunner, ChaosProfile  # type: ignore
    from server.docker_runner import DockerTestRunner  # type: ignore
    from server.perf_sentinel import PerformanceSentinel  # type: ignore
    from server.reward import compute_reward  # type: ignore
    from server.tools import (  # type: ignore
        apply_ast_patch,
        get_failure_pattern,
        inject_logging,
        list_repo_structure,
        parse_ast_summary,
        read_file_excerpt,
        resolve_target_from_evidence,
    )


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
    hypothesis_history: List[Dict[str, Any]] = field(default_factory=list)
    # ── V2 fields ─────────────────────────────────────────────────────
    chaos_pass_rate: Optional[float] = None
    chaos_baseline_pass_rate: Optional[float] = None
    perf_regression_detected: bool = False
    perf_median_ratio: float = 1.0
    infrastructure_sensitive: bool = False
    causal_graph_dict: Optional[Dict[str, Any]] = None


class FlakeForgeEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        repo_path: str = "/app/seed_repos/timing_race",
        test_id: str = "tests/test_flaky.py::test_flaky_case",
        max_steps: int = 14,
        benchmark_test_id: Optional[str] = None,
        chaos_profile: str = "cpu",
    ):
        self.repo_path = Path(repo_path)
        self.test_id = test_id
        self.max_steps = max_steps
        self.benchmark_test_id = benchmark_test_id  # e.g. "tests/test_benchmark.py::test_speed"
        self.chaos_profile = ChaosProfile(chaos_profile) if chaos_profile != "none" else ChaosProfile.NONE
        # V1: standard runner for regression checks
        self.runner = DockerTestRunner(str(self.repo_path))
        # V2 Pillar 2: chaos-capable runner
        self.chaos_runner = ChaosAmplifiedRunner(str(self.repo_path))
        # V2 Pillar 4: performance sentinel
        self.perf_sentinel = PerformanceSentinel()
        # V2 Pillar 1: causal graph builder
        self.causal_graph_builder = CrossRepoGraphBuilder(str(self.repo_path), max_depth=3)
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

        # ── V2 Pillar 2: Chaos baseline ───────────────────────────────────────
        if self.chaos_profile != ChaosProfile.NONE:
            is_sensitive, chaos_baseline = self.chaos_runner.is_infrastructure_sensitive(
                test_id=self.test_id,
                clean_pass_rate=self._episode.baseline_pass_rate,
                profile=self.chaos_profile,
                n=10,
            )
            self._episode.chaos_baseline_pass_rate = chaos_baseline
            self._episode.infrastructure_sensitive = is_sensitive
        else:
            self._episode.chaos_baseline_pass_rate = None
            self._episode.infrastructure_sensitive = False

        # ── V2 Pillar 4: Performance baseline ─────────────────────────────────
        if self.benchmark_test_id:
            try:
                self.perf_sentinel.capture_baseline(self.runner, self.benchmark_test_id)
            except Exception:
                pass  # Graceful degradation: no benchmark test available

        # ── V2 Pillar 1: Build causal graph ─────────────────────────────────
        try:
            test_file = self.test_id.split("::", 1)[0]
            test_func = self.test_id.split("::", 1)[-1] if "::" in self.test_id else ""
            if test_func:
                causal_graph = self.causal_graph_builder.build(
                    entry_file=str(self.repo_path / test_file),
                    entry_function=test_func,
                )
                self._episode.causal_graph_dict = causal_graph.to_observation_dict()
        except Exception:
            self._episode.causal_graph_dict = None  # Graceful degradation

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

        self._update_hypothesis_from_action(action)

        if action.action_type not in {"GATHER_EVIDENCE", "REVERT_LAST_PATCH"}:
            if self._episode.current_hypothesis is None:
                obs = self._build_observation(reward=-1.0, done=False)
                obs.metadata = {"reason": "action_requires_hypothesis"}
                return obs
            if self._episode.current_hypothesis.confidence < 0.3:
                obs = self._build_observation(reward=-1.0, done=False)
                obs.metadata = {"reason": "hypothesis_confidence_too_low"}
                return obs

        self._episode.step_count += 1
        self._state.step_count = self._episode.step_count
        self._episode.actions_taken.append(action.action_type)
        current_conf = self._episode.current_hypothesis.confidence if self._episode.current_hypothesis else 0.0
        self._episode.hypothesis_confidence_at_each_step.append(current_conf)
        self._episode.hypothesis_history.append({"step": self._episode.step_count, "confidence": current_conf})

        execution = self._execute_action(action)
        if execution.get("no_op"):
            obs = self._build_observation(reward=-0.5, done=False)
            obs.metadata = execution
            return obs

        post_runs = self._run_test_n_times(n=10)
        self._episode.run_history.extend(post_runs)
        self._episode.run_history = self._episode.run_history[-10:]
        self._episode.current_pass_rate = self._pass_rate(post_runs)
        self._episode.regression_detected = self.runner.check_regressions(exclude_test_id=self.test_id, timeout_seconds=30)

        # ── V2 Pillar 2: Chaos verification after each patch ─────────────────────
        if self.chaos_profile != ChaosProfile.NONE and action.action_type != "CHAOS_PROBE":
            chaos_records = self.chaos_runner.run_test_n_times_chaos(
                self.test_id, n=10, profile=self.chaos_profile
            )
            self._episode.chaos_pass_rate = self._pass_rate(chaos_records)
        elif action.action_type == "CHAOS_PROBE":
            # CHAOS_PROBE is handled in _execute_action; result already on episode
            pass

        # ── V2 Pillar 4: Performance sentinel ─────────────────────────────────
        if self.perf_sentinel.has_baseline and action.action_type not in {
            "GATHER_EVIDENCE", "CHAOS_PROBE", "DIAGNOSE_BOUNDARY"
        }:
            sentinel_result = self.perf_sentinel.check_regression(self.runner)
            self._episode.perf_regression_detected = sentinel_result.is_regression
            self._episode.perf_median_ratio = sentinel_result.median_ratio

        done = (
            self._episode.regression_detected
            or self._episode.step_count >= self.max_steps
            or self._episode.current_pass_rate >= 1.0
        )
        final_validation_runs = 0
        if done and not self._episode.regression_detected:
            terminal_runs = self._run_test_n_times(n=50)
            final_validation_runs = len(terminal_runs)
            self._episode.current_pass_rate = self._pass_rate(terminal_runs)
            self._episode.run_history.extend(terminal_runs)
            self._episode.run_history = self._episode.run_history[-10:]
        self._episode.done = done

        judge_scores = action.judge_feedback.model_dump() if action.judge_feedback is not None else {}
        if judge_scores:
            self._episode.judge_scores.append(judge_scores)
        failure_pattern = get_failure_pattern(post_runs)
        step_result = {
            "current_pass_rate": self._episode.current_pass_rate,
            "regression_detected": self._episode.regression_detected,
            "action_taken": action.action_type,
            "done": done,
            "timed_out": self._episode.step_count >= self.max_steps and self._episode.current_pass_rate < 0.9,
            "ast_diff": execution.get("ast_diff", {}),
            # ── V2 additions ──
            "chaos_pass_rate": self._episode.chaos_pass_rate,
            "chaos_baseline_pass_rate": self._episode.chaos_baseline_pass_rate,
            "perf_regression_detected": self._episode.perf_regression_detected,
            "perf_median_ratio": self._episode.perf_median_ratio,
        }
        reward, reward_breakdown = compute_reward(self._episode, step_result, judge_scores)

        self._state.current_pass_rate = self._episode.current_pass_rate
        self._state.baseline_pass_rate = self._episode.baseline_pass_rate
        self._state.done = done
        self._state.regression_detected = self._episode.regression_detected

        obs = self._build_observation(reward=reward, done=done)
        obs.metadata = {
            **execution,
            "step_result": step_result,
            "reward_breakdown": reward_breakdown,
            "failure_pattern": {
                "pass_rate": failure_pattern.pass_rate,
                "most_common_error": failure_pattern.most_common_error,
                "error_distribution": failure_pattern.error_distribution,
                "duration_mean": failure_pattern.duration_mean,
                "duration_std": failure_pattern.duration_std,
                "flakiness_score": failure_pattern.flakiness_score,
            },
            "hypothesis_history": self._episode.hypothesis_history,
            "final_validation_runs": final_validation_runs,
        }
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
            # ── V2 new fields ────────────────────────────────────────────────────
            causal_graph=self._episode.causal_graph_dict,
            chaos_pass_rate=self._episode.chaos_pass_rate,
            chaos_baseline_pass_rate=self._episode.chaos_baseline_pass_rate,
            infrastructure_sensitive=self._episode.infrastructure_sensitive,
            perf_sentinel_status={
                "regression": self._episode.perf_regression_detected,
                "median_ratio": self._episode.perf_median_ratio,
            } if self.perf_sentinel.has_baseline else None,
        )

    def _execute_action(self, action: FlakeForgeAction) -> Dict[str, Any]:
        test_file = self.test_id.split("::", 1)[0]
        test_path = self.repo_path / test_file
        target_file = str(test_path)
        evidence = self._episode.current_hypothesis.evidence if self._episode.current_hypothesis else []
        resolved_target = resolve_target_from_evidence(target_file, evidence)

        if action.action_type == "GATHER_EVIDENCE":
            injection_points = self._injection_points_from_hypothesis(resolved_target)
            patched_source = inject_logging(target_file, injection_points)
            Path(target_file).write_text(patched_source, encoding="utf-8")
            evidence_runs = self._run_test_n_times(n=5)
            self._episode.log_snippets.extend(self._extract_log_snippets(evidence_runs))
            self._episode.log_snippets = self._episode.log_snippets[-3:]
            self._episode.current_hypothesis = self._infer_hypothesis(evidence_runs)
            subprocess.run(["git", "checkout", "--", test_file], cwd=self.repo_path, check=False, capture_output=True, text=True)
            return {
                "action": action.action_type,
                "evidence_runs": len(evidence_runs),
                "resolved_target": resolved_target,
            }

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

        if action.action_type != "REVERT_LAST_PATCH" and not resolved_target.get("identifier"):
            return {"action": action.action_type, "no_op": True, "reason": "unable_to_ground_evidence"}

        patch_spec = self._build_patch_spec(action, resolved_target)
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
            "ast_diff": result.get("ast_diff", {}),
            "resolved_target": resolved_target,
        }

    def _injection_points_from_hypothesis(self, resolved_target: Dict[str, Any]) -> List[Dict[str, str]]:
        if resolved_target.get("type") == "function":
            return [{"function_name": resolved_target.get("identifier", "test_flaky_case"), "position": "entry"}]
        if not self._episode.current_hypothesis:
            return [{"function_name": "test_flaky_case", "position": "entry"}]
        points: List[Dict[str, str]] = []
        for evidence in self._episode.current_hypothesis.evidence:
            fn = evidence.split(":", 1)[0].strip() if ":" in evidence else evidence.strip()
            if fn:
                points.append({"function_name": fn, "position": "entry"})
        return points or [{"function_name": "test_flaky_case", "position": "entry"}]

    def _extract_log_snippets(self, runs: List[RunRecord]) -> List[str]:
        snippets: List[str] = []
        for run in runs:
            if run.stderr_excerpt:
                for line in run.stderr_excerpt.splitlines():
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        snippets.append(line)
            payload = {
                "passed": run.passed,
                "duration_ms": run.duration_ms,
                "error_type": run.error_type,
                "error_message": run.error_message,
            }
            snippets.append(json.dumps(payload))
        # Keep a short, high-signal window.
        return snippets[:3]

    def _build_patch_spec(self, action: FlakeForgeAction, resolved_target: Dict[str, Any]) -> Dict[str, Any]:
        test_node_identifier = "def test_"
        if action.action_type == "ADD_TIMING_GUARD":
            return {
                "operation": "insert_before",
                "target": {"type": "call", "identifier": resolved_target.get("identifier", "await")},
                "code_template": "await asyncio.sleep({delay_ms} / 1000)",
                "parameters": {"delay_ms": action.parameters["delay_ms"]},
            }
        if action.action_type == "ADD_SYNCHRONIZATION":
            primitive = action.parameters["primitive"]
            return {
                "operation": "wrap_with",
                "target": {"type": "function", "identifier": resolved_target.get("identifier", "test")},
                "code_template": "with {lock_var}:\\n    {body}",
                "parameters": {"lock_var": "_flakeforge_lock", "primitive": primitive},
            }
        if action.action_type == "MOCK_DEPENDENCY":
            return {
                "operation": "add_decorator",
                "target": {"type": "function", "identifier": resolved_target.get("identifier", "test")},
                "code_template": "@unittest.mock.patch('{target}')",
                "parameters": {"target": action.parameters["target"]},
            }
        if action.action_type == "RESET_STATE":
            return {
                "operation": "ensure_reset_fixture",
                "target": {
                    "scope": action.parameters["scope"],
                },
                "code_template": "",
                "parameters": {},
            }
        if action.action_type == "ADD_RETRY":
            return {
                "operation": "ensure_retry_wrapper",
                "target": {
                    "function_name": resolved_target.get("identifier", "test"),
                    "max_attempts": action.parameters["max_attempts"],
                    "backoff_ms": action.parameters["backoff_ms"],
                },
                "code_template": "",
                "parameters": {},
            }
        if action.action_type == "SEED_RANDOMNESS":
            return {
                "operation": "ensure_seed_call",
                "target": {
                    "function_name": resolved_target.get("identifier", "test"),
                    "library": action.parameters["library"],
                },
                "code_template": "",
                "parameters": {},
            }
        # ── V2 Deep-Action Handlers ────────────────────────────────────────────
        if action.action_type == "DIAGNOSE_BOUNDARY":
            boundary_node = action.parameters.get("boundary_node", "")
            # Rebuild the causal graph and return its dict
            try:
                test_file = self.test_id.split("::", 1)[0]
                test_func = self.test_id.split("::", 1)[-1] if "::" in self.test_id else ""
                if test_func:
                    fresh_graph = self.causal_graph_builder.build(
                        str(self.repo_path / test_file), test_func
                    )
                    self._episode.causal_graph_dict = fresh_graph.to_observation_dict()
                boundary_nodes = [
                    n for n in (self._episode.causal_graph_dict or {}).get("nodes", [])
                    if boundary_node in n.get("id", "") or n.get("boundary")
                ]
                return {
                    "action": action.action_type,
                    "boundary_node": boundary_node,
                    "found_boundaries": boundary_nodes,
                    "no_op": False,
                }
            except Exception as exc:
                return {"action": action.action_type, "no_op": True, "reason": str(exc)}

        if action.action_type == "CHAOS_PROBE":
            profile = ChaosProfile(action.parameters["profile"])
            n_runs = action.parameters.get("n_runs", 10)
            chaos_records = self.chaos_runner.run_test_n_times_chaos(
                self.test_id, n=n_runs, profile=profile
            )
            chaos_pass_rate = self._pass_rate(chaos_records)
            self._episode.chaos_pass_rate = chaos_pass_rate
            clean_pr = self._episode.current_pass_rate
            is_sensitive = chaos_pass_rate < (clean_pr - 0.2)
            self._episode.infrastructure_sensitive = is_sensitive
            return {
                "action": action.action_type,
                "profile": profile.value,
                "n_runs": n_runs,
                "chaos_pass_rate": chaos_pass_rate,
                "clean_pass_rate": clean_pr,
                "infrastructure_sensitive": is_sensitive,
                "no_op": False,
            }

        if action.action_type == "REFACTOR_CONCURRENCY":
            return {
                "operation": "refactor_concurrency_primitive",
                "target": {
                    "function_name": action.parameters["target_function"],
                    "from_primitive": action.parameters["from_primitive"],
                    "to_primitive": action.parameters["to_primitive"],
                },
                "code_template": "",
                "parameters": dict(action.parameters),
            }

        if action.action_type == "ISOLATE_BOUNDARY":
            return {
                "operation": "isolate_boundary_call",
                "target": {
                    "boundary_call": action.parameters["boundary_call"],
                    "pattern": action.parameters["pattern"],
                },
                "code_template": "",
                "parameters": dict(action.parameters),
            }

        if action.action_type == "EXTRACT_ASYNC_SCOPE":
            return {
                "operation": "extract_async_scope",
                "target": {
                    "function_name": action.parameters["target_function"],
                    "direction": action.parameters["direction"],
                },
                "code_template": "",
                "parameters": dict(action.parameters),
            }

        if action.action_type == "HARDEN_IDEMPOTENCY":
            return {
                "operation": "harden_idempotency",
                "target": {
                    "state_target": action.parameters["state_target"],
                    "key_strategy": action.parameters["key_strategy"],
                },
                "code_template": "",
                "parameters": dict(action.parameters),
            }

        raise ValueError(f"Unsupported action type for patching: {action.action_type}")

    def _update_hypothesis_from_action(self, action: FlakeForgeAction) -> None:
        if action.hypothesis is None:
            return
        try:
            self._episode.current_hypothesis = Hypothesis(**action.hypothesis.model_dump())
        except Exception:
            # Ignore malformed hypotheses and keep previous valid hypothesis.
            return

    def _infer_hypothesis(self, runs: List[RunRecord]) -> Hypothesis:
        error_types = [r.error_type or "" for r in runs if not r.passed]
        top_error = error_types[0] if error_types else ""

        category = "NONDETERMINISM"
        if "Timeout" in top_error or "TimeoutError" in top_error:
            category = "TIMING_RACE"
        elif "Connection" in top_error:
            category = "EXTERNAL_DEPENDENCY"
        elif "Assertion" in top_error:
            category = "ORDER_DEPENDENCY"

        pass_rate = self._pass_rate(runs)
        confidence = max(0.3, min(0.95, 1.0 - abs(pass_rate - 0.5)))
        evidence = [
            self.test_id.split("::")[-1],
            top_error or "intermittent_failure",
        ]
        return Hypothesis(
            root_cause_category=category,
            confidence=confidence,
            evidence=evidence,
            suggested_action="GATHER_EVIDENCE" if confidence < 0.5 else "ADD_TIMING_GUARD",
        )


# Backward compatible alias for template-generated class name.
FlakeforgeEnvironment = FlakeForgeEnvironment
