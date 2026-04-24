# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlakeForge reinforcement learning environment implementation."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    from .logger import FullTraceLogger
    from .causal_graph import EpisodeCausalTrace
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
    try:
        from FlakeForge.models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState, Hypothesis, PatchRecord, RunRecord  # type: ignore
        from FlakeForge.server.causal_graph import CrossRepoGraphBuilder, EpisodeCausalTrace  # type: ignore
        from FlakeForge.server.chaos_runner import ChaosAmplifiedRunner, ChaosProfile  # type: ignore
        from FlakeForge.server.docker_runner import DockerTestRunner  # type: ignore
        from FlakeForge.server.perf_sentinel import PerformanceSentinel  # type: ignore
        from FlakeForge.server.reward import compute_reward  # type: ignore
        from FlakeForge.server.logger import FullTraceLogger  # type: ignore
        from FlakeForge.server.tools import (  # type: ignore
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
        from server.causal_graph import CrossRepoGraphBuilder, EpisodeCausalTrace  # type: ignore
        from server.chaos_runner import ChaosAmplifiedRunner, ChaosProfile  # type: ignore
        from server.docker_runner import DockerTestRunner  # type: ignore
        from server.perf_sentinel import PerformanceSentinel  # type: ignore
        from server.reward import compute_reward  # type: ignore
        from server.logger import FullTraceLogger  # type: ignore
        from server.tools import (  # type: ignore
            apply_ast_patch,
            get_failure_pattern,
            inject_logging,
            list_repo_structure,
            parse_ast_summary,
            read_file_excerpt,
            resolve_target_from_evidence,
        )

try:
    from .state import EpisodeState
    from .hypothesis_engine import compute_duration_fingerprint, infer_hypothesis
    from .action_executor import (
        canonical_action,
        injection_points_from_hypothesis,
        extract_log_snippets,
        build_patch_spec,
    )
except ImportError:
    from server.state import EpisodeState
    from server.hypothesis_engine import compute_duration_fingerprint, infer_hypothesis
    from server.action_executor import (
        canonical_action,
        injection_points_from_hypothesis,
        extract_log_snippets,
        build_patch_spec,
    )





class FlakeForgeEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        repo_path: str = "/app/seed_repos/timing_race",
        test_id: str = "tests/test_flaky.py::test_flaky_case",
        max_steps: int = 14,
        benchmark_test_id: Optional[str] = None,
        chaos_profile: str = "none",
    ):
        self.repo_path = Path(repo_path)
        self.repo_path = self._resolve_repo_path(self.repo_path)
        
        self.test_id = self._resolve_test_id(test_id)
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
        self.trace_logger = FullTraceLogger()
        self.causal_trace = EpisodeCausalTrace()
        self.debug_enabled = os.getenv("ENV_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
        # ── RLVR Hybrid: Oracle loaded once per repo ────────────────────────
        self._manifest_oracle: Dict[str, Any] = self._load_manifest()

    def _resolve_repo_path(self, repo_path: Path) -> Path:
        root = Path(__file__).resolve().parent.parent
        candidates: List[Path] = []

        if repo_path.is_absolute() and repo_path.exists():
            return repo_path

        if not repo_path.is_absolute():
            candidates.append(root / repo_path)
        else:
            # Container-style paths like /app/seed_repos/... should map to the local workspace
            relative_parts = repo_path.parts[1:] if len(repo_path.parts) > 1 else repo_path.parts
            candidates.append(root / Path(*relative_parts))
            if "seed_repos" in repo_path.parts:
                candidates.append(root / "test_repos" / "timing_race_minimal")

        candidates.append(root / "test_repos" / "timing_race_minimal")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Fall back to the original value so callers see the failure path clearly.
        return repo_path

    def _resolve_test_id(self, test_id: str) -> str:
        test_file, _, test_func = test_id.partition("::")
        candidate_files = [self.repo_path / test_file]
        if not candidate_files[0].exists():
            candidate_files.append(self.repo_path / "tests" / "test_flaky.py")

        resolved_file = next((path for path in candidate_files if path.exists()), candidate_files[0])
        if not resolved_file.exists():
            return test_id

        if test_func and self._function_exists(resolved_file, test_func):
            return f"{test_file}::{test_func}"

        fallback_func = self._first_test_function_name(resolved_file)
        if fallback_func:
            return f"{test_file if candidate_files[0].exists() else 'tests/test_flaky.py'}::{fallback_func}"

        return test_id

    @staticmethod
    def _function_exists(path: Path, function_name: str) -> bool:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            return re.search(rf"^def\s+{re.escape(function_name)}\s*\(", source, flags=re.MULTILINE) is not None
        except Exception:
            return False

    @staticmethod
    def _first_test_function_name(path: Path) -> Optional[str]:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        match = re.search(r"^def\s+(test_[A-Za-z0-9_]+)\s*\(", source, flags=re.MULTILINE)
        return match.group(1) if match else None

    def reset(self) -> FlakeForgeObservation:
        self._restore_clean_repo()
        self._episode = EpisodeState(
            episode_id=str(uuid4()),
            max_steps=self.max_steps,
            test_identifier=self.test_id,
        )
        self.causal_trace = EpisodeCausalTrace()

        baseline_runs = self._run_test_n_times(n=5)
        self._episode.run_history = baseline_runs[-10:]
        self._episode.baseline_pass_rate = self._pass_rate(baseline_runs)
        self._episode.current_pass_rate = self._episode.baseline_pass_rate
        # Improvement 4: compute duration fingerprint from baseline runs.
        self._episode.duration_fingerprint = compute_duration_fingerprint(baseline_runs)
        # Capture baseline regression status BEFORE any patches are applied.
        # This lets step() distinguish agent-introduced regressions from
        # pre-existing failures (e.g. test_flaky_simple's 30% random failure rate).
        self._episode.baseline_regression_status = self.runner.check_regressions(
            exclude_test_id=self.test_id, timeout_seconds=30
        )


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
            prediction_error_history=[],
        )
        self.trace_logger.start_episode(
            episode_id=self._episode.episode_id,
            test_identifier=self.test_id,
            max_steps=self.max_steps,
            baseline_pass_rate=self._episode.baseline_pass_rate,
        )
        return self._build_initial_observation()

    def step(self, action: FlakeForgeAction) -> FlakeForgeObservation:  # type: ignore[override]
        if self._episode.done:
            obs = self._build_observation(reward=0.0, done=True)
            obs.metadata = {"reason": "episode_already_done"}
            return obs

        self._update_hypothesis_from_action(action)
        canonical_action_var = canonical_action(action.action_type)

        if canonical_action_var not in {"GATHER_EVIDENCE", "REVERT_LAST_PATCH", "ADD_RETRY"}:
            if self._episode.current_hypothesis is None:
                obs = self._build_observation(reward=-1.0, done=False)
                obs.metadata = {"reason": "action_requires_hypothesis"}
                self.trace_logger.log_error({"step": self._episode.step_count + 1, "error": "action_requires_hypothesis", "action": action.action_type})
                return obs
            if self._episode.current_hypothesis.confidence < 0.3:
                obs = self._build_observation(reward=-1.0, done=False)
                obs.metadata = {"reason": "hypothesis_confidence_too_low"}
                self.trace_logger.log_error({"step": self._episode.step_count + 1, "error": "hypothesis_confidence_too_low", "action": action.action_type})
                return obs

        self._episode.step_count += 1
        self._state.step_count = self._episode.step_count
        self._episode.actions_taken.append(action.action_type)
        current_conf = self._episode.current_hypothesis.confidence if self._episode.current_hypothesis else 0.0
        self._episode.hypothesis_confidence_at_each_step.append(current_conf)
        self._episode.hypothesis_history.append({"step": self._episode.step_count, "confidence": current_conf})

        try:
            execution = self._execute_action(action)
            if execution.get("no_op"):
                obs = self._build_observation(reward=-0.5, done=False)
                obs.metadata = execution
                self.trace_logger.log_error({"step": self._episode.step_count, "error": "no_op_action", "details": execution})
                return obs
        except Exception as exc:
            obs = self._build_observation(reward=-1.0, done=False)
            obs.metadata = {"action": action.action_type, "error": str(exc), "no_op": True}
            self.trace_logger.log_error({"step": self._episode.step_count, "error": "action_execution_exception", "action": action.action_type, "details": str(exc)})
            return obs

        import logging as _logging
        _log = _logging.getLogger(__name__).info

        _log("[ENV.step] start action=%s step=%d", action.action_type, self._episode.step_count + 1)
        post_runs = self._run_test_n_times(n=5)
        _log("[ENV.step] tests done: pass_rate=%.2f", self._pass_rate(post_runs))
        self._episode.run_history.extend(post_runs)
        self._episode.run_history = self._episode.run_history[-10:]
        self._episode.current_pass_rate = self._pass_rate(post_runs)
        self._episode.regression_detected = self.runner.check_regressions(
            exclude_test_id=self.test_id, timeout_seconds=30
        )
        # Guard against false regressions: if the non-target tests were already
        # failing before this episode (tracked in baseline_regression_status),
        # do not count that as a new regression introduced by the agent.
        if self._episode.regression_detected and getattr(self._episode, "baseline_regression_status", False):
            self._episode.regression_detected = False


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

        # ── RLVR Hybrid: judge_feedback on action is accepted but intentionally
        # ignored server-side — the inline Judge LLM has been removed.
        failure_pattern = get_failure_pattern(post_runs)
        repeat_action_count = _repeat_tail_count([canonical_action(a) for a in self._episode.actions_taken])
        step_result = {
            "current_pass_rate": self._episode.current_pass_rate,
            "regression_detected": self._episode.regression_detected,
            "action_taken": canonical_action_var,
            "action_parameters": action.parameters or {},
            "done": done,
            "timed_out": self._episode.step_count >= self.max_steps and self._episode.current_pass_rate < 0.9,
            "ast_diff": execution.get("ast_diff", {}),
            "lines_changed": int(execution.get("lines_changed", 0)),
            "repeat_action_count": repeat_action_count,
            # ── V2 additions ──
            "chaos_pass_rate": self._episode.chaos_pass_rate,
            "chaos_baseline_pass_rate": self._episode.chaos_baseline_pass_rate,
            "perf_regression_detected": self._episode.perf_regression_detected,
            "perf_median_ratio": self._episode.perf_median_ratio,
            # ── RLVR Hybrid ──
            "predicted_pass_rate_after": action.predicted_pass_rate_after,
        }
        # ── Accumulate CoT trajectory for Teacher Judge ───────────────────────
        self._episode.cot_trajectory.append({
            "step": self._episode.step_count,
            "action_type": action.action_type,
            "justification": action.justification or "",
            "reasoning_steps": list(
                (self._episode.current_hypothesis.reasoning_steps
                 if self._episode.current_hypothesis else [])
            ),
            "hypothesis_category": (
                self._episode.current_hypothesis.root_cause_category
                if self._episode.current_hypothesis else None
            ),
            "predicted_pass_rate_after": action.predicted_pass_rate_after,
            "actual_pass_rate": self._episode.current_pass_rate,
        })
        reward, reward_breakdown = compute_reward(
            self._episode, step_result, self._manifest_oracle
        )

        predicted = action.predicted_pass_rate_after
        prediction_error = (
            round(float(self._episode.current_pass_rate - predicted), 4)
            if predicted is not None
            else None
        )
        if prediction_error is not None:
            self._episode.prediction_error_history.append(prediction_error)
            self._state.prediction_error_history = self._episode.prediction_error_history[-10:]

        was_hypothesis_correct = bool(self._episode.current_pass_rate >= self._episode.baseline_pass_rate)
        reflection = {
            "prediction_error": prediction_error,
            "was_hypothesis_correct": was_hypothesis_correct,
            "what_learned": _learning_summary(canonical_action_var, prediction_error, was_hypothesis_correct),
            "updated_strategy": _updated_strategy(canonical_action_var, prediction_error, was_hypothesis_correct),
        }
        self._episode.reflection = reflection

        self._episode.last_outcomes.append(
            {
                "step": self._episode.step_count,
                "action": action.action_type,
                "pass_rate": self._episode.current_pass_rate,
                "reward": reward,
                "prediction_error": prediction_error,
            }
        )
        self._episode.last_outcomes = self._episode.last_outcomes[-3:]

        self._state.current_pass_rate = self._episode.current_pass_rate
        self._state.baseline_pass_rate = self._episode.baseline_pass_rate
        self._state.done = done
        self._state.regression_detected = self._episode.regression_detected

        obs = self._build_observation(reward=reward, done=done)
        obs.metadata = {
            **execution,
            "step_result": {
                **step_result,
                # Improvement 1: thread predicted_pass_rate_after into step_result
                # so reward.py can compute the prediction-error penalty.
                "predicted_pass_rate_after": (
                    action.predicted_pass_rate_after
                    if hasattr(action, "predicted_pass_rate_after")
                    else None
                ),
            },
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
            "reflection": reflection,
        }

        self._episode.failure_pattern_summary = {
            "most_common_error": failure_pattern.most_common_error,
            "error_distribution": failure_pattern.error_distribution,
            "flakiness_score": failure_pattern.flakiness_score,
            "duration_mean": failure_pattern.duration_mean,
            "duration_std": failure_pattern.duration_std,
        }

        if self._episode.current_hypothesis is not None:
            self.causal_trace.add_hypothesis(
                step=self._episode.step_count,
                hypothesis={
                    "root_cause_category": self._episode.current_hypothesis.root_cause_category,
                    "confidence": self._episode.current_hypothesis.confidence,
                    "evidence": list(self._episode.current_hypothesis.evidence),
                },
            )
        self.causal_trace.add_action(
            step=self._episode.step_count,
            action={
                "action_type": action.action_type,
                "justification": action.justification,
                "expected_outcome": action.expected_outcome,
                "predicted_pass_rate_after": action.predicted_pass_rate_after,
            },
        )
        if failure_pattern.most_common_error:
            self.causal_trace.add_symptom(failure_pattern.most_common_error)

        self.trace_logger.log_step(
            {
                "step": self._episode.step_count,
                "reasoning": {
                    "root_cause_category": (
                        self._episode.current_hypothesis.root_cause_category
                        if self._episode.current_hypothesis
                        else None
                    ),
                    "confidence": (
                        self._episode.current_hypothesis.confidence
                        if self._episode.current_hypothesis
                        else None
                    ),
                    "evidence": (
                        list(self._episode.current_hypothesis.evidence)
                        if self._episode.current_hypothesis
                        else []
                    ),
                    "reasoning_steps": (
                        list(self._episode.current_hypothesis.reasoning_steps)
                        if self._episode.current_hypothesis
                        else []
                    ),
                    "uncertainty": (
                        self._episode.current_hypothesis.uncertainty
                        if self._episode.current_hypothesis
                        else None
                    ),
                },
                "action": {
                    "action_type": action.action_type,
                    "parameters": action.parameters,
                    "justification": action.justification,
                    "predicted_pass_rate_after": action.predicted_pass_rate_after,
                    "expected_outcome": action.expected_outcome,
                    "risk_assessment": action.risk_assessment,
                    "fallback_plan": action.fallback_plan,
                },
                "execution": {
                    "pass_rate_before": self._episode.baseline_pass_rate,
                    "pass_rate_after": self._episode.current_pass_rate,
                    "runs": len(post_runs),
                    "failure_types": failure_pattern.error_distribution,
                },
                "reward_breakdown": reward_breakdown,
                "learning_signals": {
                    "prediction_error": prediction_error,
                    "was_fix_correct": was_hypothesis_correct,
                    "did_flakiness_reduce": self._episode.current_pass_rate >= self._episode.baseline_pass_rate,
                },
            }
        )

        if done:
            root_cause_identified = (
                self._episode.current_hypothesis.root_cause_category
                if self._episode.current_hypothesis
                else "unknown"
            )
            fix_summary = (
                f"{action.action_type} with params {json.dumps(action.parameters, ensure_ascii=True)}"
            )
            success = bool(self._episode.current_pass_rate >= 0.95 and not self._episode.regression_detected)
            self.causal_trace.finalize(
                final_cause=str(root_cause_identified),
                fix_applied=fix_summary,
                success=success,
            )
            efficiency_score = round(
                max(0.0, min(1.0, 1.0 - (self._episode.total_diff_lines / 100.0) - (self._episode.step_count / max(1, self.max_steps * 2)))),
                4,
            )
            # ── RLVR Hybrid: End-of-Episode Teacher Judge ─────────────────────
            # One LLM call with full trajectory + Oracle; updates reward in-place.
            teacher_reward_delta = self._finalize_episode()
            reward += teacher_reward_delta
            reward_breakdown["r_teacher_judge"] = float(teacher_reward_delta)
            reward_breakdown["total_reward"] = float(reward)
            obs.reward = reward  # propagate updated reward to observation

            self.trace_logger.set_summary(
                {
                    "baseline_pass_rate": self._episode.baseline_pass_rate,
                    "final_pass_rate": self._episode.current_pass_rate,
                    "steps_taken": self._episode.step_count,
                    "root_cause_identified": root_cause_identified,
                    "fix_summary": fix_summary,
                    "efficiency_score": efficiency_score,
                    "causal_trace": self.causal_trace.to_dict(),
                    "teacher_judge_score": self._episode.teacher_judge_score,
                    "teacher_judge_critique": self._episode.teacher_judge_critique,
                }
            )
            # Surface teacher judge info in obs metadata
            obs.metadata["teacher_judge"] = {
                "score": self._episode.teacher_judge_score,
                "critique": self._episode.teacher_judge_critique,
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

        # read_file_excerpt enforces a max 100-line window.
        test_src = read_file_excerpt(str(test_path), 1, 100) if test_path.exists() else ""
        src_under_test = read_file_excerpt(str(source_path), 1, 100) if source_path.exists() else ""

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
            # ── Improvements 4 & 5 ───────────────────────────────────────────
            duration_fingerprint=self._episode.duration_fingerprint,
            secondary_hypothesis=self._episode.secondary_hypothesis,
            last_actions=self._episode.actions_taken[-3:],
            last_outcomes=self._episode.last_outcomes[-3:],
            prediction_error_history=self._episode.prediction_error_history[-10:],
            failure_pattern_summary=self._episode.failure_pattern_summary,
            causal_hints=((self._episode.causal_graph_dict or {}).get("boundary_warnings", [])[:5]),
            reflection=self._episode.reflection,
        )

    def _execute_action(self, action: FlakeForgeAction) -> Dict[str, Any]:
        test_file = self.test_id.split("::", 1)[0]
        test_path = self.repo_path / test_file
        target_file = str(test_path)
        evidence = self._episode.current_hypothesis.evidence if self._episode.current_hypothesis else []
        resolved_target = resolve_target_from_evidence(target_file, evidence)

        canonical_action_val = canonical_action(action.action_type)

        if action.action_type == "detect_flakiness":
            probe_runs = self._run_test_n_times(n=10)
            self._episode.run_history.extend(probe_runs)
            self._episode.run_history = self._episode.run_history[-10:]
            self._episode.current_pass_rate = self._pass_rate(probe_runs)
            return {
                "action": action.action_type,
                "probe_runs": len(probe_runs),
                "distribution": {
                    "passed": sum(1 for r in probe_runs if r.passed),
                    "failed": sum(1 for r in probe_runs if not r.passed),
                },
                "lines_changed": 0,
            }

        if action.action_type == "analyze_logs":
            fp = get_failure_pattern(self._episode.run_history[-10:])
            return {
                "action": action.action_type,
                "failure_pattern": {
                    "pass_rate": fp.pass_rate,
                    "most_common_error": fp.most_common_error,
                    "error_distribution": fp.error_distribution,
                },
                "lines_changed": 0,
            }

        if action.action_type == "retry_test":
            retry_runs = self._run_test_n_times(n=5)
            self._episode.run_history.extend(retry_runs)
            self._episode.run_history = self._episode.run_history[-10:]
            self._episode.current_pass_rate = self._pass_rate(retry_runs)
            return {
                "action": action.action_type,
                "retry_runs": len(retry_runs),
                "distribution": {
                    "passed": sum(1 for r in retry_runs if r.passed),
                    "failed": sum(1 for r in retry_runs if not r.passed),
                },
                "lines_changed": 0,
            }

        if canonical_action_val == "GATHER_EVIDENCE":
            injection_points = injection_points_from_hypothesis(self._episode.current_hypothesis, resolved_target)
            patched_source = inject_logging(target_file, injection_points)
            Path(target_file).write_text(patched_source, encoding="utf-8")
            evidence_runs = self._run_test_n_times(n=5)
            self._episode.log_snippets.extend(extract_log_snippets(evidence_runs))
            self._episode.log_snippets = self._episode.log_snippets[-3:]
            self._episode.current_hypothesis = infer_hypothesis(self._episode, evidence_runs, self.test_id)
            subprocess.run(["git", "checkout", "--", test_file], cwd=self.repo_path, check=False, capture_output=True, text=True)
            return {
                "action": action.action_type,
                "evidence_runs": len(evidence_runs),
                "resolved_target": resolved_target,
            }

        if canonical_action_val == "REVERT_LAST_PATCH":
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

        if canonical_action_val == "DIAGNOSE_BOUNDARY":
            boundary_node = action.parameters.get("boundary_node", "")
            try:
                test_file_name = self.test_id.split("::", 1)[0]
                test_func = self.test_id.split("::", 1)[-1] if "::" in self.test_id else ""
                if test_func:
                    fresh_graph = self.causal_graph_builder.build(
                        str(self.repo_path / test_file_name), test_func
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

        if canonical_action_val == "CHAOS_PROBE":
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

        if canonical_action_val != "REVERT_LAST_PATCH" and not resolved_target.get("identifier"):
            return {"action": action.action_type, "no_op": True, "reason": "unable_to_ground_evidence"}

        patch_spec = build_patch_spec(action, resolved_target)
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



    def _update_hypothesis_from_action(self, action: FlakeForgeAction) -> None:
        if action.hypothesis is None:
            return
        try:
            self._episode.current_hypothesis = Hypothesis(**action.hypothesis.model_dump())
        except Exception:
            # Ignore malformed hypotheses and keep previous valid hypothesis.
            return

    # ── RLVR Hybrid helpers ───────────────────────────────────────────────────

    def _load_manifest(self) -> Dict[str, Any]:
        """Load flake_manifest.json from the repo root. Returns {} on failure."""
        manifest_path = self.repo_path / "flake_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {}

    def _finalize_episode(self) -> float:
        """Call Teacher Judge LLM once at episode end.

        Returns the reward delta from the teacher score so the caller can
        add it to the step reward before returning the final observation.
        """
        import asyncio as _asyncio
        import os as _os

        teacher_context = self._manifest_oracle.get("teacher_judge_context", {})
        # Build the prompt payload (no LLM if API key absent)
        api_key = (
            _os.environ.get("NVIDIA_API_KEY")
            or _os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()

        score: float = 0.0
        critique: str = ""

        if api_key:
            prompt_payload = json.dumps(
                {
                    "task": "grade_reasoning_trajectory",
                    "manifest_oracle": {
                        "flake_category": self._manifest_oracle.get("flake_category"),
                        "correct_actions": self._manifest_oracle.get("correct_actions"),
                        "correct_primitives": self._manifest_oracle.get("correct_primitives"),
                        "root_cause_file": self._manifest_oracle.get("root_cause_file"),
                        "root_cause_function": self._manifest_oracle.get("root_cause_function"),
                        "expected_reasoning_steps": teacher_context.get("expected_reasoning_steps", []),
                        "anti_patterns": teacher_context.get("anti_patterns", []),
                    },
                    "agent_trajectory": self._episode.cot_trajectory,
                    "final_pass_rate": self._episode.current_pass_rate,
                    "expected_pass_rate_after_fix": self._manifest_oracle.get("expected_pass_rate_after_fix"),
                    "agent_predicted_final_pass_rate": (
                        self._episode.cot_trajectory[-1].get("predicted_pass_rate_after")
                        if self._episode.cot_trajectory else None
                    ),
                },
                ensure_ascii=False,
            )
            try:
                # Run async call in a sync context.
                # Use get_running_loop() (Python 3.10+) which raises RuntimeError
                # when no loop is running — more reliable than get_event_loop().
                try:
                    running_loop = _asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None

                if running_loop is not None and running_loop.is_running():
                    # We're inside an event loop (e.g. FastAPI/uvicorn); spawn a new
                    # thread that can safely call asyncio.run() without conflict.
                    import concurrent.futures as _cf
                    with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(
                            _asyncio.run, self._call_teacher_judge(prompt_payload, api_key)
                        )
                        score, critique = future.result(timeout=90)
                else:
                    score, critique = _asyncio.run(
                        self._call_teacher_judge(prompt_payload, api_key)
                    )
            except Exception as exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "[TEACHER_JUDGE] Failed: %s", exc
                )

        self._episode.teacher_judge_score = score
        self._episode.teacher_judge_critique = critique

        # Compute reward delta using the same formula as reward.py
        r_teacher = 0.0
        if score > 0.0:
            r_teacher = (score / 10.0) * 4.0
            if score < 4.0:
                r_teacher -= (4.0 - score) * 0.5
        return r_teacher

    @staticmethod
    async def _call_teacher_judge(prompt: str, api_key: str) -> tuple:
        """Single async call to the Judge LLM.  Returns (score: float, critique: str)."""
        import logging as _logging
        from openai import AsyncOpenAI

        judge_model = os.getenv("JUDGE_MODEL", "minimaxai/minimax-m2.7").strip()
        judge_timeout = float(os.getenv("REQUEST_TIMEOUT_S", "60"))

        client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=judge_timeout,
        )
        system_msg = (
            "You are an Omniscient Teacher grading an AI agent's reasoning trajectory for debugging a flaky test. "
            "You have been given the GROUND TRUTH answer key (manifest_oracle). "
            "Score the agent's chain-of-thought on a scale of 0-10: "
            "10 = perfect causal reasoning that correctly identified the root cause step-by-step; "
            "0 = no reasoning or completely wrong diagnosis even if the correct action was stumbled upon. "
            'Reply ONLY with JSON: {"score": <0-10>, "critique": "<one actionable sentence>"}'
        )
        try:
            completion = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
                timeout=judge_timeout,
            )
            raw = completion.choices[0].message.content or ""
            _logging.getLogger(__name__).info("[TEACHER_JUDGE] raw=%s", raw[:200])
            # Extract JSON
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                parsed = json.loads(raw[start : end + 1])
                score = float(max(0.0, min(10.0, parsed.get("score", 0))))
                critique = str(parsed.get("critique", ""))[:400]
                return score, critique
        except Exception as exc:
            _logging.getLogger(__name__).warning("[TEACHER_JUDGE] API error: %s", exc)
        return 0.0, ""




def _repeat_tail_count(actions: List[str]) -> int:
    if not actions:
        return 0
    tail = actions[-1]
    count = 0
    for action in reversed(actions):
        if action == tail:
            count += 1
        else:
            break
    return count


def _learning_summary(action: str, prediction_error: Optional[float], was_hypothesis_correct: bool) -> str:
    if prediction_error is None:
        return "No explicit outcome prediction provided."
    if was_hypothesis_correct and abs(prediction_error) <= 0.1:
        return f"{action} behaved as expected with low prediction error."
    if not was_hypothesis_correct:
        return f"{action} did not improve stability; hypothesis needs revision."
    return f"{action} improved stability but calibration is off by {prediction_error:+.2f}."


def _updated_strategy(action: str, prediction_error: Optional[float], was_hypothesis_correct: bool) -> str:
    if not was_hypothesis_correct:
        return "Switch to evidence-focused probing before further code changes."
    if prediction_error is not None and abs(prediction_error) > 0.2:
        return "Reduce confidence in aggressive fixes and prefer minimal reversible actions."
    return f"Continue with targeted follow-up action after {action}."


# Backward compatible alias for template-generated class name.
FlakeforgeEnvironment = FlakeForgeEnvironment
