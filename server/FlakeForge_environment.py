"""V3 FlakeForge Environment — Unified step loop.

Key changes from V2:
- No hypothesis gating — agent outputs think+patch directly
- No judge calls — reward is fully deterministic
- Deep flakiness signals injected into observation
- Free-form patch application via search/replace
- Single-step flow: observe → generate → patch → run → reward
"""

from __future__ import annotations

import ast
import math
import os
import re
import runpy
import traceback
import uuid
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=SyntaxWarning)

from openenv.core.env_server.interfaces import Environment

try:
    from models import (
        FlakeForgeAction,
        FlakeForgeObservation,
        FlakeForgeState,
        RunRecord,
        PatchRecord,
        RewardBreakdown,
        failure_mode_entropy,
    )
    from server.state import EpisodeState
    from server.deep_flakiness import (
        build_deep_observation_signals,
        extract_failure_frontier,
    )
    from server.patch_applier import restore_repo_files, write_validated_sources
    from server.patch_validator import PatchValidator
    from server.reward import compute_verifiable_reward
    from server.oracle_engine import verify_structured_think
    from server.causal_graph import CrossRepoGraphBuilder
    try:
        from server.tools import build_agent_targeting_hints
    except ImportError:
        def build_agent_targeting_hints(**_kwargs: Any) -> List[str]:
            return []
    from server.docker_runner import DockerTestRunner
except ImportError:
    try:
        from ..models import (
            FlakeForgeAction,
            FlakeForgeObservation,
            FlakeForgeState,
            RunRecord,
            PatchRecord,
            RewardBreakdown,
            failure_mode_entropy,
        )
        from ..server.state import EpisodeState
        from ..server.deep_flakiness import (
            build_deep_observation_signals,
            extract_failure_frontier,
        )
        from ..server.patch_applier import restore_repo_files, write_validated_sources
        from ..server.patch_validator import PatchValidator
        from ..server.reward import compute_verifiable_reward
        from ..server.oracle_engine import verify_structured_think
        from ..server.causal_graph import CrossRepoGraphBuilder
        try:
            from ..server.tools import build_agent_targeting_hints
        except ImportError:
            def build_agent_targeting_hints(**_kwargs: Any) -> List[str]:
                return []
        from ..server.docker_runner import DockerTestRunner
    except (ImportError, ValueError):
        from FlakeForge.models import (
            FlakeForgeAction,
            FlakeForgeObservation,
            FlakeForgeState,
            RunRecord,
            PatchRecord,
            RewardBreakdown,
            failure_mode_entropy,
        )
        from FlakeForge.server.state import EpisodeState
        from FlakeForge.server.deep_flakiness import (
            build_deep_observation_signals,
            extract_failure_frontier,
        )
        from FlakeForge.server.patch_applier import restore_repo_files, write_validated_sources
        from FlakeForge.server.patch_validator import PatchValidator
        from FlakeForge.server.reward import compute_verifiable_reward
        from FlakeForge.server.oracle_engine import verify_structured_think
        from FlakeForge.server.causal_graph import CrossRepoGraphBuilder
        try:
            from FlakeForge.server.tools import build_agent_targeting_hints
        except ImportError:
            def build_agent_targeting_hints(**_kwargs: Any) -> List[str]:
                return []
        from FlakeForge.server.docker_runner import DockerTestRunner

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__)


_INFRA_ERROR_TYPES = {
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
    "IndentationError",
    "FileNotFoundError",
}


class FlakeForgeEnvironment(Environment[FlakeForgeAction, FlakeForgeObservation, FlakeForgeState]):
    """V3 RL environment for flaky test repair.

    Unified step loop:
    1. Build observation with deep flakiness signals
    2. Agent generates <think> + <patch> in one forward pass
    3. Apply search/replace patch atomically
    4. Run test suite to verify fix
    5. Compute 6-signal verifiable reward
    6. Return observation with updated signals
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        test_identifier: Optional[str] = None,
        max_steps: int = 8,
        num_runs: int = 10,
        runner: Optional[Any] = None,
        chaos_runner: Optional[Any] = None,
    ) -> None:
        default_repo = os.environ.get("FF_REPO_PATH", str(Path("test_repos") / "timing_race_minimal"))
        default_test = os.environ.get("FF_TEST_ID", "tests/test_flaky.py::test_fetch_should_complete")

        self.repo_path = Path(repo_path or default_repo)
        self.test_identifier = test_identifier or default_test
        self.max_steps = max_steps
        self.num_runs = num_runs
        # Default to local pytest runner if no runner provided
        self.runner = runner or DockerTestRunner(str(self.repo_path))
        self.chaos_runner = chaos_runner
        self._episode_state: Optional[EpisodeState] = None
        self._openenv_state: Optional[FlakeForgeState] = None
        # First successful reset() captures a full .py tree snapshot; every later reset() restores
        # it so consecutive episodes (and GRPO group rollouts that reset each time) do not stack patches.
        self._pristine_file_snapshots: Optional[Dict[str, str]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FlakeForgeObservation:
        """Initialize a new episode."""
        del seed

        # Allow remote clients to configure env at reset-time.
        if "repo_path" in kwargs and kwargs["repo_path"]:
            new_repo = Path(str(kwargs["repo_path"]))
            if self.repo_path.resolve() != new_repo.resolve():
                self._pristine_file_snapshots = None  # different tree — re-baseline on next line
            self.repo_path = new_repo
        if "test_identifier" in kwargs and kwargs["test_identifier"]:
            self.test_identifier = str(kwargs["test_identifier"])
        if "max_steps" in kwargs and kwargs["max_steps"] is not None:
            self.max_steps = int(kwargs["max_steps"])
        if "num_runs" in kwargs and kwargs["num_runs"] is not None:
            self.num_runs = int(kwargs["num_runs"])

        episode_id = episode_id or str(uuid.uuid4())[:8]
        logger.info("[ENV] RESET episode=%s test=%s", episode_id, self.test_identifier)

        if self._pristine_file_snapshots:
            try:
                restore_repo_files(self.repo_path, self._pristine_file_snapshots)
                logger.info(
                    "[ENV] Restored %d .py file(s) from pristine snapshot",
                    len(self._pristine_file_snapshots),
                )
            except Exception as exc:
                logger.warning("[ENV] Pristine restore failed (non-fatal): %s", exc)

        self._reset_demo_repo_if_present()

        # Read source files
        test_source, source_under_test = self._read_sources()
        file_tree = self._build_file_tree()

        # Three-stage gate: Sanity → Determinism → Flakiness.
        # skip_preflight=True reuses cached baseline from the previous reset
        # (used during GRPO rollouts where flakiness was already confirmed).
        skip_preflight = bool(kwargs.get("skip_preflight", False))
        if skip_preflight and self._episode_state is not None:
            baseline_runs = list(self._episode_state.run_history)
            baseline_pass_rate = self._episode_state.baseline_pass_rate
            baseline_entropy = self._episode_state.baseline_entropy
            preflight = {
                "runs": baseline_runs,
                "pass_rate": baseline_pass_rate,
                "failure_entropy": baseline_entropy,
                "env_type": self._episode_state.env_type or "flaky",
                "should_train": True,
                "summary": dict(self._episode_state.preflight_result)
                           if self._episode_state.preflight_result else {},
            }
        else:
            preflight = self._preflight_gate(
                quick_runs=int(kwargs.get("preflight_quick_runs", 5)),
                confirm_runs=int(kwargs.get("preflight_confirm_runs", 10)),
                drop_deterministic_bugs=bool(kwargs.get("drop_deterministic_bugs", True)),
            )
            baseline_runs = preflight["runs"]
            baseline_pass_rate = preflight["pass_rate"]
            baseline_entropy = preflight["failure_entropy"]

        # Extract failing stack trace
        failing_trace = ""
        last_error_type = None
        for r in baseline_runs:
            if not r.passed:
                failing_trace = r.stderr_excerpt or r.error_message or ""
                last_error_type = r.error_type
                break

        # Build deep flakiness signals (AST-based, <5ms)
        deep_signals = build_deep_observation_signals(self.repo_path)

        # Extract causal frontier from stack trace
        failure_frontier, call_chain, boundary_crossings = extract_failure_frontier(
            failing_trace, self.repo_path
        )

        # Check order dependency (reverse run)
        order_dep = self._check_order_dependency(baseline_pass_rate)

        # Check infrastructure sensitivity (chaos run)
        infra_sensitive = self._check_infrastructure_sensitivity(baseline_pass_rate)

        # Build causal graph
        causal_graph_data, causal_hints = self._build_causal_graph(test_source)

        # Build additional file-targeting hints from stack trace/imports/deep signals.
        targeting_hints = build_agent_targeting_hints(
            repo_path=str(self.repo_path),
            test_identifier=self.test_identifier,
            failing_stack_trace=failing_trace,
            source_under_test=source_under_test,
            causal_frontier=failure_frontier,
            deep_signals=deep_signals,
            max_hints=8,
        )
        merged_hints = list(dict.fromkeys([*causal_hints, *targeting_hints]))[:10]

        # Initialize state
        self._episode_state = EpisodeState(
            episode_id=episode_id,
            test_identifier=self.test_identifier,
            repo_path=str(self.repo_path),
            max_steps=self.max_steps,
            original_test_source=test_source,
            original_source_under_test=source_under_test,
            current_test_source=test_source,
            current_source_under_test=source_under_test,
            run_history=baseline_runs,
            baseline_pass_rate=baseline_pass_rate,
            current_pass_rate=baseline_pass_rate,
            baseline_entropy=baseline_entropy,
            env_type=preflight["env_type"],
            should_train=preflight["should_train"],
            preflight_result=preflight["summary"],
            failing_stack_trace=failing_trace,
            last_error_type=last_error_type,
            failure_frontier=failure_frontier,
            call_chain_to_frontier=call_chain,
            boundary_crossings=boundary_crossings,
            order_dependency_detected=order_dep,
            infrastructure_sensitive=infra_sensitive,
            causal_graph=causal_graph_data,
            causal_hints=merged_hints,
            file_tree=file_tree,
            **deep_signals,
        )

        observation = self._build_observation()
        observation.reward = 0.0
        observation.done = not preflight["should_train"]
        self._openenv_state = FlakeForgeState(
            episode_id=episode_id,
            step_count=0,
            done=not preflight["should_train"],
            current_pass_rate=baseline_pass_rate,
            baseline_pass_rate=baseline_pass_rate,
            env_type=preflight["env_type"],
            should_train=preflight["should_train"],
        )
        if not preflight["should_train"]:
            self._episode_state.done = True
            self._episode_state.last_done_reason = preflight["summary"]["reason"]
            observation.done = True
            observation.done_reason = self._episode_state.last_done_reason

        if self._pristine_file_snapshots is None:
            self._pristine_file_snapshots = dict(self._collect_sources())
            logger.info(
                "[ENV] Recorded pristine snapshot of %d .py file(s) for future resets",
                len(self._pristine_file_snapshots),
            )
        return observation

    def step(
        self,
        action: FlakeForgeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> FlakeForgeObservation:
        """Execute one step of the unified agent loop."""
        del timeout_s, kwargs
        if self._episode_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._episode_state.done and not self._episode_state.should_train:
            observation = self._build_observation()
            observation.done = True
            observation.done_reason = self._episode_state.last_done_reason or "preflight_rejected"
            return observation

        self._episode_state.step_count += 1
        pre_step_pass_rate = self._episode_state.current_pass_rate
        logger.info(
            "[ENV] STEP %d/%d category=%s",
            self._episode_state.step_count,
            self.max_steps,
            action.predicted_category,
        )

        # --- 1. Patch validation → apply (disk unchanged if invalid) ---
        # Snapshot sources *before* any write: oracle + PatchValidator simulation.
        pre_sources: Dict[str, str] = {}
        if action.patch_text.strip():
            pre_sources = self._collect_sources()

        patch_result: Dict[str, Any] = {
            "success": False,
            "error": "empty_patch",
            "files_modified": [],
            "lines_changed": 0,
            "diff": "",
            "rejected_by_validator": False,
            "validation_errors": [],
            "validation_warnings": [],
            "validation_score": None,
        }
        rollback_snapshots: Dict[str, str] = {}
        if action.patch_text.strip():
            validator = PatchValidator()
            validation = validator.validate(
                action.patch_text,
                repo_path=self.repo_path,
                pre_sources=pre_sources or None,
                claims=(
                    action.structured_think.claims
                    if action.structured_think is not None and action.structured_think.claims
                    else None
                ),
                default_target=self._resolve_default_target(),
                failure_frontier=self._episode_state.failure_frontier,
                call_chain=self._episode_state.call_chain_to_frontier,
            )
            rollback_snapshots = dict(
                validation.simulate_result.get("original_sources")
                or validation.simulate_result.get("rollback_snapshots")
                or {}
            )
            if not validation.is_valid:
                first_error = validation.errors[0] if validation.errors else "patch_validation_failed"
                patch_result = {
                    "success": False,
                    "error": first_error,
                    "validation_errors": list(validation.errors),
                    "validation_warnings": list(validation.warnings),
                    "validation_score": validation.score,
                    "files_modified": [],
                    "lines_changed": 0,
                    "diff": "",
                    "noop": False,
                    "protected_file": False,
                    "fuzzy_applied": False,
                    "rejected_by_validator": True,
                }
                logger.warning(
                    "[ENV] PATCH VALIDATION FAILED errors=%s warnings=%s",
                    validation.errors,
                    validation.warnings,
                )
            else:
                sim = validation.simulate_result
                try:
                    write_validated_sources(
                        self.repo_path,
                        dict(sim.get("modified_sources") or {}),
                    )
                    patch_result = {
                        "success": True,
                        "files_modified": list(sim.get("files_modified") or []),
                        "lines_changed": int(sim.get("lines_changed") or 0),
                        "hunks_applied": int(sim.get("hunks_applied") or 0),
                        "diff": sim.get("diff") or "",
                        "error": None,
                        "noop": bool(sim.get("noop", False)),
                        "protected_file": bool(sim.get("protected_file", False)),
                        "fuzzy_applied": bool(sim.get("fuzzy_applied", False)),
                    }
                except Exception as exc:
                    if rollback_snapshots:
                        restore_repo_files(self.repo_path, rollback_snapshots)
                    patch_result = {
                        "success": False,
                        "files_modified": [],
                        "lines_changed": 0,
                        "hunks_applied": 0,
                        "diff": "",
                        "error": f"validated_write_failed: {exc}",
                        "noop": False,
                        "protected_file": False,
                        "fuzzy_applied": False,
                        "rolled_back": True,
                    }
                patch_result["validation_errors"] = []
                patch_result["validation_warnings"] = list(validation.warnings)
                patch_result["validation_score"] = validation.score
                patch_result["rejected_by_validator"] = False
                logger.info(
                    "[ENV] PATCH success=%s files=%s lines=%d error=%s validation_score=%s",
                    patch_result["success"],
                    patch_result.get("files_modified", []),
                    patch_result.get("lines_changed", 0),
                    patch_result.get("error"),
                    validation.score,
                )

        # --- 2. Syntax check (sanity after apply); rollback if broken ---
        syntax_error = None
        if patch_result["success"]:
            syntax_error = self._check_syntax(patch_result.get("files_modified", []))
            patch_result["syntax_error"] = syntax_error
            if syntax_error:
                logger.warning("[ENV] SYNTAX ERROR after apply (rolling back): %s", syntax_error)
                if rollback_snapshots:
                    try:
                        restore_repo_files(self.repo_path, rollback_snapshots)
                    except Exception as exc:
                        logger.error("[ENV] Rollback failed: %s", exc)
                patch_result["success"] = False
                patch_result["error"] = "syntax_error_after_apply"
                patch_result["rolled_back"] = True

        # --- 3. Run tests ---
        post_runs: List[RunRecord] = []
        post_run_dicts: List[Dict[str, Any]] = []
        if patch_result["success"] and not syntax_error:
            post_runs = self._run_tests(self.num_runs)
            post_run_dicts = [
                {"passed": r.passed, "error_type": r.error_type, "duration_ms": r.duration_ms}
                for r in post_runs
            ]
            post_pass_rate = sum(1 for r in post_runs if r.passed) / max(len(post_runs), 1)
        else:
            post_pass_rate = self._episode_state.current_pass_rate

        # --- 4. Regression check ---
        regression_detected = post_pass_rate < self._episode_state.baseline_pass_rate - 0.1
        if patch_result["success"] and not syntax_error and self.runner is not None:
            try:
                if hasattr(self.runner, "check_regressions"):
                    regression_detected = regression_detected or bool(
                        self.runner.check_regressions(self.test_identifier)
                    )
            except Exception as exc:
                logger.debug("[ENV] Regression check failed: %s", exc)
        patch_result["regression_detected"] = regression_detected

        # --- 4b. Oracle (needs post-patch disk; run before optional performance rollback) ---
        oracle_score: Optional[float] = None
        if action.structured_think is not None and action.structured_think.claims:
            post_sources = self._collect_sources()
            try:
                annotated_think, oracle_score = verify_structured_think(
                    action.structured_think,
                    pre_sources=pre_sources,
                    post_sources=post_sources,
                    patch_hunks=(
                        action.structured_patch.hunks
                        if action.structured_patch is not None
                        else ()
                    ),
                )
                action = action.model_copy(update={"structured_think": annotated_think})
                logger.info("[ENV] ORACLE score=%.3f claims=%d", oracle_score, len(annotated_think.claims))
            except Exception as exc:
                logger.warning("[ENV] Oracle verification failed (non-fatal): %s", exc)

        # --- 4c. Tentative patch gate: rollback if regression or no meaningful improvement ---
        outcome_pass_rate_for_learning = post_pass_rate
        rolled_back_due_to_performance = False
        effective_run_extend: List[RunRecord] = []
        self._episode_state.last_environment_note = ""
        had_valid_apply_and_tests = bool(
            patch_result.get("success") and not syntax_error and bool(post_runs)
        )

        if had_valid_apply_and_tests and rollback_snapshots:
            MIN_MEANINGFUL_IMPROVEMENT = 0.02
            solved = post_pass_rate >= 1.0 - 1e-9
            gain = post_pass_rate - pre_step_pass_rate
            insufficient_gain = (not solved) and (gain < MIN_MEANINGFUL_IMPROVEMENT)
            if regression_detected or insufficient_gain:
                rolled_back_due_to_performance = True
                outcome_pass_rate_for_learning = post_pass_rate
                try:
                    restore_repo_files(self.repo_path, rollback_snapshots)
                except Exception as exc:
                    logger.error("[ENV] Performance rollback restore failed: %s", exc)
                patch_result["rolled_back_due_to_performance"] = True
                patch_result["rolled_back"] = True
                patch_result["success"] = False
                patch_result["error"] = "rolled_back_due_to_regression_or_no_improvement"
                patch_result["tentative_pass_rate"] = post_pass_rate
                patch_result["pass_rate_before_step"] = pre_step_pass_rate
                post_pass_rate = pre_step_pass_rate
                effective_run_extend = []
                if regression_detected:
                    detail = (
                        f"regression or worse vs baseline (tentative pass rate "
                        f"{outcome_pass_rate_for_learning:.2f}, baseline "
                        f"{self._episode_state.baseline_pass_rate:.2f})"
                    )
                else:
                    detail = (
                        f"no meaningful improvement (tentative {outcome_pass_rate_for_learning:.2f} vs "
                        f"{pre_step_pass_rate:.2f} before patch; require +{MIN_MEANINGFUL_IMPROVEMENT:.2f} "
                        f"gain unless all test runs pass)"
                    )
                self._episode_state.last_environment_note = (
                    "The environment reverted your last patch — "
                    + detail
                    + ". Try a different root-cause category or code location. "
                    "Do not try to undo changes yourself; the workspace was restored automatically."
                )
                logger.warning("[ENV] PERFORMANCE ROLLBACK: %s", detail)
            else:
                effective_run_extend = list(post_runs)
        elif patch_result.get("success") and not syntax_error:
            effective_run_extend = list(post_runs)

        committed_successfully = had_valid_apply_and_tests and not rolled_back_due_to_performance

        # --- 5. Compute reward ---
        pre_entropy = failure_mode_entropy(self._episode_state.run_history[-self.num_runs:])
        observation = self._build_observation()

        reward_breakdown = compute_verifiable_reward(
            action=action,
            observation=observation,
            patch_result=patch_result,
            post_run_results=post_run_dicts,
            baseline_pass_rate=self._episode_state.baseline_pass_rate,
            pre_entropy=pre_entropy,
            oracle_score=oracle_score,
            regression_detected=regression_detected,
            think_history=self._episode_state.step_think_history,
        )

        # --- 6. Update state ---
        self._episode_state.current_pass_rate = post_pass_rate
        self._episode_state.run_history.extend(effective_run_extend)
        self._episode_state.last_think_text = action.think_text
        self._episode_state.last_patch_text = action.patch_text
        self._episode_state.last_reward = reward_breakdown.total_reward
        self._episode_state.last_reward_breakdown = reward_breakdown.to_dict()
        self._episode_state.last_patch_result = patch_result

        think_summary = self._build_think_summary(
            action=action,
            oracle_score=oracle_score,
            pass_rate_after=outcome_pass_rate_for_learning,
            reward=reward_breakdown.total_reward,
        )
        if rolled_back_due_to_performance:
            think_summary["patch_reverted_by_env"] = True
            think_summary["tentative_pass_rate"] = outcome_pass_rate_for_learning
        self._episode_state.step_think_history.append(think_summary)

        if committed_successfully:
            self._episode_state.patches_applied.append(PatchRecord(
                patch_text=action.patch_text,
                target_files=patch_result.get("files_modified", []),
                lines_changed=patch_result.get("lines_changed", 0),
                pass_rate_after=outcome_pass_rate_for_learning,
                applied_successfully=True,
            ))
            self._episode_state.total_diff_lines += patch_result.get("lines_changed", 0)

        # Check for regression
        if regression_detected:
            self._episode_state.regression_detected = True

        # Re-read modified sources
        self._episode_state.current_test_source, self._episode_state.current_source_under_test = self._read_sources()

        # Determine if episode is terminal
        done = (
            post_pass_rate >= 1.0  # Full stability achieved
            or self._episode_state.step_count >= self.max_steps
        )
        self._episode_state.done = done
        self._episode_state.last_done_reason = self._done_reason(post_pass_rate, done)

        # Build final observation with updated signals
        final_observation = self._build_observation()
        final_observation.reward = reward_breakdown.total_reward
        final_observation.done = done
        final_observation.patch_result = patch_result
        final_observation.done_reason = self._episode_state.last_done_reason
        self._openenv_state = FlakeForgeState(
            episode_id=self._episode_state.episode_id,
            step_count=self._episode_state.step_count,
            done=done,
            current_pass_rate=post_pass_rate,
            baseline_pass_rate=self._episode_state.baseline_pass_rate,
            regression_detected=self._episode_state.regression_detected,
            env_type=self._episode_state.env_type,
            should_train=self._episode_state.should_train,
        )

        logger.info(
            "[ENV] REWARD total=%.4f breakdown=%s done=%s pass_rate=%.2f->%.2f",
            reward_breakdown.total_reward,
            {k: round(v, 3) for k, v in reward_breakdown.to_dict().items()},
            done,
            self._episode_state.baseline_pass_rate,
            post_pass_rate,
        )

        return final_observation

    def _build_observation(self) -> FlakeForgeObservation:
        """Build a V3 observation from current state."""
        if self._episode_state is None:
            raise RuntimeError("State not initialized")

        # Duration fingerprint
        durations = [r.duration_ms for r in self._episode_state.run_history[-self.num_runs:]]
        dur_mean = sum(durations) / max(len(durations), 1) if durations else 0
        dur_std = (
            math.sqrt(sum((d - dur_mean) ** 2 for d in durations) / max(len(durations), 1))
            if durations else 0
        )

        return FlakeForgeObservation(
            episode_id=self._episode_state.episode_id,
            test_identifier=self._episode_state.test_identifier,
            step=self._episode_state.step_count,
            steps_remaining=self._episode_state.steps_remaining,
            repo_root=str(self._episode_state.repo_path),
            test_function_source=self._episode_state.current_test_source,
            source_under_test=self._episode_state.current_source_under_test,
            relevant_imports=self._extract_imports(self._episode_state.current_test_source),
            file_tree=self._episode_state.file_tree,
            run_history=self._episode_state.run_history[-20:],
            current_pass_rate=self._episode_state.current_pass_rate,
            baseline_pass_rate=self._episode_state.baseline_pass_rate,
            env_type=self._episode_state.env_type,
            should_train=self._episode_state.should_train,
            preflight_result=dict(self._episode_state.preflight_result),
            patches_applied=self._episode_state.patches_applied,
            total_diff_lines=self._episode_state.total_diff_lines,
            # V3 deep signals
            module_cache_violations=self._episode_state.module_cache_violations,
            fixture_scope_risks=self._episode_state.fixture_scope_risks,
            mock_residue_sites=self._episode_state.mock_residue_sites,
            import_side_effect_files=self._episode_state.import_side_effect_files,
            async_contamination_alive=self._episode_state.async_contamination_alive,
            # Causal frontier
            failure_frontier=self._episode_state.failure_frontier,
            call_chain_to_frontier=self._episode_state.call_chain_to_frontier,
            boundary_crossings=self._episode_state.boundary_crossings,
            # iDFlakies
            order_dependency_detected=self._episode_state.order_dependency_detected,
            infrastructure_sensitive=self._episode_state.infrastructure_sensitive,
            # Causal graph
            causal_graph=self._episode_state.causal_graph,
            causal_hints=self._episode_state.causal_hints,
            # Failure analysis
            failing_stack_trace=self._episode_state.failing_stack_trace,
            duration_fingerprint={"mean": dur_mean, "std": dur_std},
            # Episode context
            last_think_text=self._episode_state.last_think_text,
            last_patch_text=self._episode_state.last_patch_text,
            last_reward=self._episode_state.last_reward,
            reward_breakdown=self._episode_state.last_reward_breakdown,
            patch_result=self._episode_state.last_patch_result,
            done_reason=self._episode_state.last_done_reason,
            reward=self._episode_state.last_reward,
            think_history=list(self._episode_state.step_think_history),
            last_environment_note=self._episode_state.last_environment_note,
        )

    def _build_think_summary(
        self,
        action: FlakeForgeAction,
        oracle_score: Optional[float],
        pass_rate_after: float,
        reward: float,
    ) -> Dict[str, Any]:
        """Build a compact think summary dict for history tracking."""
        categories: List[str] = []
        entities: List[str] = []
        reason_signatures: List[str] = []

        if action.structured_think and action.structured_think.claims:
            for claim in action.structured_think.claims:
                categories.append(claim.category)
                if claim.entity:
                    entities.append(claim.entity)
                if claim.reason:
                    reason_signatures.append(claim.reason[:35].lower().strip())
        elif action.predicted_category:
            categories = [action.predicted_category]

        return {
            "step": self._episode_state.step_count,
            "categories": categories,
            "entities": entities,
            "reason_signatures": reason_signatures,
            "oracle_score": round(oracle_score, 3) if oracle_score is not None else None,
            "pass_rate_after": round(pass_rate_after, 3),
            "reward": round(reward, 4),
        }

    def _preflight_gate(
        self,
        *,
        quick_runs: int = 10,
        confirm_runs: int = 20,
        drop_deterministic_bugs: bool = True,
    ) -> Dict[str, Any]:
        """Classify environment before training: sanity → determinism → flakiness.

        Pass rate alone is not enough: 0/N can be a deterministic bug, infra
        breakage, or a hard flaky case where success was not observed yet.
        This gate separates those cases using error consistency/entropy.
        """
        quick_runs = max(1, quick_runs)
        confirm_runs = max(1, confirm_runs)

        sanity = self._run_tests(1)
        runs: List[RunRecord] = list(sanity)
        sanity_record = sanity[0] if sanity else RunRecord(
            passed=False,
            duration_ms=0,
            error_type="RunnerError",
            error_message="runner returned no result",
        )

        if self._is_infra_failure(sanity_record):
            return self._preflight_result(
                runs=runs,
                env_type="infra_broken",
                should_train=False,
                reason="preflight_infra_broken",
                stage="sanity",
                quick_runs=quick_runs,
                confirm_runs=confirm_runs,
            )

        # Stage 2: cheap deterministic baseline. We already ran one sanity pass.
        remaining_quick = max(quick_runs - len(runs), 0)
        if remaining_quick:
            runs.extend(self._run_tests(remaining_quick))

        quick_passes = sum(1 for r in runs[:quick_runs] if r.passed)
        if quick_passes == quick_runs:
            return self._preflight_result(
                runs=runs[:quick_runs],
                env_type="stable",
                should_train=False,
                reason="preflight_stable_pass",
                stage="determinism",
                quick_runs=quick_runs,
                confirm_runs=confirm_runs,
            )
        if 0 < quick_passes < quick_runs:
            return self._preflight_result(
                runs=runs[:quick_runs],
                env_type="flaky",
                should_train=True,
                reason="preflight_mixed_pass_fail",
                stage="determinism",
                quick_runs=quick_runs,
                confirm_runs=confirm_runs,
            )

        # Stage 3: 0/N quick passes. Do not drop yet; confirm failure type.
        runs.extend(self._run_tests(confirm_runs))
        confirm_window = runs[quick_runs:quick_runs + confirm_runs]
        confirm_passes = sum(1 for r in confirm_window if r.passed)
        if confirm_passes > 0:
            return self._preflight_result(
                runs=runs,
                env_type="flaky",
                should_train=True,
                reason="preflight_late_success_after_zero_quick_passes",
                stage="flakiness_confirm",
                quick_runs=quick_runs,
                confirm_runs=confirm_runs,
            )

        failure_keys = self._failure_keys(runs)
        unique_failures = set(failure_keys)
        if len(unique_failures) <= 1:
            return self._preflight_result(
                runs=runs,
                env_type="deterministic_bug",
                should_train=not drop_deterministic_bugs,
                reason=(
                    "preflight_deterministic_bug_dropped"
                    if drop_deterministic_bugs
                    else "preflight_deterministic_bug_labeled"
                ),
                stage="flakiness_confirm",
                quick_runs=quick_runs,
                confirm_runs=confirm_runs,
            )

        return self._preflight_result(
            runs=runs,
            env_type="deterministic_bug",
            should_train=not drop_deterministic_bugs,
            reason=(
                "preflight_deterministic_bug_dropped"
                if drop_deterministic_bugs
                else "preflight_deterministic_multi_error_labeled"
            ),
            stage="flakiness_confirm",
            quick_runs=quick_runs,
            confirm_runs=confirm_runs,
        )

    def _preflight_result(
        self,
        *,
        runs: List[RunRecord],
        env_type: str,
        should_train: bool,
        reason: str,
        stage: str,
        quick_runs: int,
        confirm_runs: int,
    ) -> Dict[str, Any]:
        pass_count = sum(1 for r in runs if r.passed)
        pass_rate = pass_count / max(len(runs), 1)
        failure_keys = self._failure_keys(runs)
        error_distribution = dict(Counter(failure_keys))
        unique_failure_types = len(error_distribution)
        entropy = failure_mode_entropy(runs)

        summary = {
            "env_type": env_type,
            "should_train": should_train,
            "reason": reason,
            "stage": stage,
            "runs": len(runs),
            "passes": pass_count,
            "pass_rate": round(pass_rate, 4),
            "quick_runs": quick_runs,
            "confirm_runs": confirm_runs,
            "unique_failure_types": unique_failure_types,
            "failure_entropy": entropy,
            "error_distribution": error_distribution,
        }
        logger.info("[ENV] PREFLIGHT %s", summary)
        return {
            "runs": runs,
            "pass_rate": pass_rate,
            "failure_entropy": entropy,
            "env_type": env_type,
            "should_train": should_train,
            "summary": summary,
        }

    _TIMING_RE = re.compile(r"\b\d+\.\d+s\b")

    def _failure_keys(self, runs: List[RunRecord]) -> List[str]:
        keys: List[str] = []
        for r in runs:
            if r.passed:
                continue
            error_type = r.error_type or "UnknownError"
            message = (r.error_message or r.stderr_excerpt or "").strip()
            msg_sig = message[:80] if message else ""
            # Strip wall-clock durations ("0.17s", "1.23s") from the signature
            # so that the same deterministic failure with slightly different
            # timing doesn't inflate failure-entropy into a false "flaky" verdict.
            msg_sig = self._TIMING_RE.sub("Xs", msg_sig)
            keys.append(f"{error_type}:{msg_sig}")
        return keys

    def _is_infra_failure(self, run: RunRecord) -> bool:
        if run.passed:
            return False
        error_type = run.error_type or ""
        message = f"{run.error_message or ''}\n{run.stderr_excerpt or ''}"
        if error_type in _INFRA_ERROR_TYPES:
            return True
        infra_needles = (
            "ImportError",
            "ModuleNotFoundError",
            "SyntaxError",
            "IndentationError",
            "ERROR collecting",
            "collected 0 items",
            "fixture",
            "pytest timed out",
        )
        return any(needle in message for needle in infra_needles)

    def _run_tests(self, n: int) -> List[RunRecord]:
        """Run the target test n times, collecting results."""
        if self.runner is None:
            logger.warning("[ENV] No runner configured — returning synthetic runs")
            return self._synthetic_runs(n)

        try:
            if hasattr(self.runner, "run_test_n_times"):
                batch = self.runner.run_test_n_times(self.test_identifier, n)
                if isinstance(batch, list) and batch:
                    return batch
        except Exception as exc:
            logger.debug("[ENV] run_test_n_times failed, falling back to loop: %s", exc)

        results: List[RunRecord] = []
        for _ in range(n):
            try:
                result = self.runner.run_test(self.test_identifier)
                results.append(result)
            except Exception as exc:
                results.append(RunRecord(
                    passed=False,
                    duration_ms=0,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:200],
                    stderr_excerpt=None,
                ))
        return results

    def _synthetic_runs(self, n: int) -> List[RunRecord]:
        """Generate synthetic run results for development/testing."""
        import random
        from server.docker_runner import RunRecord
        results = []
        for _ in range(n):
            passed = random.random() > 0.5
            results.append(RunRecord(
                passed=passed,
                duration_ms=random.randint(10, 500),
                error_type=None if passed else "TimeoutError",
                error_message=None if passed else "Operation timed out",
                stderr_excerpt=None,
            ))
        return results

    def _read_sources(self) -> Tuple[str, str]:
        """Read test and source-under-test files."""
        test_source = ""
        source_under_test = ""

        # Find test file
        test_parts = self.test_identifier.split("::")
        test_file_hint = test_parts[0] if test_parts else ""

        if test_file_hint:
            test_path = self.repo_path / test_file_hint
            if test_path.exists():
                try:
                    test_source = test_path.read_text(encoding="utf-8", errors="ignore")[:8000]
                except Exception:
                    pass

        # Try to find source under test from imports
        if test_source:
            for candidate in self._source_candidates_from_test(test_source, test_file_hint):
                if candidate.exists() and candidate.is_file():
                    try:
                        source_under_test = candidate.read_text(encoding="utf-8", errors="ignore")[:8000]
                        break
                    except Exception:
                        pass

        return test_source, source_under_test

    def _source_candidates_from_test(self, test_source: str, test_file_hint: str) -> List[Path]:
        """Resolve likely source files from imports in the target test."""
        candidates: List[Path] = []
        test_dir = (self.repo_path / test_file_hint).parent if test_file_hint else self.repo_path

        def add_module_candidates(module: str) -> None:
            if not module:
                return
            parts = module.replace(".", "/")
            candidates.extend([
                self.repo_path / f"{parts}.py",
                self.repo_path / parts / "__init__.py",
                self.repo_path / "src" / f"{parts}.py",
                self.repo_path / "src" / parts / "__init__.py",
            ])

        try:
            tree = ast.parse(test_source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        add_module_candidates(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    base_module = node.module or ""
                    if node.level:
                        base_dir = test_dir
                        for _ in range(max(node.level - 1, 0)):
                            base_dir = base_dir.parent
                        if base_module:
                            candidates.extend([
                                base_dir / f"{base_module.replace('.', '/')}.py",
                                base_dir / base_module.replace(".", "/") / "__init__.py",
                            ])
                        for alias in node.names:
                            if alias.name != "*":
                                candidates.append(base_dir / f"{alias.name}.py")
                    else:
                        add_module_candidates(base_module)
                        for alias in node.names:
                            if alias.name != "*":
                                add_module_candidates(f"{base_module}.{alias.name}" if base_module else alias.name)
        except Exception:
            for imp in self._extract_imports(test_source):
                add_module_candidates(imp)

        seen = set()
        unique_candidates: List[Path] = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_candidates.append(candidate)
        return unique_candidates

    def _build_file_tree(self) -> List[str]:
        """Build a compact file tree of the repo."""
        tree: List[str] = []
        skip = {"__pycache__", ".git", "node_modules", "venv", ".venv", ".pytest_cache", ".tox"}
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in skip]
            rel = os.path.relpath(root, self.repo_path)
            depth = rel.count(os.sep)
            if depth > 3:
                continue
            for f in files:
                if f.endswith(".py"):
                    tree.append(os.path.join(rel, f).replace("\\", "/"))
        return sorted(tree)[:50]

    def _extract_imports(self, source: str) -> List[str]:
        """Extract import statements from source."""
        imports = []
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
        except Exception:
            pass
        return imports

    def _check_syntax(self, files: List[str]) -> Optional[str]:
        """Check that modified files have valid Python syntax."""
        for f in files:
            path = self.repo_path / f
            if not path.exists() or path.suffix != ".py":
                continue
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
                ast.parse(source)
            except SyntaxError as exc:
                return f"{path.name}:{exc.lineno}: {exc.msg}"
        return None

    def _check_order_dependency(self, baseline_rate: float) -> bool:
        """Check for order dependency by running in reverse order."""
        if self.runner is None:
            return False
        try:
            if hasattr(self.runner, "run_reversed"):
                result = self.runner.run_reversed(self.test_identifier)
                reverse_rate = result.get("pass_rate", baseline_rate)
                return abs(reverse_rate - baseline_rate) > 0.15
        except Exception:
            pass
        return False

    def _check_infrastructure_sensitivity(self, baseline_rate: float) -> bool:
        """Check if the test is sensitive to infrastructure pressure."""
        if self.chaos_runner is None:
            return False
        try:
            result = self.chaos_runner.run_single(self.test_identifier)
            chaos_rate = 1.0 if result.get("passed", False) else 0.0
            return abs(chaos_rate - baseline_rate) > 0.2
        except Exception:
            pass
        return False

    def _build_causal_graph(self, test_source: str) -> Tuple[Optional[Dict], List[str]]:
        """Build causal graph for the test."""
        del test_source
        try:
            test_file, _, test_func = self.test_identifier.partition("::")
            entry_file = str(self.repo_path / test_file)
            entry_function = test_func or ""

            if not entry_function:
                return None, []

            builder = CrossRepoGraphBuilder(str(self.repo_path), max_depth=3)
            graph = builder.build(entry_file=entry_file, entry_function=entry_function)
            graph_dict = graph.to_observation_dict()
            hints = list(graph_dict.get("boundary_warnings", []))[:5]
            return graph_dict, hints
        except Exception as exc:
            logger.debug("[ENV] Causal graph construction failed: %s", exc)
            return None, []

    def _resolve_default_target(self) -> str:
        """Resolve the default target file for patches."""
        test_parts = self.test_identifier.split("::")
        return test_parts[0] if test_parts else ""

    def _collect_sources(self) -> Dict[str, str]:
        """Collect current on-disk source texts keyed by path relative to repo_path.

        Used to feed the oracle engine with pre/post source snapshots.
        This is called *before* a patch is applied for pre-sources and *after*
        for post-sources, so it just reads the current disk state.
        """
        sources: Dict[str, str] = {}
        if self._episode_state is None:
            return sources
        skip = {"__pycache__", ".git", "node_modules", "venv", ".venv", ".pytest_cache"}
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in skip]
            for f in files:
                if not f.endswith(".py"):
                    continue
                full = Path(root) / f
                rel = str(full.relative_to(self.repo_path)).replace("\\", "/")
                try:
                    sources[rel] = full.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
        return sources

    def _done_reason(self, pass_rate: float, done: bool) -> str:
        if not done:
            return "in_progress"
        if pass_rate >= 1.0:
            return "fully_stable"
        if self._episode_state and self._episode_state.regression_detected:
            return "regression_detected"
        if self._episode_state and self._episode_state.step_count >= self.max_steps:
            return "max_steps_reached"
        return "unknown"

    def _reset_demo_repo_if_present(self) -> None:
        """Restore bundled demo repos before each episode when they provide a reset script."""
        if os.environ.get("FF_SKIP_DEMO_RESET", "0") == "1":
            return
        reset_script = self.repo_path / "reset_demo.py"
        if not reset_script.exists():
            return
        try:
            runpy.run_path(str(reset_script), run_name="__main__")
        except Exception as exc:
            logger.warning("[ENV] Demo reset script failed: %s", exc)

    @property
    def state(self) -> FlakeForgeState:
        if self._openenv_state is not None:
            return self._openenv_state
        if self._episode_state is not None:
            return FlakeForgeState(
                episode_id=self._episode_state.episode_id,
                step_count=self._episode_state.step_count,
                done=self._episode_state.done,
                current_pass_rate=self._episode_state.current_pass_rate,
                baseline_pass_rate=self._episode_state.baseline_pass_rate,
                regression_detected=self._episode_state.regression_detected,
                env_type=self._episode_state.env_type,
                should_train=self._episode_state.should_train,
            )
        return FlakeForgeState(
            episode_id="",
            step_count=0,
            done=False,
            current_pass_rate=0.0,
            baseline_pass_rate=0.0,
            regression_detected=False,
            env_type="unknown",
            should_train=True,
        )


# ── Factory for OpenEnv ──────────────────────────────────────────────────────

def create_flakeforge_environment(
    repo_path: str,
    test_identifier: str,
    **kwargs: Any,
) -> FlakeForgeEnvironment:
    """Create a FlakeForge V3 environment instance."""
    return FlakeForgeEnvironment(
        repo_path=repo_path,
        test_identifier=test_identifier,
        **kwargs,
    )