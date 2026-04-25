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
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.types import (
    Action,
    EnvInfo,
    Observation,
    State,
    StepOutput,
)

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
    from server.patch_applier import apply_search_replace_patch
    from server.reward import compute_verifiable_reward
    from server.causal_graph import CausalGraphBuilder
except ImportError:
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
    from ..server.patch_applier import apply_search_replace_patch
    from ..server.reward import compute_verifiable_reward
    from ..server.causal_graph import CausalGraphBuilder

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__)


class FlakeForgeEnvironment:
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
        repo_path: str,
        test_identifier: str,
        max_steps: int = 8,
        num_runs: int = 10,
        runner: Optional[Any] = None,
        chaos_runner: Optional[Any] = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.test_identifier = test_identifier
        self.max_steps = max_steps
        self.num_runs = num_runs
        self.runner = runner
        self.chaos_runner = chaos_runner
        self.state: Optional[EpisodeState] = None

    def reset(self) -> StepOutput:
        """Initialize a new episode."""
        episode_id = str(uuid.uuid4())[:8]
        logger.info("[ENV] RESET episode=%s test=%s", episode_id, self.test_identifier)

        # Read source files
        test_source, source_under_test = self._read_sources()
        file_tree = self._build_file_tree()

        # Run baseline tests
        baseline_runs = self._run_tests(self.num_runs)
        baseline_pass_rate = sum(1 for r in baseline_runs if r.passed) / max(len(baseline_runs), 1)
        baseline_entropy = failure_mode_entropy(baseline_runs)

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

        # Initialize state
        self.state = EpisodeState(
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
            failing_stack_trace=failing_trace,
            last_error_type=last_error_type,
            failure_frontier=failure_frontier,
            call_chain_to_frontier=call_chain,
            boundary_crossings=boundary_crossings,
            order_dependency_detected=order_dep,
            infrastructure_sensitive=infra_sensitive,
            causal_graph=causal_graph_data,
            causal_hints=causal_hints,
            file_tree=file_tree,
            **deep_signals,
        )

        observation = self._build_observation()

        return StepOutput(
            observation=observation,
            state=FlakeForgeState(
                episode_id=episode_id,
                step_count=0,
                done=False,
                current_pass_rate=baseline_pass_rate,
                baseline_pass_rate=baseline_pass_rate,
            ),
            reward=0.0,
            done=False,
            info={
                "episode_id": episode_id,
                "baseline_pass_rate": baseline_pass_rate,
                "baseline_entropy": baseline_entropy,
                "deep_signals": {k: len(v) if isinstance(v, list) else v for k, v in deep_signals.items()},
            },
        )

    def step(self, action: FlakeForgeAction) -> StepOutput:
        """Execute one step of the unified agent loop."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.state.step_count += 1
        logger.info(
            "[ENV] STEP %d/%d category=%s",
            self.state.step_count,
            self.max_steps,
            action.predicted_category,
        )

        # --- 1. Apply patch ---
        patch_result = {"success": False, "error": "empty_patch", "files_modified": [], "lines_changed": 0, "diff": ""}
        if action.patch_text.strip():
            patch_result = apply_search_replace_patch(
                repo_path=self.repo_path,
                patch_text=action.patch_text,
                default_target=self._resolve_default_target(),
            )
            logger.info(
                "[ENV] PATCH success=%s files=%s lines=%d error=%s",
                patch_result["success"],
                patch_result.get("files_modified", []),
                patch_result.get("lines_changed", 0),
                patch_result.get("error"),
            )

        # --- 2. Syntax check ---
        syntax_error = None
        if patch_result["success"]:
            syntax_error = self._check_syntax(patch_result.get("files_modified", []))
            if syntax_error:
                logger.warning("[ENV] SYNTAX ERROR: %s", syntax_error)

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
            post_pass_rate = self.state.current_pass_rate

        # --- 4. Compute reward ---
        pre_entropy = failure_mode_entropy(self.state.run_history[-self.num_runs:])
        observation = self._build_observation()  # Build before reward for causal proximity

        reward_breakdown = compute_verifiable_reward(
            action=action,
            observation=observation,
            patch_result=patch_result,
            post_run_results=post_run_dicts,
            baseline_pass_rate=self.state.baseline_pass_rate,
            pre_entropy=pre_entropy,
        )

        # --- 5. Update state ---
        self.state.current_pass_rate = post_pass_rate
        self.state.run_history.extend(post_runs)
        self.state.last_think_text = action.think_text
        self.state.last_patch_text = action.patch_text
        self.state.last_reward = reward_breakdown.total_reward
        self.state.last_reward_breakdown = reward_breakdown.to_dict()

        if patch_result["success"]:
            self.state.patches_applied.append(PatchRecord(
                patch_text=action.patch_text,
                target_files=patch_result.get("files_modified", []),
                lines_changed=patch_result.get("lines_changed", 0),
                pass_rate_after=post_pass_rate,
                applied_successfully=True,
            ))
            self.state.total_diff_lines += patch_result.get("lines_changed", 0)

        # Check for regression
        if post_pass_rate < self.state.baseline_pass_rate - 0.1:
            self.state.regression_detected = True

        # Re-read modified sources
        self.state.current_test_source, self.state.current_source_under_test = self._read_sources()

        # Determine if episode is terminal
        done = (
            post_pass_rate >= 1.0  # Full stability achieved
            or self.state.step_count >= self.max_steps
            or self.state.regression_detected
        )
        self.state.done = done

        # Build final observation with updated signals
        final_observation = self._build_observation()
        final_observation.reward = reward_breakdown.total_reward
        final_observation.done = done

        logger.info(
            "[ENV] REWARD total=%.4f breakdown=%s done=%s pass_rate=%.2f→%.2f",
            reward_breakdown.total_reward,
            {k: round(v, 3) for k, v in reward_breakdown.to_dict().items()},
            done,
            self.state.baseline_pass_rate,
            post_pass_rate,
        )

        return StepOutput(
            observation=final_observation,
            state=FlakeForgeState(
                episode_id=self.state.episode_id,
                step_count=self.state.step_count,
                done=done,
                current_pass_rate=post_pass_rate,
                baseline_pass_rate=self.state.baseline_pass_rate,
                regression_detected=self.state.regression_detected,
            ),
            reward=reward_breakdown.total_reward,
            done=done,
            info={
                "reward_breakdown": reward_breakdown.to_dict(),
                "patch_result": {
                    "success": patch_result["success"],
                    "error": patch_result.get("error"),
                    "files_modified": patch_result.get("files_modified", []),
                    "lines_changed": patch_result.get("lines_changed", 0),
                },
                "pass_rate_delta": round(post_pass_rate - self.state.baseline_pass_rate, 4),
                "step": self.state.step_count,
                "done_reason": self._done_reason(post_pass_rate, done),
            },
        )

    def _build_observation(self) -> FlakeForgeObservation:
        """Build a V3 observation from current state."""
        if self.state is None:
            raise RuntimeError("State not initialized")

        # Duration fingerprint
        durations = [r.duration_ms for r in self.state.run_history[-self.num_runs:]]
        dur_mean = sum(durations) / max(len(durations), 1) if durations else 0
        dur_std = (
            math.sqrt(sum((d - dur_mean) ** 2 for d in durations) / max(len(durations), 1))
            if durations else 0
        )

        return FlakeForgeObservation(
            episode_id=self.state.episode_id,
            test_identifier=self.state.test_identifier,
            step=self.state.step_count,
            steps_remaining=self.state.steps_remaining,
            test_function_source=self.state.current_test_source,
            source_under_test=self.state.current_source_under_test,
            relevant_imports=self._extract_imports(self.state.current_test_source),
            file_tree=self.state.file_tree,
            run_history=self.state.run_history[-20:],
            current_pass_rate=self.state.current_pass_rate,
            baseline_pass_rate=self.state.baseline_pass_rate,
            patches_applied=self.state.patches_applied,
            total_diff_lines=self.state.total_diff_lines,
            # V3 deep signals
            module_cache_violations=self.state.module_cache_violations,
            fixture_scope_risks=self.state.fixture_scope_risks,
            mock_residue_sites=self.state.mock_residue_sites,
            import_side_effect_files=self.state.import_side_effect_files,
            async_contamination_alive=self.state.async_contamination_alive,
            # Causal frontier
            failure_frontier=self.state.failure_frontier,
            call_chain_to_frontier=self.state.call_chain_to_frontier,
            boundary_crossings=self.state.boundary_crossings,
            # iDFlakies
            order_dependency_detected=self.state.order_dependency_detected,
            infrastructure_sensitive=self.state.infrastructure_sensitive,
            # Causal graph
            causal_graph=self.state.causal_graph,
            causal_hints=self.state.causal_hints,
            # Failure analysis
            failing_stack_trace=self.state.failing_stack_trace,
            duration_fingerprint={"mean": dur_mean, "std": dur_std},
            # Episode context
            last_think_text=self.state.last_think_text,
            last_patch_text=self.state.last_patch_text,
            last_reward=self.state.last_reward,
        )

    def _run_tests(self, n: int) -> List[RunRecord]:
        """Run the target test n times, collecting results."""
        if self.runner is None:
            logger.warning("[ENV] No runner configured — returning synthetic runs")
            return self._synthetic_runs(n)

        results: List[RunRecord] = []
        for _ in range(n):
            try:
                result = self.runner.run_single(self.test_identifier)
                results.append(RunRecord(
                    passed=result.get("passed", False),
                    duration_ms=result.get("duration_ms", 0),
                    error_type=result.get("error_type"),
                    error_message=result.get("error_message"),
                    stderr_excerpt=result.get("stderr", "")[:500],
                ))
            except Exception as exc:
                results.append(RunRecord(
                    passed=False,
                    duration_ms=0,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:200],
                ))
        return results

    def _synthetic_runs(self, n: int) -> List[RunRecord]:
        """Generate synthetic run results for development/testing."""
        import random
        results = []
        for _ in range(n):
            passed = random.random() > 0.5
            results.append(RunRecord(
                passed=passed,
                duration_ms=random.randint(10, 500),
                error_type=None if passed else "TimeoutError",
                error_message=None if passed else "Operation timed out",
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
            imports = self._extract_imports(test_source)
            for imp in imports:
                # Convert import to file path
                parts = imp.replace(".", "/")
                candidates = [
                    self.repo_path / f"{parts}.py",
                    self.repo_path / "src" / f"{parts}.py",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        try:
                            source_under_test = candidate.read_text(encoding="utf-8", errors="ignore")[:8000]
                            break
                        except Exception:
                            pass
                if source_under_test:
                    break

        return test_source, source_under_test

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
            path = Path(f)
            if not path.exists() or not path.suffix == ".py":
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
        try:
            builder = CausalGraphBuilder(str(self.repo_path))
            graph = builder.build(test_source, self.test_identifier)
            hints = builder.get_hints() if hasattr(builder, "get_hints") else []
            return graph, hints
        except Exception as exc:
            logger.debug("[ENV] Causal graph construction failed: %s", exc)
            return None, []

    def _resolve_default_target(self) -> str:
        """Resolve the default target file for patches."""
        test_parts = self.test_identifier.split("::")
        return test_parts[0] if test_parts else ""

    def _done_reason(self, pass_rate: float, done: bool) -> str:
        if not done:
            return "in_progress"
        if pass_rate >= 1.0:
            return "fully_stable"
        if self.state and self.state.regression_detected:
            return "regression_detected"
        if self.state and self.state.step_count >= self.max_steps:
            return "max_steps_reached"
        return "unknown"


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
