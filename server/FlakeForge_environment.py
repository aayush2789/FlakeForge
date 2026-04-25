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
    from server.patch_applier import apply_search_replace_patch
    from server.reward import compute_verifiable_reward
    from server.causal_graph import CrossRepoGraphBuilder
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
        from ..server.patch_applier import apply_search_replace_patch
        from ..server.reward import compute_verifiable_reward
        from ..server.causal_graph import CrossRepoGraphBuilder
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
        from FlakeForge.server.patch_applier import apply_search_replace_patch
        from FlakeForge.server.reward import compute_verifiable_reward
        from FlakeForge.server.causal_graph import CrossRepoGraphBuilder
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
        # Default to DockerTestRunner if no runner provided
        self.runner = runner or DockerTestRunner(str(self.repo_path))
        self.chaos_runner = chaos_runner
        self._episode_state: Optional[EpisodeState] = None
        self._openenv_state: Optional[FlakeForgeState] = None

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
            self.repo_path = Path(str(kwargs["repo_path"]))
        if "test_identifier" in kwargs and kwargs["test_identifier"]:
            self.test_identifier = str(kwargs["test_identifier"])
        if "max_steps" in kwargs and kwargs["max_steps"] is not None:
            self.max_steps = int(kwargs["max_steps"])
        if "num_runs" in kwargs and kwargs["num_runs"] is not None:
            self.num_runs = int(kwargs["num_runs"])

        episode_id = episode_id or str(uuid.uuid4())[:8]
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
        observation.reward = 0.0
        observation.done = False
        self._openenv_state = FlakeForgeState(
            episode_id=episode_id,
            step_count=0,
            done=False,
            current_pass_rate=baseline_pass_rate,
            baseline_pass_rate=baseline_pass_rate,
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

        self._episode_state.step_count += 1
        logger.info(
            "[ENV] STEP %d/%d category=%s",
            self._episode_state.step_count,
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
            post_pass_rate = self._episode_state.current_pass_rate

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

        # --- 4. Compute reward ---
        pre_entropy = failure_mode_entropy(self._episode_state.run_history[-self.num_runs:])
        observation = self._build_observation()  # Build before reward for causal proximity

        reward_breakdown = compute_verifiable_reward(
            action=action,
            observation=observation,
            patch_result=patch_result,
            post_run_results=post_run_dicts,
            baseline_pass_rate=self._episode_state.baseline_pass_rate,
            pre_entropy=pre_entropy,
            regression_detected=regression_detected,
        )

        # --- 5. Update state ---
        self._episode_state.current_pass_rate = post_pass_rate
        self._episode_state.run_history.extend(post_runs)
        self._episode_state.last_think_text = action.think_text
        self._episode_state.last_patch_text = action.patch_text
        self._episode_state.last_reward = reward_breakdown.total_reward
        self._episode_state.last_reward_breakdown = reward_breakdown.to_dict()
        self._episode_state.last_patch_result = patch_result

        if patch_result["success"]:
            self._episode_state.patches_applied.append(PatchRecord(
                patch_text=action.patch_text,
                target_files=patch_result.get("files_modified", []),
                lines_changed=patch_result.get("lines_changed", 0),
                pass_rate_after=post_pass_rate,
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
        )

        logger.info(
            "[ENV] REWARD total=%.4f breakdown=%s done=%s pass_rate=%.2f→%.2f",
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
            test_function_source=self._episode_state.current_test_source,
            source_under_test=self._episode_state.current_source_under_test,
            relevant_imports=self._extract_imports(self._episode_state.current_test_source),
            file_tree=self._episode_state.file_tree,
            run_history=self._episode_state.run_history[-20:],
            current_pass_rate=self._episode_state.current_pass_rate,
            baseline_pass_rate=self._episode_state.baseline_pass_rate,
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
        )

    def _run_tests(self, n: int) -> List[RunRecord]:
        """Run the target test n times, collecting results."""
        if self.runner is None:
            logger.warning("[ENV] No runner configured — returning synthetic runs")
            return self._synthetic_runs(n)

        results: List[RunRecord] = []
        for _ in range(n):
            try:
                # DockerTestRunner uses run_test, not run_single
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
            )
        return FlakeForgeState(
            episode_id="",
            step_count=0,
            done=False,
            current_pass_rate=0.0,
            baseline_pass_rate=0.0,
            regression_detected=False,
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