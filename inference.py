"""Inference loop — unified agent with verifiable reward."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
except Exception:
    EnvClient = None  # type: ignore[assignment]
    StepResult = None  # type: ignore[assignment]

try:
    from models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState
    from agent.tool_loop import ToolContext
    from agent.unified_agent import ToolAugmentedFlakeForgeAgent, UnifiedFlakeForgeAgent, build_unified_prompt
    from server.FlakeForge_environment import FlakeForgeEnvironment
except ImportError:
    from .models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState
    from .agent.tool_loop import ToolContext
    from .agent.unified_agent import ToolAugmentedFlakeForgeAgent, UnifiedFlakeForgeAgent, build_unified_prompt
    from .server.FlakeForge_environment import FlakeForgeEnvironment

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from .utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__, log_file=Path("outputs/inference.log"))

if load_dotenv is not None:
    load_dotenv()


if EnvClient is not None:
    class FlakeForgeEnvClient(EnvClient[FlakeForgeAction, FlakeForgeObservation, FlakeForgeState]):
        """Concrete OpenEnv client for FlakeForge's action/observation models."""

        def _step_payload(self, action: FlakeForgeAction) -> Dict[str, Any]:
            return action.model_dump()

        def _parse_result(self, payload: Dict[str, Any]) -> Any:
            observation_data = payload.get("observation", {})
            observation = FlakeForgeObservation.model_validate(observation_data)
            reward = payload.get("reward", getattr(observation, "reward", 0.0))
            done = payload.get("done", getattr(observation, "done", False))
            info = payload.get("info") or _info_from_observation(observation)

            if StepResult is not None:
                result = StepResult(
                    observation=observation,
                    reward=float(reward or 0.0),
                    done=bool(done),
                )
            else:
                result = {
                    "observation": observation,
                    "reward": float(reward or 0.0),
                    "done": bool(done),
                }

            state = FlakeForgeState(
                episode_id=str(getattr(observation, "episode_id", "")),
                step_count=int(getattr(observation, "step", 0)),
                done=bool(done),
                current_pass_rate=float(getattr(observation, "current_pass_rate", 0.0)),
                baseline_pass_rate=float(getattr(observation, "baseline_pass_rate", 0.0)),
            )
            setattr(result, "state", state)
            setattr(result, "info", info)
            return result

        def _parse_state(self, payload: Dict[str, Any]) -> FlakeForgeState:
            return FlakeForgeState.model_validate(payload)
else:
    FlakeForgeEnvClient = None  # type: ignore[assignment]


def _as_step_output_like(value: Any) -> Any:
    """Normalize env return values to a StepOutput-like object.

    Supports both:
    - OpenEnv client/server style return object with observation/reward/done/state/info
    - Direct observation returns from local environment implementations
    """
    if (
        hasattr(value, "observation")
        and hasattr(value, "done")
        and hasattr(value, "state")
        and hasattr(value, "info")
    ):
        if not getattr(value, "info", None):
            setattr(value, "info", _info_from_observation(value.observation))
        return value

    class _StepLike:
        def __init__(self, source: Any) -> None:
            observation = getattr(source, "observation", source)
            self.observation = observation
            self.reward = float(
                getattr(source, "reward", getattr(observation, "reward", 0.0)) or 0.0
            )
            self.done = bool(getattr(source, "done", getattr(observation, "done", False)))
            self.info = getattr(source, "info", {}) or _info_from_observation(observation)

            self.state = getattr(source, "state", None) or FlakeForgeState(
                episode_id=str(getattr(observation, "episode_id", "")),
                step_count=int(getattr(observation, "step", 0)),
                done=self.done,
                current_pass_rate=float(getattr(observation, "current_pass_rate", 0.0)),
                baseline_pass_rate=float(getattr(observation, "baseline_pass_rate", 0.0)),
                regression_detected=False,
            )

    return _StepLike(value)


def _info_from_observation(observation: Any) -> Dict[str, Any]:
    """Recover step metadata carried on plain OpenEnv observations."""
    info: Dict[str, Any] = {}
    patch_result = getattr(observation, "patch_result", None)
    if patch_result:
        info["patch_result"] = patch_result
    reward_breakdown = getattr(observation, "reward_breakdown", None)
    if reward_breakdown:
        info["reward_breakdown"] = reward_breakdown
    done_reason = getattr(observation, "done_reason", None)
    if done_reason:
        info["done_reason"] = done_reason
    return info


class LLMBackend:
    """LLM backend that calls OpenAI-compatible APIs."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.model_name = model_name or os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
        self.api_base = api_base or os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")
        
        self.max_tokens = int(max_tokens or os.environ.get("MAX_TOKENS", 4096))
        self.temperature = float(temperature or os.environ.get("TEMPERATURE", 0.2))

    def _openai_json_schema_response_format(self) -> Dict[str, Any]:
        """Strict response_format for OpenAI-compatible backends."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "flakeforge_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["think", "patch"],
                    "properties": {
                        "think": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["claims", "confidence"],
                            "properties": {
                                "claims": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": True,
                                        "required": [
                                            "category",
                                            "entity",
                                            "location",
                                            "polarity",
                                            "reason",
                                        ],
                                        "properties": {
                                            "category": {
                                                "type": "string",
                                                "enum": [
                                                    "async_wait",
                                                    "concurrency",
                                                    "test_order_dependency",
                                                    "resource_leak",
                                                    "shared_state",
                                                    "network",
                                                    "platform_dependency",
                                                    "nondeterminism",
                                                    "import_side_effect",
                                                    "module_cache_pollution",
                                                    "fixture_scope_leak",
                                                    "mock_residue",
                                                    "unknown",
                                                ],
                                            },
                                            "entity": {"type": "string"},
                                            "location": {"type": "string"},
                                            "polarity": {"type": "string", "enum": ["present", "absent"]},
                                            "reason": {"type": "string"},
                                        },
                                    },
                                },
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                        },
                        "patch": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["hunks"],
                            "properties": {
                                "hunks": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": True,
                                        "required": [
                                            "file",
                                            "search",
                                            "replace",
                                        ],
                                        "properties": {
                                            "file": {"type": "string"},
                                            "search": {"type": "string"},
                                            "replace": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

    def generate(self, prompt: str, *, system_prompt: str) -> str:
        """Generate a completion using either OpenAI or Ollama natively."""
        fallback_response = {
            "think": {
                "claims": [
                    {
                        "category": "unknown",
                        "entity": "",
                        "location": "",
                        "polarity": "present",
                        "reason": "LLM call failed before root cause could be verified.",
                    }
                ],
                "confidence": 0.1,
            },
            "patch": {"hunks": []},
        }
        # Use native Ollama if configured
        if os.environ.get("OLLAMA_API_KEY") or "11434" in self.api_base:
            try:
                import ollama
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                    format="json",
                )
                return response["message"]["content"] or ""
            except ImportError:
                logger.warning("[INFERENCE] ollama package not found, falling back to OpenAI compatibility layer")
            except Exception as e:
                logger.error(f"[INFERENCE] Ollama call failed: {e}")
                return json.dumps(fallback_response)

        # Default to OpenAI compatibility layer
        try:
            import openai
            client = openai.OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
            request = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_format": self._openai_json_schema_response_format(),
            }
            try:
                response = client.chat.completions.create(**request)
            except Exception as response_format_error:
                logger.warning(
                    "[INFERENCE] response_format=json_schema unsupported by backend; retrying with json_object: %s",
                    response_format_error,
                )
                request["response_format"] = {"type": "json_object"}
                try:
                    response = client.chat.completions.create(**request)
                except Exception as json_object_error:
                    logger.warning(
                        "[INFERENCE] response_format=json_object unsupported by backend; retrying without it: %s",
                        json_object_error,
                    )
                    request.pop("response_format", None)
                    response = client.chat.completions.create(**request)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[INFERENCE] LLM call failed: {e} (base={self.api_base} model={self.model_name})")
            return json.dumps(fallback_response)


def _build_default_runner(repo_path: str) -> Optional[Any]:
    """Wrap the local pytest runner for FlakeForgeEnvironment (expects run_test)."""
    try:
        from server.docker_runner import DockerTestRunner
    except Exception:
        try:
            from .server.docker_runner import DockerTestRunner
        except Exception:
            return None

    base_runner = DockerTestRunner(repo_path)

    class _RunnerAdapter:
        def run_test(self, test_identifier: str):
            return base_runner.run_test(test_identifier)

        def run_single(self, test_identifier: str) -> Dict[str, Any]:
            record = self.run_test(test_identifier)
            return {
                "passed": bool(record.passed),
                "duration_ms": int(record.duration_ms),
                "error_type": record.error_type,
                "error_message": record.error_message,
                "stderr": record.stderr_excerpt or "",
            }

    return _RunnerAdapter()


def _should_use_remote_env() -> bool:
    return False


def _run_async(coro: Any) -> Any:
    """Run a coroutine in CLI and notebook contexts without masking real errors."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        msg = str(exc)
        if "asyncio.run() cannot be called from a running event loop" not in msg:
            raise

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


async def run_episode(
    env: FlakeForgeEnvironment,
    agent: Union[UnifiedFlakeForgeAgent, ToolAugmentedFlakeForgeAgent],
    verbose: bool = True,
    reset_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a single episode of the unified inference loop.

    Returns:
        Episode result dict with trajectory, rewards, and metadata.
    """
    episode_result: Dict[str, Any] = {
        "trajectory": [],
        "total_reward": 0.0,
        "steps": 0,
        "final_pass_rate": 0.0,
        "done_reason": "in_progress",
        "reward_breakdown_history": [],
    }

    reset_result = env.reset(**(reset_kwargs or {}))
    if asyncio.iscoroutine(reset_result):
        reset_result = await reset_result
    step_output = _as_step_output_like(reset_result)
    observation = step_output.observation

    if verbose:
        logger.info(
            "[EPISODE] START test=%s baseline_pass_rate=%.2f deep_signals=%s",
            observation.test_identifier,
            observation.baseline_pass_rate,
            step_output.info.get("deep_signals", {}),
        )

    while not step_output.done:
        tool_ctx = ToolContext(
            repo_root=getattr(observation, "repo_root", "") or "",
            observation=observation,
            env=env,
        )
        action = agent.generate(observation, tool_context=tool_ctx)

        if verbose:
            logger.info(
                "[EPISODE] STEP %d -> category=%s confidence=%.2f patch_len=%d",
                observation.step + 1,
                action.predicted_category,
                action.predicted_confidence,
                len(action.patch_text),
            )

        step_result = env.step(action)
        if asyncio.iscoroutine(step_result):
            step_result = await step_result
        step_output = _as_step_output_like(step_result)
        observation = step_output.observation

        reward = step_output.reward
        breakdown = step_output.info.get("reward_breakdown", {})
        done = step_output.done
        pass_rate_after = step_output.state.current_pass_rate
        pass_rate_before = observation.baseline_pass_rate

        logger.info(
            f"[EPISODE] RESULT step={step_output.state.step_count} reward={reward:.4f} "
            f"pass_rate={pass_rate_before:.2f}->{pass_rate_after:.2f} "
            f"done={done} reason={step_output.info.get('done_reason', '')}"
        )
        if breakdown and verbose:
            logger.info(f"    Breakdown: {breakdown}")

        if not step_output.info.get("patch_result", {}).get("success", False) and verbose:
            logger.warning(f"    [DEBUG] Patch failed to apply. Raw response excerpt: {action.raw_response[:200]}...")

        step_data = {
            "step": step_output.state.step_count,
            "predicted_category": action.predicted_category,
            "predicted_confidence": action.predicted_confidence,
            "think_text": action.think_text[:500],
            "patch_text": action.patch_text,
            "patch_applied": step_output.info.get("patch_result", {}).get("success", False),
            "reward": step_output.reward,
            "reward_breakdown": step_output.info.get("reward_breakdown", {}),
            "pass_rate": step_output.state.current_pass_rate,
            "done": step_output.done,
        }
        episode_result["trajectory"].append(step_data)
        episode_result["total_reward"] += step_output.reward
        episode_result["reward_breakdown_history"].append(
            step_output.info.get("reward_breakdown", {})
        )

        if verbose:
            logger.info(
                "[EPISODE] RESULT step=%d reward=%.4f pass_rate=%.2f->%.2f done=%s reason=%s",
                step_data["step"],
                step_output.reward,
                observation.baseline_pass_rate,
                step_output.state.current_pass_rate,
                step_output.done,
                step_output.info.get("done_reason", ""),
            )

    episode_result["steps"] = step_output.state.step_count
    episode_result["final_pass_rate"] = step_output.state.current_pass_rate
    episode_result["done_reason"] = step_output.info.get("done_reason", "unknown")

    return episode_result


def run_inference(
    repo_path: str,
    test_identifier: str,
    model_name: Optional[str] = None,
    max_steps: Optional[int] = None,
    num_runs: int = 20,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a FlakeForge inference episode.

    Creates the environment and agent, runs an episode, and returns
    the result dict with trajectory, rewards, and metadata.
    """
    max_steps = int(max_steps or os.environ.get("INFERENCE_MAX_STEPS", 8))
    backend = LLMBackend(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
    )

    agent = ToolAugmentedFlakeForgeAgent(backend=backend)

    if _should_use_remote_env():
        env_url = (os.environ.get("ENV_BASE_URL") or "http://localhost:5000").strip()
        if verbose:
            logger.info("[INFERENCE] Using remote OpenEnv client at %s", env_url)
        if FlakeForgeEnvClient is None:
            raise RuntimeError("Remote execution requested but openenv.core.EnvClient is unavailable")
        env = FlakeForgeEnvClient(base_url=env_url)
        reset_payload = {
            "repo_path": repo_path,
            "test_identifier": test_identifier,
            "max_steps": max_steps,
            "num_runs": num_runs,
        }

        async def _run_remote() -> Dict[str, Any]:
            connect = getattr(env, "connect", None)
            if callable(connect):
                maybe_connected = connect()
                if asyncio.iscoroutine(maybe_connected):
                    await maybe_connected
            try:
                return await run_episode(
                    env,
                    agent,
                    verbose=verbose,
                    reset_kwargs=reset_payload,
                )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    maybe_closed = close()
                    if asyncio.iscoroutine(maybe_closed):
                        await maybe_closed

        result = _run_async(_run_remote())
    else:
        runner = _build_default_runner(repo_path)
        if runner is None and verbose:
            logger.warning("[INFERENCE] Could not create pytest runner adapter; environment will use synthetic runs.")

        # Create environment
        env = FlakeForgeEnvironment(
            repo_path=repo_path,
            test_identifier=test_identifier,
            max_steps=max_steps,
            num_runs=num_runs,
            runner=runner,
        )

        # Run episode
        result = _run_async(
            run_episode(
                env,
                agent,
                verbose=verbose,
                reset_kwargs={
                    "preflight_quick_runs": max(3, min(num_runs, 10)),
                    "preflight_confirm_runs": num_runs,
                },
            )
        )

    if verbose:
        logger.info(
            "[INFERENCE] COMPLETE steps=%d total_reward=%.4f final_pass_rate=%.2f reason=%s",
            result["steps"],
            result["total_reward"],
            result["final_pass_rate"],
            result["done_reason"],
        )

    return result


def _default_repo_path() -> str:
    return os.environ.get("FF_REPO_PATH", str(Path("test_repos") / "timing_race_minimal"))


def _default_test_id() -> str:
    return os.environ.get("FF_TEST_ID", "tests/test_flaky.py::test_fetch_should_complete")


def _default_seed_root() -> str:
    env_root = os.environ.get("FF_SEED_ROOT")
    if env_root:
        return env_root

    project_root = Path(__file__).resolve().parent
    candidates = [
        Path(r"C:\CodingNest\seed_repos\idoft"),
        Path(r"C:\CodingNest\seed_repos\idof"),
        Path("/CodingNest/seed_repos/idoft"),
        Path("/CodingNest/seed_repos/idof"),
        (project_root / ".." / "seed_repos" / "idoft"),
        (project_root / ".." / "seed_repos" / "idof"),
        (project_root / ".." / ".." / "seed_repos" / "idoft"),
        (project_root / ".." / ".." / "seed_repos" / "idof"),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and any(resolved.glob("*/flake_manifest.json")):
            return str(resolved)
    return str(candidates[0])


def _load_seed_cases(seed_root: str | Path) -> List[Dict[str, Any]]:
    root = Path(seed_root).expanduser().resolve()
    cases: List[Dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/flake_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[INFERENCE] Skipping unreadable manifest %s: %s", manifest_path, exc)
            continue

        test_id = manifest.get("flaky_test_path") or manifest.get("test_identifier")
        if not test_id:
            logger.warning("[INFERENCE] Skipping manifest without flaky_test_path: %s", manifest_path)
            continue

        cases.append({
            "case_id": manifest_path.parent.name,
            "repo_path": manifest_path.parent,
            "manifest_path": manifest_path,
            "test_id": str(test_id),
            "repo_name": manifest.get("repo_name", manifest_path.parent.name),
            "flake_category": manifest.get("flake_category", "UNKNOWN"),
            "difficulty": manifest.get("difficulty", "medium"),
        })

    if not cases:
        raise FileNotFoundError(f"No usable flake_manifest.json files found under {root}")
    return cases


def _ignore_episode_copy(_: str, names: List[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} & set(names)
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def _ensure_case_backup(
    case: Dict[str, Any],
    *,
    backup_root: str | Path,
    enabled: bool = True,
) -> Optional[Path]:
    """Create a one-time immutable backup copy of an original seed repo."""
    if not enabled:
        return None

    source = Path(case["repo_path"]).resolve()
    root = Path(backup_root).expanduser().resolve()
    backup_path = root / case["case_id"]

    if backup_path.exists():
        return backup_path

    root.mkdir(parents=True, exist_ok=True)
    tmp_path = root / f".{case['case_id']}.tmp-{uuid.uuid4().hex[:8]}"
    shutil.copytree(source, tmp_path, ignore=_ignore_episode_copy)
    tmp_path.replace(backup_path)
    return backup_path


def _materialize_case_repo(
    case: Dict[str, Any],
    isolate: bool = True,
    backup_path: Optional[Path] = None,
) -> Path:
    repo_path = Path(case["repo_path"]).resolve()
    if not isolate:
        return repo_path

    episode_root = Path(os.environ.get("FF_INFERENCE_REPO_ROOT", "outputs/inference_repos")).resolve()
    episode_root.mkdir(parents=True, exist_ok=True)
    worktree = episode_root / f"{case['case_id']}-{uuid.uuid4().hex[:8]}"
    if worktree.exists():
        shutil.rmtree(worktree)
    shutil.copytree(backup_path or repo_path, worktree, ignore=_ignore_episode_copy)
    return worktree


def _select_seed_cases(
    cases: List[Dict[str, Any]],
    *,
    case: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    selected = cases
    if case:
        needle = case.lower()
        selected = [
            item for item in selected
            if needle in item["case_id"].lower() or needle in str(item.get("repo_name", "")).lower()
        ]
        if not selected:
            raise ValueError(f"No seed case matched {case!r}")

    if limit is not None:
        selected = selected[:max(limit, 0)]
    return selected


def run_seed_inference(
    *,
    seed_root: str | Path,
    case: Optional[str],
    limit: Optional[int],
    isolate: bool,
    backup: bool,
    backup_root: str | Path,
    model_name: Optional[str],
    max_steps: Optional[int],
    num_runs: int,
    api_base: Optional[str],
    api_key: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    cases = _select_seed_cases(_load_seed_cases(seed_root), case=case, limit=limit)
    results: List[Dict[str, Any]] = []

    for idx, seed_case in enumerate(cases, start=1):
        backup_path = _ensure_case_backup(seed_case, backup_root=backup_root, enabled=backup)
        worktree = _materialize_case_repo(seed_case, isolate=isolate, backup_path=backup_path)
        if verbose:
            logger.info(
                "[INFERENCE] Seed case %d/%d id=%s category=%s test=%s repo=%s backup=%s",
                idx,
                len(cases),
                seed_case["case_id"],
                seed_case["flake_category"],
                seed_case["test_id"],
                worktree,
                backup_path or "<disabled>",
            )

        try:
            result = run_inference(
                repo_path=str(worktree),
                test_identifier=seed_case["test_id"],
                model_name=model_name,
                max_steps=max_steps,
                num_runs=num_runs,
                api_base=api_base,
                api_key=api_key,
                verbose=verbose,
            )
            results.append({
                "case_id": seed_case["case_id"],
                "repo_name": seed_case["repo_name"],
                "flake_category": seed_case["flake_category"],
                "difficulty": seed_case["difficulty"],
                "test_id": seed_case["test_id"],
                "source_repo_path": str(seed_case["repo_path"]),
                "backup_repo_path": str(backup_path) if backup_path else None,
                "run_repo_path": str(worktree),
                "result": result,
            })
        except Exception as exc:
            logger.error("[INFERENCE] Seed case failed id=%s: %s", seed_case["case_id"], exc)
            results.append({
                "case_id": seed_case["case_id"],
                "repo_name": seed_case["repo_name"],
                "flake_category": seed_case["flake_category"],
                "difficulty": seed_case["difficulty"],
                "test_id": seed_case["test_id"],
                "source_repo_path": str(seed_case["repo_path"]),
                "backup_repo_path": str(backup_path) if backup_path else None,
                "run_repo_path": str(worktree),
                "error": type(exc).__name__,
                "message": str(exc),
            })

    return {
        "seed_root": str(Path(seed_root).resolve()),
        "backup_root": str(Path(backup_root).resolve()) if backup else None,
        "isolated": isolate,
        "count": len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlakeForge unified inference episode")
    parser.add_argument("--repo-path", default=_default_repo_path(), help="Path to target repo")
    parser.add_argument("--test-id", default=_default_test_id(), help="Target test identifier")
    parser.add_argument(
        "--seed-root",
        default=None,
        help=f"Run manifest-backed local inference for repos under this seed root (default candidate: {_default_seed_root()})",
    )
    parser.add_argument("--case", default=None, help="Substring filter for seed case directory/repo_name")
    parser.add_argument("--limit", type=int, default=None, help="Max seed cases to run")
    parser.add_argument("--list-cases", action="store_true", help="List seed cases from --seed-root and exit")
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Patch seed repos in place instead of copying to outputs/inference_repos first",
    )
    parser.add_argument(
        "--backup-root",
        default=os.environ.get("FF_SEED_BACKUP_ROOT", "outputs/seed_repo_backups"),
        help="Where immutable seed repo backups are stored before local inference",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable seed repo backup creation. Not recommended for batch runs.",
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"), help="LLM model name")
    parser.add_argument("--max-steps", type=int, default=None, help="Max episode steps")
    parser.add_argument("--num-runs", type=int, default=int(os.environ.get("NUM_RUNS", 10)), help="Repeated test runs per step")
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_API_BASE"), help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")
    args = parser.parse_args()

    try:
        if args.seed_root or args.list_cases:
            seed_root = args.seed_root or _default_seed_root()
            if args.list_cases:
                cases = _select_seed_cases(_load_seed_cases(seed_root), case=args.case, limit=args.limit)
                result = {
                    "seed_root": str(Path(seed_root).resolve()),
                    "count": len(cases),
                    "cases": [
                        {
                            "case_id": item["case_id"],
                            "repo_name": item["repo_name"],
                            "flake_category": item["flake_category"],
                            "difficulty": item["difficulty"],
                            "test_id": item["test_id"],
                            "repo_path": str(item["repo_path"]),
                        }
                        for item in cases
                    ],
                }
            else:
                result = run_seed_inference(
                    seed_root=seed_root,
                    case=args.case,
                    limit=args.limit,
                    isolate=not args.no_isolation,
                    backup=not args.no_backup,
                    backup_root=args.backup_root,
                    model_name=args.model,
                    max_steps=args.max_steps,
                    num_runs=args.num_runs,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    verbose=not args.quiet,
                )
        else:
            result = run_inference(
                repo_path=args.repo_path,
                test_identifier=args.test_id,
                model_name=args.model,
                max_steps=args.max_steps,
                num_runs=args.num_runs,
                api_base=args.api_base,
                api_key=args.api_key,
                verbose=not args.quiet,
            )
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        logger.error("[INFERENCE] FATAL: %s", exc)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()


def flakeforge_reward_fn(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> list:
    """Reward function compatible with TRL's GRPOTrainer.

    Takes prompts and completions (from the model), evaluates each
    completion using the V3 reward architecture.

    This is a simplified wrapper for training — the full environment
    step is not used here. Instead, we parse the completion and
    compute format + reasoning consistency rewards.
    """
    from agent.unified_agent import (
        extract_think,
        extract_patch,
        extract_category_from_think,
        extract_confidence_from_think,
        infer_category_from_patch,
    )
    from server.reward import (
        compute_format_reward,
        compute_reasoning_consistency,
    )

    rewards = []
    for prompt, completion in zip(prompts, completions):
        completion_text = completion if isinstance(completion, str) else str(completion)

        action = FlakeForgeAction(
            raw_response=completion_text,
            think_text=extract_think(completion_text),
            patch_text=extract_patch(completion_text),
            predicted_category=extract_category_from_think(extract_think(completion_text)),
            predicted_confidence=extract_confidence_from_think(extract_think(completion_text)),
        )

        format_score = compute_format_reward(action)
        inferred_cat = infer_category_from_patch(action.patch_text)
        consistency_score = compute_reasoning_consistency(
            action.predicted_category, inferred_cat, action.think_text, action.patch_text
        )

        total = format_score * 1.0 + consistency_score * 0.5
        rewards.append(total)

    return rewards
