"""V3 Inference Loop — unified agent with verifiable reward.

Replaces the V2 two-phase inference (analyze → fix) with a single
unified loop: observe → think+patch → apply → verify → reward.

No judge calls. No hypothesis gating. Just execution-verified reward.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

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
    from agent.unified_agent import UnifiedFlakeForgeAgent, build_unified_prompt
    from server.FlakeForge_environment import FlakeForgeEnvironment
except ImportError:
    from .models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState
    from .agent.unified_agent import UnifiedFlakeForgeAgent, build_unified_prompt
    from .server.FlakeForge_environment import FlakeForgeEnvironment

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from .utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__)

if load_dotenv is not None:
    load_dotenv()


def _observation_info(observation: FlakeForgeObservation) -> Dict[str, Any]:
    """Extract step metadata carried inside OpenEnv observations."""
    patch_result = getattr(observation, "patch_result", {}) or {}
    if not patch_result and getattr(observation, "patches_applied", None):
        patch_result = {"success": True}

    return {
        "reward_breakdown": getattr(observation, "reward_breakdown", {}) or {},
        "patch_result": patch_result,
        "done_reason": getattr(observation, "done_reason", "") or "",
        "deep_signals": {
            "module_cache_violations": getattr(observation, "module_cache_violations", []),
            "fixture_scope_risks": getattr(observation, "fixture_scope_risks", []),
            "mock_residue_sites": getattr(observation, "mock_residue_sites", []),
            "import_side_effect_files": getattr(observation, "import_side_effect_files", []),
            "async_contamination_alive": getattr(observation, "async_contamination_alive", False),
        },
    }


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
            setattr(result, "info", _observation_info(observation))
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
        return value

    class _StepLike:
        def __init__(self, source: Any) -> None:
            observation = getattr(source, "observation", source)
            self.observation = observation
            self.reward = float(
                getattr(source, "reward", getattr(observation, "reward", 0.0)) or 0.0
            )
            self.done = bool(getattr(source, "done", getattr(observation, "done", False)))
            self.info = getattr(source, "info", None) or _observation_info(observation)

            self.state = getattr(source, "state", None) or FlakeForgeState(
                episode_id=str(getattr(observation, "episode_id", "")),
                step_count=int(getattr(observation, "step", 0)),
                done=self.done,
                current_pass_rate=float(getattr(observation, "current_pass_rate", 0.0)),
                baseline_pass_rate=float(getattr(observation, "baseline_pass_rate", 0.0)),
                regression_detected=False,
            )

    return _StepLike(value)


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

    def generate(self, prompt: str, *, system_prompt: str) -> str:
        """Generate a completion using an OpenAI-compatible API."""
        try:
            import openai
            client = openai.OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(
                "[INFERENCE] LLM call failed: %s (base=%s model=%s)",
                exc,
                self.api_base,
                self.model_name,
            )
            return f"<think>\nRoot Cause: unknown (confidence: 0.1)\nLLM call failed: {exc}\nStrategy: Unable to generate fix.\n</think>\n<patch>\n</patch>"


def _build_default_runner(repo_path: str) -> Optional[Any]:
    """Create an environment runner adapter that exposes run_single()."""
    try:
        from server.docker_runner import DockerTestRunner
    except Exception:
        try:
            from .server.docker_runner import DockerTestRunner
        except Exception:
            return None

    base_runner = DockerTestRunner(repo_path)

    class _RunnerAdapter:
        def run_single(self, test_identifier: str) -> Dict[str, Any]:
            record = base_runner.run_test(test_identifier)
            return {
                "passed": bool(record.passed),
                "duration_ms": int(record.duration_ms),
                "error_type": record.error_type,
                "error_message": record.error_message,
                "stderr": record.stderr_excerpt or "",
            }

    return _RunnerAdapter()


def _should_use_remote_env() -> bool:
    base_url = (os.environ.get("ENV_BASE_URL") or "").strip()
    use_docker = os.environ.get("USE_DOCKER_IMAGE", "0").strip().lower() in {"1", "true", "yes"}
    return bool(base_url) or use_docker


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
    agent: UnifiedFlakeForgeAgent,
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

    # Reset environment
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
        # Generate unified think+patch
        action = agent.generate(observation)

        if verbose:
            logger.info(
                "[EPISODE] STEP %d → category=%s confidence=%.2f patch_len=%d",
                observation.step + 1,
                action.predicted_category,
                action.predicted_confidence,
                len(action.patch_text),
            )

        # Execute step
        step_result = env.step(action)
        if asyncio.iscoroutine(step_result):
            step_result = await step_result
        step_output = _as_step_output_like(step_result)
        observation = step_output.observation

        # Log result
        reward = step_output.reward
        breakdown = step_output.info.get("reward_breakdown", {})
        patch_result = step_output.info.get("patch_result", {})
        done = step_output.done
        pass_rate_after = step_output.state.current_pass_rate
        pass_rate_before = observation.baseline_pass_rate

        logger.info(
            f"[EPISODE] RESULT step={step_output.state.step_count} reward={reward:.4f} "
            f"pass_rate={pass_rate_before:.2f}→{pass_rate_after:.2f} "
            f"done={done} reason={step_output.info.get('done_reason', '')}"
        )
        if breakdown and verbose:
            logger.info(f"    Breakdown: {breakdown}")

        if action.patch_text.strip() and not patch_result.get("success", False) and verbose:
            logger.warning(f"    [DEBUG] Patch failed to apply. Raw response excerpt: {action.raw_response[:200]}...")

        # Track trajectory
        step_data = {
            "step": step_output.state.step_count,
            "predicted_category": action.predicted_category,
            "predicted_confidence": action.predicted_confidence,
            "think_text": action.think_text[:500],
            "patch_applied": patch_result.get("success", False) and not patch_result.get("rolled_back", False),
            "patch_rolled_back": patch_result.get("rolled_back", False),
            "reward": step_output.reward,
            "reward_breakdown": breakdown,
            "pass_rate": step_output.state.current_pass_rate,
            "done": step_output.done,
        }
        episode_result["trajectory"].append(step_data)
        episode_result["total_reward"] += step_output.reward
        episode_result["reward_breakdown_history"].append(breakdown)

    # Finalize
    episode_result["steps"] = step_output.state.step_count
    episode_result["final_pass_rate"] = step_output.state.current_pass_rate
    episode_result["done_reason"] = step_output.info.get("done_reason", "unknown")

    return episode_result


def run_inference(
    repo_path: str,
    test_identifier: str,
    model_name: Optional[str] = None,
    max_steps: Optional[int] = None,
    num_runs: int = 10,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a FlakeForge V3 inference episode.

    This is the main entry point for running the unified agent on a
    flaky test. It creates the environment and agent, runs an episode,
    and returns the result.
    """
    # Load defaults from environment
    max_steps = int(max_steps or os.environ.get("INFERENCE_MAX_STEPS", 8))
    # Create LLM backend
    backend = LLMBackend(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
    )

    # Create unified agent
    agent = UnifiedFlakeForgeAgent(backend=backend)

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
            logger.warning("[INFERENCE] Could not create DockerTestRunner adapter; environment will use synthetic runs.")

        # Create environment
        env = FlakeForgeEnvironment(
            repo_path=repo_path,
            test_identifier=test_identifier,
            max_steps=max_steps,
            num_runs=num_runs,
            runner=runner,
        )

        # Run episode
        result = _run_async(run_episode(env, agent, verbose=verbose))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlakeForge V3 unified inference episode")
    parser.add_argument("--repo-path", default=_default_repo_path(), help="Path to target repo")
    parser.add_argument("--test-id", default=_default_test_id(), help="Target test identifier")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"), help="LLM model name")
    parser.add_argument("--max-steps", type=int, default=None, help="Max episode steps")
    parser.add_argument("--num-runs", type=int, default=10, help="Repeated test runs per step")
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_API_BASE"), help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")
    args = parser.parse_args()

    try:
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


# ── Reward Function for GRPO Training ─────────────────────────────────────

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

        # Parse the completion
        action = FlakeForgeAction(
            raw_response=completion_text,
            think_text=extract_think(completion_text),
            patch_text=extract_patch(completion_text),
            predicted_category=extract_category_from_think(extract_think(completion_text)),
            predicted_confidence=extract_confidence_from_think(extract_think(completion_text)),
        )

        # Format reward
        format_score = compute_format_reward(action)

        # Reasoning consistency
        inferred_cat = infer_category_from_patch(action.patch_text)
        consistency_score = compute_reasoning_consistency(
            action.predicted_category, inferred_cat, action.think_text, action.patch_text
        )

        # Composite training reward (no execution signals in offline mode)
        total = format_score * 1.0 + consistency_score * 0.5
        rewards.append(total)

    return rewards
