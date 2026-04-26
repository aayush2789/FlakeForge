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
from typing import Any, Dict, List, Optional

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
    from agent.unified_agent import (
        UnifiedFlakeForgeAgent,
        build_unified_prompt,
        finalize_for_inference,
    )
    from server.FlakeForge_environment import FlakeForgeEnvironment
except ImportError:
    from .models import FlakeForgeAction, FlakeForgeObservation, FlakeForgeState
    from .agent.unified_agent import (
        UnifiedFlakeForgeAgent,
        build_unified_prompt,
        finalize_for_inference,
    )
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



# ── Inference-mode safety helpers ────────────────────────────────────────────

# LLM calls per environment step: do not call env.step until format + grounded patch succeed.
_MAX_INFERENCE_LLM_RETRIES: int = int(os.environ.get("FF_INFERENCE_LLM_RETRIES", "7"))
_MAX_CONSECUTIVE_FAILURES = 10  # consecutive bad steps before early-stopping the episode
ALLOW_FUZZY_MATCH: bool = os.environ.get("FF_ALLOW_FUZZY_GROUNDING", "1") not in ("0", "false", "False")


def _fuzzy_grounding_ok(search: str, blob: str) -> bool:
    """7B-friendly: a key line or approximate block may still apply via patch_applier fuzzy match."""
    if not (search or "").strip():
        return True
    if search in blob:
        return True
    blob_n = blob.replace("\r\n", "\n")
    for line in search.splitlines():
        t = line.strip()
        if len(t) < 2:
            continue
        if t in blob or t in blob_n:
            return True
    return False


def _check_patch_grounding(
    action: "FlakeForgeAction",
    observation: "FlakeForgeObservation",
    repo_path: Optional[Path] = None,
) -> str:
    """Return '' if the patch is plausibly grounded, else a short error for the next retry.

    Uses full on-disk SUT when ``repo_path`` + ``observation.source_file`` are available so
    checks are not truncated by observation limits. Fuzzy is allowed for ``search`` when
    :data:`ALLOW_FUZZY_MATCH` is true (aligns with ``fuzzy_applied`` in the applier).
    """
    hunks = getattr(getattr(action, "structured_patch", None), "hunks", None) or []
    if not hunks:
        return ""

    full_text = "\n".join(
        filter(
            None,
            [
                getattr(observation, "source_under_test", "") or "",
                getattr(observation, "test_function_source", "") or "",
            ],
        )
    )
    if repo_path and getattr(observation, "source_file", None):
        sp = (Path(repo_path) / str(observation.source_file)).resolve()
        try:
            sp.relative_to(Path(repo_path).resolve())
        except ValueError:
            return "source_file path escapes repo"
        if sp.is_file():
            try:
                full_text = sp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
    if not (full_text or "").strip():
        return ""

    for hunk in hunks:
        ln = getattr(hunk, "line_number", None)
        if ln is not None and int(ln) > 0:
            n = len(full_text.splitlines())
            if 1 <= int(ln) <= n:
                continue
            return f"line_number {ln} is out of range (file has {n} lines)"
        search = (getattr(hunk, "search", "") or "").strip()
        if not search:
            return "hunk has no search and no line_number; use TARGET LINE for search or line_number+replace"
        if search in full_text or search in full_text.replace("\r\n", "\n"):
            continue
        if ALLOW_FUZZY_MATCH and _fuzzy_grounding_ok(search, full_text):
            continue
        preview = search[:60].replace("\n", "↵")
        return f"search not found in target file: \"{preview}...\" (use the TARGET LINE line verbatim or line_number+replace)"
    return ""


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
                                        "required": ["replace"],
                                        "properties": {
                                            "hunk_id": {"type": "string"},
                                            "search": {"type": "string"},
                                            "replace": {"type": "string"},
                                            "line_number": {"type": "integer", "minimum": 1},
                                            "rationale": {"type": "string"},
                                            "addresses_claim": {"type": "string"},
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
    """Wrap DockerTestRunner for FlakeForgeEnvironment (expects run_test)."""
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
    agent: UnifiedFlakeForgeAgent,
    verbose: bool = True,
    reset_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a single episode of the unified inference loop.

    Inference-mode guards (not used for training the same way):
    - Re-generate up to ``_MAX_INFERENCE_LLM_RETRIES`` (``FF_INFERENCE_LLM_RETRIES``) times
      per env step: invalid JSON, loose grounding, or empty patch *after finalize* (no
      ``env.step`` until success or the episode ends with
      ``done_reason=inference_invalid_exhausted_retries``).
    - ``finalize_for_inference`` sets ``file`` from ``observation.source_file`` and expands
      ``line_number`` → ``search`` for the patch string.
    - Grounding: substring or, when ``FF_ALLOW_FUZZY_GROUNDING`` is on, per-line fuzzy
      match; full file is read from disk when possible (not only the observation excerpt).
    - Regression: ``build_unified_prompt`` nudges the model after a bad apply/pass drop.
    - Early stop: ``_MAX_CONSECUTIVE_FAILURES`` regressed/invalid apply steps in a row.

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

    consecutive_failures = 0
    repo_root: Optional[Path] = getattr(env, "repo_path", None)

    while not step_output.done:
        # ── Per-step: retry until valid + grounded, or else abort (no bad env.step) ──
        retry_hint: Optional[str] = None
        action: Optional[FlakeForgeAction] = None

        for attempt in range(_MAX_INFERENCE_LLM_RETRIES):
            candidate = agent.generate(observation, retry_hint=retry_hint)
            if repo_root is not None:
                candidate = finalize_for_inference(candidate, observation, Path(repo_root))

            think_ok = getattr(
                getattr(candidate, "structured_think", None), "format_penalty", -1.0
            ) >= 0.0
            patch_ok_fmt = getattr(
                getattr(candidate, "structured_patch", None), "format_penalty", -1.0
            ) >= 0.0

            if not think_ok or not patch_ok_fmt:
                retry_hint = (
                    "FORMAT_ERROR: return only one JSON object with think + patch. "
                    "No markdown, no text outside JSON."
                )
                if verbose:
                    logger.warning(
                        "[EPISODE] step=%d attempt=%d FORMAT invalid (think=%s patch=%s) — retrying",
                        observation.step + 1,
                        attempt,
                        think_ok,
                        patch_ok_fmt,
                    )
                continue

            grounding_err = _check_patch_grounding(
                candidate, observation, repo_path=repo_root
            )
            if grounding_err:
                retry_hint = (
                    f"GROUNDING: {grounding_err} — copy the line from TARGET LINE / SOURCE, "
                    "or use {line_number, replace} (1-based) with the full new line."
                )
                if verbose:
                    logger.warning(
                        "[EPISODE] step=%d attempt=%d GROUNDING: %s — retrying",
                        observation.step + 1,
                        attempt,
                        grounding_err,
                    )
                continue

            if not (getattr(candidate, "patch_text", None) or "").strip():
                retry_hint = (
                    "EMPTY_PATCH: add patch.hunks with search+replace, or line_number+replace, "
                    "as in the system prompt. File is assigned for you; do not pick a file path."
                )
                if verbose:
                    logger.warning(
                        "[EPISODE] step=%d attempt=%d empty patch after finalize — retrying",
                        observation.step + 1,
                        attempt,
                    )
                continue

            action = candidate
            break
        else:
            if verbose:
                logger.error(
                    "[EPISODE] step=%d gave up after %d LLM attempts (invalid/grounding)",
                    observation.step + 1,
                    _MAX_INFERENCE_LLM_RETRIES,
                )
            episode_result["done_reason"] = "inference_invalid_exhausted_retries"
            break

        assert action is not None  # only reachable after successful for-loop break

        if verbose:
            logger.info(
                "[EPISODE] STEP %d -> category=%s confidence=%.2f patch_len=%d",
                observation.step + 1,
                action.predicted_category,
                action.predicted_confidence,
                len(action.patch_text),
            )

        # ── Environment step ─────────────────────────────────────────────────
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

        # ── Failure / regression accounting ─────────────────────────────────
        patch_applied = step_output.info.get("patch_result", {}).get("success", False)
        regression = pass_rate_after < pass_rate_before - 0.05

        if not patch_applied or regression:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        logger.info(
            "[EPISODE] RESULT step=%d reward=%.4f pass_rate=%.2f->%.2f "
            "patch_applied=%s regression=%s consecutive_failures=%d done=%s reason=%s",
            step_output.state.step_count,
            reward,
            pass_rate_before,
            pass_rate_after,
            patch_applied,
            regression,
            consecutive_failures,
            done,
            step_output.info.get("done_reason", ""),
        )
        if breakdown and verbose:
            logger.info("    Breakdown: %s", breakdown)

        if not patch_applied and verbose:
            logger.warning(
                "    [DEBUG] Patch failed to apply. Raw response excerpt: %s...",
                action.raw_response[:200],
            )

        step_data = {
            "step": step_output.state.step_count,
            "predicted_category": action.predicted_category,
            "predicted_confidence": action.predicted_confidence,
            "think_text": action.think_text[:500],
            "patch_text": action.patch_text,
            "patch_applied": patch_applied,
            "reward": step_output.reward,
            "reward_breakdown": step_output.info.get("reward_breakdown", {}),
            "pass_rate": pass_rate_after,
            "done": step_output.done,
        }
        episode_result["trajectory"].append(step_data)
        episode_result["total_reward"] += step_output.reward
        episode_result["reward_breakdown_history"].append(breakdown)

        if verbose:
            logger.info(
                "[EPISODE] RESULT step=%d reward=%.4f pass_rate=%.2f->%.2f done=%s reason=%s",
                step_data["step"],
                step_output.reward,
                observation.baseline_pass_rate,
                pass_rate_after,
                step_output.done,
                step_output.info.get("done_reason", ""),
            )

        # ── Early stopping on collapse ────────────────────────────────────────
        if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES and not done:
            logger.warning(
                "[EPISODE] EARLY STOP: %d consecutive failures (patch_applied=%s regression=%s)",
                consecutive_failures,
                patch_applied,
                regression,
            )
            episode_result["done_reason"] = "early_stop_consecutive_failures"
            break

    episode_result["steps"] = step_output.state.step_count
    episode_result["final_pass_rate"] = step_output.state.current_pass_rate
    if episode_result["done_reason"] == "in_progress":
        episode_result["done_reason"] = step_output.info.get("done_reason", "unknown")

    return episode_result


def run_inference(
    repo_path: str,
    test_identifier: str,
    model_name: Optional[str] = None,
    max_steps: Optional[int] = None,
    num_runs: int = 10,
    test_timeout: Optional[int] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a FlakeForge inference episode.

    Creates the environment and agent, runs an episode, and returns
    the result dict with trajectory, rewards, and metadata.
    """
    if test_timeout is not None:
        os.environ["FF_TEST_TIMEOUT_SECONDS"] = str(test_timeout)
    max_steps = int(max_steps or os.environ.get("INFERENCE_MAX_STEPS", 8))
    backend = LLMBackend(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
    )

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


def _materialize_case_repo(case: Dict[str, Any], isolate: bool = True) -> Path:
    repo_path = Path(case["repo_path"]).resolve()
    if not isolate:
        return repo_path

    episode_root = Path(os.environ.get("FF_INFERENCE_REPO_ROOT", "outputs/inference_repos")).resolve()
    episode_root.mkdir(parents=True, exist_ok=True)
    worktree = episode_root / f"{case['case_id']}-{uuid.uuid4().hex[:8]}"
    if worktree.exists():
        shutil.rmtree(worktree)
    shutil.copytree(repo_path, worktree, ignore=_ignore_episode_copy)
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


# Categories whose flakiness requires state to accumulate across runs in the
# SAME directory (leaked files, global vars, CSV rows, etc.).  In Docker every
# run is a fresh --rm container so state is always clean → always passes.
# Force local mode for these so the persistent episode worktree lets state leak.
_LOCAL_MODE_CATEGORIES: set = {"RESOURCE_LEAK", "ORDER_DEPENDENCY", "SHARED_STATE"}

# Categories that need their own project deps (numpy, aioredis …) before pytest
# can even collect tests.  Auto-install is always enabled for IDoFT seed runs
# via _apply_category_env.


def _apply_category_env(category: str) -> Dict[str, str]:
    """Return env-var overrides for the given flake category, and apply them.

    IDoFT seed runs ALWAYS use local mode (USE_DOCKER_IMAGE=0) because:
    - Docker's ephemeral --rm containers re-download deps on every run (slow/timeout).
    - Pip's local cache makes re-installs nearly instant after the first run.
    - RESOURCE_LEAK/ORDER_DEPENDENCY need persistent state between runs, which
      only the local runner provides.
    """
    overrides: Dict[str, str] = {
        "FF_AUTO_INSTALL_DEPS": "1",
        "FF_FULL_FILE_MODE": "0",
        "USE_DOCKER_IMAGE": "0",  # always local for IDoFT; see docstring
    }
    if category.upper() in _LOCAL_MODE_CATEGORIES:
        # Run full test file so polluter tests can leak state into the victim.
        overrides["FF_FULL_FILE_MODE"] = "1"
    # Apply to process environment so DockerTestRunner.__init__ picks them up.
    for k, v in overrides.items():
        os.environ[k] = v
    return overrides


def run_seed_inference(
    *,
    seed_root: str | Path,
    case: Optional[str],
    limit: Optional[int],
    isolate: bool,
    model_name: Optional[str],
    max_steps: Optional[int],
    num_runs: int,
    test_timeout: Optional[int],
    api_base: Optional[str],
    api_key: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    cases = _select_seed_cases(_load_seed_cases(seed_root), case=case, limit=limit)
    results: List[Dict[str, Any]] = []

    for idx, seed_case in enumerate(cases, start=1):
        worktree = _materialize_case_repo(seed_case, isolate=isolate)
        category = seed_case["flake_category"]
        env_overrides = _apply_category_env(category)
        if verbose:
            logger.info(
                "[INFERENCE] Seed case %d/%d id=%s category=%s test=%s repo=%s env=%s",
                idx,
                len(cases),
                seed_case["case_id"],
                category,
                seed_case["test_id"],
                worktree,
                env_overrides,
            )

        try:
            result = run_inference(
                repo_path=str(worktree),
                test_identifier=seed_case["test_id"],
                model_name=model_name,
                max_steps=max_steps,
                num_runs=num_runs,
                test_timeout=test_timeout,
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
                "run_repo_path": str(worktree),
                "error": type(exc).__name__,
                "message": str(exc),
            })

    return {
        "seed_root": str(Path(seed_root).resolve()),
        "count": len(results),
        "results": results,
    }


def _probe_all_cases(
    *,
    seed_root: str | Path,
    case: Optional[str],
    limit: Optional[int],
    num_runs: int,
    test_timeout: Optional[int],
    verbose: bool,
) -> Dict[str, Any]:
    """Quick-scan every IDoFT case under seed_root (max-steps=0) to find flaky ones.

    Returns a summary dict with cases bucketed as: flaky / stable / broken.
    Prints a compact progress line per case so you can watch live.
    """
    cases = _select_seed_cases(_load_seed_cases(seed_root), case=case, limit=limit)
    flaky: List[Dict] = []
    stable: List[Dict] = []
    broken: List[Dict] = []

    for idx, seed_case in enumerate(cases, start=1):
        case_id = seed_case["case_id"]
        category = seed_case["flake_category"]
        difficulty = seed_case["difficulty"]
        env_overrides = _apply_category_env(category)
        worktree = _materialize_case_repo(seed_case, isolate=True)
        print(
            f"[{idx:3d}/{len(cases)}] {case_id}  ({category}/{difficulty}) ...",
            end=" ",
            flush=True,
        )
        try:
            result = run_inference(
                repo_path=str(worktree),
                test_identifier=seed_case["test_id"],
                model_name=None,
                max_steps=0,
                num_runs=num_runs,
                test_timeout=test_timeout,
                api_base=None,
                api_key=None,
                verbose=False,
            )
            reason = result.get("done_reason", "")
            pass_rate = result.get("final_pass_rate", 0.0)
            entry = {
                "case_id": case_id,
                "flake_category": category,
                "idoft_category": seed_case.get("idoft_category", ""),
                "difficulty": difficulty,
                "test_id": seed_case["test_id"],
                "done_reason": reason,
                "pass_rate": pass_rate,
            }
            if "flaky" in reason or (0.05 < pass_rate < 0.95):
                bucket = flaky
                label = f"FLAKY  pass_rate={pass_rate:.2f}"
            elif "stable" in reason or pass_rate >= 0.95:
                bucket = stable
                label = f"stable pass_rate={pass_rate:.2f}"
            else:
                bucket = broken
                label = f"BROKEN reason={reason}"
            bucket.append(entry)
            print(label, flush=True)
        except Exception as exc:
            broken.append({
                "case_id": case_id,
                "flake_category": category,
                "difficulty": difficulty,
                "test_id": seed_case["test_id"],
                "error": str(exc),
            })
            print(f"ERROR  {exc}", flush=True)

    print(
        f"\n=== Probe complete: {len(flaky)} flaky / {len(stable)} stable / {len(broken)} broken ===",
        flush=True,
    )
    if flaky:
        print("\nFLAKY cases (ready for training):")
        for c in flaky:
            print(f"  --case {c['case_id']}  ({c['flake_category']}/{c['difficulty']})  pass_rate={c['pass_rate']:.2f}")

    return {
        "seed_root": str(Path(seed_root).resolve()),
        "total": len(cases),
        "flaky": flaky,
        "stable": stable,
        "broken": broken,
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
        "--probe-all",
        action="store_true",
        help=(
            "Quick-scan every case under --seed-root (--max-steps 0) and print a table "
            "of which cases are genuinely flaky vs stable/broken."
        ),
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Patch seed repos in place instead of copying to outputs/inference_repos first",
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"), help="LLM model name")
    parser.add_argument("--max-steps", type=int, default=None, help="Max episode steps")
    parser.add_argument("--num-runs", type=int, default=int(os.environ.get("NUM_RUNS", 10)), help="Repeated test runs per step")
    parser.add_argument(
        "--test-timeout",
        type=int,
        default=None,
        help="Seconds allowed for each pytest/docker run (default: FF_TEST_TIMEOUT_SECONDS or 30)",
    )
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_API_BASE"), help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")
    args = parser.parse_args()

    try:
        if args.seed_root or args.list_cases or args.probe_all:
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
            elif args.probe_all:
                result = _probe_all_cases(
                    seed_root=seed_root,
                    case=args.case,
                    limit=args.limit,
                    num_runs=args.num_runs,
                    test_timeout=args.test_timeout,
                    verbose=not args.quiet,
                )
            else:
                result = run_seed_inference(
                    seed_root=seed_root,
                    case=args.case,
                    limit=args.limit,
                    isolate=not args.no_isolation,
                    model_name=args.model,
                    max_steps=args.max_steps,
                    num_runs=args.num_runs,
                    test_timeout=args.test_timeout,
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
                test_timeout=args.test_timeout,
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
