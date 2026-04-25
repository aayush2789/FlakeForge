"""V3 Inference Loop — unified agent with verifiable reward.

Replaces the V2 two-phase inference (analyze → fix) with a single
unified loop: observe → think+patch → apply → verify → reward.

No judge calls. No hypothesis gating. Just execution-verified reward.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from models import FlakeForgeAction, FlakeForgeObservation
    from agent.unified_agent import UnifiedFlakeForgeAgent, build_unified_prompt
    from server.FlakeForge_environment import FlakeForgeEnvironment
except ImportError:
    from .models import FlakeForgeAction, FlakeForgeObservation
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
        self.temperature = float(temperature or os.environ.get("TEMPERATURE", 0.7))

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
            logger.error("[INFERENCE] LLM call failed: %s", exc)
            return f"<think>\nRoot Cause: unknown (confidence: 0.1)\nLLM call failed: {exc}\nStrategy: Unable to generate fix.\n</think>\n<patch>\n</patch>"


async def run_episode(
    env: FlakeForgeEnvironment,
    agent: UnifiedFlakeForgeAgent,
    verbose: bool = True,
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
    step_output = env.reset()
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
        step_output = env.step(action)
        observation = step_output.observation

        # Track trajectory
        step_data = {
            "step": step_output.state.step_count,
            "predicted_category": action.predicted_category,
            "predicted_confidence": action.predicted_confidence,
            "think_text": action.think_text[:500],
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
                "[EPISODE] RESULT step=%d reward=%.4f pass_rate=%.2f→%.2f done=%s reason=%s",
                step_data["step"],
                step_output.reward,
                observation.baseline_pass_rate,
                step_output.state.current_pass_rate,
                step_output.done,
                step_output.info.get("done_reason", ""),
            )

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
    # Create environment
    env = FlakeForgeEnvironment(
        repo_path=repo_path,
        test_identifier=test_identifier,
        max_steps=max_steps,
        num_runs=num_runs,
    )

    # Create LLM backend
    backend = LLMBackend(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
    )

    # Create unified agent
    agent = UnifiedFlakeForgeAgent(backend=backend)

    # Run episode
    try:
        result = asyncio.run(run_episode(env, agent, verbose=verbose))
    except RuntimeError:
        # If event loop is already running (e.g., in Jupyter)
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(run_episode(env, agent, verbose=verbose))

    if verbose:
        logger.info(
            "[INFERENCE] COMPLETE steps=%d total_reward=%.4f final_pass_rate=%.2f reason=%s",
            result["steps"],
            result["total_reward"],
            result["final_pass_rate"],
            result["done_reason"],
        )

    return result


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
