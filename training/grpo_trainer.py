"""V3 GRPO Trainer — Group Relative Policy Optimization for FlakeForge.

Uses TRL's GRPOTrainer with a single verifiable reward function
that combines format compliance + reasoning consistency signals.

For online training with execution feedback, the full environment
step should be used (FlakeForgeEnvironment.step()), which adds
stability delta, compile check, causal proximity, entropy reduction,
and anti-hack penalty signals.

Research basis:
- DeepSeek R1: GRPO with group-relative advantage estimation
- Open-RS: Verifiable reward without critic model
- RLEF: Execution-verified reward for code generation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from models import FlakeForgeAction
except ImportError:
    from ..models import FlakeForgeAction

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__)


def build_reward_function(use_execution: bool = False):
    """Build the reward function for GRPO training.

    Args:
        use_execution: If True, includes execution-based signals (requires environment).
                      If False, uses offline format+consistency signals only.

    Returns:
        A reward function compatible with TRL's GRPOTrainer.
    """
    if use_execution:
        return _execution_reward_fn
    return _offline_reward_fn


def _offline_reward_fn(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> list:
    """Offline reward: format + reasoning consistency only.

    Used for initial GRPO training before online environment is available.
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
        text = completion if isinstance(completion, str) else str(completion)

        think = extract_think(text)
        patch = extract_patch(text)
        category = extract_category_from_think(think)
        confidence = extract_confidence_from_think(think)

        action = FlakeForgeAction(
            raw_response=text,
            think_text=think,
            patch_text=patch,
            predicted_category=category,
            predicted_confidence=confidence,
        )

        format_score = compute_format_reward(action)
        inferred_cat = infer_category_from_patch(patch)
        consistency = compute_reasoning_consistency(
            category, inferred_cat, think, patch
        )

        total = format_score * 1.0 + consistency * 0.5
        rewards.append(total)

    return rewards


def _execution_reward_fn(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> list:
    """Online reward: full six-signal execution-verified reward.

    Requires a running FlakeForge environment. Each completion is
    applied as a patch and tested.
    """
    from agent.unified_agent import (
        extract_think,
        extract_patch,
        extract_category_from_think,
        extract_confidence_from_think,
    )

    # Get environment from kwargs
    env = kwargs.get("env")
    if env is None:
        logger.warning("[TRAINER] No environment provided — falling back to offline reward")
        return _offline_reward_fn(prompts, completions, **kwargs)

    rewards = []
    for prompt, completion in zip(prompts, completions):
        text = completion if isinstance(completion, str) else str(completion)

        think = extract_think(text)
        patch = extract_patch(text)

        action = FlakeForgeAction(
            raw_response=text,
            think_text=think,
            patch_text=patch,
            predicted_category=extract_category_from_think(think),
            predicted_confidence=extract_confidence_from_think(think),
        )

        try:
            step_output = env.step(action)
            rewards.append(step_output.reward)
        except Exception as exc:
            logger.error("[TRAINER] Execution reward failed: %s", exc)
            rewards.append(0.0)

    return rewards


def create_trainer(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_dir: str = "./outputs/flakeforge_v3",
    sft_data_path: Optional[str] = None,
    use_execution: bool = False,
    **trainer_kwargs: Any,
) -> Any:
    """Create a GRPOTrainer for FlakeForge V3.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Output directory for checkpoints
        sft_data_path: Path to SFT JSONL data for initial training
        use_execution: Whether to use execution-verified reward
        **trainer_kwargs: Additional kwargs for GRPOTrainer

    Returns:
        Configured GRPOTrainer instance
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "FlakeForge V3 training requires: pip install trl transformers datasets"
        ) from exc

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    # Load dataset
    dataset = None
    if sft_data_path and Path(sft_data_path).exists():
        dataset = load_dataset("json", data_files=sft_data_path, split="train")
        logger.info("[TRAINER] Loaded %d training examples from %s", len(dataset), sft_data_path)

    # Build reward function
    reward_fn = build_reward_function(use_execution=use_execution)

    # Configure training
    config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=trainer_kwargs.pop("batch_size", 4),
        num_train_epochs=trainer_kwargs.pop("num_epochs", 3),
        learning_rate=trainer_kwargs.pop("learning_rate", 5e-6),
        logging_steps=trainer_kwargs.pop("logging_steps", 10),
        save_steps=trainer_kwargs.pop("save_steps", 100),
        max_completion_length=trainer_kwargs.pop("max_completion_length", 4096),
        num_generations=trainer_kwargs.pop("num_generations", 4),
        **trainer_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        config=config,
        reward_funcs=reward_fn,
        train_dataset=dataset,
    )

    logger.info(
        "[TRAINER] Created GRPOTrainer model=%s execution=%s dataset_size=%s",
        model_name,
        use_execution,
        len(dataset) if dataset else "none",
    )

    return trainer
