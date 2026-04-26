"""V3 GRPO Trainer — Group Relative Policy Optimization for FlakeForge.

Uses TRL's GRPOTrainer with a multi-signal verifiable reward function.

Key design decisions (aligned with the training notebook):
- Qwen/Qwen2.5-7B-Instruct as the base model (strong code reasoning)
- LoRA r=64, α=128 across attention + FFN (keeps trainable params ~1%)
- Flash Attention 2 for 2× memory efficiency
- bfloat16 over float16 for RL training stability
- GRPO group size G=8: 8 rollouts per prompt, baseline = group mean
- KL coefficient 0.04 (increase → 0.1 if reward collapses; decrease → 0.01 if too slow)
- DeepSpeed ZeRO-3 compatible (device_map=None, model placement handled externally)
- Special tokens: <think>, </think>, <patch>, </patch>, <tool_call>, </tool_call>

Research basis:
- DeepSeek-R1: GRPO without value network; group advantage via reward normalization
- Open-RS / RLEF: Execution-verified reward for code generation tasks
- LoRA (Hu 2021): Low-rank adaptation at ~1% parameter overhead
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from models import FlakeForgeAction
except ImportError:
    try:
        from ..models import FlakeForgeAction
    except (ImportError, ValueError):
        try:
            from FlakeForge.models import FlakeForgeAction
        except ImportError:
            FlakeForgeAction = None  # type: ignore[assignment,misc]

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except (ImportError, ValueError):
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__)


# ── Special tokens injected into the tokenizer ───────────────────────────────

FLAKEFORGE_SPECIAL_TOKENS: Dict[str, List[str]] = {
    "additional_special_tokens": [
        "<think>", "</think>",
        "<patch>", "</patch>",
        "<tool_call>", "</tool_call>",
        "<tool_result>", "</tool_result>",
    ]
}

# ── Default model ─────────────────────────────────────────────────────────────
# Qwen-2.5-7B-Instruct outperforms Llama-3.1-8B on code tasks with comparable size.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


# ── Reward functions ──────────────────────────────────────────────────────────

def build_reward_function(use_execution: bool = False):
    """Build the reward function for GRPO training.

    Args:
        use_execution: If True, includes execution-based signals (requires environment).
                      If False, uses offline format + reasoning consistency signals only.

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
) -> List[float]:
    """Offline reward: format compliance + reasoning consistency.

    Used for initial GRPO warm-up before the live Docker environment is
    available.  TRL handles group-relative normalization internally; this
    function returns raw per-completion scalars.

    Scale: roughly [−1.5, 1.5].  Negative values gate the KL penalty.
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

    rewards: List[float] = []
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

        format_score = compute_format_reward(action)          # 0.0 – 1.0
        inferred_cat = infer_category_from_patch(patch)
        consistency = compute_reasoning_consistency(           # −0.5 – 0.5
            category, inferred_cat, think, patch
        )

        # Confidence bonus: reward calibrated confidence (not always 1.0)
        confidence_bonus = 0.1 if 0.6 <= confidence <= 0.95 else 0.0

        # Penalise empty outputs hard so they don't anchor the group mean
        if not think.strip() and not patch.strip():
            rewards.append(-1.5)
            continue

        total = format_score * 1.0 + consistency * 0.5 + confidence_bonus
        rewards.append(round(total, 4))

    return rewards


def _execution_reward_fn(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> List[float]:
    """Online reward: full six-signal execution-verified reward.

    Requires a running FlakeForge environment passed via kwargs["env"].
    Falls back to offline reward if no environment is available.
    """
    from agent.unified_agent import (
        extract_think,
        extract_patch,
        extract_category_from_think,
        extract_confidence_from_think,
    )

    env = kwargs.get("env")
    if env is None:
        logger.warning("[TRAINER] No environment in kwargs — falling back to offline reward")
        return _offline_reward_fn(prompts, completions, **kwargs)

    rewards: List[float] = []
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
            rewards.append(float(step_output.reward))
        except Exception as exc:
            logger.error("[TRAINER] Execution reward failed: %s", exc)
            rewards.append(0.0)

    return rewards


# ── Single-episode rollout (used outside TRL for manual GRPO loops) ───────────

def run_episode(
    model: Any,
    tokenizer: Any,
    env: Any,
    config: Dict[str, Any],
) -> Tuple[str, str, float]:
    """Run a single RL episode: observe → generate → step → reward.

    Args:
        model: Loaded (LoRA-wrapped) causal LM.
        tokenizer: Tokenizer with special tokens added.
        env: A FlakeForgeEnvironment instance (already reset externally).
        config: Training config dict with generation hyperparameters.

    Returns:
        (prompt_text, completion_text, scalar_reward)
    """
    import torch
    from agent.unified_agent import build_unified_prompt, UnifiedFlakeForgeAgent

    observation = env.reset()
    prompt_text = build_unified_prompt(observation)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=config.get("max_prompt_length", 3072),
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.get("max_new_tokens", 2048),
            temperature=config.get("temperature", 0.8),
            top_p=config.get("top_p", 0.95),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    from agent.unified_agent import extract_think, extract_patch, extract_category_from_think, extract_confidence_from_think
    from models import FlakeForgeAction

    action = FlakeForgeAction(
        raw_response=completion_text,
        think_text=extract_think(completion_text),
        patch_text=extract_patch(completion_text),
        predicted_category=extract_category_from_think(extract_think(completion_text)),
        predicted_confidence=extract_confidence_from_think(extract_think(completion_text)),
    )

    try:
        step_obs = env.step(action)
        reward = float(step_obs.reward)
    except Exception as exc:
        logger.error("[run_episode] step failed: %s", exc)
        reward = -1.0

    return prompt_text, completion_text, reward


def build_grpo_batch(
    model: Any,
    tokenizer: Any,
    env: Any,
    config: Dict[str, Any],
) -> Tuple[List[str], List[str], List[float], List[float]]:
    """Generate G rollouts for one prompt (GRPO group).

    The GRPO advantage for each response is:
        A_g = (r_g − mean(r)) / (std(r) + ε)

    Args:
        model: LoRA-wrapped causal LM.
        tokenizer: Tokenizer.
        env: Environment instance (reset happens inside run_episode).
        config: Training config; must contain 'grpo_group_size' (G).

    Returns:
        (prompts, completions, raw_rewards, advantages)
    """
    import numpy as np

    G = config.get("grpo_group_size", 8)
    prompts, completions, rewards = [], [], []

    for _ in range(G):
        prompt, completion, reward = run_episode(model, tokenizer, env, config)
        prompts.append(prompt)
        completions.append(completion)
        rewards.append(reward)

    # Clip then normalise within group (GRPO core)
    reward_clip = config.get("reward_clip", 5.0)
    arr = np.clip(np.array(rewards, dtype=float), -reward_clip, reward_clip)
    mean, std = arr.mean(), arr.std()
    advantages = ((arr - mean) / (std + 1e-8)).tolist() if config.get("normalize_rewards", True) else rewards

    logger.info(
        "[build_grpo_batch] G=%d mean_reward=%.3f std=%.3f max=%.3f min=%.3f",
        G, mean, std, arr.max(), arr.min(),
    )
    return prompts, completions, rewards, advantages


# ── DeepSpeed ZeRO-3 config ───────────────────────────────────────────────────

def generate_deepspeed_config(output_path: str = "configs/deepspeed_z3.json") -> str:
    """Write a DeepSpeed ZeRO-3 config and return its path."""
    config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": int(1e9),
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": int(1e9),
            "stage3_max_reuse_distance": int(1e9),
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 0.5,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("[TRAINER] DeepSpeed ZeRO-3 config written to %s", path)
    return str(path)


# ── Model + tokenizer factory ─────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_flash_attention: bool = True,
    use_unsloth: bool = True,
    max_seq_length: int = 4096,
    dtype: Optional[Any] = None,
    load_in_4bit: bool = True,
):
    """Load model with Unsloth (preferred) or fallback to HF+PEFT.

    Unsloth gives 2-5x faster training, better memory efficiency, and
    native GRPO support. Uses 4-bit QLoRA by default for 7B models.
    """
    # Unsloth is now **mandatory** — no fallback allowed (per user request)
    try:
        from unsloth import FastLanguageModel
        import torch
        logger.info("[TRAINER] Loading with Unsloth (mandatory — fastest path) for %s", model_name)

        if dtype is None:
            dtype = None  # Unsloth auto-detects

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )

        # Add our special tokens for FlakeForge structured output
        tokenizer.add_special_tokens(FLAKEFORGE_SPECIAL_TOKENS)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Apply LoRA via Unsloth (much faster + better memory than PEFT)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # 30% less VRAM
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )

        logger.info("[TRAINER] Successfully loaded with Unsloth + 4-bit QLoRA (r=%d)", lora_r)
        return model, tokenizer

    except ImportError:
        raise ImportError(
            "Unsloth is now **mandatory**. Please install it with:\n"
            "    pip install --upgrade --force-reinstall --no-deps \\\n"
            "        unsloth[cu121-torch240]@git+https://github.com/unslothai/unsloth.git"
        ) from None
    except Exception as e:
        raise RuntimeError(f"Unsloth model loading failed: {e}") from e


# ── Main GRPOTrainer factory ──────────────────────────────────────────────────

def create_trainer(
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = "./outputs/flakeforge_v3",
    sft_data_path: Optional[str] = None,
    use_execution: bool = False,
    use_lora: bool = True,
    use_flash_attention: bool = True,
    wandb_project: Optional[str] = "flakeforge-rl",
    wandb_run_name: Optional[str] = None,
    deepspeed_config_path: Optional[str] = None,
    **trainer_kwargs: Any,
) -> Any:
    """Create a fully configured GRPOTrainer for FlakeForge V3.

    Args:
        model_name: HuggingFace model ID (default: Qwen2.5-7B-Instruct).
        output_dir: Checkpoint output directory.
        sft_data_path: Path to a JSONL file produced by data_generator.py.
        use_execution: Use live environment reward (True) or offline format reward (False).
        use_lora: Apply LoRA adapters (strongly recommended for 7B models).
        use_flash_attention: Enable Flash Attention 2.
        wandb_project: W&B project name; set to None to disable W&B.
        wandb_run_name: W&B run name; auto-generated if None.
        deepspeed_config_path: Path to DeepSpeed JSON; auto-generated if provided path
            is "auto".
        **trainer_kwargs: Override any GRPOConfig parameter.

    Returns:
        Configured GRPOTrainer instance.
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "FlakeForge V3 training requires: pip install trl transformers datasets peft"
        ) from exc

    # ── W&B ──────────────────────────────────────────────────────────────────
    if wandb_project:
        try:
            import wandb
            grpo_group_size = trainer_kwargs.get("num_generations", 8)
            run_name = wandb_run_name or f"grpo-{model_name.split('/')[-1]}-G{grpo_group_size}"
            wandb.init(project=wandb_project, name=run_name, config=trainer_kwargs)
            logger.info("[TRAINER] W&B initialized: project=%s run=%s", wandb_project, run_name)
        except ImportError:
            logger.warning("[TRAINER] wandb not installed — skipping W&B init")
        except Exception as exc:
            logger.warning("[TRAINER] W&B init failed (non-fatal): %s", exc)

    # ── DeepSpeed ────────────────────────────────────────────────────────────
    if deepspeed_config_path == "auto":
        deepspeed_config_path = generate_deepspeed_config(
            os.path.join(output_dir, "deepspeed_z3.json")
        )

    # ── Model + tokenizer ────────────────────────────────────────────────────
    # For DeepSpeed multi-GPU, device_map must be None.
    device_map = None if deepspeed_config_path else "auto"
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        use_lora=use_lora,
        use_flash_attention=use_flash_attention,
        device_map=device_map,
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = None
    if sft_data_path and Path(sft_data_path).exists():
        dataset = load_dataset("json", data_files=sft_data_path, split="train")
        logger.info(
            "[TRAINER] Loaded %d training examples from %s",
            len(dataset), sft_data_path,
        )

    # ── Reward function ───────────────────────────────────────────────────────
    reward_fn = build_reward_function(use_execution=use_execution)

    # Unsloth GRPOTrainer is now **mandatory** (no TRL fallback)
    try:
        from unsloth import GRPOTrainer as UnslothGRPOTrainer
        logger.info("[TRAINER] Using Unsloth GRPOTrainer (mandatory — 2-5x faster, optimized for Qwen2.5)")
    except ImportError:
        raise ImportError(
            "Unsloth is now **mandatory** for training. Please install with:\n"
            "    pip install --upgrade --force-reinstall --no-deps \\\n"
            "        unsloth[cu121-torch240]@git+https://github.com/unslothai/unsloth.git"
        ) from None

    # Unsloth GRPOTrainer
    trainer = UnslothGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=None,  # Unsloth uses its own defaults + trainer_kwargs
        train_dataset=dataset,
        reward_func=reward_fn,
        max_length=4096,
        temperature=0.8,
        top_p=0.95,
        num_generations=8,
        **trainer_kwargs,
    )

    logger.info(
        "[TRAINER] Created UnslothGRPOTrainer model=%s 4bit=%s G=8 dataset_size=%s",
        model_name,
        getattr(model, "is_loaded_in_4bit", False),
        len(dataset) if dataset else "none",
    )

    return trainer
