"""V3 GRPO Trainer — Group Relative Policy Optimization for FlakeForge.

Uses TRL's GRPOTrainer with a **verifiable offline** reward (format + consistency).

**Execution / env rewards and stock GRPOTrainer — incompatible**

TRL's ``GRPOTrainer`` calls the reward with *all G completions in one batch* and
expects one scalar per completion. A naive ``env.step()`` loop would call ``step`` G
times on the *same* episode **without** ``reset()`` between completions, stacking G
patches on one repo. **Do not** wire ``FlakeForgeEnvironment.step``-based rewards into
``GRPOTrainer`` without a custom trainer that does ``reset()`` + one ``step()`` per
completion.

**Path A (recommended for warm-up):** ``create_trainer(use_execution=False)`` and a
``prompt`` dataset — offline reward only, stable and fast.

**Path B (live env):** use :func:`run_episode` / :func:`build_grpo_batch` inside your
own loop. ``run_episode`` calls ``env.reset()`` at the start of each rollout; ensure
:meth:`FlakeForgeEnvironment.reset` restores the repo (pristine snapshot). You must
still implement the policy / GRPO loss and ``backward()`` — :func:`build_grpo_batch`
only returns advantages; it does not run the optimizer.

Key design decisions (aligned with the training notebook):
- Qwen/Qwen2.5-7B-Instruct as the base model (strong code reasoning)
- LoRA r=64, α=128 across attention + FFN (keeps trainable params ~1%)
- Flash Attention 2 for 2× memory efficiency
- bfloat16 over float16 for RL training stability
- GRPO group size G=8: 8 rollouts per prompt, baseline = group mean
- KL coefficient 0.04 (increase → 0.1 if reward collapses; decrease → 0.01 if too slow)
- DeepSpeed ZeRO-2 by default for LoRA 7B (ZeRO-3 optional if VRAM tight)
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
    """Build the reward function for **TRL GRPOTrainer** (offline only).

    Args:
        use_execution: Must stay ``False`` for stock ``GRPOTrainer``. Execution-based
            rewards need ``env.step()`` once per completion with a fresh tree; use
            :func:`run_episode` / a custom training loop instead (see module docstring).

    Returns:
        A reward function compatible with TRL's ``GRPOTrainer``.

    Raises:
        ValueError: if ``use_execution`` is True — prevents silently broken training.
    """
    if use_execution:
        raise ValueError(
            "use_execution=True is incompatible with TRL's GRPOTrainer: the trainer "
            "scores a group of G completions in one call without env.reset() between "
            "them, so patches would stack. Use use_execution=False for TRL, or use "
            "run_episode / build_grpo_batch with a custom loop and FlakeForgeEnvironment "
            "with pristine reset (see FlakeForgeEnvironment._pristine_file_snapshots)."
        )
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
    """[Experimental] Per-completion env.step reward — **not** for TRL's GRPOTrainer.

    A single call handles *len(completions)* items; a correct integration must call
    ``env.reset()`` before *each* ``env.step()``. Stock ``GRPOTrainer`` does not do
    that, so it must not use this. Prefer :func:`run_episode` (resets every rollout).
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
    """Run a single RL episode: **reset** → observe → generate → step → reward.

    Calls ``env.reset()`` at the start of **every** rollout so each completion is
    scored on a clean repo (given ``FlakeForgeEnvironment`` pristine restore).

    Args:
        model: Loaded (LoRA-wrapped) causal LM.
        tokenizer: Tokenizer with special tokens added.
        env: A ``FlakeForgeEnvironment`` instance.
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
    """Generate G rollouts for one GRPO group (manual / custom training loop **only**).

    **This does not call ``loss.backward()`` or ``optimizer.step()``** — you must
    compute the GRPO / policy loss from ``advantages`` and the model, then backprop
    in your own trainer. :func:`run_episode` is invoked G times; each call
    ``env.reset()``s first, so the repo is not a stack of G patches (requires env
    pristine restore in :meth:`FlakeForgeEnvironment.reset`).

    Group advantage (for reference; TRL may normalize differently)::
        A_g = (r_g − mean(r)) / (std(r) + ε)

    **reward_clip** default 15.0 so terminal bonuses (e.g. +4) from the env are not
    cut off; tune via ``config["reward_clip"]``.

    Args:
        model: LoRA-wrapped causal LM.
        tokenizer: Tokenizer.
        env: ``FlakeForgeEnvironment`` (or compatible).
        config: May include ``grpo_group_size`` (G), ``reward_clip`` (default 15.0),
            ``normalize_rewards`` (default True; used only here, not in TRL).

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

    reward_clip = float(config.get("reward_clip", 15.0))
    arr = np.clip(np.array(rewards, dtype=float), -reward_clip, reward_clip)
    mean, std = arr.mean(), arr.std()
    advantages = ((arr - mean) / (std + 1e-8)).tolist() if config.get("normalize_rewards", True) else arr.tolist()

    logger.info(
        "[build_grpo_batch] G=%d mean_reward=%.3f std=%.3f max=%.3f min=%.3f",
        G, float(mean), float(std), float(arr.max()), float(arr.min()),
    )
    return prompts, completions, rewards, advantages


# ── DeepSpeed ZeRO-2/3 config ────────────────────────────────────────────────
# Prefer ZeRO-2 for 7B + LoRA: less collective overhead than stage 3; use 3 if VRAM is tight.

def generate_deepspeed_config(
    output_path: str = "configs/deepspeed_z2.json",
    *,
    zero_stage: int = 2,
) -> str:
    """Write a DeepSpeed ZeRO config and return its path.

    Args:
        output_path: Where to write JSON.
        zero_stage: 2 (default, good for LoRA) or 3 (shard params; more comms).
    """
    if zero_stage == 3:
        zconf = {
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
        }
    else:
        zconf = {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
        }
    config = {
        "zero_optimization": zconf,
        "bf16": {"enabled": True},
        "gradient_clipping": 0.5,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(
        "[TRAINER] DeepSpeed ZeRO-%d config written to %s", zero_stage, path
    )
    return str(path)


# ── Model + tokenizer factory ─────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_flash_attention: bool = True,
    device_map: Optional[str] = None,
):
    """Load Qwen-2.5-7B-Instruct with LoRA and FlakeForge special tokens.

    Args:
        model_name: HuggingFace model ID.
        use_lora: Whether to wrap with LoRA (default True; set False for eval only).
        lora_r: LoRA rank.  r=64 gives ~1% trainable params at 7B scale.
        lora_alpha: LoRA scaling factor (alpha/r = 2.0 is standard).
        lora_dropout: LoRA dropout rate.
        use_flash_attention: Enable Flash Attention 2 for 2× memory efficiency.
        device_map: Passed to from_pretrained.  Leave None for DeepSpeed; 'auto' for single-GPU.

    Returns:
        (model, tokenizer) — model has LoRA applied and embeddings resized.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("pip install torch transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",   # Required for batch left-padded generation
    )
    tokenizer.add_special_tokens(FLAKEFORGE_SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if use_flash_attention else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,     # Numerically more stable than float16 for RL
        attn_implementation=attn_impl,
        device_map=device_map,          # None = let DeepSpeed / accelerate place shards
    )
    # Required when using gradient checkpointing (DeepSpeed / TRL) — avoids re-entrant
    # cache warnings and subtle generation bugs in training.
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            raise ImportError("pip install peft") from exc

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",    # Attention
                "gate_proj", "up_proj", "down_proj",         # FFN / MLP
            ],
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    logger.info("[TRAINER] Loaded %s (lora=%s flash=%s)", model_name, use_lora, use_flash_attention)
    return model, tokenizer


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
    """Create a fully configured GRPOTrainer for FlakeForge V3 (offline reward only).

    Args:
        model_name: HuggingFace model ID (default: Qwen2.5-7B-Instruct).
        output_dir: Checkpoint output directory.
        sft_data_path: Path to a JSON/JSONL with a ``prompt`` column. If missing, a
            placeholder :class:`datasets.Dataset` is created (see log).
        use_execution: **Unsupported** — must be ``False`` (raises). Execution rewards
            belong in a custom loop with :func:`run_episode` / :func:`build_grpo_batch`.
        use_lora: Apply LoRA adapters (strongly recommended for 7B models).
        use_flash_attention: Enable Flash Attention 2.
        wandb_project: W&B project name; set to None to disable W&B.
        wandb_run_name: W&B run name; auto-generated if None.
        deepspeed_config_path: Path to DeepSpeed JSON, or ``"auto"`` for generated ZeRO-2.
        **trainer_kwargs: Override any ``GRPOConfig`` parameter. Keys that are not part
            of your TRL version's ``GRPOConfig`` are dropped with a warning.

    Returns:
        Configured GRPOTrainer instance.
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset, load_dataset
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
        # Default ZeRO-2 for 7B + LoRA; set zero_stage=3 in generate_deepspeed_config if you need it.
        deepspeed_config_path = generate_deepspeed_config(
            os.path.join(output_dir, "deepspeed_z2.json"), zero_stage=2
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

    # ── Dataset (TRL GRPO requires a non-empty train_dataset) ────────────────
    dataset = None
    if sft_data_path and Path(sft_data_path).exists():
        dataset = load_dataset("json", data_files=sft_data_path, split="train")
        logger.info(
            "[TRAINER] Loaded %d training examples from %s",
            len(dataset), sft_data_path,
        )
    if dataset is None:
        n = int(os.environ.get("FF_GRPO_MIN_DATASET_SIZE", "128"))
        placeholder = os.environ.get(
            "FF_GRPO_PROMPT_PLACEHOLDER",
            "You are FlakeForge. Fix the flaky test; respond with one JSON object with think and patch keys.",
        )
        dataset = Dataset.from_dict({"prompt": [placeholder] * n})
        logger.warning(
            "[TRAINER] No valid sft_data_path: using a placeholder Dataset (%d rows). "
            "Set sft_data_path=... or FF_GRPO_MIN_DATASET_SIZE / FF_GRPO_PROMPT_PLACEHOLDER.",
            n,
        )

    # ── Reward function ───────────────────────────────────────────────────────
    reward_fn = build_reward_function(use_execution=use_execution)

    # ── GRPOConfig — all critical knobs with research-backed defaults ────────
    # trainer_kwargs can override any of these.
    config_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,

        # Batch sizes
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,

        # Training schedule
        "num_train_epochs": 3,
        "learning_rate": 5e-6,        # LoRA + RL: slightly stronger than 1e-6; tune to loss noise
        "lr_scheduler_type": "cosine",
        "warmup_steps": 50,
        "weight_decay": 0.01,

        # GRPO group — G=8 gives stable group variance without excessive Docker I/O
        "num_generations": 8,

        # Generation (rollout)
        "max_completion_length": 2048,
        "temperature": 0.8,
        "top_p": 0.95,

        # KL / clipping
        "beta": 0.04,                   # KL penalty (increase → 0.1 if reward collapses)
        # Note: do not put reward_clip / normalize_rewards here — not standard TRL
        # GRPOConfig fields; TRL normalizes group rewards internally. For manual
        # :func:`build_grpo_batch`, set reward_clip in that config dict (default 15.0).

        # Stability
        "max_grad_norm": 0.5,           # Gradient clipping — critical for RL
        "seed": 42,

        # Logging / checkpointing
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 100,
        "report_to": "wandb" if wandb_project else "none",
    }
    config_kwargs.update(trainer_kwargs)

    # Common foot-guns: these are not part of TRL's GRPOConfig in many versions.
    for k in ("reward_clip", "normalize_rewards"):
        if k in config_kwargs:
            logger.warning(
                "[TRAINER] Removing %r from config — not a standard GRPOConfig key; "
                "TRL normalizes per-group rewards. For manual :func:`build_grpo_batch`, pass reward_clip in that dict.",
                k,
            )
            del config_kwargs[k]

    # Handle version-specific GRPOConfig param names
    try:
        grpo_config = GRPOConfig(**config_kwargs)
    except TypeError:
        # Older TRL (<0.9) may not have all params — strip unknowns gracefully
        import inspect
        valid_keys = set(inspect.signature(GRPOConfig).parameters.keys())
        filtered = {k: v for k, v in config_kwargs.items() if k in valid_keys}
        logger.warning(
            "[TRAINER] GRPOConfig dropped unsupported keys: %s",
            set(config_kwargs) - valid_keys,
        )
        grpo_config = GRPOConfig(**filtered)

    # ── DeepSpeed integration ─────────────────────────────────────────────────
    if deepspeed_config_path:
        try:
            grpo_config.deepspeed = deepspeed_config_path
        except Exception:
            pass  # Not all TRL versions expose this field

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        config=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=dataset,
    )

    logger.info(
        "[TRAINER] Created GRPOTrainer model=%s lora=%s execution=%s G=%d dataset_size=%s",
        model_name,
        use_lora,
        use_execution,
        config_kwargs["num_generations"],
        len(dataset) if dataset else "none",
    )

    return trainer
