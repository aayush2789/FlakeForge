"""GRPO Trainer for FlakeForge — Unsloth + TRL warm-up + custom online loop.

Two complementary entry points share one Unsloth-loaded LoRA model:

**Phase 1 (warm-up) — :func:`create_trainer`**
    Stock TRL ``GRPOTrainer`` with an offline reward (format + reasoning
    consistency). The model is loaded by Unsloth (4-bit QLoRA, fast attention)
    and then handed to TRL's trainer untouched. This stabilises JSON output
    format on real prompts pulled from the curated IDoFT manifests; no Docker,
    no pytest, no env reset. Fast (~5 min / 100 steps on a single A100 LoRA).

**Phase 2 (online RL) — :class:`OnlineGRPOLoop`**
    Custom GRPO loop that drives :class:`FlakeForgeEnvironment` per rollout:

        for episode:
            case = curriculum.sample()
            env  = make_env(case)
            obs0 = env.reset()                  # baseline pass-rate, preflight
            prompt = build_unified_prompt(obs0)
            seqs = model.generate(prompt, num_return_sequences=G)
            for g in range(G):
                env.reset()                     # pristine snapshot restores tree
                action = decode(seqs[g])
                obs    = env.step(action)
                rewards[g] = obs.reward         # full multi-signal reward
            advantages = (rewards - rewards.mean()) / (rewards.std() + eps)
            policy_logp = model(seqs).logp(completion_tokens)
            with model.disable_adapter():
                ref_logp = model(seqs).logp(completion_tokens)
            kl   = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
            loss = -(advantages * seq_logp).mean() + beta * kl.mean()
            loss.backward(); optimizer.step()

    The pristine snapshot in :meth:`FlakeForgeEnvironment.reset` is what makes
    G rollouts on one repo safe — without it, patches would stack.

**Why we don't put env rewards inside TRL's GRPOTrainer:** TRL scores all G
completions in *one* reward call, with no env reset between them. Chaining G
real ``env.step()`` calls there would apply G patches to the same tree.

Defaults:
- ``Qwen/Qwen2.5-Coder-7B-Instruct`` (chosen for code patches)
- LoRA r=64, alpha=128 over attn+FFN
- 4-bit QLoRA via Unsloth
- KL coefficient 0.04 (raise to 0.1 if reward collapses)
- GRPO group size G=8

Research basis:
- DeepSeek-R1 / DeepSeekMath: GRPO with group-mean baseline, no value head
- TRL GRPOTrainer documentation (offline reward path)
- Unsloth: 2-5x memory-efficient LoRA training for 7B class
- Hu 2021 (LoRA), Ng/Harada/Russell 1999 (potential-based shaping in reward.py)
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
# Qwen2.5-Coder-7B-Instruct: best open 7B for code patches as of Q1 2026.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"


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
    use_unsloth: bool = True,
    max_seq_length: int = 4096,
    dtype: Optional[Any] = None,
    load_in_4bit: bool = True,
    device_map: Optional[Any] = None,
):
    """Load (Unsloth-patched) model + tokenizer with LoRA adapters attached.

    Unsloth gives 2-5x faster training and native GRPO compatibility with TRL.
    Uses 4-bit QLoRA by default for 7B models. ``device_map`` is accepted for
    DeepSpeed compatibility (must be ``None`` under DeepSpeed) and otherwise
    forwarded to ``FastLanguageModel.from_pretrained`` only when supplied
    explicitly.
    """
    del use_flash_attention, use_unsloth  # accepted for API compat; Unsloth handles both
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is required for FlakeForge training. Install with:\n"
            "    pip install --upgrade --force-reinstall --no-deps \\\n"
            "        \"unsloth[cu121-torch240]@git+https://github.com/unslothai/unsloth.git\""
        ) from exc

    logger.info("[TRAINER] Loading %s via Unsloth (4bit=%s, max_seq=%d)", model_name, load_in_4bit, max_seq_length)

    from_pretrained_kwargs: Dict[str, Any] = dict(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    if device_map is not None:
        from_pretrained_kwargs["device_map"] = device_map

    model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)

    tokenizer.add_special_tokens(FLAKEFORGE_SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    if use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        logger.info("[TRAINER] Attached LoRA r=%d alpha=%d", lora_r, lora_alpha)

    return model, tokenizer


# ── Main GRPOTrainer factory ──────────────────────────────────────────────────

def create_trainer(
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = "./outputs/flakeforge_v3",
    sft_data_path: Optional[str] = None,
    seed_root: Optional[str] = "seed_repos/idoft",
    use_execution: bool = False,
    use_lora: bool = True,
    use_flash_attention: bool = True,
    wandb_project: Optional[str] = "flakeforge-rl",
    wandb_run_name: Optional[str] = None,
    deepspeed_config_path: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    num_generations: int = 8,
    max_prompt_length: int = 3072,
    max_completion_length: int = 1024,
    learning_rate: float = 1e-5,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    **trainer_kwargs: Any,
) -> Any:
    """Build a TRL ``GRPOTrainer`` configured for the FlakeForge warm-up phase.

    The model is loaded by Unsloth (so we get 4-bit QLoRA + fast attention) and
    then handed to TRL untouched — TRL's ``GRPOTrainer`` works directly on the
    Unsloth-patched module.

    Args:
        model_name: HuggingFace model ID (default: Qwen2.5-Coder-7B-Instruct).
        output_dir: Checkpoint output directory.
        sft_data_path: Optional JSON/JSONL with a ``prompt`` column. If missing,
            we build a real prompt dataset from ``seed_root``.
        seed_root: Root containing IDoFT-style ``*/flake_manifest.json`` files
            used to construct warm-up prompts when ``sft_data_path`` is absent.
        use_execution: Must stay ``False`` for TRL's ``GRPOTrainer``. Online
            execution reward lives in :class:`OnlineGRPOLoop`.
        use_lora: Apply LoRA adapters (strongly recommended for 7B models).
        use_flash_attention: Forwarded to Unsloth (kept for API compat).
        wandb_project: W&B project name; set to None to disable W&B.
        wandb_run_name: W&B run name; auto-generated if None.
        deepspeed_config_path: Path to DeepSpeed JSON, or ``"auto"`` for ZeRO-2.
        model / tokenizer: Reuse an already-loaded pair (e.g. when called from
            :mod:`training.train_grpo` so warm-up and online phases share weights).
        num_generations: GRPO group size G.
        **trainer_kwargs: Extra ``GRPOConfig`` overrides. Unknown keys are
            dropped with a warning so older TRL versions don't crash.

    Returns:
        ``trl.GRPOTrainer`` ready for ``trainer.train()``.
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise ImportError(
            "FlakeForge training requires: pip install trl transformers datasets peft"
        ) from exc

    if wandb_project:
        try:
            import wandb
            run_name = wandb_run_name or f"grpo-warmup-{model_name.split('/')[-1]}-G{num_generations}"
            if wandb.run is None:
                wandb.init(project=wandb_project, name=run_name, config={
                    "phase": "warmup",
                    "model": model_name,
                    "G": num_generations,
                    **trainer_kwargs,
                })
            logger.info("[TRAINER] W&B run: project=%s run=%s", wandb_project, wandb.run.name if wandb.run else run_name)
        except ImportError:
            logger.warning("[TRAINER] wandb not installed — skipping W&B init")
        except Exception as exc:
            logger.warning("[TRAINER] W&B init failed (non-fatal): %s", exc)

    if deepspeed_config_path == "auto":
        deepspeed_config_path = generate_deepspeed_config(
            os.path.join(output_dir, "deepspeed_z2.json"), zero_stage=2
        )

    device_map = None if deepspeed_config_path else "auto"
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            use_lora=use_lora,
            use_flash_attention=use_flash_attention,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device_map=device_map,
        )

    dataset = None
    if sft_data_path and Path(sft_data_path).exists():
        dataset = load_dataset("json", data_files=sft_data_path, split="train")
        logger.info("[TRAINER] Loaded %d examples from %s", len(dataset), sft_data_path)

    if dataset is None and seed_root:
        try:
            from .data_generator import build_prompt_dataset_from_idoft
        except ImportError:
            from data_generator import build_prompt_dataset_from_idoft  # type: ignore[no-redef]
        try:
            dataset = build_prompt_dataset_from_idoft(seed_root=seed_root)
            logger.info("[TRAINER] Built warm-up dataset from %s: %d rows", seed_root, len(dataset))
        except Exception as exc:
            logger.warning("[TRAINER] Could not build dataset from %s: %s", seed_root, exc)

    if dataset is None:
        n = int(os.environ.get("FF_GRPO_MIN_DATASET_SIZE", "64"))
        placeholder = os.environ.get(
            "FF_GRPO_PROMPT_PLACEHOLDER",
            "You are FlakeForge. Fix the flaky test; respond with one JSON object "
            "with think and patch keys per the unified schema.",
        )
        dataset = Dataset.from_dict({"prompt": [placeholder] * n})
        logger.warning(
            "[TRAINER] Falling back to placeholder Dataset (%d rows). Set sft_data_path or "
            "ensure %s contains */flake_manifest.json files.", n, seed_root,
        )

    reward_fn = build_reward_function(use_execution=use_execution)

    grpo_kwargs: Dict[str, Any] = dict(
        output_dir=output_dir,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        learning_rate=learning_rate,
        per_device_train_batch_size=trainer_kwargs.pop("per_device_train_batch_size", 1),
        gradient_accumulation_steps=trainer_kwargs.pop("gradient_accumulation_steps", 4),
        bf16=trainer_kwargs.pop("bf16", True),
        logging_steps=trainer_kwargs.pop("logging_steps", 1),
        save_steps=trainer_kwargs.pop("save_steps", 50),
        report_to=["wandb"] if wandb_project else "none",
        beta=trainer_kwargs.pop("beta", 0.04),
        temperature=trainer_kwargs.pop("temperature", 0.8),
        top_p=trainer_kwargs.pop("top_p", 0.95),
    )
    if deepspeed_config_path:
        grpo_kwargs["deepspeed"] = deepspeed_config_path
    max_steps = trainer_kwargs.pop("max_steps", None)
    if max_steps is not None:
        grpo_kwargs["max_steps"] = max_steps
    grpo_kwargs.update(trainer_kwargs)

    try:
        config = GRPOConfig(**grpo_kwargs)
    except TypeError as exc:
        bad = str(exc)
        logger.warning("[TRAINER] GRPOConfig rejected some kwargs (%s); retrying with conservative subset.", bad)
        safe = {k: v for k, v in grpo_kwargs.items() if k in {
            "output_dir", "num_generations", "learning_rate", "logging_steps",
            "save_steps", "bf16", "report_to", "max_steps",
        }}
        config = GRPOConfig(**safe)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    logger.info(
        "[TRAINER] Built TRL GRPOTrainer model=%s 4bit=%s G=%d dataset=%d",
        model_name,
        getattr(model, "is_loaded_in_4bit", False),
        num_generations,
        len(dataset) if dataset is not None else 0,
    )
    return trainer


# ── Online GRPO loop with FlakeForge environment reward ─────────────────────

class OnlineGRPOLoop:
    """Custom GRPO loop that drives FlakeForgeEnvironment for live reward.

    One ``step()`` call:
      1. Sample a curriculum case (repo + flaky test).
      2. Build a fresh ``FlakeForgeEnvironment`` and ``reset()`` it (preflight,
         pristine snapshot of the source tree).
      3. Tokenize the unified prompt and generate G completions in one batched
         ``model.generate`` call.
      4. For each completion: ``env.reset()`` (restore pristine tree) then
         ``env.step(action)``; collect ``observation.reward`` (full multi-signal
         reward already computed inside the env).
      5. Compute group-relative advantages
         ``A_g = (r_g - mean(r)) / (std(r) + eps)``.
      6. Compute per-token logprobs of the generated tokens with the policy
         (LoRA-active) and a frozen reference (LoRA disabled via PEFT
         ``disable_adapter`` context). The KL term uses the DeepSeek-R1 /
         TRL approximation ``exp(log_ratio) - log_ratio - 1``.
      7. Loss = ``-(advantages * seq_logp).mean() + beta * kl.mean()``,
         then ``loss.backward()`` and ``optimizer.step()``.

    Designed to share an Unsloth-loaded LoRA model with :func:`create_trainer`
    so warm-up and online phases continue training the same adapter weights.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        curriculum: Any,
        *,
        output_dir: str = "outputs/flakeforge-online",
        group_size: int = 8,
        max_prompt_length: int = 3072,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        kl_beta: float = 0.04,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        num_runs: int = 6,
        max_env_steps: int = 1,
        use_chat_template: bool = True,
        compact_prompt: bool = True,
        system_prompt: Optional[str] = None,
        runner_factory: Optional[Any] = None,
        wandb_run: Any = None,
        log_every: int = 1,
        checkpoint_every: int = 50,
        env_preflight_quick_runs: int = 3,
        env_preflight_confirm_runs: int = 3,
    ) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.curriculum = curriculum
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.G = int(group_size)
        self.max_prompt_length = int(max_prompt_length)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.kl_beta = float(kl_beta)
        self.grad_clip = float(grad_clip)
        self.num_runs = int(num_runs)
        self.max_env_steps = int(max_env_steps)
        self.use_chat_template = bool(use_chat_template)
        self.compact_prompt = bool(compact_prompt)
        self.runner_factory = runner_factory
        self.wandb_run = wandb_run
        self.log_every = int(log_every)
        self.checkpoint_every = int(checkpoint_every)
        self.env_preflight_quick_runs = int(env_preflight_quick_runs)
        self.env_preflight_confirm_runs = int(env_preflight_confirm_runs)

        # Small-model friendly system prompt (matches training/train_grpo_tinker.py)
        self.system_prompt = system_prompt or _COMPACT_SYSTEM_PROMPT

        self.device = next(model.parameters()).device

        params = [p for p in self.model.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(params, lr=learning_rate, weight_decay=weight_decay)
            logger.info("[ONLINE] Using bitsandbytes AdamW8bit optimizer (lr=%g)", learning_rate)
        except Exception:
            self.optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
            logger.info("[ONLINE] Using torch.optim.AdamW (lr=%g)", learning_rate)

        self.episode = 0
        self._wb_step = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def train(self, max_episodes: int) -> Dict[str, Any]:
        """Run ``max_episodes`` online GRPO steps."""
        history: List[Dict[str, Any]] = []
        for ep in range(max_episodes):
            self.episode = ep + 1
            try:
                metrics = self.step()
            except Exception as exc:
                logger.exception("[ONLINE] Episode %d crashed: %s", self.episode, exc)
                metrics = {"episode": self.episode, "error": f"{type(exc).__name__}: {exc}"}

            history.append(metrics)
            if "error" not in metrics and (self.episode % self.log_every == 0):
                logger.info(
                    "[ONLINE] ep=%d stage=%s reward(mean=%.3f std=%.3f max=%.3f) loss=%.4f kl=%.4f",
                    self.episode,
                    metrics.get("stage", "?"),
                    metrics.get("mean_reward", 0.0),
                    metrics.get("std_reward", 0.0),
                    metrics.get("max_reward", 0.0),
                    metrics.get("loss", 0.0),
                    metrics.get("kl", 0.0),
                )

            if self.checkpoint_every and self.episode % self.checkpoint_every == 0:
                self.save_checkpoint(self.output_dir / "checkpoints" / f"ep{self.episode}")

        return {"episodes": history, "last_episode": self.episode}

    def step(self) -> Dict[str, Any]:
        """Run a single GRPO online episode and return its metrics."""
        import torch

        case = self.curriculum.sample()
        if case is None:
            return {"episode": self.episode, "skipped": True, "reason": "empty_curriculum"}

        env = self._make_env(case)
        try:
            obs0 = env.reset(
                preflight_quick_runs=self.env_preflight_quick_runs,
                preflight_confirm_runs=self.env_preflight_confirm_runs,
            )
        except Exception as exc:
            logger.warning("[ONLINE] env.reset failed for %s: %s", case.get("case_id"), exc)
            return {"episode": self.episode, "skipped": True, "reason": f"reset_failed:{exc}"}

        if not getattr(obs0, "should_train", True):
            return {
                "episode": self.episode,
                "skipped": True,
                "reason": getattr(obs0, "done_reason", "preflight_rejected"),
                "case_id": case.get("case_id"),
            }

        prompt_text = self._build_prompt(obs0)
        prompt_ids = self._tokenize_prompt(prompt_text)
        prompt_len = prompt_ids.shape[1]

        sequences = self._generate_group(prompt_ids)

        rewards: List[float] = []
        completions: List[str] = []
        for g in range(self.G):
            comp_ids = sequences[g, prompt_len:]
            comp_text = self.tokenizer.decode(comp_ids, skip_special_tokens=False)
            completions.append(comp_text)

            try:
                env.reset(
                    preflight_quick_runs=self.env_preflight_quick_runs,
                    preflight_confirm_runs=self.env_preflight_confirm_runs,
                )
                action = self._completion_to_action(comp_text)
                step_obs = env.step(action)
                rewards.append(float(getattr(step_obs, "reward", 0.0)))
            except Exception as exc:
                logger.warning("[ONLINE] env.step crashed (rollout %d): %s", g, exc)
                rewards.append(-1.0)

        loss_metrics = self._policy_update(sequences, prompt_len, rewards)

        max_reward = max(rewards) if rewards else 0.0
        self.curriculum.record(max_reward)

        metrics: Dict[str, Any] = {
            "episode": self.episode,
            "case_id": case.get("case_id"),
            "stage": self.curriculum.current_stage.name,
            "mean_reward": float(sum(rewards) / max(len(rewards), 1)),
            "max_reward": float(max_reward),
            "min_reward": float(min(rewards)) if rewards else 0.0,
            "std_reward": float(_std(rewards)),
            **loss_metrics,
            "completion_chars_avg": int(sum(len(c) for c in completions) / max(len(completions), 1)),
        }
        self._maybe_wandb_log(metrics)
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except Exception as exc:
            logger.warning("[ONLINE] save_pretrained failed (%s); saving state_dict instead", exc)
            import torch
            torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path / "ckpt.pt")
        try:
            (path / "curriculum_state.json").write_text(
                json.dumps(self.curriculum.state_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[ONLINE] Could not write curriculum state: %s", exc)
        logger.info("[ONLINE] Checkpoint saved -> %s", path)

    # ── Internals ──────────────────────────────────────────────────────────

    def _make_env(self, case: Dict[str, Any]) -> Any:
        try:
            from server.FlakeForge_environment import FlakeForgeEnvironment
            from server.docker_runner import DockerTestRunner
        except ImportError:
            from ..server.FlakeForge_environment import FlakeForgeEnvironment
            from ..server.docker_runner import DockerTestRunner

        repo_path = str(case["repo_path"])
        test_id = case.get("test_identifier") or case.get("test_id") or case["manifest"].get("flaky_test_path")
        runner = self.runner_factory(repo_path) if self.runner_factory is not None else DockerTestRunner(repo_path)
        return FlakeForgeEnvironment(
            repo_path=repo_path,
            test_identifier=test_id,
            max_steps=self.max_env_steps,
            num_runs=self.num_runs,
            runner=runner,
        )

    def _build_prompt(self, observation: Any) -> str:
        if self.compact_prompt:
            user_body = _compact_observation_from_env_observation(observation)
            sys_prompt = self.system_prompt
        else:
            try:
                from agent.unified_agent import build_unified_prompt, UNIFIED_SYSTEM_PROMPT
            except ImportError:
                from ..agent.unified_agent import build_unified_prompt, UNIFIED_SYSTEM_PROMPT  # type: ignore[no-redef]
            user_body = build_unified_prompt(observation)
            sys_prompt = UNIFIED_SYSTEM_PROMPT

        user_body = user_body[:3000]
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_body},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{sys_prompt}\n\n{user_body}"

    def _tokenize_prompt(self, prompt_text: str) -> Any:
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length,
        )
        return enc["input_ids"].to(self.device)

    def _generate_group(self, prompt_ids: Any) -> Any:
        import torch

        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(self.model)
        except Exception:
            self.model.eval()

        with torch.no_grad():
            out = self.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=self.G,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_training(self.model)
        except Exception:
            self.model.train()
        return out

    def _completion_to_action(self, completion_text: str) -> Any:
        try:
            from agent.unified_agent import (
                extract_think, extract_patch,
                extract_category_from_think, extract_confidence_from_think,
            )
            from models import FlakeForgeAction as _Action
        except ImportError:
            from ..agent.unified_agent import (  # type: ignore[no-redef]
                extract_think, extract_patch,
                extract_category_from_think, extract_confidence_from_think,
            )
            from ..models import FlakeForgeAction as _Action  # type: ignore[no-redef]

        think = extract_think(completion_text)
        patch = extract_patch(completion_text)
        return _Action(
            raw_response=completion_text,
            think_text=think,
            patch_text=patch,
            predicted_category=extract_category_from_think(think),
            predicted_confidence=extract_confidence_from_think(think),
        )

    def _policy_update(self, sequences: Any, prompt_len: int, rewards: List[float]) -> Dict[str, float]:
        """GRPO policy gradient update with KL to frozen reference (LoRA disabled)."""
        import torch

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std(unbiased=False) + 1e-8)

        attention_mask = (sequences != self.tokenizer.pad_token_id).long()

        completion_mask = torch.zeros_like(sequences, dtype=torch.float32)
        completion_mask[:, prompt_len:] = (sequences[:, prompt_len:] != self.tokenizer.pad_token_id).float()
        comp_mask = completion_mask[:, 1:]

        policy_logp = self._sequence_logprobs(sequences, attention_mask, requires_grad=True)
        try:
            with self.model.disable_adapter():
                with torch.no_grad():
                    ref_logp = self._sequence_logprobs(sequences, attention_mask, requires_grad=False)
        except (AttributeError, ValueError) as exc:
            logger.debug("[ONLINE] disable_adapter unavailable (%s); falling back to KL=0", exc)
            ref_logp = policy_logp.detach()

        log_ratio = ref_logp - policy_logp
        kl_per_tok = torch.exp(log_ratio) - log_ratio - 1.0
        denom = comp_mask.sum(dim=1).clamp(min=1.0)
        seq_logp = (policy_logp * comp_mask).sum(dim=1) / denom
        seq_kl = (kl_per_tok * comp_mask).sum(dim=1) / denom

        policy_loss = -(advantages.detach() * seq_logp).mean()
        kl_loss = self.kl_beta * seq_kl.mean()
        loss = policy_loss + kl_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.grad_clip,
            )
        self.optimizer.step()

        return {
            "loss": float(loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "kl": float(seq_kl.mean().detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "advantage_max": float(advantages.max().detach().item()),
            "advantage_min": float(advantages.min().detach().item()),
        }

    def _sequence_logprobs(self, sequences: Any, attention_mask: Any, *, requires_grad: bool) -> Any:
        """Per-token logprobs of ``sequences[:, 1:]`` under the current model.

        Returns a tensor shaped (B, L-1) where L is the sequence length. The
        slot at index ``t`` holds ``log p(sequences[:, t+1] | sequences[:, :t+1])``.
        """
        import torch

        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            outputs = self.model(input_ids=sequences, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            target = sequences[:, 1:]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    def _maybe_wandb_log(self, metrics: Dict[str, Any]) -> None:
        if self.wandb_run is None:
            return
        try:
            self._wb_step += 1
            payload = {f"online/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.wandb_run.log(payload, step=self._wb_step)
        except Exception as exc:
            logger.debug("[ONLINE] wandb log failed: %s", exc)


def _std(values: List[float]) -> float:
    """Population std without numpy (avoid the import in the hot path)."""
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


_COMPACT_SYSTEM_PROMPT = """\
You are FlakeForge, a debugging agent that fixes flaky Python tests.

Reply with ONE JSON object. No markdown, no XML, no commentary.

Shape:
{"think":{"claims":[{"category":"<cat>","entity":"<symbol>","location":"<file>::<func>","polarity":"present","reason":"<short>"}],"confidence":0.85},"patch":{"hunks":[{"file":"<path>","search":"<one line from source>","replace":"<fixed line>"}]}}

CATEGORIES (pick ONE):
async_wait, concurrency, test_order_dependency, resource_leak, shared_state,
network, platform_dependency, nondeterminism, import_side_effect,
module_cache_pollution, fixture_scope_leak, mock_residue, unknown.

RULES:
1. "search" = ONE verbatim line from SOURCE UNDER TEST (same indentation).
2. "replace" keeps the same indentation.
3. Prefer the smallest fix. No sleep(), no retry, no pytest.mark.skip.
4. Patch source.py, not the test, unless the bug is in the test itself.
5. If unsure, set confidence < 0.3 and return empty hunks.
"""


def _compact_observation_from_env_observation(observation: Any) -> str:
    """Compact observation string (small-model friendly).

    Mirrors `training/train_grpo_tinker.py::build_compact_observation` but uses
    fields already present in `FlakeForgeObservation` produced by the env.
    """
    test_id = str(getattr(observation, "test_identifier", "tests/test_flaky.py") or "tests/test_flaky.py")
    baseline = float(getattr(observation, "baseline_pass_rate", 0.0) or 0.0)
    current = float(getattr(observation, "current_pass_rate", 0.0) or 0.0)

    source_under_test = str(getattr(observation, "source_under_test", "") or "")[:1200]
    test_src = str(getattr(observation, "test_function_source", "") or "")[:800]
    trace = str(getattr(observation, "failing_stack_trace", "") or "")
    trace = trace[-600:] if trace else ""

    parts: List[str] = [
        "=== TASK ===",
        f"Test: {test_id}",
        f"Pass rate: baseline={baseline:.2f}  current={current:.2f}  goal=1.00",
        "",
    ]
    if source_under_test.strip():
        parts += ["=== SOURCE UNDER TEST ===", source_under_test, ""]
    if test_src.strip():
        parts += ["=== TEST FUNCTION ===", test_src, ""]
    if trace.strip():
        parts += ["=== LAST FAILURE (tail) ===", trace, ""]
    parts += [
        'Reply with ONE JSON object: {"think": {...}, "patch": {...}}',
    ]
    return "\n".join(parts)
