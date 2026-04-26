#!/usr/bin/env python3
"""FlakeForge GRPO training on Tinker (Thinking Machines cloud GPU).

Uses Tinker's remote GPU infrastructure for GRPO training of Qwen3-8B on
FlakeForge flaky-test-fix tasks.  No local GPU required — training, sampling,
and gradient computation all happen on Tinker's servers via the API.

Architecture
------------
Tinker's GRPO loop (importance-sampling policy gradient):

    for step in range(max_steps):
        sampling_client = training_client.save_weights_and_get_sampling_client()
        for prompt in batch:
            sequences = sampling_client.sample(prompt, num_samples=G)
            rewards   = [reward_fn(seq) for seq in sequences]
            advantages = group_relative(rewards)
            datums    += build_datums(prompt, sequences, advantages)
        training_client.forward_backward(datums, loss_fn="importance_sampling")
        training_client.optim_step(adam_params)

After training, weights are downloaded locally via Tinker's checkpoint API
and optionally exported as a merged HuggingFace model or PEFT adapter.

Default hyperparameters (``--help`` overrides)
----------------------------------------------
Tuned for a **~3–3.5 hour wall-clock** Tinker run (time budget, not local GPU).

  - **G=8** GRPO groups and **LoRA rank 128** are kept for quality.
  - **Batch size 12** (not 16) and **shorter max completion tokens** make each
    step a bit faster so more **optimizer steps** fit in 3–3.5h.
  - Default **100 steps** targets ~2.0 min/step on average; after step 0, read
    the printed ETA and use ``--max-steps`` to land near 3h or 3.5h.

  If steps are *faster* than ~1.5 min, raise ``--max-steps`` (e.g. 115–130).
  If *slower* than ~2.5 min, lower ``--max-steps`` (e.g. 70–85) or reduce
  ``--batch-size`` to 10.

Prompt design
-------------
Compact but **rich enough** for 8B when budget allows:
  - Shorter system prompt than the full unified agent
  - Source / test excerpts capped (see module constants) — no DEEP SIGNALS block
  - Single-claim JSON shape (reduces nesting errors)

Requirements
------------
    pip install tinker tinker-cookbook python-dotenv datasets torch

Environment
-----------
    TINKER_API_KEY  — set in .env or exported in shell
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Tinker imports ────────────────────────────────────────────────────────────

try:
    import tinker
    from tinker import TensorData
except ImportError as exc:
    raise ImportError(
        "Tinker SDK is required for this trainer. Install with:\n"
        "    pip install tinker tinker-cookbook"
    ) from exc

try:
    import torch
except ImportError as exc:
    raise ImportError("PyTorch is required: pip install torch") from exc

# ── FlakeForge imports ────────────────────────────────────────────────────────

from training.data_generator import build_prompt_dataset_from_idoft
from training.curriculum import CurriculumScheduler

try:
    from agent.unified_agent import (
        extract_think,
        extract_patch,
        extract_category_from_think,
        extract_confidence_from_think,
        infer_category_from_patch,
    )
    from server.reward import compute_format_reward, compute_reasoning_consistency
    from models import FlakeForgeAction
except ImportError:
    from FlakeForge.agent.unified_agent import (
        extract_think,
        extract_patch,
        extract_category_from_think,
        extract_confidence_from_think,
        infer_category_from_patch,
    )
    from FlakeForge.server.reward import compute_format_reward, compute_reasoning_consistency
    from FlakeForge.models import FlakeForgeAction


# ── Constants — high-budget defaults (override via CLI) ─────────────────────

TINKER_MODEL_ID = "Qwen/Qwen3-8B"
# LoRA: 128 = more adapter capacity than 64; costs more train tokens per step.
LORA_RANK = 128
# GRPO group size G: 8 matches Tinker RL tutorials + local FlakeForge default.
GROUP_SIZE = 8
# Cap generation length — lower = faster sampling per rollout (3–3.5h budget).
MAX_COMPLETION_TOKENS = 1200
# User / source / test caps (chars). Slightly below “unlimited” to cut prefill time.
COMPACT_USER_MAX_CHARS = 4000
COMPACT_SOURCE_CHARS = 2000
COMPACT_TEST_CHARS = 1200
# Sampling: slightly higher diversity helps GRPO find better-than-mean rollouts.
SAMPLE_TEMPERATURE = 0.85
SAMPLE_TOP_P = 0.98
# Optimizer: mild weight decay often helps LoRA generalization.
ADAM_WEIGHT_DECAY = 0.01

# Lighter system prompt for 8B — fewer rules, less nesting, single claim
SYSTEM_PROMPT_8B = """\
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


# ── Reward function (offline, Tinker-compatible) ──────────────────────────────

def reward_fn(completion_text: str, prompt_text: str) -> float:
    """Score a single completion. Returns a scalar in roughly [-1.5, 1.5]."""
    think = extract_think(completion_text)
    patch = extract_patch(completion_text)
    category = extract_category_from_think(think)
    confidence = extract_confidence_from_think(think)

    action = FlakeForgeAction(
        raw_response=completion_text,
        think_text=think,
        patch_text=patch,
        predicted_category=category,
        predicted_confidence=confidence,
    )

    format_score = compute_format_reward(action)
    inferred_cat = infer_category_from_patch(patch)
    consistency = compute_reasoning_consistency(category, inferred_cat, think, patch)

    confidence_bonus = 0.1 if 0.6 <= confidence <= 0.95 else 0.0

    if not think.strip() and not patch.strip():
        return -1.5

    return round(format_score * 1.0 + consistency * 0.5 + confidence_bonus, 4)


# ── Prompt builder (compact for 8B) ──────────────────────────────────────────

def build_compact_prompt(prompt_text: str) -> List[Dict[str, str]]:
    """Wrap a FlakeForge observation into a chat message list for Tinker rendering.

    Keeps the system prompt short and caps the user message to avoid blowing
    through the 8B model's effective attention window.
    """
    user_content = prompt_text[:COMPACT_USER_MAX_CHARS]
    return [
        {"role": "system", "content": SYSTEM_PROMPT_8B},
        {"role": "user", "content": user_content},
    ]


def build_compact_observation(case: Dict[str, Any]) -> str:
    """Render a curriculum case into a compact observation string.

    Shorter than the full unified_agent prompt — tuned for 8B context budget.
    """
    manifest = case.get("manifest", {})
    test_id = case.get("test_identifier", "tests/test_flaky.py")
    # Normalise raw category values (e.g. TIMING, ORDER_DEPENDENCY, RACE_CONDITION)
    # that live in the manifests in uppercase and non-standard forms.
    _raw_cat = str(manifest.get("flake_category") or manifest.get("category") or "unknown").lower()
    _MANIFEST_NORM = {
        "timing": "async_wait",
        "race_condition": "concurrency",
        "order_dependency": "test_order_dependency",
        "shared_state": "shared_state",
        "nondeterminism": "nondeterminism",
        "resource_leak": "resource_leak",
    }
    category = _MANIFEST_NORM.get(_raw_cat, _raw_cat)
    difficulty = str(manifest.get("difficulty") or "medium").lower()

    repo_dir = Path(case.get("repo_path", ""))
    source_text = _try_read_file(repo_dir, manifest, max_chars=COMPACT_SOURCE_CHARS)
    test_text = _try_read_test(repo_dir, test_id, max_chars=COMPACT_TEST_CHARS)

    parts = [
        "=== TASK ===",
        f"Test: {test_id}",
        "Pass rate: baseline=0.00  current=0.00  goal=1.00",
        "",
    ]
    if source_text:
        parts += ["=== SOURCE UNDER TEST ===", source_text, ""]
    if test_text:
        parts += ["=== TEST FUNCTION ===", test_text, ""]
    parts += [
        f"Likely root cause: {category} (difficulty: {difficulty})",
        "",
        'Reply with ONE JSON object: {"think": {...}, "patch": {...}}',
    ]
    return "\n".join(parts)


def _try_read_file(repo_dir: Path, manifest: Dict[str, Any], max_chars: int = 1200) -> str:
    for key in ("root_cause_file", "source_file", "fix_file"):
        rel = manifest.get(key)
        if not rel:
            continue
        # 1. Try exact relative path first
        candidate = repo_dir / rel
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except Exception:
            pass
        # 2. Fallback: rglob by filename anywhere under repo_dir (handles
        #    repos where the manifest path is relative to a sub-directory)
        try:
            for match in repo_dir.rglob(Path(rel).name):
                if match.is_file():
                    return match.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except Exception:
            pass
    return ""


def _try_read_test(repo_dir: Path, test_id: str, max_chars: int = 800) -> str:
    if not test_id:
        return ""
    file_part = test_id.split("::", 1)[0].rstrip("?").strip()
    if not file_part:
        return ""
    candidate = repo_dir / file_part
    try:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        pass
    return ""


# ── GRPO training loop ───────────────────────────────────────────────────────

async def run_grpo_training(
    *,
    max_steps: int = 100,
    batch_size: int = 12,
    group_size: int = GROUP_SIZE,
    learning_rate: float = 5e-5,
    lora_rank: int = LORA_RANK,
    curriculum_root: str = "seed_repos/idoft",
    output_dir: str = "outputs/flakeforge-tinker-qwen3-8b",
    use_curriculum: bool = True,
    wandb_project: Optional[str] = None,
    temperature: float = SAMPLE_TEMPERATURE,
    top_p: float = SAMPLE_TOP_P,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    weight_decay: float = ADAM_WEIGHT_DECAY,
    checkpoint_every: int = 15,
) -> Dict[str, Any]:
    """Run the full GRPO training loop on Tinker.

    Returns a summary dict with metrics history and the path to downloaded weights.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n=== FlakeForge GRPO on Tinker ===")
    print(f"  model           : {TINKER_MODEL_ID}")
    print(f"  lora_rank       : {lora_rank}")
    print(f"  group_size (G)  : {group_size}")
    print(f"  batch_size      : {batch_size}")
    print(f"  max_steps       : {max_steps}")
    print(f"  learning_rate   : {learning_rate}")
    print(f"  weight_decay    : {weight_decay}")
    print(f"  temperature     : {temperature}  top_p={top_p}")
    print(f"  max_new_tokens  : {max_completion_tokens}")
    print(f"  checkpoint_every: {checkpoint_every}")
    print(f"  output_dir      : {output_dir}")
    print()

    # ── 1. Connect to Tinker and create training client ───────────────────
    print("[TINKER] Creating service client and training client ...", flush=True)
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=TINKER_MODEL_ID,
        rank=lora_rank,
    )
    tokenizer = training_client.get_tokenizer()

    from tinker_cookbook.renderers import get_renderer, get_text_content
    # Disable Qwen3's native <think> mode — FlakeForge puts reasoning inside the
    # JSON "think" key, so native thinking tokens would waste context on 8B.
    renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    sampling_params = tinker.SamplingParams(
        max_tokens=max_completion_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
        top_p=top_p,
    )
    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=weight_decay,
    )
    print(f"[TINKER] Training client ready (model={TINKER_MODEL_ID}, rank={lora_rank})", flush=True)

    # ── 2. Build prompt dataset ───────────────────────────────────────────
    prompts: List[str] = []
    if use_curriculum:
        scheduler = CurriculumScheduler(synthetic_root=curriculum_root)
        total_cases = sum(len(s.cases) for s in scheduler.stages)
        print(f"[DATA] Curriculum: {total_cases} cases across {len(scheduler.stages)} stages", flush=True)

        if total_cases > 0:
            for _ in range(max_steps * batch_size):
                case = scheduler.sample()
                if case:
                    obs_text = build_compact_observation(case)
                    prompts.append(obs_text)
        else:
            print("[DATA] No curriculum cases found, falling back to IDoFT prompts", flush=True)

    if not prompts:
        try:
            dataset = build_prompt_dataset_from_idoft(seed_root=curriculum_root)
            prompts = [row["prompt"] for row in dataset]
            print(f"[DATA] Loaded {len(prompts)} prompts from IDoFT manifests", flush=True)
        except Exception as exc:
            print(f"[DATA] Could not build dataset: {exc}", flush=True)

    if not prompts:
        placeholder = (
            "=== TASK ===\n"
            "Test: tests/test_example.py::test_flaky\n"
            "Pass rate: baseline=0.00  current=0.00  goal=1.00\n\n"
            "=== SOURCE UNDER TEST ===\n"
            "import asyncio\n\nasync def fetch_data(url, timeout=0.5):\n"
            "    return await asyncio.wait_for(session.get(url), timeout=timeout)\n\n"
            "Likely root cause: async_wait (difficulty: easy)\n\n"
            'Reply with ONE JSON object: {"think": {...}, "patch": {...}}'
        )
        prompts = [placeholder] * (max_steps * batch_size)
        print(f"[DATA] Using {len(prompts)} placeholder prompts", flush=True)

    print(f"[DATA] Total prompts available: {len(prompts)}", flush=True)

    # ── 3. W&B init (optional) ────────────────────────────────────────────
    wandb_run = None
    if wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                name=f"tinker-grpo-{TINKER_MODEL_ID.split('/')[-1]}-G{group_size}",
                config={
                    "model": TINKER_MODEL_ID, "lora_rank": lora_rank,
                    "group_size": group_size, "batch_size": batch_size,
                    "max_steps": max_steps, "learning_rate": learning_rate,
                    "weight_decay": weight_decay, "temperature": temperature,
                    "top_p": top_p, "max_completion_tokens": max_completion_tokens,
                },
            )
        except Exception as exc:
            print(f"[WANDB] Init failed (non-fatal): {exc}", flush=True)

    # ── 4. GRPO training loop ─────────────────────────────────────────────
    metrics_history: List[Dict[str, Any]] = []
    prompt_idx = 0

    for step in range(max_steps):
        t0 = time.time()

        # Grab this step's batch of prompts
        batch_prompts: List[str] = []
        for _ in range(batch_size):
            batch_prompts.append(prompts[prompt_idx % len(prompts)])
            prompt_idx += 1

        # Save current weights → get a sampling client
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        # Sample G completions per prompt concurrently
        sample_coros = []
        rendered_prompts: List[tinker.ModelInput] = []
        for prompt_text in batch_prompts:
            messages = build_compact_prompt(prompt_text)
            model_input = renderer.build_generation_prompt(messages)
            rendered_prompts.append(model_input)
            sample_coros.append(
                sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=group_size,
                    sampling_params=sampling_params,
                )
            )
        sample_results = await asyncio.gather(*sample_coros)

        # Grade completions and compute group-relative advantages
        datums: List[tinker.Datum] = []
        all_rewards: List[float] = []
        n_degenerate = 0

        for sample_result, model_input, prompt_text in zip(
            sample_results, rendered_prompts, batch_prompts
        ):
            rewards_G: List[float] = []
            tokens_G: List[List[int]] = []
            logprobs_G: List[List[float]] = []

            for seq in sample_result.sequences:
                tokens_G.append(seq.tokens)
                logprobs_G.append(seq.logprobs)

                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg) if parsed_msg else ""
                r = reward_fn(content, prompt_text)
                rewards_G.append(r)

            mean_r = sum(rewards_G) / len(rewards_G) if rewards_G else 0.0
            advantages_G = [r - mean_r for r in rewards_G]
            all_rewards.append(mean_r)

            if all(a == 0.0 for a in advantages_G):
                n_degenerate += 1
                continue

            ob_len = model_input.length - 1
            for tokens, logprobs, advantage in zip(tokens_G, logprobs_G, advantages_G):
                full_input = model_input.append(
                    tinker.EncodedTextChunk(tokens=tokens[:-1])
                )
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (full_input.length - ob_len)

                datum = tinker.Datum(
                    model_input=full_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums.append(datum)

        # Forward-backward + optimizer step
        if datums:
            fwd_bwd = await training_client.forward_backward_async(
                datums, loss_fn="importance_sampling"
            )
            optim = await training_client.optim_step_async(adam_params)
            fwd_bwd_result = await fwd_bwd.result_async()
            await optim.result_async()
            loss_val = getattr(fwd_bwd_result, "loss", None)
        else:
            loss_val = None

        elapsed = time.time() - t0
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        frac_degen = n_degenerate / batch_size

        step_metrics = {
            "step": step,
            "mean_reward": mean_reward,
            "frac_degenerate": frac_degen,
            "n_datums": len(datums),
            "loss": float(loss_val) if loss_val is not None else None,
            "elapsed_s": round(elapsed, 1),
        }
        metrics_history.append(step_metrics)

        print(
            f"Step {step:3d} | reward={mean_reward:+.3f} | "
            f"datums={len(datums):3d} | degen={frac_degen:.0%} | "
            f"loss={loss_val if loss_val is not None else 'n/a'} | "
            f"{elapsed:.1f}s",
            flush=True,
        )
        if step == 0 and elapsed > 1.0:
            est_min = elapsed * max_steps / 60.0
            print(
                f"  [TIME] If ~{elapsed:.0f}s/step continues → ~{est_min:.0f} min for "
                f"all {max_steps} steps (tune --max-steps for a 3–3.5h target).",
                flush=True,
            )

        if wandb_run:
            try:
                wandb_run.log({f"tinker/{k}": v for k, v in step_metrics.items() if v is not None}, step=step)
            except Exception:
                pass

        # Periodic checkpoint (Tinker cloud — resume via save_state path in console)
        if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            ckpt_name = f"step_{step+1:04d}"
            try:
                ckpt = await training_client.save_state_async(ckpt_name)
                ckpt_path = ckpt.path
                print(f"  [CKPT] Saved checkpoint: {ckpt_path}", flush=True)
            except Exception as exc:
                print(f"  [CKPT] Save failed: {exc}", flush=True)

    # ── 5. Save final weights and download ────────────────────────────────
    print("\n[TINKER] Saving final weights ...", flush=True)
    try:
        save_result = training_client.save_weights_for_sampler("final")
        sampler_path = save_result.result().path
        print(f"[TINKER] Final weights saved at: {sampler_path}", flush=True)

        adapter_dir = output_path / "adapter"
        print(f"[TINKER] Downloading adapter to {adapter_dir} ...", flush=True)
        try:
            from tinker_cookbook import weights as tinker_weights
            downloaded = tinker_weights.download(
                tinker_path=sampler_path,
                output_dir=str(adapter_dir),
            )
            print(f"[TINKER] Adapter downloaded to: {downloaded}", flush=True)
        except Exception as exc:
            print(f"[TINKER] tinker_cookbook download failed ({exc}); trying CLI ...", flush=True)
            try:
                import subprocess
                subprocess.run(
                    ["tinker", "checkpoint", "download", sampler_path, "--output", str(adapter_dir)],
                    check=True,
                )
                print(f"[TINKER] Adapter downloaded via CLI to: {adapter_dir}", flush=True)
            except Exception as exc2:
                print(f"[TINKER] CLI download also failed: {exc2}", flush=True)
                print(f"[TINKER] You can manually download with:", flush=True)
                print(f"         tinker checkpoint download {sampler_path}", flush=True)

        # Export merged HF model
        merged_dir = output_path / "merged_model"
        try:
            from tinker_cookbook import weights as tinker_weights
            tinker_weights.build_hf_model(
                base_model=TINKER_MODEL_ID,
                adapter_path=str(adapter_dir),
                output_path=str(merged_dir),
            )
            print(f"[TINKER] Merged HF model saved to: {merged_dir}", flush=True)
        except Exception as exc:
            print(f"[TINKER] Merge to HF model failed ({exc}); adapter is still available", flush=True)

        # Also build a PEFT adapter for vLLM serving
        peft_dir = output_path / "peft_adapter"
        try:
            from tinker_cookbook import weights as tinker_weights
            tinker_weights.build_lora_adapter(
                base_model=TINKER_MODEL_ID,
                adapter_path=str(adapter_dir),
                output_path=str(peft_dir),
            )
            print(f"[TINKER] PEFT adapter saved to: {peft_dir}", flush=True)
        except Exception as exc:
            print(f"[TINKER] PEFT adapter export failed ({exc})", flush=True)

    except Exception as exc:
        print(f"[TINKER] Weight save failed: {exc}", flush=True)
        sampler_path = None

    # ── 6. Save training summary ──────────────────────────────────────────
    summary = {
        "model": TINKER_MODEL_ID,
        "lora_rank": lora_rank,
        "group_size": group_size,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_completion_tokens,
        "checkpoint_every": checkpoint_every,
        "metrics": metrics_history,
        "sampler_path": sampler_path,
    }
    summary_path = output_path / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[DONE] Summary -> {summary_path}", flush=True)

    if wandb_run:
        try:
            wandb_run.finish()
        except Exception:
            pass

    return summary


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="FlakeForge GRPO training on Tinker (Thinking Machines GPU cloud)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--max-steps", type=int, default=100,
        help="Optimizer steps. Default 100 targets ~3–3.5h wall-clock; use step-0 [TIME] line to adjust.",
    )
    p.add_argument(
        "--batch-size", type=int, default=12,
        help="Prompts per step. 12 is a 3h-budget default (smaller = faster step, more steps fit in 3.5h).",
    )
    p.add_argument(
        "--group-size", type=int, default=GROUP_SIZE,
        help="GRPO group size G — completions per prompt; 8 is standard for GRPO here.",
    )
    p.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="AdamW learning rate. Slightly above 4e-5 for faster signal with large batches.",
    )
    p.add_argument(
        "--weight-decay", type=float, default=ADAM_WEIGHT_DECAY,
        help="AdamW weight decay on LoRA params.",
    )
    p.add_argument("--lora-rank", type=int, default=LORA_RANK)
    p.add_argument(
        "--temperature", type=float, default=SAMPLE_TEMPERATURE,
        help="Rollout temperature (GRPO exploration).",
    )
    p.add_argument("--top-p", type=float, default=SAMPLE_TOP_P, help="Nucleus sampling.")
    p.add_argument(
        "--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS,
        help="Max new tokens per completion. Lower = faster steps (default 1200 for ~3h runs).",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=15,
        help="Save Tinker training state every N steps (0 = disable; 15 saves a bit of API time).",
    )
    p.add_argument("--curriculum-root", default="seed_repos/idoft")
    p.add_argument("--output-dir", default="outputs/flakeforge-tinker-qwen3-8b")
    p.add_argument("--no-curriculum", action="store_true",
                   help="Skip curriculum and use IDoFT prompts directly.")
    p.add_argument("--wandb-project", default=None)
    args = p.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        print("[ERROR] TINKER_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_grpo_training(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        curriculum_root=args.curriculum_root,
        output_dir=args.output_dir,
        use_curriculum=not args.no_curriculum,
        wandb_project=args.wandb_project,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        weight_decay=args.weight_decay,
        checkpoint_every=args.checkpoint_every,
    ))


if __name__ == "__main__":
    main()
