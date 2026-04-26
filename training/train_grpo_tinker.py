#!/usr/bin/env python3
"""FlakeForge GRPO training on Tinker — real environment rewards.

Tinker handles remote GPU (sampling, forward_backward, optim_step).
Local machine runs the real FlakeForgeEnvironment (env.reset + env.step)
to compute execution-based rewards: patches are applied, pytest is executed
locally, and the verifiable reward (pass-rate delta, format, reasoning,
oracle) drives GRPO.

Architecture
------------
    for step in range(max_steps):
        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Local: build envs, reset, get real observations (parallel threads)
        envs, observations = parallel_reset(cases)
        prompts = [observation_to_prompt(obs) for obs in observations]

        # Remote (Tinker): sample G completions per prompt
        sample_results = await gather(sample(prompt, G) for prompt in prompts)

        # Local: for each completion, env.reset → env.step → real reward
        rewards = parallel_rollouts(envs, completions)

        # Remote (Tinker): GRPO policy update
        datums = build_datums(prompts, completions, rewards)
        training_client.forward_backward(datums)
        training_client.optim_step()

Default hyperparameters (``--help`` overrides)
----------------------------------------------
Tuned for **~3–3.5 hour wall-clock** with real env execution:

  - **batch_size=4**: fewer repos per step → parallelized env work stays fast.
  - **G=4**: fewer rollouts per prompt → fewer pytest invocations per step.
  - **num_runs=4**: pytest passes per env.step (enough for pass-rate signal).
  - **Preflight**: fast (2 quick + 2 confirm first reset; 1+1 for rollout resets).
  - **max_steps=80**: ~2–2.5 min/step → fits in 3h. Step-0 ETA printed.

Requirements
------------
    pip install tinker tinker-cookbook python-dotenv datasets torch

Environment
-----------
    TINKER_API_KEY       — set in .env or exported in shell
    USE_DOCKER_IMAGE=0   — local pytest (default, fastest)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from training.curriculum import CurriculumScheduler

try:
    from agent.unified_agent import (
        extract_think,
        extract_patch,
        extract_category_from_think,
        extract_confidence_from_think,
    )
    from models import FlakeForgeAction
    from server.FlakeForge_environment import FlakeForgeEnvironment
    from server.docker_runner import DockerTestRunner
except ImportError:
    from FlakeForge.agent.unified_agent import (
        extract_think,
        extract_patch,
        extract_category_from_think,
        extract_confidence_from_think,
    )
    from FlakeForge.models import FlakeForgeAction
    from FlakeForge.server.FlakeForge_environment import FlakeForgeEnvironment
    from FlakeForge.server.docker_runner import DockerTestRunner


# ── Constants ─────────────────────────────────────────────────────────────────

TINKER_MODEL_ID = "Qwen/Qwen3-8B"
LORA_RANK = 128
GROUP_SIZE = 4
MAX_COMPLETION_TOKENS = 1200
COMPACT_USER_MAX_CHARS = 4000
COMPACT_SOURCE_CHARS = 2000
COMPACT_TEST_CHARS = 1200
SAMPLE_TEMPERATURE = 0.85
SAMPLE_TOP_P = 0.98
ADAM_WEIGHT_DECAY = 0.01
ENV_NUM_RUNS = 4
PREFLIGHT_QUICK = 2
PREFLIGHT_CONFIRM = 2
ROLLOUT_PREFLIGHT_QUICK = 1
ROLLOUT_PREFLIGHT_CONFIRM = 1

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


# ── Environment helpers ───────────────────────────────────────────────────────

def _make_env(case: Dict[str, Any], num_runs: int = ENV_NUM_RUNS) -> FlakeForgeEnvironment:
    """Build a FlakeForgeEnvironment for one curriculum case."""
    repo_path = str(case["repo_path"])
    test_id = (
        case.get("test_identifier")
        or case.get("test_id")
        or case.get("manifest", {}).get("flaky_test_path")
    )
    runner = DockerTestRunner(repo_path)
    return FlakeForgeEnvironment(
        repo_path=repo_path,
        test_identifier=test_id,
        max_steps=1,
        num_runs=num_runs,
        runner=runner,
    )


def _completion_to_action(completion_text: str) -> FlakeForgeAction:
    """Parse a model completion into a FlakeForgeAction."""
    think = extract_think(completion_text)
    patch = extract_patch(completion_text)
    return FlakeForgeAction(
        raw_response=completion_text,
        think_text=think,
        patch_text=patch,
        predicted_category=extract_category_from_think(think),
        predicted_confidence=extract_confidence_from_think(think),
    )


def _observation_to_prompt(observation: Any) -> str:
    """Convert a real FlakeForgeObservation into a compact prompt string."""
    test_id = str(getattr(observation, "test_identifier", "tests/test_flaky.py") or "tests/test_flaky.py")
    baseline = float(getattr(observation, "baseline_pass_rate", 0.0) or 0.0)
    current = float(getattr(observation, "current_pass_rate", 0.0) or 0.0)

    source = str(getattr(observation, "source_under_test", "") or "")[:COMPACT_SOURCE_CHARS]
    test_src = str(getattr(observation, "test_function_source", "") or "")[:COMPACT_TEST_CHARS]
    trace = str(getattr(observation, "failing_stack_trace", "") or "")
    trace = trace[-600:] if trace else ""

    parts: List[str] = [
        "=== TASK ===",
        f"Test: {test_id}",
        f"Pass rate: baseline={baseline:.2f}  current={current:.2f}  goal=1.00",
        "",
    ]
    if source.strip():
        parts += ["=== SOURCE UNDER TEST ===", source, ""]
    if test_src.strip():
        parts += ["=== TEST FUNCTION ===", test_src, ""]
    if trace.strip():
        parts += ["=== LAST FAILURE (tail) ===", trace, ""]
    parts.append('Reply with ONE JSON object: {"think": {...}, "patch": {...}}')
    return "\n".join(parts)


def _build_chat_messages(prompt_text: str) -> List[Dict[str, str]]:
    """Wrap prompt text into chat messages for Tinker rendering."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_8B},
        {"role": "user", "content": prompt_text[:COMPACT_USER_MAX_CHARS]},
    ]


def _reset_env(
    case: Dict[str, Any],
    num_runs: int,
    quick: int,
    confirm: int,
) -> Tuple[Optional[FlakeForgeEnvironment], Optional[Any], Optional[str]]:
    """Build env, reset with preflight, return (env, observation, prompt) or Nones on failure."""
    try:
        env = _make_env(case, num_runs=num_runs)
        obs = env.reset(
            preflight_quick_runs=quick,
            preflight_confirm_runs=confirm,
        )
        if not getattr(obs, "should_train", True):
            return None, None, None
        prompt = _observation_to_prompt(obs)
        return env, obs, prompt
    except Exception as exc:
        print(f"  [ENV] reset failed for {case.get('case_id', '?')}: {exc}", flush=True)
        return None, None, None


def _rollout_one(
    env: FlakeForgeEnvironment,
    completion_text: str,
    quick: int = ROLLOUT_PREFLIGHT_QUICK,
    confirm: int = ROLLOUT_PREFLIGHT_CONFIRM,
) -> float:
    """Reset env to pristine, apply completion as action, return real reward."""
    try:
        env.reset(preflight_quick_runs=quick, preflight_confirm_runs=confirm)
        action = _completion_to_action(completion_text)
        step_obs = env.step(action)
        return float(getattr(step_obs, "reward", 0.0))
    except Exception as exc:
        print(f"  [ENV] rollout failed: {exc}", flush=True)
        return -1.0


# ── GRPO training loop ───────────────────────────────────────────────────────

async def run_grpo_training(
    *,
    max_steps: int = 80,
    batch_size: int = 4,
    group_size: int = GROUP_SIZE,
    learning_rate: float = 5e-5,
    lora_rank: int = LORA_RANK,
    num_runs: int = ENV_NUM_RUNS,
    curriculum_root: str = "seed_repos/idoft",
    output_dir: str = "outputs/flakeforge-tinker-qwen3-8b",
    use_curriculum: bool = True,
    wandb_project: Optional[str] = None,
    temperature: float = SAMPLE_TEMPERATURE,
    top_p: float = SAMPLE_TOP_P,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    weight_decay: float = ADAM_WEIGHT_DECAY,
    checkpoint_every: int = 20,
) -> Dict[str, Any]:
    """Run the full GRPO loop: Tinker GPU + local FlakeForge env rewards.

    Returns a summary dict with metrics history and path to downloaded weights.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n=== FlakeForge GRPO on Tinker (real env rewards) ===")
    print(f"  model           : {TINKER_MODEL_ID}")
    print(f"  lora_rank       : {lora_rank}")
    print(f"  group_size (G)  : {group_size}")
    print(f"  batch_size      : {batch_size}")
    print(f"  num_runs (env)  : {num_runs}")
    print(f"  max_steps       : {max_steps}")
    print(f"  learning_rate   : {learning_rate}")
    print(f"  weight_decay    : {weight_decay}")
    print(f"  temperature     : {temperature}  top_p={top_p}")
    print(f"  max_new_tokens  : {max_completion_tokens}")
    print(f"  checkpoint_every: {checkpoint_every}")
    print(f"  output_dir      : {output_dir}")
    print()

    # ── 1. Connect to Tinker ──────────────────────────────────────────────
    print("[TINKER] Creating service client and training client ...", flush=True)
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=TINKER_MODEL_ID,
        rank=lora_rank,
    )
    tokenizer = training_client.get_tokenizer()

    from tinker_cookbook.renderers import get_renderer, get_text_content
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

    # ── 2. Curriculum ─────────────────────────────────────────────────────
    scheduler: Optional[CurriculumScheduler] = None
    if use_curriculum:
        scheduler = CurriculumScheduler(synthetic_root=curriculum_root)
        total_cases = sum(len(s.cases) for s in scheduler.stages)
        print(f"[DATA] Curriculum: {total_cases} cases across {len(scheduler.stages)} stages", flush=True)
        if total_cases == 0:
            print("[DATA] No curriculum cases — cannot proceed without real repos.", flush=True)
            return {"error": "no_curriculum_cases"}
    else:
        print("[DATA] Curriculum disabled — cannot run real env without repo cases.", flush=True)
        return {"error": "curriculum_required"}

    # ── 3. W&B init (optional) ────────────────────────────────────────────
    wandb_run = None
    if wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                name=f"tinker-grpo-env-{TINKER_MODEL_ID.split('/')[-1]}-G{group_size}",
                config={
                    "model": TINKER_MODEL_ID, "lora_rank": lora_rank,
                    "group_size": group_size, "batch_size": batch_size,
                    "num_runs": num_runs, "max_steps": max_steps,
                    "learning_rate": learning_rate, "weight_decay": weight_decay,
                    "temperature": temperature, "top_p": top_p,
                    "max_completion_tokens": max_completion_tokens,
                    "real_env": True,
                },
            )
        except Exception as exc:
            print(f"[WANDB] Init failed (non-fatal): {exc}", flush=True)

    # ── 4. GRPO training loop ─────────────────────────────────────────────
    metrics_history: List[Dict[str, Any]] = []

    for step in range(max_steps):
        t0 = time.time()

        # ── 4a. Sample cases and build envs (parallel threads) ────────────
        cases: List[Dict[str, Any]] = []
        for _ in range(batch_size):
            c = scheduler.sample()
            if c is not None:
                cases.append(c)
        if not cases:
            print(f"Step {step:3d} | SKIP (no cases sampled)", flush=True)
            continue

        t_env_start = time.time()
        with ThreadPoolExecutor(max_workers=len(cases)) as pool:
            reset_futures = {
                pool.submit(_reset_env, case, num_runs, PREFLIGHT_QUICK, PREFLIGHT_CONFIRM): i
                for i, case in enumerate(cases)
            }
            slot_results: Dict[int, Tuple] = {}
            for fut in as_completed(reset_futures):
                idx = reset_futures[fut]
                slot_results[idx] = fut.result()

        envs: List[FlakeForgeEnvironment] = []
        prompts: List[str] = []
        valid_cases: List[Dict[str, Any]] = []
        for i in range(len(cases)):
            env, obs, prompt = slot_results.get(i, (None, None, None))
            if env is not None and prompt is not None:
                envs.append(env)
                prompts.append(prompt)
                valid_cases.append(cases[i])

        if not envs:
            print(f"Step {step:3d} | SKIP (all resets failed/rejected)", flush=True)
            continue

        t_env_reset = time.time() - t_env_start

        # ── 4b. Tinker: snapshot weights → sample G completions ───────────
        t_sample_start = time.time()
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        sample_coros = []
        rendered_prompts: List[tinker.ModelInput] = []
        for prompt_text in prompts:
            messages = _build_chat_messages(prompt_text)
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
        t_sample = time.time() - t_sample_start

        # ── 4c. Decode completions ────────────────────────────────────────
        all_completions: List[List[str]] = []
        for sample_result in sample_results:
            group_texts: List[str] = []
            for seq in sample_result.sequences:
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg) if parsed_msg else ""
                group_texts.append(content)
            all_completions.append(group_texts)

        # ── 4d. Local env rollouts → real rewards (parallel threads) ──────
        t_roll_start = time.time()

        def _run_all_rollouts_for_case(
            env: FlakeForgeEnvironment,
            completions: List[str],
        ) -> List[float]:
            rewards = []
            for comp in completions:
                r = _rollout_one(env, comp)
                rewards.append(r)
            return rewards

        with ThreadPoolExecutor(max_workers=len(envs)) as pool:
            rollout_futures = {
                pool.submit(_run_all_rollouts_for_case, env, comps): i
                for i, (env, comps) in enumerate(zip(envs, all_completions))
            }
            rewards_by_case: Dict[int, List[float]] = {}
            for fut in as_completed(rollout_futures):
                idx = rollout_futures[fut]
                try:
                    rewards_by_case[idx] = fut.result()
                except Exception as exc:
                    print(f"  [ENV] rollout batch failed: {exc}", flush=True)
                    rewards_by_case[idx] = [-1.0] * group_size

        t_rollout = time.time() - t_roll_start

        # ── 4e. Build GRPO datums with group-relative advantages ──────────
        datums: List[tinker.Datum] = []
        all_mean_rewards: List[float] = []
        n_degenerate = 0

        for i, (sample_result, model_input) in enumerate(
            zip(sample_results, rendered_prompts)
        ):
            rewards_G = rewards_by_case.get(i, [-1.0] * group_size)
            mean_r = sum(rewards_G) / len(rewards_G) if rewards_G else 0.0
            advantages_G = [r - mean_r for r in rewards_G]
            all_mean_rewards.append(mean_r)

            if all(a == 0.0 for a in advantages_G):
                n_degenerate += 1
                continue

            ob_len = model_input.length - 1
            for seq, advantage in zip(sample_result.sequences, advantages_G):
                tokens = seq.tokens
                logprobs = seq.logprobs
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

        # ── 4f. Tinker: forward-backward + optimizer step ─────────────────
        loss_val = None
        if datums:
            fwd_bwd = await training_client.forward_backward_async(
                datums, loss_fn="importance_sampling"
            )
            optim = await training_client.optim_step_async(adam_params)
            fwd_bwd_result = await fwd_bwd.result_async()
            await optim.result_async()
            loss_val = getattr(fwd_bwd_result, "loss", None)

        # ── 4g. Logging ──────────────────────────────────────────────────
        elapsed = time.time() - t0
        mean_reward = sum(all_mean_rewards) / len(all_mean_rewards) if all_mean_rewards else 0.0
        max_reward = max(
            (r for rews in rewards_by_case.values() for r in rews),
            default=0.0,
        )
        frac_degen = n_degenerate / len(envs) if envs else 1.0

        step_metrics = {
            "step": step,
            "mean_reward": round(mean_reward, 4),
            "max_reward": round(max_reward, 4),
            "frac_degenerate": round(frac_degen, 2),
            "n_datums": len(datums),
            "n_envs": len(envs),
            "loss": float(loss_val) if loss_val is not None else None,
            "t_reset_s": round(t_env_reset, 1),
            "t_sample_s": round(t_sample, 1),
            "t_rollout_s": round(t_rollout, 1),
            "elapsed_s": round(elapsed, 1),
        }
        metrics_history.append(step_metrics)

        scheduler.record(max_reward)

        print(
            f"Step {step:3d} | reward={mean_reward:+.3f} (max={max_reward:+.3f}) | "
            f"datums={len(datums):3d} | envs={len(envs)} | degen={frac_degen:.0%} | "
            f"loss={loss_val if loss_val is not None else 'n/a'} | "
            f"reset={t_env_reset:.0f}s sample={t_sample:.0f}s roll={t_rollout:.0f}s "
            f"total={elapsed:.0f}s",
            flush=True,
        )
        if step == 0 and elapsed > 1.0:
            est_h = elapsed * max_steps / 3600.0
            print(
                f"  [TIME] ~{elapsed:.0f}s/step -> ~{est_h:.1f}h for {max_steps} steps. "
                f"Tune --max-steps to land near 3-3.5h.",
                flush=True,
            )

        if wandb_run:
            try:
                wandb_run.log(
                    {f"tinker/{k}": v for k, v in step_metrics.items() if v is not None},
                    step=step,
                )
            except Exception:
                pass

        if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            ckpt_name = f"step_{step+1:04d}"
            try:
                ckpt = await training_client.save_state_async(ckpt_name)
                print(f"  [CKPT] Saved: {ckpt.path}", flush=True)
            except Exception as exc:
                print(f"  [CKPT] Save failed: {exc}", flush=True)

    # ── 5. Save final weights and download ────────────────────────────────
    print("\n[TINKER] Saving final weights ...", flush=True)
    sampler_path = None
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
                    ["tinker", "checkpoint", "download", sampler_path,
                     "--output", str(adapter_dir)],
                    check=True,
                )
                print(f"[TINKER] Adapter downloaded via CLI to: {adapter_dir}", flush=True)
            except Exception as exc2:
                print(f"[TINKER] CLI download also failed: {exc2}", flush=True)
                print(f"         tinker checkpoint download {sampler_path}", flush=True)

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
            print(f"[TINKER] Merge failed ({exc}); adapter still available", flush=True)

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

    # ── 6. Save training summary ──────────────────────────────────────────
    summary = {
        "model": TINKER_MODEL_ID,
        "lora_rank": lora_rank,
        "group_size": group_size,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_completion_tokens,
        "checkpoint_every": checkpoint_every,
        "real_env": True,
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
        description="FlakeForge GRPO on Tinker with real env rewards (local pytest)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--max-steps", type=int, default=80,
        help="Optimizer steps (~2-2.5 min each with real env; 80 ~ 3h).",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Repos per step (processed in parallel threads). More = slower step.",
    )
    p.add_argument(
        "--group-size", type=int, default=GROUP_SIZE,
        help="GRPO group size G -- rollouts per prompt. 4 is fast + enough signal.",
    )
    p.add_argument(
        "--num-runs", type=int, default=ENV_NUM_RUNS,
        help="Pytest runs per env.step (pass-rate precision). 4 is fast + reliable.",
    )
    p.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="AdamW learning rate.",
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
    p.add_argument("--top-p", type=float, default=SAMPLE_TOP_P)
    p.add_argument(
        "--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS,
        help="Max new tokens per completion.",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=20,
        help="Save Tinker training state every N steps (0 = disable).",
    )
    p.add_argument("--curriculum-root", default="seed_repos/idoft")
    p.add_argument("--output-dir", default="outputs/flakeforge-tinker-qwen3-8b")
    p.add_argument("--no-curriculum", action="store_true",
                   help="Disable curriculum (not recommended -- real env needs repo cases).")
    p.add_argument("--wandb-project", default=None)
    args = p.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        print("[ERROR] TINKER_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_grpo_training(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        num_runs=args.num_runs,
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
