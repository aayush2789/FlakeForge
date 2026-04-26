#!/usr/bin/env python3
"""FlakeForge GRPO training entrypoint -- Unsloth + TRL warm-up + online RL.

Two-phase training that shares one Unsloth-loaded LoRA model:

  Phase 1 -- "warmup" : TRL GRPOTrainer with offline reward (format +
            reasoning consistency). Real prompts are pulled from the curated
            IDoFT manifests via :func:`build_prompt_dataset_from_idoft`. This
            phase teaches the model to emit valid JSON without ever touching
            the environment, so it is fast and free.

  Phase 2 -- "online" : :class:`OnlineGRPOLoop` runs FlakeForgeEnvironment
            on each curriculum case, generates G completions per prompt with
            a single batched ``model.generate`` call, applies each completion
            on a pristine copy of the repo (env.reset between rollouts),
            collects the multi-signal verifiable reward from the env, and
            does a manual policy-gradient update with KL to the LoRA-disabled
            reference policy.

Default model: ``Qwen/Qwen2.5-Coder-7B-Instruct`` (best open 7B for code as of
Q1 2026). Curriculum default: ``seed_repos/idoft`` (the 40 curated cases you
already have on disk). Pass ``--curriculum-root test_repos/synthetic`` to mix
in or replace with synthetic cases.

Examples
--------

Local smoke test (small model, no Docker, single rollout per group)::

    python -m training.train_grpo \
        --phase warmup \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --no-load-in-4bit \
        --max-warmup-steps 5 \
        --group-size 2

Full local run on a 24GB consumer GPU (4-bit QLoRA)::

    python -m training.train_grpo \
        --phase both \
        --max-warmup-steps 200 \
        --max-online-episodes 500 \
        --group-size 8 \
        --num-runs 6

HF Job from Colab (uses your HF credits, runs on H100 / A100)::

    # in Colab
    !pip install -q huggingface_hub
    !python training/launch_hf_job.py \
        --hardware a100-large \
        --max-warmup-steps 200 \
        --max-online-episodes 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.curriculum import CurriculumScheduler  # noqa: E402
from training.grpo_trainer import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    OnlineGRPOLoop,
    create_trainer,
    load_model_and_tokenizer,
)


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FlakeForge GRPO trainer: Unsloth + TRL warm-up + online RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--phase", choices=["warmup", "online", "both"], default="both",
                   help="Which phase(s) to run. 'both' runs warmup then online sequentially.")
    p.add_argument("--model", default=DEFAULT_MODEL_NAME,
                   help="HF model ID. Defaults to Qwen2.5-Coder-7B-Instruct.")
    p.add_argument("--curriculum-root", default="seed_repos/idoft",
                   help="Directory containing */flake_manifest.json files for the curriculum.")
    p.add_argument("--extra-curriculum-roots", nargs="*", default=[],
                   help="Additional roots to mix in (e.g. test_repos/synthetic).")

    p.add_argument("--output-dir", default="outputs/flakeforge-coder7b")
    p.add_argument("--resume", default=None,
                   help="Path to a previously saved adapter / checkpoint to load before training.")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Save online-phase checkpoint every N episodes.")

    p.add_argument("--group-size", type=int, default=8, help="GRPO group size G.")
    p.add_argument("--max-warmup-steps", type=int, default=200)
    p.add_argument("--max-online-episodes", type=int, default=500)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-prompt-length", type=int, default=3072)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--compact-prompt", dest="compact_prompt", action="store_true", default=True,
                   help="Use the small-model prompt style (like train_grpo_tinker.py).")
    p.add_argument("--full-prompt", dest="compact_prompt", action="store_false",
                   help="Use the full unified_agent prompt (bigger, more rules).")

    p.add_argument("--load-in-4bit", dest="load_in_4bit", type=_str2bool, default=True)
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)

    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)

    p.add_argument("--num-runs", type=int, default=6,
                   help="Pytest reps the env runs per env.step (online phase).")
    p.add_argument("--use-docker", action="store_true",
                   help="Set USE_DOCKER_IMAGE=1 so the runner uses the sandbox image.")
    p.add_argument("--env-quick-runs", type=int, default=3,
                   help="Preflight quick-stage runs per reset (online phase).")
    p.add_argument("--env-confirm-runs", type=int, default=3,
                   help="Preflight confirm-stage runs per reset (online phase).")

    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "flakeforge-rl"))
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")

    p.add_argument("--sft-data", default=None,
                   help="Optional pre-built JSONL with a 'prompt' column. "
                        "If absent we build one from --curriculum-root.")
    p.add_argument("--seed", type=int, default=42)
    return p


def _maybe_init_wandb(args: argparse.Namespace) -> Any:
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[TRAIN] wandb not installed -- running without W&B logging", flush=True)
        return None
    if wandb.run is not None:
        return wandb.run
    name = args.wandb_run_name or f"grpo-{args.phase}-{args.model.split('/')[-1]}-G{args.group_size}"
    try:
        return wandb.init(project=args.wandb_project, name=name, config=vars(args))
    except Exception as exc:
        print(f"[TRAIN] wandb.init failed (non-fatal): {exc}", flush=True)
        return None


def _maybe_resume(model: Any, resume_path: Optional[str]) -> None:
    if not resume_path:
        return
    path = Path(resume_path)
    if not path.exists():
        print(f"[TRAIN] --resume {path} does not exist; ignoring", flush=True)
        return
    try:
        model.load_adapter(str(path), adapter_name="default")
        print(f"[TRAIN] Loaded LoRA adapter from {path}", flush=True)
    except Exception:
        try:
            from peft import PeftModel
            PeftModel.from_pretrained(model, str(path))
            print(f"[TRAIN] Resumed PEFT adapter from {path}", flush=True)
        except Exception as exc:
            print(f"[TRAIN] Resume failed ({exc}); continuing from scratch", flush=True)


def run_warmup(args: argparse.Namespace, model: Any, tokenizer: Any, wandb_run: Any) -> None:
    print(f"\n=== Phase 1 (warm-up): TRL GRPOTrainer x {args.max_warmup_steps} steps ===", flush=True)
    trainer = create_trainer(
        model_name=args.model,
        output_dir=str(Path(args.output_dir) / "warmup"),
        sft_data_path=args.sft_data,
        seed_root=args.curriculum_root,
        use_execution=False,
        use_lora=True,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        model=model,
        tokenizer=tokenizer,
        num_generations=args.group_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_new_tokens,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_steps=args.max_warmup_steps,
        beta=args.kl_beta,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    trainer.train()
    out = Path(args.output_dir) / "warmup" / "final"
    out.mkdir(parents=True, exist_ok=True)
    try:
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
        print(f"[TRAIN] Warm-up adapter saved -> {out}", flush=True)
    except Exception as exc:
        print(f"[TRAIN] save_model failed: {exc}", flush=True)


def run_online(args: argparse.Namespace, model: Any, tokenizer: Any, wandb_run: Any) -> None:
    print(f"\n=== Phase 2 (online): OnlineGRPOLoop x {args.max_online_episodes} episodes ===", flush=True)

    if args.use_docker:
        os.environ["USE_DOCKER_IMAGE"] = "1"

    curriculum = CurriculumScheduler(
        synthetic_root=args.curriculum_root,
        extra_roots=args.extra_curriculum_roots or None,
    )
    total = sum(len(s.cases) for s in curriculum.stages)
    if total == 0:
        print(f"[TRAIN] Curriculum has 0 cases under {args.curriculum_root}; aborting online phase", flush=True)
        return
    print(f"[TRAIN] Curriculum: {total} cases across {len(curriculum.stages)} stages", flush=True)

    loop = OnlineGRPOLoop(
        model=model,
        tokenizer=tokenizer,
        curriculum=curriculum,
        output_dir=str(Path(args.output_dir) / "online"),
        group_size=args.group_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        kl_beta=args.kl_beta,
        learning_rate=args.learning_rate,
        num_runs=args.num_runs,
        wandb_run=wandb_run,
        checkpoint_every=args.checkpoint_every,
        env_preflight_quick_runs=args.env_quick_runs,
        env_preflight_confirm_runs=args.env_confirm_runs,
        compact_prompt=args.compact_prompt,
    )
    summary = loop.train(max_episodes=args.max_online_episodes)

    summary_path = Path(args.output_dir) / "online" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[TRAIN] Online summary -> {summary_path}", flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()

    import random
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n=== FlakeForge GRPO ===", flush=True)
    print(f"  phase                : {args.phase}", flush=True)
    print(f"  model                : {args.model}", flush=True)
    print(f"  curriculum root      : {args.curriculum_root}", flush=True)
    print(f"  group size G         : {args.group_size}", flush=True)
    print(f"  max-warmup-steps     : {args.max_warmup_steps}", flush=True)
    print(f"  max-online-episodes  : {args.max_online_episodes}", flush=True)
    print(f"  load_in_4bit         : {args.load_in_4bit}", flush=True)
    print(f"  output dir           : {args.output_dir}", flush=True)

    wandb_run = _maybe_init_wandb(args)

    print("\n[TRAIN] Loading model + tokenizer (Unsloth) ...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    _maybe_resume(model, args.resume)

    if args.phase in ("warmup", "both"):
        run_warmup(args, model, tokenizer, wandb_run)

    if args.phase in ("online", "both"):
        run_online(args, model, tokenizer, wandb_run)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    print("\n[TRAIN] Done. Adapters and logs are under", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
