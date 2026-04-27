#!/usr/bin/env python3
"""Submit ``training/train_grpo.py`` as a Hugging Face Job.

Use this when you want to spend your HF compute credits on a real GPU
(A10G / A100 / H100) instead of training on your laptop. It works from any
machine where you can ``pip install huggingface_hub`` and run ``hf auth login``,
including Google Colab.

How it works
------------
The HF Jobs runtime is a stock Docker container -- it does not know anything
about FlakeForge. To make the job runnable we ship a small bootstrap shell
command that does, in order:

1. ``git clone <git_url>`` -- the public (or token-authenticated) URL of this
   repository. You push your local FlakeForge to GitHub or to an HF Spaces
   repo first; this clones it inside the container.
2. ``pip install`` Unsloth + the rest of ``training-requirements.txt`` plus
   ``server/requirements.txt`` so the env reward / runner work.
3. ``python -m training.train_grpo <forwarded args>`` -- the same CLI you would
   run locally, with whatever flags you pass to this launcher.

W&B logging continues to work because we forward ``WANDB_API_KEY`` as a job
secret.

Quick start (Colab)
-------------------

    !pip install -q huggingface_hub
    from huggingface_hub import login
    login()                       # paste a Pro HF token

    !python training/launch_hf_job.py \\
        --git-url https://github.com/<you>/FlakeForge.git \\
        --hardware a100-large \\
        --timeout 4h \\
        --max-warmup-steps 200 \\
        --max-online-episodes 500

Locally
-------

    hf auth login
    python training/launch_hf_job.py --git-url https://github.com/<you>/FlakeForge.git --hardware a10g-large

Notes
-----
* HF Jobs are a Pro / Team feature.
* The default image is ``pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`` to match
  Unsloth's CUDA wheels. Override with ``--image`` if you have a custom one.
* Pass ``--branch`` to clone a specific git branch.
* The job *does not* see your local ``seed_repos/idoft/`` -- it gets a fresh
  clone of the repo via ``git clone``. So make sure the curated 40 repos are
  committed in the branch you point at.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import List, Optional

DEFAULT_IMAGE = "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Submit FlakeForge GRPO training as a Hugging Face Job",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--git-url", required=True,
                   help="HTTPS URL of the FlakeForge git repo to clone inside the job. "
                        "For private repos, embed a token: "
                        "https://oauth2:<TOKEN>@github.com/<user>/FlakeForge.git")
    p.add_argument("--branch", default="main", help="Branch to check out after clone.")
    p.add_argument("--image", default=DEFAULT_IMAGE,
                   help="Docker image. Default matches Unsloth's CUDA 12.1 wheels.")
    p.add_argument("--hardware", default="a10g-large",
                   help="HF flavor: cpu-basic, t4-small, a10g-small, a10g-large, a10g-largex2, a100-large, ...")
    p.add_argument("--timeout", default="2h",
                   help="Job timeout. Accepts s/m/h/d suffixes or raw seconds.")
    p.add_argument("--namespace", default=None,
                   help="HF user or org under which the job runs. Defaults to your account.")
    p.add_argument("--no-wait", action="store_true",
                   help="Submit and exit. Default streams logs until the job completes.")
    p.add_argument("--extra-pip", default="",
                   help="Extra pip packages to install inside the job (space-separated).")

    forwarded = p.add_argument_group(
        "Forwarded to train_grpo.py",
        "Anything in this group is passed verbatim. Defaults match the local CLI.",
    )
    forwarded.add_argument("--phase", choices=["warmup", "online", "both"], default="both")
    forwarded.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    forwarded.add_argument("--curriculum-root", default="seed_repos/idoft")
    forwarded.add_argument("--group-size", type=int, default=8)
    forwarded.add_argument("--max-warmup-steps", type=int, default=200)
    forwarded.add_argument("--max-online-episodes", type=int, default=500)
    forwarded.add_argument("--max-seq-length", type=int, default=4096)
    forwarded.add_argument("--max-new-tokens", type=int, default=1024)
    forwarded.add_argument("--learning-rate", type=float, default=1e-5)
    forwarded.add_argument("--kl-beta", type=float, default=0.04)
    forwarded.add_argument("--num-runs", type=int, default=6)
    forwarded.add_argument("--output-dir", default="outputs/flakeforge-coder7b")
    forwarded.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "flakeforge-rl"))
    forwarded.add_argument("--wandb-run-name", default=None)
    forwarded.add_argument("--no-wandb", action="store_true")
    return p


def _build_train_command(args: argparse.Namespace) -> List[str]:
    cmd = ["python", "-m", "training.train_grpo",
           "--phase", args.phase,
           "--model", args.model,
           "--curriculum-root", args.curriculum_root,
           "--group-size", str(args.group_size),
           "--max-warmup-steps", str(args.max_warmup_steps),
           "--max-online-episodes", str(args.max_online_episodes),
           "--max-seq-length", str(args.max_seq_length),
           "--max-new-tokens", str(args.max_new_tokens),
           "--learning-rate", str(args.learning_rate),
           "--kl-beta", str(args.kl_beta),
           "--num-runs", str(args.num_runs),
           "--output-dir", args.output_dir,
           "--wandb-project", args.wandb_project]
    if args.wandb_run_name:
        cmd += ["--wandb-run-name", args.wandb_run_name]
    if args.no_wandb:
        cmd += ["--no-wandb"]
    return cmd


def _build_bootstrap(args: argparse.Namespace) -> List[str]:
    """Return the full ``["bash", "-c", "..."]`` command for the HF Job."""
    train_cmd = " ".join(shlex.quote(a) for a in _build_train_command(args))
    extra_pip = args.extra_pip.strip()

    # Single shell pipeline: clone, install, train. We chain with `&&` so any
    # failure short-circuits and the HF Job reports it via job status.
    script = (
        "set -euxo pipefail; "
        "apt-get update -qq && apt-get install -y --no-install-recommends git ca-certificates >/dev/null; "
        f"git clone --depth 1 --branch {shlex.quote(args.branch)} {shlex.quote(args.git_url)} /workspace; "
        "cd /workspace; "
        "python -m pip install --upgrade pip wheel setuptools; "
        "if [ -f training-requirements.txt ]; then pip install -r training-requirements.txt; fi; "
        "if [ -f server/requirements.txt ]; then pip install -r server/requirements.txt; fi; "
        + (f"pip install {extra_pip}; " if extra_pip else "")
        + f"{train_cmd}"
    )
    return ["bash", "-c", script]


def _gather_secrets() -> dict:
    """Pull HF + W&B credentials from env so the job can authenticate."""
    secrets: dict = {}
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "WANDB_API_KEY"):
        val = os.environ.get(key)
        if val:
            target = "HF_TOKEN" if key.startswith("HF") or key == "HUGGINGFACE_TOKEN" else key
            secrets[target] = val
    return secrets


def _gather_env() -> dict:
    env: dict = {"PYTHONUNBUFFERED": "1", "TRANSFORMERS_VERBOSITY": "info"}
    for passthrough in ("WANDB_PROJECT", "WANDB_ENTITY"):
        val = os.environ.get(passthrough)
        if val:
            env[passthrough] = val
    return env


def submit(args: argparse.Namespace) -> Optional[str]:
    try:
        from huggingface_hub import run_job
    except ImportError as exc:
        print("[HF-JOB] huggingface_hub is required. Install with: pip install -U huggingface_hub", file=sys.stderr)
        raise SystemExit(2) from exc

    command = _build_bootstrap(args)
    secrets = _gather_secrets()
    env = _gather_env()

    print("[HF-JOB] Submitting:")
    print(f"  image      = {args.image}")
    print(f"  flavor     = {args.hardware}")
    print(f"  timeout    = {args.timeout}")
    print(f"  namespace  = {args.namespace or '<your account>'}")
    print(f"  command[0] = bash -c '...'")
    print(f"  command[1] (preview) = {command[2][:240]}...")
    print(f"  secrets    = {sorted(secrets.keys())}  (values redacted)")

    job_kwargs = dict(
        image=args.image,
        command=command,
        flavor=args.hardware,
        timeout=args.timeout,
        env=env or None,
        secrets=secrets or None,
    )
    if args.namespace:
        job_kwargs["namespace"] = args.namespace

    job = run_job(**job_kwargs)
    print(f"[HF-JOB] Submitted. id={job.id}  url={job.url}")

    if args.no_wait:
        return job.id

    try:
        from huggingface_hub import fetch_job_logs, inspect_job
    except ImportError:
        return job.id

    print("[HF-JOB] Streaming logs until job completes (Ctrl+C to detach)...")
    try:
        for log in fetch_job_logs(job_id=job.id):
            print(log, flush=True)
    except KeyboardInterrupt:
        print("\n[HF-JOB] Detached -- job is still running. Inspect with:")
        print(f"  hf jobs inspect {job.id}")
        return job.id

    info = inspect_job(job_id=job.id)
    print(f"[HF-JOB] Final status: {info.status.stage}")
    if info.status.message:
        print(f"[HF-JOB] Message: {info.status.message}")
    return job.id


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except Exception:
        pass

    submit(args)


if __name__ == "__main__":
    main()
