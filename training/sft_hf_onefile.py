#!/usr/bin/env python3
"""One-file HF SFT pipeline (clone 40 IDoFT repos + build SFT + train).

This single script does EVERYTHING:
1) Reads the curated 40 IDoFT manifests already in this repo under `seed_repos/idoft/*/flake_manifest.json`
2) Clones the upstream repos (`repo_url`) into a fresh working directory
3) Builds an SFT JSONL dataset by converting `solution/fix.diff` into FlakeForge's JSON action format
4) Fine-tunes a model with **SFT only** (no GRPO) using Unsloth + LoRA
5) Optionally submits the whole thing as a Hugging Face Job (so you spend HF credits)

Typical use (Colab -> HF Job)
-----------------------------

In Colab:

```python
!pip -q install -U huggingface_hub
from huggingface_hub import login
login()  # paste HF token (Pro)
```

Clone your FlakeForge fork (must include `seed_repos/idoft/` and the `solution/fix.diff` files):

```bash
!git clone https://github.com/<YOU>/FlakeForge.git
%cd FlakeForge
```

Export secrets:

```python
import os
os.environ["HF_TOKEN"] = "hf_..."
os.environ["WANDB_API_KEY"] = "..."  # optional
```

Submit the job (runs on HF infra):

```bash
!python training/sft_hf_onefile.py submit \
  --git-url https://github.com/<YOU>/FlakeForge.git \
  --branch main \
  --hardware a100-large \
  --timeout 4h \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir outputs/sft-qwen2.5-coder-7b
```

Local run (no HF credits)
-------------------------

```bash
python training/sft_hf_onefile.py run --max-steps 50 --model Qwen/Qwen2.5-Coder-1.5B-Instruct
```
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_ROOT_DEFAULT = REPO_ROOT / "seed_repos" / "idoft"


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _run_stream(cmd: List[str], *, cwd: Optional[Path] = None, prefix: str = "") -> None:
    """Run a command and stream stdout/stderr so long steps show progress."""
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    last_line = time.time()
    for line in p.stdout:
        last_line = time.time()
        msg = line.rstrip("\n")
        if msg:
            print(f"{prefix}{msg}", flush=True)
    rc = p.wait()
    if rc != 0:
        # If the process produced no output for a long time before failing, say so.
        idle_s = max(0.0, time.time() - last_line)
        raise subprocess.CalledProcessError(rc, cmd, output=f"no output for {idle_s:.1f}s before exit")


def _safe_read_text(path: Path, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _find_file(repo_dir: Path, rel_or_name: str) -> Optional[Path]:
    if not rel_or_name:
        return None
    candidate = repo_dir / rel_or_name
    if candidate.is_file():
        return candidate
    name = Path(rel_or_name).name
    for p in repo_dir.rglob(name):
        if p.is_file():
            return p
    return None


def _pip_install(args: List[str], *, cwd: Optional[Path] = None, prefix: str = "") -> None:
    """Run `python -m pip ...` with streaming output."""
    cmd = [sys.executable, "-m", "pip", *args]
    _run_stream(cmd, cwd=cwd, prefix=prefix)


def _compact_system_prompt() -> str:
    # Matches training/train_grpo_tinker.py SYSTEM_PROMPT_8B (small prompt).
    return """\
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


def _compact_prompt(test_id: str, source_text: str, test_text: str, category: str, difficulty: str) -> str:
    parts: List[str] = [
        "=== TASK ===",
        f"Test: {test_id}",
        "Pass rate: baseline=0.00  current=0.00  goal=1.00",
        "",
    ]
    if source_text.strip():
        parts += ["=== SOURCE UNDER TEST ===", source_text[:1200], ""]
    if test_text.strip():
        parts += ["=== TEST FUNCTION ===", test_text[:800], ""]
    parts += [
        f"Likely root cause: {category} (difficulty: {difficulty})",
        "",
        'Reply with ONE JSON object: {"think": {...}, "patch": {...}}',
    ]
    return "\n".join(parts)


@dataclass(frozen=True)
class IdofTCase:
    slug: str
    repo_url: str
    test_identifier: str
    difficulty: str
    category: str
    seed_repo_dir: Path  # contains flake_manifest + solution/fix.diff


def load_idoft_cases(seed_root: Path) -> List[IdofTCase]:
    cases: List[IdofTCase] = []
    for manifest_path in sorted(seed_root.glob("*/flake_manifest.json")):
        slug = manifest_path.parent.name
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(m, dict):
            continue
        repo_url = str(m.get("repo_url") or "")
        test_id = str(m.get("test_identifier") or m.get("flaky_test_path") or "")
        if not repo_url or not test_id:
            continue
        cases.append(
            IdofTCase(
                slug=slug,
                repo_url=repo_url,
                test_identifier=test_id,
                difficulty=str(m.get("difficulty") or "medium").lower(),
                category=str(m.get("flake_category") or m.get("category") or "unknown").lower(),
                seed_repo_dir=manifest_path.parent,
            )
        )
    return cases


def clone_upstream_repos(cases: List[IdofTCase], out_root: Path, *, depth: int = 1) -> Dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    cloned: Dict[str, Path] = {}
    total = len(cases)
    t0 = time.time()
    for i, c in enumerate(cases, start=1):
        target = out_root / c.slug
        if target.exists():
            cloned[c.slug] = target
            if i == 1 or i % 5 == 0 or i == total:
                print(f"[CLONE] {i}/{total} already exists: {c.slug}", flush=True)
            continue
        cmd = ["git", "clone", "--progress", "--depth", str(depth), c.repo_url, str(target)]
        print(f"[CLONE] {i}/{total} {c.slug} <- {c.repo_url}", flush=True)
        try:
            _run_stream(cmd, prefix=f"[CLONE:{c.slug}] ")
        except Exception as exc:
            print(f"[CLONE] FAIL {c.slug}: {exc}", flush=True)
            # continue cloning others; dataset build will skip missing clones
            continue
        cloned[c.slug] = target
        if i == 1 or i % 5 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"[CLONE] progress {i}/{total} elapsed={elapsed:.1f}s", flush=True)
    return cloned


def install_repo_dependencies(
    cases: List[IdofTCase],
    cloned_repo_paths: Dict[str, Path],
    *,
    always: Optional[List[str]] = None,
    editable: bool = False,
) -> Dict[str, Any]:
    """Best-effort install of repo deps so repo imports won't error.

    NOTE: SFT dataset building in this script only reads files and diffs, so
    repo deps are not strictly required. This step exists because some users
    want a more robust pipeline that can also run lightweight import checks or
    later reuse the same cloned repos for execution.
    """
    always = always or []
    total = len(cases)
    ok = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, c in enumerate(cases, start=1):
        repo_dir = cloned_repo_paths.get(c.slug)
        if repo_dir is None or not repo_dir.exists():
            skipped += 1
            continue

        marker = repo_dir / ".flakeforge_repo_deps_ready"
        if marker.exists():
            skipped += 1
            if i == 1 or i % 5 == 0 or i == total:
                print(f"[DEPS] {i}/{total} already done: {c.slug}", flush=True)
            continue

        print(f"[DEPS] {i}/{total} installing: {c.slug}", flush=True)
        try:
            if always:
                _pip_install(["install", *always], prefix=f"[DEPS:{c.slug}] ")

            req = repo_dir / "requirements.txt"
            pyproject = repo_dir / "pyproject.toml"
            setup_py = repo_dir / "setup.py"

            if req.exists() and req.stat().st_size > 0:
                _pip_install(["install", "-r", str(req)], prefix=f"[DEPS:{c.slug}] ")
            elif editable and (pyproject.exists() or setup_py.exists()):
                # Editable installs can fail for older repos; keep it optional.
                _pip_install(["install", "-e", "."], cwd=repo_dir, prefix=f"[DEPS:{c.slug}] ")

            marker.write_text("ok\n", encoding="utf-8")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[DEPS] FAIL {c.slug}: {exc}", flush=True)

        if i == 1 or i % 5 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"[DEPS] progress {i}/{total} ok={ok} skipped={skipped} failed={failed} elapsed={elapsed:.1f}s", flush=True)

    return {"ok": ok, "skipped": skipped, "failed": failed}


def _parse_unified_diff(diff_text: str) -> List[Tuple[str, List[str], List[str]]]:
    """Parse minimal info: [(file_path, deleted_lines, added_lines), ...] per hunk.

    This is a deliberately simple parser that is robust across IDoFT diffs.
    We only need enough to create search/replace hunks for SFT targets.
    """
    file_path = ""
    hunks: List[Tuple[str, List[str], List[str]]] = []
    del_lines: List[str] = []
    add_lines: List[str] = []

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            file_path = line[len("+++ b/") :].strip()
            continue
        if line.startswith("@@ "):
            if file_path and (del_lines or add_lines):
                hunks.append((file_path, del_lines, add_lines))
            del_lines, add_lines = [], []
            continue
        if not file_path:
            continue
        if line.startswith("-") and not line.startswith("---"):
            del_lines.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            add_lines.append(line[1:])

    if file_path and (del_lines or add_lines):
        hunks.append((file_path, del_lines, add_lines))
    return hunks


def diff_to_json_hunks(diff_text: str) -> List[Dict[str, str]]:
    """Convert unified diff to FlakeForge JSON hunks (best-effort).

    Heuristic:
    - pick the first deleted line as the search anchor
    - replace with the entire added block joined with '\\n' (can be multi-line)
    - if no deletions, skip (can't build search anchor)
    """
    hunks_out: List[Dict[str, str]] = []
    for file_path, dels, adds in _parse_unified_diff(diff_text):
        if not dels:
            continue
        search = dels[0].rstrip("\n")
        replace_block = "\n".join(adds).rstrip("\n") if adds else ""
        if not replace_block:
            continue
        hunks_out.append({"file": file_path, "search": search, "replace": replace_block})
    return hunks_out


def build_sft_rows(
    cases: List[IdofTCase],
    cloned_repo_paths: Dict[str, Path],
) -> List[Dict[str, str]]:
    """Build rows: {prompt, completion} where completion is ONE JSON object."""
    rows: List[Dict[str, str]] = []
    sys_prompt = _compact_system_prompt()
    total = len(cases)
    t0 = time.time()

    # Skip counters (so silent `continue` becomes visible)
    skipped_missing_clone = 0
    skipped_missing_fix = 0
    skipped_diff_empty = 0
    skipped_no_hunks = 0
    skipped_no_prompt = 0

    for i, c in enumerate(cases, start=1):
        repo_dir = cloned_repo_paths.get(c.slug)
        if repo_dir is None or not repo_dir.exists():
            skipped_missing_clone += 1
            if i == 1 or i % 5 == 0 or i == total:
                print(f"[DATA] {i}/{total} missing clone: {c.slug}", flush=True)
            continue

        fix_path = c.seed_repo_dir / "solution" / "fix.diff"
        if not fix_path.exists():
            skipped_missing_fix += 1
            continue
        diff_text = _safe_read_text(fix_path, max_chars=200_000)
        if not diff_text.strip():
            skipped_diff_empty += 1
            continue
        hunks = diff_to_json_hunks(diff_text)
        if not hunks:
            skipped_no_hunks += 1
            continue

        manifest = json.loads((c.seed_repo_dir / "flake_manifest.json").read_text(encoding="utf-8"))
        src_file = _find_file(repo_dir, str(manifest.get("root_cause_file") or "")) or _find_file(repo_dir, "source.py")
        test_file = _find_file(repo_dir, c.test_identifier.split("::", 1)[0].rstrip("?"))

        src_text = _safe_read_text(src_file, 1200) if src_file else ""
        test_text = _safe_read_text(test_file, 800) if test_file else ""

        prompt = _compact_prompt(
            test_id=c.test_identifier,
            source_text=src_text,
            test_text=test_text,
            category=c.category,
            difficulty=c.difficulty,
        )
        if not prompt.strip():
            skipped_no_prompt += 1
            continue

        completion_obj = {
            "think": {
                "claims": [
                    {
                        "category": c.category,
                        "entity": "",
                        "location": "",
                        "polarity": "present",
                        "reason": "supervised fix from solution diff",
                    }
                ],
                "confidence": 0.85,
            },
            "patch": {"hunks": hunks},
        }
        completion = json.dumps(completion_obj, ensure_ascii=False)

        full_prompt = f"{sys_prompt}\n\n{prompt}"
        rows.append({"prompt": full_prompt, "completion": completion})

        if i == 1 or i % 5 == 0 or i == total:
            elapsed = time.time() - t0
            print(
                f"[DATA] progress {i}/{total} rows={len(rows)} "
                f"skips(clone={skipped_missing_clone} fix={skipped_missing_fix} "
                f"emptydiff={skipped_diff_empty} no_hunks={skipped_no_hunks} noprompt={skipped_no_prompt}) "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return rows


def write_jsonl(rows: Iterable[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_pipeline_graph(path: Path) -> None:
    """Write a simple Mermaid graph describing the SFT pipeline."""
    graph = """flowchart TD
    A[seed_repos/idoft manifests] --> B[Load 40 cases]
    B --> C[Clone upstream repos via repo_url]
    C --> D[Read solution/fix.diff]
    D --> E[Parse diff -> JSON hunks]
    C --> F[Read source + test snippets]
    E --> G[Build SFT rows: prompt + JSON completion]
    F --> G
    G --> H[Write JSONL dataset]
    H --> I[Unsloth load model + attach LoRA]
    I --> J[TRL SFTTrainer.train()]
    J --> K[Save adapter checkpoint]
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph, encoding="utf-8")


def run_sft(
    dataset_path: Path,
    *,
    model_name: str,
    output_dir: Path,
    max_steps: int,
    lr: float,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
) -> None:
    """SFT only (TRL SFTTrainer) on {prompt, completion} JSONL."""
    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise SystemExit(f"SFT deps missing. Install training-requirements.txt. error={exc}") from exc

    try:
        import wandb
    except Exception:
        wandb = None  # type: ignore[assignment]

    # Always make Trainer log to stdout (W&B is optional).
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_info()
    except Exception:
        pass

    # Unsloth load (mandatory for the target setup)
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise SystemExit(f"Unsloth missing in the job environment: {exc}") from exc

    print(f"[SFT] Loading model via Unsloth: {model_name}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"[SFT] Dataset rows: {len(ds)}", flush=True)
    print(f"[SFT] First row prompt chars: {len(ds[0]['prompt']) if len(ds) else 0}", flush=True)

    if wandb and wandb_project:
        try:
            if wandb.run is None:
                wandb.init(project=wandb_project, name=wandb_run_name, config={
                    "phase": "sft",
                    "model": model_name,
                    "max_steps": max_steps,
                    "lr": lr,
                    "dataset": str(dataset_path),
                })
        except Exception as exc:
            print(f"[SFT] wandb init failed (non-fatal): {exc}", flush=True)

    cfg = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=1,
        save_steps=max(10, max_steps // 5),
        bf16=True,
        report_to=["wandb"] if wandb_project else [],
        dataset_text_field="prompt",  # we feed prompt+completion via formatting_func
        packing=False,
    )

    # Fallback stdout logger (so it never looks frozen if W&B is broken).
    try:
        from transformers import TrainerCallback

        class _StdoutCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                # keep it compact but visible
                keys = {k: logs[k] for k in sorted(logs) if isinstance(logs[k], (int, float))}
                if keys:
                    print(f"[SFT:LOG] step={state.global_step} {keys}", flush=True)

        stdout_cb: Optional[Any] = _StdoutCallback()
    except Exception:
        stdout_cb = None

    def formatting_func(examples: Dict[str, List[str]]) -> List[str]:
        # TRL SFTTrainer expects a single text field. We concatenate prompt + completion.
        out: List[str] = []
        for p, c in zip(examples["prompt"], examples["completion"]):
            out.append(f"{p}\n{c}")
        return out

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        train_dataset=ds,
        formatting_func=formatting_func,
    )
    if stdout_cb is not None:
        trainer.add_callback(stdout_cb)
    print("[SFT] Starting trainer.train() ...", flush=True)
    trainer.train()
    print("[SFT] trainer.train() finished.", flush=True)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[SFT] Saved -> {final_dir}", flush=True)


def submit_hf_job(
    *,
    git_url: str,
    branch: str,
    flavor: str,
    timeout: str,
    image: str,
    namespace: Optional[str],
    args_to_forward: List[str],
) -> None:
    from huggingface_hub import run_job

    # The job clones your FlakeForge repo, installs deps, runs THIS script in `run` mode.
    forwarded = " ".join(shlex.quote(a) for a in args_to_forward)
    script = (
        "set -euxo pipefail; "
        "apt-get update -qq && apt-get install -y --no-install-recommends git ca-certificates >/dev/null; "
        f"git clone --depth 1 --branch {shlex.quote(branch)} {shlex.quote(git_url)} /workspace; "
        "cd /workspace; "
        "python -m pip install --upgrade pip wheel setuptools; "
        "pip install -r training-requirements.txt; "
        f"python training/sft_hf_onefile.py run {forwarded}"
    )

    secrets: Dict[str, str] = {}
    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_tok:
        secrets["HF_TOKEN"] = hf_tok
    wb = os.environ.get("WANDB_API_KEY")
    if wb:
        secrets["WANDB_API_KEY"] = wb

    env = {"PYTHONUNBUFFERED": "1"}
    if os.environ.get("WANDB_PROJECT"):
        env["WANDB_PROJECT"] = os.environ["WANDB_PROJECT"]

    kwargs: Dict[str, Any] = dict(
        image=image,
        command=["bash", "-c", script],
        flavor=flavor,
        timeout=timeout,
        env=env,
        secrets=secrets or None,
    )
    if namespace:
        kwargs["namespace"] = namespace

    job = run_job(**kwargs)
    print(f"[HF-JOB] Submitted. id={job.id} url={job.url}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-file HF SFT pipeline (clone 40 IDoFT repos + SFT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s_submit = sub.add_parser("submit", help="Submit this pipeline as an HF Job (spend HF credits).")
    s_submit.add_argument("--git-url", required=True, help="Git URL of YOUR FlakeForge fork (must include seed_repos/idoft).")
    s_submit.add_argument("--branch", default="main")
    s_submit.add_argument("--hardware", default="a100-large")
    s_submit.add_argument("--timeout", default="4h")
    s_submit.add_argument("--image", default="pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel")
    s_submit.add_argument("--namespace", default=None)

    # Forwarded training args
    s_submit.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    s_submit.add_argument("--output-dir", default="outputs/sft-qwen2.5-coder-7b")
    s_submit.add_argument("--max-steps", type=int, default=200)
    s_submit.add_argument("--lr", type=float, default=2e-5)
    s_submit.add_argument("--max-seq-length", type=int, default=2048)
    s_submit.add_argument("--load-in-4bit", action="store_true", default=True)
    s_submit.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    s_submit.add_argument("--lora-r", type=int, default=64)
    s_submit.add_argument("--lora-alpha", type=int, default=128)
    s_submit.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "flakeforge-sft"))
    s_submit.add_argument("--wandb-run-name", default=None)
    s_submit.add_argument("--max-repos", type=int, default=30,
                          help="Safety guard: only use the first N IDoFT cases found under seed_repos/idoft.")
    s_submit.add_argument("--install-repo-deps", action="store_true", default=False,
                          help="Optionally install each cloned repo's deps inside the job (slower).")
    s_submit.add_argument("--deps-editable", action="store_true", default=False,
                          help="If set, attempt `pip install -e .` when requirements.txt is missing (can fail).")

    s_run = sub.add_parser("run", help="Run the pipeline locally/in-job (no HF submit).")
    s_run.add_argument("--seed-root", default=str(SEED_ROOT_DEFAULT))
    s_run.add_argument("--workdir", default="outputs/sft_workdir")
    s_run.add_argument("--clone-depth", type=int, default=1)
    s_run.add_argument("--max-repos", type=int, default=None, help="Debug: only use first N repos.")
    s_run.add_argument("--install-repo-deps", action="store_true", default=False,
                       help="Optionally install dependencies of each cloned repo (best-effort, slower).")
    s_run.add_argument("--deps-editable", action="store_true", default=False,
                       help="If set, attempt `pip install -e .` when requirements.txt is missing (can fail on older repos).")

    s_run.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    s_run.add_argument("--output-dir", default="outputs/sft-qwen2.5-coder-7b")
    s_run.add_argument("--max-steps", type=int, default=200)
    s_run.add_argument("--lr", type=float, default=2e-5)
    s_run.add_argument("--max-seq-length", type=int, default=2048)
    s_run.add_argument("--load-in-4bit", action="store_true", default=True)
    s_run.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    s_run.add_argument("--lora-r", type=int, default=64)
    s_run.add_argument("--lora-alpha", type=int, default=128)
    s_run.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "flakeforge-sft"))
    s_run.add_argument("--wandb-run-name", default=None)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "submit":
        forwarded = [
            "--seed-root",
            str(SEED_ROOT_DEFAULT),
            "--workdir",
            "outputs/sft_workdir",
            "--max-repos",
            str(args.max_repos),
            "--model",
            args.model,
            "--output-dir",
            args.output_dir,
            "--max-steps",
            str(args.max_steps),
            "--lr",
            str(args.lr),
            "--max-seq-length",
            str(args.max_seq_length),
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--wandb-project",
            args.wandb_project,
        ]
        if args.wandb_run_name:
            forwarded += ["--wandb-run-name", args.wandb_run_name]
        if not args.load_in_4bit:
            forwarded += ["--no-load-in-4bit"]
        if args.install_repo_deps:
            forwarded += ["--install-repo-deps"]
        if args.deps_editable:
            forwarded += ["--deps-editable"]

        submit_hf_job(
            git_url=args.git_url,
            branch=args.branch,
            flavor=args.hardware,
            timeout=args.timeout,
            image=args.image,
            namespace=args.namespace,
            args_to_forward=forwarded,
        )
        return

    if args.cmd == "run":
        seed_root = Path(args.seed_root)
        workdir = Path(args.workdir)
        repo_clone_root = workdir / "cloned_repos"
        dataset_path = workdir / "datasets" / "idoft_sft.jsonl"
        graph_path = workdir / "graphs" / "sft_pipeline.mmd"

        print(f"[RUN] seed_root={seed_root}", flush=True)
        cases = load_idoft_cases(seed_root)
        if args.max_repos is not None:
            cases = cases[: int(args.max_repos)]
        print(f"[RUN] cases={len(cases)}", flush=True)
        if not cases:
            raise SystemExit(f"No cases found under {seed_root}. Did you commit seed_repos/idoft/?")

        print(f"[RUN] workdir={workdir}", flush=True)
        write_pipeline_graph(graph_path)
        print(f"[GRAPH] wrote mermaid pipeline graph -> {graph_path}", flush=True)
        print(f"[RUN] cloning upstream repos to {repo_clone_root} ...", flush=True)
        cloned = clone_upstream_repos(cases, repo_clone_root, depth=int(args.clone_depth))
        print(f"[RUN] cloned={len(cloned)}/{len(cases)}", flush=True)

        if args.install_repo_deps:
            print("[RUN] installing repo dependencies (best-effort) ...", flush=True)
            deps_summary = install_repo_dependencies(
                cases,
                cloned,
                always=["pytest", "pytest-asyncio"],
                editable=bool(args.deps_editable),
            )
            print(f"[RUN] deps summary: {deps_summary}", flush=True)

        print("[RUN] building dataset ...", flush=True)
        rows = build_sft_rows(cases, cloned)
        print(f"[DATA] SFT rows={len(rows)}", flush=True)
        if not rows:
            raise SystemExit("No SFT rows could be built (missing fix.diff or diff parse failed).")
        write_jsonl(rows, dataset_path)
        print(f"[DATA] wrote {dataset_path}", flush=True)

        run_sft(
            dataset_path=dataset_path,
            model_name=args.model,
            output_dir=Path(args.output_dir),
            max_steps=int(args.max_steps),
            lr=float(args.lr),
            max_seq_length=int(args.max_seq_length),
            load_in_4bit=bool(args.load_in_4bit),
            lora_r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            wandb_project=None if not args.wandb_project else args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
        return


if __name__ == "__main__":
    main()

