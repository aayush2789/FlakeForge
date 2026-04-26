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
# Default SFT dataset is synthetic (local repos).
SEED_ROOT_DEFAULT = REPO_ROOT / "test_repos" / "synthetic"


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


def _infer_imported_packages(repo_dir: Path) -> List[str]:
    """Infer minimal pip packages needed to import and run tests for a synthetic repo.

    We intentionally do NOT trust requirements.txt for synthetic repos because
    it may contain old pins (e.g. sympy==0.7.5) that are incompatible with the
    job's Python. Instead we install only what the code actually imports.
    """
    import ast

    # Python 3.10+ provides stdlib module names.
    try:
        stdlib = set(sys.stdlib_module_names)  # type: ignore[attr-defined]
    except Exception:
        stdlib = set()

    # Local modules that should not be treated as pip deps.
    local_names: set[str] = {"source", "tests", "__future__"}
    # Some common stdlib aliases that may not appear in stdlib_module_names on older versions.
    stdlib |= {"typing", "pathlib", "dataclasses", "collections", "concurrent", "asyncio", "unittest"}

    py_files: List[Path] = []
    src = repo_dir / "source.py"
    if src.exists():
        py_files.append(src)
    tests_dir = repo_dir / "tests"
    if tests_dir.exists():
        py_files.extend(sorted(tests_dir.rglob("*.py")))

    imports: set[str] = set()
    for p in py_files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        imports.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".", 1)[0])

    # Filter stdlib + local.
    pkgs: List[str] = []
    for name in sorted(imports):
        if name in local_names:
            continue
        if name in stdlib:
            continue
        # pytest is installed separately
        if name == "pytest":
            continue
        pkgs.append(name)

    # Map common import->pip name mismatches (add more if encountered).
    rename = {
        "yaml": "pyyaml",
        "PIL": "pillow",
        "sklearn": "scikit-learn",
    }
    pkgs = [rename.get(p, p) for p in pkgs]

    # Safety: never install ancient sympy from requirements; if imported, force modern.
    if "sympy" in pkgs:
        pkgs = [p for p in pkgs if p != "sympy"]
        pkgs.append("sympy>=1.10")

    return pkgs


def _apply_hunks_in_place(repo_dir: Path, hunks: List[Dict[str, str]]) -> Dict[Path, str]:
    """Apply FlakeForge-style search/replace hunks. Returns {path: original_text} for rollback."""
    originals: Dict[Path, str] = {}
    for h in hunks:
        rel = h.get("file") or ""
        search = h.get("search") or ""
        replace = h.get("replace") or ""
        if not rel or not search:
            continue
        path = repo_dir / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if search not in text:
            continue
        if path not in originals:
            originals[path] = text
        path.write_text(text.replace(search, replace, 1), encoding="utf-8")
    return originals


def _rollback(repo_dir: Path, originals: Dict[Path, str]) -> None:
    for path, text in originals.items():
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass


def _pytest_passes(repo_dir: Path, test_identifier: str, *, runs: int = 3, timeout_s: int = 30) -> bool:
    """Run pytest multiple times on the same test id. True if all runs pass."""
    cmd = ["pytest", test_identifier, "-q", "--no-header", "--tb=short"]
    for i in range(1, runs + 1):
        try:
            t0 = time.time()
            subprocess.run(cmd, cwd=str(repo_dir), check=True, timeout=timeout_s)
            dt = time.time() - t0
            print(f"[VERIFY] run {i}/{runs} PASS ({dt:.2f}s) test={test_identifier}", flush=True)
        except Exception:
            print(f"[VERIFY] run {i}/{runs} FAIL test={test_identifier}", flush=True)
            return False
    return True


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


def _pip_install(
    args: List[str],
    *,
    cwd: Optional[Path] = None,
    prefix: str = "",
    constraints: Optional[Path] = None,
) -> None:
    """Run `python -m pip ...` with streaming output."""
    cmd = [sys.executable, "-m", "pip", *args]
    if constraints is not None:
        cmd += ["-c", str(Path(constraints).resolve())]
    # Always print the exact command so debugging is easy in HF Jobs logs.
    print(prefix + "[PIP] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
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


# ── Synthetic dataset support (local repos, no upstream clone) ──────────────

@dataclass(frozen=True)
class SyntheticCase:
    slug: str
    test_identifier: str
    difficulty: str
    category: str
    repo_dir: Path
    manifest_path: Path


def load_synthetic_cases(seed_root: Path) -> List[SyntheticCase]:
    cases: List[SyntheticCase] = []
    for manifest_path in sorted(seed_root.glob("*/flake_manifest.json")):
        slug = manifest_path.parent.name
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(m, dict):
            continue
        test_id = str(m.get("flaky_test_path") or m.get("test_identifier") or "")
        if not test_id:
            continue
        cases.append(
            SyntheticCase(
                slug=slug,
                test_identifier=test_id,
                difficulty=str(m.get("difficulty") or "medium").lower(),
                category=str(m.get("flake_category") or m.get("category") or "unknown").lower(),
                repo_dir=manifest_path.parent,
                manifest_path=manifest_path,
            )
        )
    return cases


def select_balanced_synthetic(cases: List[SyntheticCase], *, total: int = 40) -> List[SyntheticCase]:
    """Pick a near-equal easy/medium/hard subset (default total=40).

    IMPORTANT: We do NOT pre-filter by patchability here. Patchability is decided
    later by actually generating a candidate patch and verifying it with pytest.
    Pre-filtering would under-select and reduce dataset size.
    """
    by: Dict[str, List[SyntheticCase]] = {"easy": [], "medium": [], "hard": []}
    for c in cases:
        if c.difficulty in by:
            by[c.difficulty].append(c)
    for k in by:
        by[k] = sorted(by[k], key=lambda x: x.slug)

    if total == 40:
        need = {"easy": 13, "medium": 14, "hard": 13}
    else:
        base = total // 3
        rem = total - 3 * base
        need = {"easy": base, "medium": base, "hard": base}
        for k in ("medium", "easy", "hard"):
            if rem <= 0:
                break
            need[k] += 1
            rem -= 1

    picked: List[SyntheticCase] = []
    for d in ("easy", "medium", "hard"):
        picked.extend(by[d][: need[d]])

    return picked


def _synthetic_fix_hunks(manifest: Dict[str, Any], repo_dir: Path) -> List[Dict[str, str]]:
    """Generate a best-effort supervised patch for synthetic repos.

    Synthetic repos do not ship solution diffs. We use small category-based edits
    against root_cause_file (usually source.py). If we can't generate a safe edit,
    return [] and the caller skips the repo.
    """
    category = str(manifest.get("flake_category") or "unknown").lower()
    root_file = str(manifest.get("root_cause_file") or "source.py")
    target = repo_dir / root_file
    if not target.exists():
        target = repo_dir / "source.py"
    src = _safe_read_text(target, 200_000)
    if not src.strip():
        return []

    rel = str(target.relative_to(repo_dir)).replace("\\", "/")

    def one(search_line: str, replace_block: str) -> List[Dict[str, str]]:
        """Create a single hunk. `search_line` must match exact line text (including indentation)."""
        if not replace_block:
            return []
        if search_line in src:
            return [{"file": rel, "search": search_line, "replace": replace_block}]
        return []

    if category == "shared_state":
        # Reset singleton-owned dict each instantiation (synthetic singleton template).
        return one("            cls._instance._settings = {}", "            cls._instance._settings = {}  # reset each time")
    if category == "nondeterminism":
        # Synthetic token generator template: test expects alpha-only tokens.
        if "string.ascii_lowercase + string.digits" in src:
            return one(
                "    chars = string.ascii_lowercase + string.digits",
                "    chars = string.ascii_lowercase",
            )
        # Fallback: seed random if used.
        if "import random" in src and "random.seed(" not in src:
            return one("import random", "import random\nrandom.seed(0)")
        return []
    if category == "network":
        return one("timeout: float = 0.1", "timeout: float = 0.5")
    if category == "async_wait":
        return one("timeout=0.1", "timeout=1.0")
    if category == "concurrency":
        # Add a lock + wrap read-modify-write in a critical section (synthetic counter template).
        hunks: List[Dict[str, str]] = []
        lock_hunk = one("        # Bug: no lock protecting _value", "        self._lock = threading.Lock()")
        if lock_hunk:
            hunks.extend(lock_hunk)
        # Wrap increment.
        if "current = self._value" in src and "with self._lock" not in src:
            hunks.append({
                "file": rel,
                "search": "        current = self._value",
                "replace": "        with self._lock:\n            current = self._value\n            self._value = current + 1",
            })
        return hunks
    if category == "import_side_effect":
        # Guard import-time registration so it doesn't accumulate.
        return one(
            "_auto_register()  # Bug: runs every time module is imported/reloaded",
            "if not _plugins:\n    _auto_register()  # guarded auto-register",
        )
    if category == "test_order_dependency":
        # Hard to safely generalize across templates; skip by default.
        return []
    if category == "resource_leak":
        # If we see an obvious open() without context manager, patch anchor line.
        m = re.search(r"^\s*f\s*=\s*open\(", src, flags=re.M)
        if m:
            line = m.group(0).rstrip("\n")
            return one(line, line.replace("f = open(", "with open(") + " as f:")
        return []
    if category == "module_cache_pollution":
        # If lru_cache is used, clear it at call sites (synthetic cache template).
        if "@lru_cache" in src and "cache_clear()" in src:
            return []
        if "@lru_cache" in src:
            return one("@lru_cache(maxsize=None)", "@lru_cache(maxsize=None)\n# cache is cleared by tests as needed")
        return []
    if category == "platform_dependency":
        return []
    if category == "fixture_scope_leak":
        return []
    if category == "mock_residue":
        return []
    return []


def build_sft_rows_synthetic(
    cases: List[SyntheticCase],
    *,
    verify_runs: int = 3,
    pytest_timeout_s: int = 30,
    install_deps: bool = True,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    sys_prompt = _compact_system_prompt()
    used: List[str] = []
    total = len(cases)
    t0 = time.time()

    skipped_no_hunks: List[str] = []
    skipped_verify_failed: List[str] = []
    deps_failed: List[str] = []

    for i, c in enumerate(cases, start=1):
        repo_t0 = time.time()
        print(f"\n[SYN] {i}/{total} slug={c.slug} diff={c.difficulty} cat={c.category}", flush=True)
        try:
            manifest = json.loads(c.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        print(f"[SYN] test={c.test_identifier} root_file={manifest.get('root_cause_file','source.py')}", flush=True)

        if install_deps:
            try:
                # Always ensure pytest exists for verification.
                _pip_install(["install", "-U", "pytest", "pytest-asyncio"], prefix=f"[SYNDEPS:{c.slug}] ")
                pkgs = _infer_imported_packages(c.repo_dir)
                print(f"[SYNDEPS] inferred_packages={pkgs}", flush=True)
                if pkgs:
                    _pip_install(["install", "-U", *pkgs], prefix=f"[SYNDEPS:{c.slug}] ")
                else:
                    print("[SYNDEPS] no external imports detected (stdlib-only)", flush=True)
            except Exception as exc:
                deps_failed.append(c.slug)
                print(f"[SYNDEPS] FAIL {c.slug}: {exc}", flush=True)

        hunks = _synthetic_fix_hunks(manifest, c.repo_dir)
        if not hunks:
            skipped_no_hunks.append(c.slug)
            if i == 1 or i % 5 == 0 or i == total:
                print(f"[DATA] {i}/{total} skip(no_hunks): {c.slug} cat={c.category}", flush=True)
            continue

        print(f"[SYN] candidate_hunks={len(hunks)}", flush=True)
        for h in hunks[:3]:
            print(f"[SYN] hunk file={h.get('file')} search={repr((h.get('search') or '')[:80])}", flush=True)
        if len(hunks) > 3:
            print(f"[SYN] ... {len(hunks)-3} more hunks", flush=True)

        # Verify hunks by executing pytest multiple times (this is the label generation step).
        print("[SYN] applying hunks -> verify -> rollback", flush=True)
        originals = _apply_hunks_in_place(c.repo_dir, hunks)
        print(f"[SYN] touched_files={len(originals)}", flush=True)
        ok = _pytest_passes(
            c.repo_dir,
            c.test_identifier,
            runs=int(verify_runs),
            timeout_s=int(pytest_timeout_s),
        )
        _rollback(c.repo_dir, originals)
        print("[SYN] rollback done", flush=True)
        if not ok:
            skipped_verify_failed.append(c.slug)
            if i == 1 or i % 5 == 0 or i == total:
                print(f"[DATA] {i}/{total} skip(verify_failed): {c.slug}", flush=True)
            continue

        source_text = _safe_read_text(c.repo_dir / str(manifest.get("root_cause_file") or "source.py"), 1200)
        test_path = c.test_identifier.split("::", 1)[0]
        test_text = _safe_read_text(c.repo_dir / test_path, 800)
        prompt = _compact_prompt(
            test_id=c.test_identifier,
            source_text=source_text,
            test_text=test_text,
            category=c.category,
            difficulty=c.difficulty,
        )
        completion_obj = {
            "think": {
                "claims": [
                    {
                        "category": c.category,
                        "entity": "",
                        "location": "",
                        "polarity": "present",
                        "reason": "synthetic rule-based fix",
                    }
                ],
                "confidence": 0.7,
            },
            "patch": {"hunks": hunks},
        }
        rows.append({"prompt": f"{sys_prompt}\n\n{prompt}", "completion": json.dumps(completion_obj, ensure_ascii=False)})
        used.append(c.slug)

        if i == 1 or i % 5 == 0 or i == total:
            print(f"[DATA] progress {i}/{total} rows={len(rows)}", flush=True)
        print(f"[SYN] OK labeled in {time.time()-repo_t0:.1f}s", flush=True)

    setattr(build_sft_rows_synthetic, "_used_slugs", sorted(set(used)))  # type: ignore[attr-defined]
    print("\n[SYN] labeling summary:", flush=True)
    print(f"[SYN] total_selected={total} labeled={len(used)} elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[SYN] skipped_no_hunks={len(skipped_no_hunks)} skipped_verify_failed={len(skipped_verify_failed)} deps_failed={len(deps_failed)}", flush=True)
    return rows


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
    safe_mode: bool = True,
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
    failed_slugs: List[str] = []
    t0 = time.time()

    def _write_filtered_requirements(req_path: Path) -> Optional[Path]:
        """Filter/patch requirements.txt to be compatible with modern Python.

        Strategy:
        - Drop/comment lines that are known to break on py3.11+ (very old pins).
        - Replace some pins with a modern lower bound (e.g. sympy>=1.10).
        - Keep the file in-repo so it shows up in logs and is debuggable.
        """
        try:
            raw = req_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return None

        out_lines: List[str] = []
        changed = False

        # Known incompatible pins for py3.11+ seen in the wild
        sympy_any = re.compile(r"^\s*sympy(\[.*\])?\s*([<>=!~]=?.*)?$", re.I)

        for line in raw:
            s = line.strip()
            if not s or s.startswith("#"):
                out_lines.append(line)
                continue

            if sympy_any.match(s):
                # Drop any sympy pin and re-add a modern compatible one at the end.
                changed = True
                continue

            out_lines.append(line)

        if not changed:
            return None

        out_lines.append("sympy>=1.10  # patched by FlakeForge safe_mode (py3.11+)")
        filtered = req_path.parent / ".flakeforge_requirements.safe.txt"
        filtered.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return filtered

    def _write_constraints(repo_dir: Path) -> Path:
        """Constraints file applied to all pip installs for this repo."""
        constraints = repo_dir / ".flakeforge_constraints.txt"
        # Sympy <1.10 breaks on py3.11+ (inspect.getargspec removal).
        constraints.write_text("sympy>=1.10\n", encoding="utf-8")
        return constraints

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
            constraints = _write_constraints(repo_dir) if safe_mode else None
            if always:
                _pip_install(["install", *always], prefix=f"[DEPS:{c.slug}] ", constraints=constraints)

            req = repo_dir / "requirements.txt"
            pyproject = repo_dir / "pyproject.toml"
            setup_py = repo_dir / "setup.py"

            if req.exists() and req.stat().st_size > 0:
                try:
                    # Pre-install sympy constraint first so even transitive deps won't pull 0.7.x.
                    if safe_mode:
                        _pip_install(["install", "sympy>=1.10"], prefix=f"[DEPS:{c.slug}] ", constraints=constraints)
                    _pip_install(["install", "-r", str(req)], prefix=f"[DEPS:{c.slug}] ", constraints=constraints)
                except Exception as exc:
                    if not safe_mode:
                        raise
                    print(f"[DEPS] requirements.txt failed for {c.slug}: {exc}", flush=True)
                    filtered = _write_filtered_requirements(req)
                    if filtered is None:
                        print(f"[DEPS] no safe_mode patch available for {c.slug}; continuing (deps may be incomplete)", flush=True)
                    else:
                        print(f"[DEPS] retry with filtered requirements: {filtered.name}", flush=True)
                        if safe_mode:
                            _pip_install(["install", "sympy>=1.10"], prefix=f"[DEPS:{c.slug}] ", constraints=constraints)
                        _pip_install(["install", "-r", str(filtered)], prefix=f"[DEPS:{c.slug}] ", constraints=constraints)
            elif editable and (pyproject.exists() or setup_py.exists()):
                # Editable installs can fail for older repos; keep it optional.
                _pip_install(["install", "-e", "."], cwd=repo_dir, prefix=f"[DEPS:{c.slug}] ", constraints=constraints)

            marker.write_text("ok\n", encoding="utf-8")
            ok += 1
        except Exception as exc:
            failed += 1
            failed_slugs.append(c.slug)
            print(f"[DEPS] FAIL {c.slug}: {exc}", flush=True)

        if i == 1 or i % 5 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"[DEPS] progress {i}/{total} ok={ok} skipped={skipped} failed={failed} elapsed={elapsed:.1f}s", flush=True)

    return {"ok": ok, "skipped": skipped, "failed": failed, "failed_slugs": sorted(set(failed_slugs))}


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
    used_slugs: List[str] = []

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
        used_slugs.append(c.slug)

        if i == 1 or i % 5 == 0 or i == total:
            elapsed = time.time() - t0
            print(
                f"[DATA] progress {i}/{total} rows={len(rows)} "
                f"skips(clone={skipped_missing_clone} fix={skipped_missing_fix} "
                f"emptydiff={skipped_diff_empty} no_hunks={skipped_no_hunks} noprompt={skipped_no_prompt}) "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    # Attach metadata for the caller (without changing return type):
    setattr(build_sft_rows, "_used_slugs", sorted(set(used_slugs)))  # type: ignore[attr-defined]
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
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_private: bool = True,
    hub_commit_message: Optional[str] = None,
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

    if push_to_hub:
        if not hub_repo_id:
            raise SystemExit("--push-to-hub requires --hub-repo-id (e.g. HarshTri007/flakeforge-sft-qwen2.5-coder-7b)")
        try:
            from huggingface_hub import HfApi, create_repo, upload_folder
        except Exception as exc:
            raise SystemExit(f"huggingface_hub missing for push_to_hub: {exc}") from exc

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN is not set; cannot push to Hub.")

        # Ensure repo exists (idempotent).
        try:
            create_repo(repo_id=hub_repo_id, token=token, private=hub_private, exist_ok=True)
        except Exception:
            # fall back to API method in older hub versions
            HfApi(token=token).create_repo(repo_id=hub_repo_id, private=hub_private, exist_ok=True)

        commit_message = hub_commit_message or "Upload FlakeForge SFT adapter"
        print(f"[HUB] Uploading adapter to {hub_repo_id} ...", flush=True)
        upload_folder(
            repo_id=hub_repo_id,
            folder_path=str(final_dir),
            path_in_repo=".",
            token=token,
            commit_message=commit_message,
        )
        print(f"[HUB] Uploaded: https://huggingface.co/{hub_repo_id}", flush=True)


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
        "echo '[JOB] repo HEAD='$(git rev-parse HEAD); "
        # Global pip constraints for ancient pinned deps on modern Python (py3.11+).
        # pip honors PIP_CONSTRAINT for *all* installs, including build isolation.
        "echo 'sympy>=1.10' > /workspace/.flakeforge_global_constraints.txt; "
        "export PIP_CONSTRAINT=/workspace/.flakeforge_global_constraints.txt; "
        "echo '[JOB] PIP_CONSTRAINT='${PIP_CONSTRAINT}; "
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
        description="One-file HF SFT pipeline (synthetic or IDoFT) + SFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s_submit = sub.add_parser("submit", help="Submit this pipeline as an HF Job (spend HF credits).")
    s_submit.add_argument("--git-url", required=True, help="Git URL of YOUR FlakeForge fork (must include test_repos/synthetic).")
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
    s_submit.add_argument("--max-repos", type=int, default=40,
                          help="How many synthetic repos to use (balanced easy/medium/hard).")
    s_submit.add_argument("--install-repo-deps", action="store_true", default=False,
                          help="Optionally install each cloned repo's deps inside the job (slower).")
    s_submit.add_argument("--deps-editable", action="store_true", default=False,
                          help="If set, attempt `pip install -e .` when requirements.txt is missing (can fail).")
    s_submit.add_argument("--deps-safe-mode", action="store_true", default=True,
                          help="Patch known-incompatible pins (e.g. sympy 0.7.x) and retry installs (recommended).")
    s_submit.add_argument("--no-deps-safe-mode", dest="deps_safe_mode", action="store_false",
                          help="Disable safe-mode patching for dependency installs.")
    s_submit.add_argument("--push-to-hub", action="store_true", default=False,
                          help="Upload the final LoRA adapter to the Hugging Face Hub at the end of training.")
    s_submit.add_argument("--hub-repo-id", default=None,
                          help="Target repo on Hub, e.g. 'HarshTri007/flakeforge-sft-qwen2.5-coder-7b'.")
    s_submit.add_argument("--hub-private", action="store_true", default=True,
                          help="Create/upload to a private repo (default).")
    s_submit.add_argument("--hub-public", dest="hub_private", action="store_false",
                          help="Create/upload to a public repo.")

    s_run = sub.add_parser("run", help="Run the pipeline locally/in-job (no HF submit).")
    s_run.add_argument("--seed-root", default=str(SEED_ROOT_DEFAULT))
    s_run.add_argument("--workdir", default="outputs/sft_workdir")
    s_run.add_argument("--clone-depth", type=int, default=1)
    s_run.add_argument("--max-repos", type=int, default=40, help="How many synthetic repos to use (balanced easy/medium/hard).")
    s_run.add_argument("--verify-runs", type=int, default=3, help="How many pytest runs to verify a synthetic fix label.")
    s_run.add_argument("--pytest-timeout-s", type=int, default=30, help="Timeout per pytest run (seconds).")
    s_run.add_argument("--no-install-synth-deps", dest="install_synth_deps", action="store_false",
                       help="Skip pip installing requirements.txt for synthetic repos.")
    s_run.set_defaults(install_synth_deps=True)
    s_run.add_argument("--install-repo-deps", action="store_true", default=False,
                       help="Optionally install dependencies of each cloned repo (best-effort, slower).")
    s_run.add_argument("--deps-editable", action="store_true", default=False,
                       help="If set, attempt `pip install -e .` when requirements.txt is missing (can fail on older repos).")
    s_run.add_argument("--deps-safe-mode", action="store_true", default=True,
                       help="Patch known-incompatible pins (e.g. sympy 0.7.x) and retry installs (recommended).")
    s_run.add_argument("--no-deps-safe-mode", dest="deps_safe_mode", action="store_false",
                       help="Disable safe-mode patching and fail fast on bad requirements pins.")

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
    s_run.add_argument("--push-to-hub", action="store_true", default=False,
                       help="Upload the final LoRA adapter to the Hugging Face Hub at the end of training.")
    s_run.add_argument("--hub-repo-id", default=None,
                       help="Target repo on Hub, e.g. 'HarshTri007/flakeforge-sft-qwen2.5-coder-7b'.")
    s_run.add_argument("--hub-private", action="store_true", default=True)
    s_run.add_argument("--hub-public", dest="hub_private", action="store_false")
    s_run.add_argument("--hub-commit-message", default=None)

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
        if not args.deps_safe_mode:
            forwarded += ["--no-deps-safe-mode"]
        if args.push_to_hub:
            forwarded += ["--push-to-hub"]
            if args.hub_repo_id:
                forwarded += ["--hub-repo-id", args.hub_repo_id]
            if not args.hub_private:
                forwarded += ["--hub-public"]

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
        dataset_path = workdir / "datasets" / "sft.jsonl"
        graph_path = workdir / "graphs" / "sft_pipeline.mmd"
        summary_path = workdir / "reports" / "run_summary.json"

        print(f"[RUN] seed_root={seed_root}", flush=True)
        # Synthetic default path: local repos already exist, no upstream clone needed.
        syn_all = load_synthetic_cases(seed_root)
        syn = select_balanced_synthetic(syn_all, total=int(args.max_repos))
        print(f"[RUN] synthetic available={len(syn_all)} selected={len(syn)}", flush=True)
        if len(syn) < int(args.max_repos):
            print(f"[RUN] WARN: could only select {len(syn)} repos for requested {args.max_repos}", flush=True)

        print(f"[RUN] workdir={workdir}", flush=True)
        write_pipeline_graph(graph_path)
        print(f"[GRAPH] wrote mermaid pipeline graph -> {graph_path}", flush=True)
        # No cloning required for synthetic repos.
        cloned = {}

        deps_summary = {"ok": 0, "skipped": 0, "failed": 0, "failed_slugs": []}

        print("[RUN] building dataset ...", flush=True)
        rows = build_sft_rows_synthetic(
            syn,
            verify_runs=int(args.verify_runs),
            pytest_timeout_s=int(args.pytest_timeout_s),
            install_deps=bool(args.install_synth_deps),
        )
        print(f"[DATA] SFT rows={len(rows)}", flush=True)
        if not rows:
            raise SystemExit("No SFT rows could be built (missing fix.diff or diff parse failed).")
        write_jsonl(rows, dataset_path)
        print(f"[DATA] wrote {dataset_path}", flush=True)

        used_slugs = getattr(build_sft_rows_synthetic, "_used_slugs", [])  # type: ignore[attr-defined]
        summary = {
            "dataset": "synthetic",
            "cases_total": len(syn),
            "cloned_total": 0,
            "sft_rows": len(rows),
            "used_slugs": used_slugs,
            "deps": deps_summary,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[REPORT] wrote summary -> {summary_path}", flush=True)
        if deps_summary.get("failed_slugs"):
            print(f"[REPORT] deps failed for {len(deps_summary['failed_slugs'])} repos (continuing anyway):", flush=True)
            for s in deps_summary["failed_slugs"][:20]:
                print(f"  - {s}", flush=True)
            if len(deps_summary["failed_slugs"]) > 20:
                print("  ... (see run_summary.json for full list)", flush=True)

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
            push_to_hub=bool(args.push_to_hub),
            hub_repo_id=args.hub_repo_id,
            hub_private=bool(args.hub_private),
            hub_commit_message=args.hub_commit_message,
        )
        return


if __name__ == "__main__":
    main()

