"""
Curate seed_repos/idoft to exactly 50 repos.

## Quick install cells (Colab / local)

If you're running this in **Colab** (recommended for quick bootstraps), run:

```bash
!python -m pip install -U pip setuptools wheel
!python -m pip install -r training-requirements.txt
```

If you're running locally, first activate your venv, then:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r training-requirements.txt
```

Selection criteria:
- 17 easy / 17 medium / 16 hard
- Max category diversity across RESOURCE_LEAK, SHARED_STATE, ORDER_DEPENDENCY,
  TIMING, RACE_CONDITION, NETWORK, NONDETERMINISM, ENVIRONMENT_SENSITIVE
- Prefer repos with fewer / lighter dependencies
- Avoid: Apache Beam (massive), keras (heavy ML), robot-arm hardware repos,
  repos needing live network or API credentials, heavy monorepos like demisto-sdk
- Avoid 3+ slugs from the same upstream project
"""
from __future__ import annotations
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def _force_remove(func, path, _exc_info):
    """onerror handler: clear read-only flag then retry (needed for .git on Windows)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────
# THE 50 KEEPERS
# ─────────────────────────────────────────────────────────────────
KEEP: set[str] = {
    # ── EASY (17) ──────────────────────────────────────────────
    "api-python__test_query_no_api_key",          # SHARED_STATE
    "bottle-neck__test_router_mount_pass",         # ORDER_DEPENDENCY
    "codespell__test_dictionary_looping",          # ORDER_DEPENDENCY  (stdlib only)
    "devtracker__test_end_time_total",             # RESOURCE_LEAK
    "expstock__test_append_param",                 # SHARED_STATE
    "jsonextra__test_disable_rex",                 # SHARED_STATE
    "ljson__test_unique_check",                    # RESOURCE_LEAK
    "mdutils__test_create_md_file",               # ORDER_DEPENDENCY
    "observed__test_discard",                      # RESOURCE_LEAK
    "parso__test_permission_error",                # RESOURCE_LEAK  (parso = light)
    "python-fs__test_mkdir",                       # RESOURCE_LEAK
    "python-fs__test_rename_directory",            # RESOURCE_LEAK
    "redisqueue__test_mock_queue_connection",      # SHARED_STATE
    "redisqueue__test_mock_queue_put_get",         # ORDER_DEPENDENCY
    "stats_arrays__test_random_variables",         # NONDETERMINISM (numpy)
    "typeguard__test_check_call_args",             # ORDER_DEPENDENCY
    "Utter-More__test_ibu_aut",                    # RESOURCE_LEAK

    # ── MEDIUM (17) ────────────────────────────────────────────
    "accessify__test_implements_no_implementat",  # ORDER_DEPENDENCY
    "aiopylimit__test_exception",                  # TIMING
    "aiotasks__test_memory_delay_add_task_non",   # TIMING
    "cloudnetpy__test_fix_old_data_2",            # SHARED_STATE
    "cloudnetpy__test_l2_norm",                   # NONDETERMINISM (numpy)
    "coinbase-commerce-python__test_create",       # ORDER_DEPENDENCY
    "confight__test_it_should_load_and_merge_",   # ORDER_DEPENDENCY
    "devtracker__test_full_report",               # TIMING
    "elemental__test_send_request_should_call_",  # ORDER_DEPENDENCY
    "fishbase__test_yaml_conf_as_dict_01",        # SHARED_STATE
    "krllint__test_rule_with_fix",                # ORDER_DEPENDENCY
    "ljson__test_contains",                        # ORDER_DEPENDENCY
    "logx__test_formatted_output",                # ORDER_DEPENDENCY
    "observed__test_callbacks",                   # ORDER_DEPENDENCY
    "pydash__test_unique_id",                     # NONDETERMINISM
    "runium__test_processing",                    # TIMING
    "yunomi__test_count_calls_decorator",         # ORDER_DEPENDENCY

    # ── HARD (16) ──────────────────────────────────────────────
    "aiotasks__test_build_manager_invalid_pre",   # TIMING
    "django-beam__test_delete",                   # SHARED_STATE
    "django-beam__test_list",                     # SHARED_STATE
    "django-beam__test_list_search",              # SHARED_STATE
    "mythx-cli__test_report_json",               # ORDER_DEPENDENCY
    "osc-tiny__test_get",                        # ORDER_DEPENDENCY
    "panamah-sdk-python__test_events",           # TIMING
    "panamah-sdk-python__test_recover_failures", # TIMING
    "plcx__test_clientx_context321",             # RACE_CONDITION
    "plcx__test_clientx_error",                  # RACE_CONDITION
    "proxy.py__test_new_socket_connection_ipv",  # RACE_CONDITION
    "pybrake__test_celery_integration",          # RESOURCE_LEAK
    "rxpy-backpressure__test_on_next_drop_new_message_",  # RACE_CONDITION
    "stats_arrays__pretty_close",                # NONDETERMINISM
    "tokendito__test_set_okta_password",         # ORDER_DEPENDENCY
    "tokendito__test_set_okta_username",         # ORDER_DEPENDENCY
}

ROOT = Path("seed_repos/idoft")

# ─────────────────────────────────────────────────────────────────
# Step 1 — delete repos NOT in KEEP
# ─────────────────────────────────────────────────────────────────
def curate() -> tuple[list[str], list[str]]:
    kept, deleted = [], []
    for repo_dir in sorted(ROOT.iterdir()):
        if not repo_dir.is_dir():
            continue
        if repo_dir.name in KEEP:
            kept.append(repo_dir.name)
        else:
            print(f"  [DELETE] {repo_dir.name}")
            shutil.rmtree(repo_dir, onerror=_force_remove)
            deleted.append(repo_dir.name)
    return kept, deleted


# ─────────────────────────────────────────────────────────────────
# Step 2 — install deps for each kept repo
# ─────────────────────────────────────────────────────────────────
ALWAYS = ["pytest", "pytest-asyncio"]

def install_deps(repo_dir: Path) -> dict:
    marker = repo_dir / ".flakeforge_deps_ready"
    if marker.exists():
        return {"slug": repo_dir.name, "status": "already_done"}

    cmds: list[list[str]] = [
        [sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location", *ALWAYS],
    ]

    req = repo_dir / "requirements.txt"
    req_test = repo_dir / "requirements-test.txt"
    req_dev  = repo_dir / "requirements-dev.txt"

    # Install requirements files only.  NEVER `pip install -e .` on seed
    # repos — editable installs pollute the venv with cross-repo packages,
    # causing massive import contamination between unrelated projects.
    if req.exists():
        cmds.append([sys.executable, "-m", "pip", "install", "-q",
                     "--no-warn-script-location", "-r", "requirements.txt"])
    if req_test.exists():
        cmds.append([sys.executable, "-m", "pip", "install", "-q",
                     "--no-warn-script-location", "-r", "requirements-test.txt"])
    if req_dev.exists():
        cmds.append([sys.executable, "-m", "pip", "install", "-q",
                     "--no-warn-script-location", "-r", "requirements-dev.txt"])

    errors = []
    for cmd in cmds:
        print(f"    $ {' '.join(cmd[-4:])}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=repo_dir,
                timeout=180,
            )
            if proc.returncode != 0:
                combined = (proc.stderr or "") + (proc.stdout or "")
                # Ignore exit=1 that is solely a pip "new release" notice
                real_error = "\n".join(
                    l for l in combined.splitlines()
                    if not l.startswith("[notice]") and l.strip()
                )
                if not real_error:
                    continue
                snippet = real_error[:400].strip()
                errors.append(snippet)
                print(f"    [WARN] exit={proc.returncode}: {snippet[:120]}")
        except subprocess.TimeoutExpired:
            errors.append("timeout")
            print(f"    [WARN] install timed out after 180s")
        except Exception as exc:
            errors.append(str(exc))
            print(f"    [WARN] {exc}")

    if not errors:
        marker.write_text("ok\n", encoding="utf-8")
        return {"slug": repo_dir.name, "status": "ok"}
    return {"slug": repo_dir.name, "status": "partial_errors", "errors": errors}


# ─────────────────────────────────────────────────────────────────
# Step 3 — ensure manifest has flaky_test_path + difficulty
# ─────────────────────────────────────────────────────────────────
def ensure_manifest_fields() -> list[str]:
    from build_idoft_dataset import REPOS, _slug
    spec_by_slug = {_slug(s.repo_url, s.test_path): s for s in REPOS}
    fixed = []
    for repo_dir in sorted(ROOT.iterdir()):
        if not repo_dir.is_dir():
            continue
        mf = repo_dir / "flake_manifest.json"
        if not mf.exists():
            continue
        data = json.loads(mf.read_text(encoding="utf-8"))
        spec = spec_by_slug.get(repo_dir.name)
        changed = False
        if not data.get("flaky_test_path") and spec:
            data["flaky_test_path"] = spec.test_path
            changed = True
        if not data.get("test_identifier") and spec:
            data["test_identifier"] = spec.test_path
            changed = True
        if not data.get("difficulty") and spec:
            data["difficulty"] = spec.difficulty
            changed = True
        if changed:
            mf.write_text(json.dumps(data, indent=2), encoding="utf-8")
            fixed.append(repo_dir.name)
    return fixed


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1 — curating to 50 repos")
    print("=" * 70)
    kept, deleted = curate()
    print(f"\nKept: {len(kept)}   Deleted: {len(deleted)}")

    print("\n" + "=" * 70)
    print("STEP 2 — fixing manifest fields (flaky_test_path, difficulty)")
    print("=" * 70)
    fixed = ensure_manifest_fields()
    print(f"Manifests updated: {len(fixed)}")

    print("\n" + "=" * 70)
    print("STEP 3 — installing deps for each kept repo")
    print("=" * 70)
    results = []
    for slug in sorted(KEEP):
        repo_dir = ROOT / slug
        if not repo_dir.exists():
            print(f"\n  [MISSING] {slug} — skipping")
            continue
        print(f"\n[{slug}]")
        r = install_deps(repo_dir)
        results.append(r)
        print(f"  -> {r['status']}")

    ok     = [r for r in results if r["status"] in ("ok", "already_done")]
    errors = [r for r in results if r["status"] == "partial_errors"]
    print("\n" + "=" * 70)
    print(f"DONE.  ok={len(ok)}  partial_errors={len(errors)}")
    if errors:
        print("Repos with partial install errors:")
        for r in errors:
            print(f"  {r['slug']}: {r.get('errors', [])[:1]}")
