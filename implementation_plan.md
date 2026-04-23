# FlakeForge FACTORY_ERROR — Full Fix Plan

## Root Cause Analysis

The `FACTORY_ERROR` is raised by OpenEnv when the `FlakeForgeEnvironment.__init__` throws **any uncaught exception**. There are **4 layered bugs** causing this:

---

## Bug #1 — [EpisodeState](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py#10-47) Dataclass Signature Mismatch (PRIMARY CRASH)

### Problem
`FlakeForgeEnvironment.__init__` (line 135) calls:
```python
EpisodeState(episode_id=..., max_steps=..., test_identifier=...)
```
But [server/state.py](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py)'s [EpisodeState](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py#10-47) dataclass **requires** two extra positional fields:
- `target_function_source: str`
- `source_under_test: str`

These were added to [state.py](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py) but callers were never updated. This is the immediate crash.

### Fix — [server/state.py](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py)
- Give `target_function_source` and `source_under_test` default values of `""`.
- Add the `test_identifier` and `step_count` attributes that [FlakeForgeEnvironment](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/FlakeForge_environment.py#108-806) assigns/reads but [EpisodeState](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py#10-47) doesn't currently declare:
  - `test_identifier: str = ""`
  - `step_count: int = 0` (alias for [step](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/client.py#72-122))
  - `patches_applied`, `actions_taken`, `hypothesis_history`, `hypothesis_confidence_at_each_step`, `last_outcomes`, `chaos_pass_rate`, `chaos_baseline_pass_rate`, `perf_regression_detected`, `perf_median_ratio`, `total_diff_lines`, [judge_scores](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/client.py#123-125), `reflection`, [done](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py#45-47)

---

## Bug #2 — [EpisodeState](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py#10-47) Missing Attributes Used by [FlakeForgeEnvironment](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/FlakeForge_environment.py#108-806)

### Problem
`FlakeForgeEnvironment.reset()` and [step()](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/client.py#72-122) read/write these attributes on `_episode` (an [EpisodeState](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py#10-47)):
- `step_count` — doesn't exist (only [step](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/client.py#72-122) is declared)
- `patches_applied` — missing
- `actions_taken` — missing
- `hypothesis_history` — missing
- `hypothesis_confidence_at_each_step` — missing
- `last_outcomes` — missing (`last_outcomes` in state stores `float`, but [FlakeForgeEnvironment](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/FlakeForge_environment.py#108-806) stores dicts)
- `chaos_pass_rate` — missing
- `chaos_baseline_pass_rate` — missing
- `perf_regression_detected` — missing
- `perf_median_ratio` — missing
- `total_diff_lines` — missing
- [judge_scores](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/client.py#123-125) — missing
- `reflection` — declared as `Optional[str]` but environment writes dicts
- [done](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py#45-47) — missing
- `failure_pattern_summary` — declared as [str](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/chaos_runner.py#157-186), environment writes dicts

All of these will cause `AttributeError` at runtime after init if init happens to succeed.

### Fix — [server/state.py](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py)
Add all missing fields with correct types and defaults.

---

## Bug #3 — [hypothesis_engine.py](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/hypothesis_engine.py) import uses relative path inside absolute-import context

### Problem
Line 9 of [hypothesis_engine.py](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/hypothesis_engine.py):
```python
from .state import EpisodeState
```
This relative import **works only when module is loaded as part of a package**. The triple-fallback import chain in [FlakeForge_environment.py](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/FlakeForge_environment.py) ends up importing `hypothesis_engine` as a flat module (`from server.hypothesis_engine import ...`), which loads [hypothesis_engine.py](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/hypothesis_engine.py) in a non-package context, making the `.state` relative import fail with `ImportError`.

### Fix — [server/hypothesis_engine.py](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/hypothesis_engine.py)
Wrap the [EpisodeState](file:///c:/Users/Krish%20Raghuwanshi/Work/FlakeForge/server/state.py#10-47) import in the same try/except relative→absolute pattern used everywhere else:
```python
try:
    from .state import EpisodeState
except ImportError:
    from server.state import EpisodeState  # type: ignore
```

---

## Bug #4 — Seed Repo Path Resolution Falls Back to Non-Existent Directory

### Problem
`FlakeForgeEnvironment._resolve_repo_path()` maps `/app/seed_repos/timing_race` → tries `<root>/seed_repos/timing_race` which **doesn't exist**. The actual directories are:
- `seed_repos/cpu_timing_race`
- `seed_repos/db_commit_scope`
- `seed_repos/async_lock_deadlock`

The fallback `test_repos/timing_race_minimal` also doesn't exist. So `self.repo_path` is set to a nonexistent path. While this doesn't crash `__init__`, it means `reset()` runs `git checkout` on nothing and pytest finds no tests.

### Fix — `FlakeForge_environment.py`
Update `_resolve_repo_path` to:
1. Scan `seed_repos/` for any directory containing a `tests/` folder
2. Use `seed_repos/cpu_timing_race` as the canonical default fallback

Also update the `__init__` default `repo_path` from `/app/seed_repos/timing_race` → `seed_repos/cpu_timing_race`.

---

## Proposed Changes

### `server/state.py` [MODIFY]
Complete rewrite to add all missing fields with correct types and defaults. This is the primary fix.

### `server/FlakeForge_environment.py` [MODIFY]
- Change `repo_path` default to `seed_repos/cpu_timing_race`
- Fix `_resolve_repo_path` to auto-discover available seed repos
- Fix `reset()` and `step()` to not crash if `_episode` attributes are missing (defensive `getattr`)

### `server/hypothesis_engine.py` [MODIFY]
- Fix the `from .state import EpisodeState` to use a try/except import chain

---

## Verification Plan

### Automated (can be run after fixes)
1. **Import smoke test** — run from project root:
   ```
   cd c:\Users\Krish Raghuwanshi\Work\FlakeForge
   python -c "from server.FlakeForge_environment import FlakeForgeEnvironment; e = FlakeForgeEnvironment(); print('Init OK')"
   ```

2. **Full pytest** (existing unit tests):
   ```
   cd c:\Users\Krish Raghuwanshi\Work\FlakeForge
   python -m pytest tests/ -x -q --ignore=tests/test_flakeforge_integration.py
   ```

3. **End-to-end** (requires server already running on `http://localhost:8000`):
   ```
   python inference.py
   ```

> [!NOTE]
> The integration test (`tests/test_flakeforge_integration.py`) requires Docker and the built image, so it is excluded from a quick unit test run but is the gold standard for the full pipeline.
