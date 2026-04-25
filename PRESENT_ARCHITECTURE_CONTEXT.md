# FlakeForge Present Architecture Context

Date: 2026-04-25

Purpose: this document is a complete handoff context for another LLM or engineer. It explains what FlakeForge currently is, how the code is wired, what the runtime loop does, what files matter, what environment variables control behavior, and what is known to be weak or incomplete.

This is not an aspirational architecture. This is the current implemented architecture as of this repo state.

## One-Line Summary

FlakeForge is an OpenEnv-compatible reinforcement learning environment for flaky test repair. A language model observes a flaky test, emits a structured reasoning block plus a search/replace patch, the environment applies the patch to a target repo, repeatedly runs pytest, computes deterministic verifier rewards, and returns the next observation/reward/state.

## Current Architecture Version

The repo has gone through multiple architecture versions. The current working direction is V3.

### V1/V2 Ideas Still Visible In Docs

Older docs and some comments mention:

- Analyzer and Fixer roles.
- Two LoRA adapters.
- Frozen LLM judge.
- Seven hardcoded action types.
- Judge-scored hypotheses and patches.
- Docker-isolated runner.
- `seed_repos/`.

Those are not the current hot path.

### Current V3 Reality

The current execution path is:

- One unified model call.
- Model emits `<think>` and `<patch>`.
- Patch is a search/replace hunk, not a hardcoded action.
- Environment applies patch.
- Environment runs pytest repeatedly.
- Reward is deterministic and verifier-based.
- No LLM judge is used in the active reward path.
- OpenEnv server exposes the environment over WebSocket/FastAPI.

Important: docs such as `README.md` may still describe V2. The most accurate code-level sources are:

- `__init__.py`
- `agent/unified_agent.py`
- `inference.py`
- `server/FlakeForge_environment.py`
- `server/reward.py`
- `server/patch_applier.py`
- `models.py`

## High-Level Runtime Flow

The runtime loop is:

```text
User runs inference.py
  -> inference.py creates an LLM backend
  -> UnifiedFlakeForgeAgent builds prompt from observation
  -> model returns raw text
  -> parser extracts:
       think_text
       patch_text
       predicted_category
       predicted_confidence
  -> FlakeForgeEnvironment.step(action)
       applies patch
       checks syntax
       runs target test N times
       computes reward
       updates episode state
       returns observation with reward metadata
  -> inference.py logs result and repeats until done
```

The current episode ends when:

- post-patch pass rate reaches `1.0`,
- max step count is reached,
- or regression is detected.

## Main Files And Responsibilities

### `models.py`

This is the schema source of truth.

It defines:

- `FlakeForgeAction`
- `FlakeForgeObservation`
- `FlakeForgeState`
- `RunRecord`
- `PatchRecord`
- `RewardBreakdown`
- root cause category constants
- helper functions like `failure_mode_entropy`

#### `FlakeForgeAction`

Current action fields:

- `raw_response`: full raw model response.
- `think_text`: parsed reasoning block.
- `patch_text`: parsed patch block.
- `predicted_category`: root cause category extracted from `think_text`.
- `predicted_confidence`: confidence extracted from `think_text`.
- `action_type`: legacy compatibility field, defaults to `"UNIFIED_PATCH"`.
- `parameters`: legacy compatibility dict.

Current action style is not the old seven-action enum. The model effectively sends a free-form repair patch.

#### `FlakeForgeObservation`

The observation contains:

- episode metadata:
  - `episode_id`
  - `test_identifier`
  - `step`
  - `steps_remaining`
- code context:
  - `test_function_source`
  - `source_under_test`
  - `relevant_imports`
  - `file_tree`
- execution history:
  - `run_history`
  - `current_pass_rate`
  - `baseline_pass_rate`
- patch history:
  - `patches_applied`
  - `total_diff_lines`
- deep flakiness signals:
  - `module_cache_violations`
  - `fixture_scope_risks`
  - `mock_residue_sites`
  - `import_side_effect_files`
  - `async_contamination_alive`
- causal fields:
  - `failure_frontier`
  - `call_chain_to_frontier`
  - `boundary_crossings`
  - `causal_graph`
  - `causal_hints`
- failure/debug context:
  - `failure_pattern_summary`
  - `duration_fingerprint`
  - `failing_stack_trace`
- previous step context:
  - `last_think_text`
  - `last_patch_text`
  - `last_reward`
  - `reward_breakdown`
  - `patch_result`
  - `done_reason`
  - `reward`
  - `done`

`patch_result` and `done_reason` were added because OpenEnv remote results were dropping `info` metadata. The client now reconstructs `info` from observation fields.

#### `FlakeForgeState`

Compact state:

- `episode_id`
- `step_count`
- `done`
- `current_pass_rate`
- `baseline_pass_rate`
- `regression_detected`

### `server/state.py`

Defines internal `EpisodeState`.

This is server-side state, richer than public `FlakeForgeState`.

Important fields:

- current source snapshots:
  - `original_test_source`
  - `original_source_under_test`
  - `current_test_source`
  - `current_source_under_test`
- run stats:
  - `run_history`
  - `baseline_pass_rate`
  - `current_pass_rate`
  - `baseline_entropy`
- patch tracking:
  - `patches_applied`
  - `total_diff_lines`
- last action/result tracking:
  - `last_think_text`
  - `last_patch_text`
  - `last_reward`
  - `last_reward_breakdown`
  - `last_patch_result`
  - `last_done_reason`
- flakiness signals:
  - module cache
  - fixtures
  - mocks
  - import side effects
  - async contamination
- causal fields and file tree.

The environment keeps one `EpisodeState` in memory per environment instance.

### `agent/unified_agent.py`

This is the model-facing agent layer.

It contains:

- `UNIFIED_SYSTEM_PROMPT`
- `build_unified_prompt`
- parsing helpers:
  - `extract_think`
  - `extract_patch`
  - `extract_category_from_think`
  - `extract_confidence_from_think`
  - `infer_category_from_patch`
- `UnifiedFlakeForgeAgent`

#### Prompt Contract

The model is told to return exactly two XML-ish blocks:

```xml
<think>
Root Cause: async_wait (confidence: 0.85)
Evidence: ...
Strategy: ...
</think>

<patch>
--- source.py
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
</patch>
```

The patch block must use search/replace hunks.

The prompt tells the model:

- no Markdown fences,
- exact search text,
- include separators,
- do not add sleep/retry hacks,
- prefer minimal surgical fixes,
- fix test or source depending on root cause.

#### Prompt Contents

`build_unified_prompt` includes:

- test id,
- step count,
- current pass rate,
- baseline pass rate,
- test source,
- source under test,
- recent run history,
- failing stack trace,
- deep flakiness signals,
- causal frontier,
- file tree,
- previous patches,
- previous reasoning.

This prompt is currently plain text, not tool-calling.

#### Parsing

`extract_patch` is intentionally tolerant. It handles:

- normal `<patch>...</patch>`,
- Markdown fences such as ```xml,
- fenced patch/diff blocks,
- last-resort hunk extraction if tags are missing.

This was added because local/small models often wrapped outputs in Markdown.

#### Category Inference

`extract_category_from_think` maps phrases like:

- `timeout` -> `async_wait`
- `race` -> `concurrency`
- `mock` -> `mock_residue`
- `fixture` -> `fixture_scope_leak`

`infer_category_from_patch` is a rough heuristic based on patch text:

- timeout/wait_for -> `async_wait`
- lock/semaphore -> `concurrency`
- fixture/teardown -> `fixture_scope_leak`
- mock/monkeypatch -> `mock_residue`
- cache clear -> `module_cache_pollution`
- seed/random -> `nondeterminism`

This heuristic is used for reasoning-consistency reward.

### `inference.py`

This is the main local CLI/inference driver.

It provides:

- `LLMBackend`
- `FlakeForgeEnvClient`
- `_as_step_output_like`
- `_info_from_observation`
- `run_episode`
- `run_inference`
- CLI `main`
- a simplified `flakeforge_reward_fn` for GRPO-style offline reward

#### Environment Selection

`run_inference` decides remote vs local mode using:

- `ENV_BASE_URL`
- `USE_DOCKER_IMAGE`

If remote mode is enabled, it connects to an OpenEnv server at:

```text
ENV_BASE_URL or http://localhost:5000
```

In practice the user has been running:

```powershell
uv run server --port 8000
python inference.py
```

and setting:

```powershell
$env:ENV_BASE_URL = "http://localhost:8000"
```

or relying on existing env configuration.

If not remote, `inference.py` constructs a local `FlakeForgeEnvironment`.

#### LLM Backend

`LLMBackend` supports:

- OpenAI-compatible APIs,
- Ollama native path if `OLLAMA_API_KEY` exists or API base includes `11434`.

Important env vars:

- `MODEL_NAME`
- `API_BASE_URL`
- `OPENAI_API_BASE`
- `NVIDIA_API_KEY`
- `OPENAI_API_KEY`
- `OLLAMA_API_KEY`
- `MAX_TOKENS`
- `TEMPERATURE`

Default temperature was lowered to `0.2` to make patch formatting more deterministic.

#### Run Loop

`run_episode`:

1. Calls `env.reset`.
2. Logs baseline pass rate.
3. While not done:
   - calls `agent.generate(observation)`,
   - logs category/confidence/patch length,
   - calls `env.step(action)`,
   - normalizes the step result,
   - logs reward, pass rate, done reason,
   - records trajectory item.
4. Returns result dict.

Trajectory fields:

- `step`
- `predicted_category`
- `predicted_confidence`
- `think_text`
- `patch_applied`
- `reward`
- `reward_breakdown`
- `pass_rate`
- `done`

#### Remote Metadata Issue And Fix

Earlier, remote OpenEnv `info` was set to `{}` in `_parse_result`, which caused false `patch_applied=false` even when server applied the patch.

Current fix:

- server puts `patch_result`, `reward_breakdown`, `done_reason` on `FlakeForgeObservation`,
- inference reconstructs `info` from observation via `_info_from_observation`.

### `server/FlakeForge_environment.py`

This is the OpenEnv environment implementation.

It subclasses:

```python
Environment[FlakeForgeAction, FlakeForgeObservation, FlakeForgeState]
```

#### Constructor

Important constructor args:

- `repo_path`
- `test_identifier`
- `max_steps`
- `num_runs`
- `runner`
- `chaos_runner`

Defaults:

```python
default_repo = os.environ.get("FF_REPO_PATH", "test_repos/timing_race_minimal")
default_test = os.environ.get("FF_TEST_ID", "tests/test_flaky.py::test_fetch_should_complete")
```

If no runner is provided, it creates:

```python
DockerTestRunner(str(self.repo_path))
```

Despite the name, this runner currently runs local pytest subprocesses.

#### `reset`

`reset`:

1. Allows `repo_path`, `test_identifier`, `max_steps`, `num_runs` override from kwargs.
2. Reads test source and source under test.
3. Builds file tree.
4. Runs baseline tests `num_runs` times.
5. Computes baseline pass rate.
6. Computes baseline failure entropy.
7. Extracts failing trace/error type from failed run.
8. Builds deep flakiness signals.
9. Extracts failure frontier and call chain.
10. Checks order dependency and infrastructure sensitivity if supported.
11. Builds causal graph if possible.
12. Initializes `EpisodeState`.
13. Builds first observation.
14. Sets `_openenv_state`.

Baseline pass rate is computed from the current repo contents. If the target repo was already patched, baseline can be `1.00`.

That behavior caused confusion during demos, because inference modifies target repos in-place.

#### `step`

`step`:

1. Increments step count.
2. Applies patch with `apply_search_replace_patch`.
3. Checks syntax on modified files.
4. Runs target test `num_runs` times if patch applied and syntax is okay.
5. Computes post pass rate.
6. Computes reward using `compute_verifiable_reward`.
7. Updates episode state:
   - current pass rate
   - run history
   - last think/patch
   - last reward
   - last reward breakdown
   - last patch result
   - patches applied
   - total diff lines
   - regression flag
8. Re-reads sources.
9. Determines done:
   - pass rate >= 1.0
   - or step count >= max steps
   - or regression detected
10. Builds final observation.
11. Adds reward/patch/done metadata.
12. Returns observation.

#### Done Reasons

`_done_reason` returns:

- `in_progress`
- `fully_stable`
- `regression_detected`
- `max_steps_reached`
- `unknown`

#### Source Reading

`_read_sources`:

- Reads test file from `test_identifier`.
- Extracts imports from test source.
- Tries to find source file by import name:
  - `<repo>/<module>.py`
  - `<repo>/src/<module>.py`

This works for toy repos like `from source import ...`.

#### Test Running

`_run_tests(n)` calls:

```python
self.runner.run_test(self.test_identifier)
```

for each of `n` runs.

If no runner exists, it uses random synthetic runs. That is useful for development but dangerous for serious evaluation.

### `server/docker_runner.py`

Despite the name, this is currently a local pytest runner.

`DockerTestRunner.run_test`:

```python
subprocess.run(
    ["pytest", test_id, "--tb=short", "-q", "--no-header"],
    cwd=self.repo_path,
)
```

It returns `RunRecord`.

Important recent fix:

```python
passed = proc.returncode == 0
```

Previously it required `"1 passed"` in output. That broke quiet pytest output and made successful runs count as failed.

Methods:

- `run_test`
- `run_test_n_times`
- `check_regressions`

`check_regressions` runs all tests under `tests/` except the target file and returns whether regressions exist. However, regression checking is not yet central in the V3 reward loop.

### `server/patch_applier.py`

This module parses and applies model patches.

Patch format:

```text
--- path/to/file.py
<<<<<<< SEARCH
exact old text
=======
new text
>>>>>>> REPLACE
```

Core functions:

- `parse_search_replace_hunks`
- `apply_search_replace_patch`
- `_apply_single_hunk`
- `_apply_fuzzy_hunk`
- `_normalize_whitespace`
- `_count_lines_changed`
- `_make_unified_diff`

#### Apply Behavior

`apply_search_replace_patch`:

1. Parses hunks.
2. For each hunk:
   - resolves file path,
   - falls back to finding candidate by filename,
   - reads original file,
   - attempts exact replacement,
   - attempts fuzzy replacement,
   - writes modified content,
   - records diff.
3. Uses in-memory rollback if any hunk fails.
4. Returns dict:
   - `success`
   - `files_modified`
   - `lines_changed`
   - `hunks_applied`
   - `diff`
   - `error`

Recent fix:

- Missing target files now fail instead of being silently skipped.
- Partial multi-hunk failure rolls back.

Known weakness:

- It does not yet enforce path allowlists/denylists.
- It does not yet treat no-op patches as failure.
- Fuzzy replacement could be improved.
- It applies directly to the target repo, not an isolated copy.

### `server/reward.py`

This is the deterministic reward system.

Reward components:

1. Format compliance.
2. Compile/syntax result.
3. Stability/pass-rate improvement.
4. Causal proximity.
5. Failure entropy reduction.
6. Anti-hack penalty.
7. Reasoning consistency.
8. Terminal bonus.

#### Format Reward

`compute_format_reward(action)`:

- checks `think_text` contains root cause and confidence,
- checks `patch_text` contains search/replace markers,
- returns up to `1.0`.

#### Compile Reward

`compute_compile_reward(patch_applied_successfully, syntax_error)`:

- `1.0` for applied and no syntax error,
- `-0.5` for syntax error,
- `-1.0` if patch did not apply.

Note: patch parse failure is currently treated similarly to compile failure.

#### Stability Reward

`compute_stability_reward(baseline_pass_rate, current_pass_rate)`:

- `2.0` if current pass rate is `1.0`,
- positive scaled reward for improvement,
- negative for regression,
- `-0.1` for no change.

This is weighted heavily in final reward.

#### Causal Proximity Reward

Rewards patches near failure frontier/call chain.

Weakness: if failure frontier is empty, returns `0.0`.

#### Entropy Reward

Rewards reduced diversity of failure modes.

#### Anti-Hack Penalty

Penalizes:

- assertion deletion,
- added sleeps,
- broad `except`,
- skip decorators,
- excessive patch size.

Missing anti-hack checks:

- weakening tests while keeping asserts,
- editing pytest config,
- editing runner/env files,
- modifying imports to fake source,
- hardcoding only the target test,
- global monkeypatching.

#### Reasoning Consistency

Compares category from reasoning with category inferred from patch text.

This is useful but heuristic.

#### Composite Reward

`compute_verifiable_reward` combines:

```text
format * 0.5
compile * 1.0
stability * 2.0
causal_proximity * 0.5
failure_entropy * 0.5
anti_hack * 1.5
reasoning_consistency * 0.5
terminal_bonus * 1.0
```

Output is `RewardBreakdown`.

### `server/deep_flakiness.py`

Static analysis for flakiness signals.

Detects:

- module cache pollution,
- fixture scope leaks,
- monkeypatch/mock residue,
- import side effects,
- async/thread contamination.

It also builds combined observation signals.

This is a good architecture piece because it makes observations richer than raw pytest output.

Limitations:

- Static heuristics are approximate.
- Many signals may be empty for toy repos.
- Runtime evidence is still more important for real flaky tests.

### `server/causal_graph.py`

Builds a lightweight cross-repo call graph from test entry file/function.

Used by environment in reset:

```python
builder = CrossRepoGraphBuilder(str(self.repo_path), max_depth=3)
graph = builder.build(entry_file=entry_file, entry_function=entry_function)
```

Result can populate:

- `causal_graph`
- `causal_hints`
- boundary warnings.

This is ambitious, but not yet central to reward quality.

### `server/app.py`

FastAPI/OpenEnv app wrapper.

It creates:

```python
app = create_app(FlakeForgeEnvironment, FlakeForgeAction, FlakeForgeObservation)
```

It runs uvicorn with:

```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Entry point in `pyproject.toml`:

```toml
server = "FlakeForge.server.app:main"
```

Common local command:

```powershell
uv run server --port 8000
```

### `client.py`

Thin client wrapper.

Functions:

- generate action from observation,
- parse raw model response,
- run remote episode using HTTP-style endpoints.

Important issue:

- `env_url` default is `http://localhost:8080`, but most of the app uses port `8000`.

This should be aligned.

### `training/grpo_trainer.py`

Training scaffold.

It provides:

- `build_reward_function(use_execution=False)`
- `_offline_reward_fn`
- `_execution_reward_fn`
- `create_trainer`

#### Offline Reward

Only evaluates:

- format reward,
- reasoning consistency.

This is useful for format bootstrapping but does not verify real patches.

#### Execution Reward

Attempts to call:

```python
env.step(action)
```

and use `step_output.reward`.

Weakness:

- It assumes env is passed through kwargs.
- It does not clearly reset per sample/generation.
- It is not yet aligned with TRL's recommended OpenEnv `environment_factory` approach.
- It risks cross-sample contamination if the env mutates a shared repo.

#### Trainer Creation

Uses:

- `GRPOConfig`
- `GRPOTrainer`
- HuggingFace model/tokenizer loading.

This is a scaffold, not proven production training.

## OpenEnv Integration

OpenEnv pieces:

- `server/app.py` uses OpenEnv `create_app`.
- `models.py` subclasses OpenEnv action/observation/state base classes.
- `inference.py` optionally uses `EnvClient`.
- `openenv.yaml` declares app, runtime, port.

`openenv.yaml`:

```yaml
spec_version: 1
name: FlakeForge
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

Current remote mode is WebSocket-based through OpenEnv client.

## Environment Variables

Common runtime variables:

### Target Repo/Test

```powershell
$env:FF_REPO_PATH = "test_repos/timing_race_minimal"
$env:FF_TEST_ID = "tests/test_flaky.py::test_fetch_should_complete"
```

For harder demo:

```powershell
$env:FF_REPO_PATH = "test_repos/multi_step_flaky"
$env:FF_TEST_ID = "tests/test_flaky.py::test_profile_fetch_should_be_stable"
```

### Episode

```powershell
$env:INFERENCE_MAX_STEPS = "5"
```

`num_runs` defaults to CLI argument `--num-runs 10` unless changed.

### Remote Env

```powershell
$env:ENV_BASE_URL = "http://localhost:8000"
```

If `ENV_BASE_URL` is set, `inference.py` uses remote OpenEnv client.

### Model

```powershell
$env:MODEL_NAME = "..."
$env:API_BASE_URL = "..."
$env:OPENAI_API_BASE = "..."
$env:OPENAI_API_KEY = "..."
$env:NVIDIA_API_KEY = "..."
$env:TEMPERATURE = "0.2"
$env:MAX_TOKENS = "4096"
```

### Ollama

If API base includes `11434` or `OLLAMA_API_KEY` is set, `LLMBackend` uses native Ollama path.

## Demo Repositories

### `test_repos/timing_race_minimal`

Original flaky behavior:

```python
timeout = 0.05 if random.random() < 0.8 else 0.5
```

The async operation takes about `0.15s`, so `0.05s` causes intermittent timeout.

Stable fix:

```python
timeout = 0.5
```

Target:

```text
tests/test_flaky.py::test_fetch_should_complete
```

Important: inference patches this repo in-place. If it is already fixed, baseline pass rate will be `1.00`.

### `test_repos/multi_step_flaky`

Created as a harder demo repo.

Target:

```text
tests/test_flaky.py::test_profile_fetch_should_be_stable
```

It has two flaky gates:

1. async timeout race:
   - `0.03s` timeout sometimes wraps operation taking about `0.10s`.
2. payload nondeterminism:
   - `build_payload` randomly returns `"stable-request"` or `"stale-request"`.

Reset command:

```powershell
python test_repos/multi_step_flaky/reset_demo.py
```

Run command:

```powershell
$env:FF_REPO_PATH = "test_repos/multi_step_flaky"
$env:FF_TEST_ID = "tests/test_flaky.py::test_profile_fetch_should_be_stable"
$env:INFERENCE_MAX_STEPS = "5"
python inference.py
```

Important: strong models may patch both issues in one step.

## Current Known Runtime Behaviors

### Baseline Pass Rate Can Be 1.00

Baseline is measured from current repo contents during `reset`.

If previous inference already patched source, baseline can become `1.00`.

This is expected with current in-place mutation architecture.

### Server Port Conflict

If this appears:

```text
WinError 10048
only one usage of each socket address is normally permitted
```

it means port `8000` is already occupied by an existing server.

### Server May Close After Client

The OpenEnv/WebSocket session closes after inference. This is normal at connection level. If uvicorn itself shuts down, the terminal/server process was stopped or provider lifecycle ended.

### Patch Applied But `patch_applied=false`

This was a bug caused by dropped `info` metadata in remote mode. It was fixed by carrying `patch_result` through observation and reconstructing info.

### Pass Rate Stayed 0.00 After Good Patch

This was caused by `DockerTestRunner` requiring `"1 passed"` in pytest output. Quiet pytest outputs `.`. Fixed by trusting return code.

## Current Weaknesses

### 1. No Isolated Workspaces

The environment mutates target repos directly.

Consequences:

- baseline changes across runs,
- demos require reset scripts,
- GRPO rollouts can contaminate each other,
- not safe for parallel training.

Needed architecture:

```text
seed repo -> temp episode workspace -> patch/test -> archive diff -> cleanup
```

### 2. Runner Name Is Misleading

`DockerTestRunner` is local subprocess pytest, not Docker.

Either rename it or implement real Docker isolation.

### 3. GRPO Integration Is Incomplete

The current GRPO file is a scaffold. It does not yet implement the recommended TRL/OpenEnv `environment_factory` pattern.

Expected future shape:

```python
class FlakeForgeToolEnv:
    def reset(self, **kwargs) -> str:
        ...

    def propose_patch(self, root_cause: str, patch: str) -> str:
        """Apply patch and return observation text."""
        ...

def reward_func(environments, **kwargs):
    return [env.reward for env in environments]
```

### 4. Reward Can Be Hacked

Need stronger checks around:

- editing tests,
- weakening assertions,
- skip/xfail,
- editing pytest config,
- modifying runner/environment,
- changing import path,
- hardcoding answers,
- adding broad mocks,
- deleting flaky behavior in unrealistic ways.

### 5. Docs Are Stale

`README.md` is not aligned with V3.

Another LLM should not trust README without cross-checking code.

The new review file:

```text
HACKATHON_ARCHITECTURE_REVIEW.md
```

contains recommendations.

This file:

```text
PRESENT_ARCHITECTURE_CONTEXT.md
```

is intended to describe current reality.

### 6. Client Port Default

`client.py` defaults to `8080`, while server uses `8000`.

### 7. Reward and Verifier Are Entangled

Currently reward computation directly calculates component scores. A cleaner architecture would separate:

```text
VerifierResult -> RewardProjection
```

That would make reward auditing easier.

## Recommended Mental Model For Another LLM

Think of FlakeForge as three layers:

### Layer 1: Environment

Owns:

- repo state,
- reset,
- patch application,
- test execution,
- reward,
- done condition.

Main file:

```text
server/FlakeForge_environment.py
```

### Layer 2: Agent

Owns:

- prompt construction,
- model call,
- parsing raw model response into action.

Main files:

```text
agent/unified_agent.py
inference.py
```

### Layer 3: Training Adapter

Currently weak/scaffold.

Owns:

- TRL/GRPO integration,
- reward function compatibility,
- model training.

Main file:

```text
training/grpo_trainer.py
```

Future should add:

```text
training/openenv_grpo_adapter.py
```

## Important Commands

### Start Server

```powershell
uv run server --port 8000
```

Run this in one terminal.

### Run Basic Inference

```powershell
python inference.py
```

### Run Timing Race Target

```powershell
$env:FF_REPO_PATH = "test_repos/timing_race_minimal"
$env:FF_TEST_ID = "tests/test_flaky.py::test_fetch_should_complete"
$env:INFERENCE_MAX_STEPS = "5"
python inference.py
```

### Run Multi-Step Demo Target

```powershell
$env:FF_REPO_PATH = "test_repos/multi_step_flaky"
$env:FF_TEST_ID = "tests/test_flaky.py::test_profile_fetch_should_be_stable"
$env:INFERENCE_MAX_STEPS = "5"
python test_repos/multi_step_flaky/reset_demo.py
python inference.py
```

### Run Focused Regression Tests

```powershell
python -m pytest tests/test_action_and_judge_regressions.py -q
```

### Run Integration Tests

```powershell
python -m pytest tests/test_flakeforge_integration.py -q
```

## Recent Fixes Already Made

These were recently changed and are part of current expected behavior:

- Prompt example in `agent/unified_agent.py` was fixed to show a complete patch hunk.
- Prompt now tells model not to use Markdown fences.
- `extract_patch` handles fenced XML/patch responses.
- Default inference temperature lowered to `0.2`.
- `patch_applier` now fails missing files instead of silently skipping.
- `patch_applier` rolls back partial multi-hunk failures.
- `FlakeForgeObservation` now carries `patch_result` and `done_reason`.
- `EpisodeState` now stores last patch result and done reason.
- `inference.py` reconstructs `info` metadata from observation.
- `DockerTestRunner` now treats pytest return code `0` as pass.
- Added `test_repos/multi_step_flaky`.
- Added `HACKATHON_ARCHITECTURE_REVIEW.md`.

## Suggested Next Code Changes

If another LLM is asked to continue, highest-impact changes are:

1. Add `server/workspace.py` for isolated per-episode repo copies.
2. Change environment reset/step to patch isolated workspace, not seed repo.
3. Rename `DockerTestRunner` to `LocalPytestRunner`.
4. Add `server/verifiers.py` to separate verifier outputs from reward weights.
5. Add protected-file policy.
6. Add regression-suite reward into V3 step.
7. Add `test_repos/catalog.yaml`.
8. Add `training/openenv_grpo_adapter.py` following TRL `environment_factory`.
9. Rewrite README to match V3.
10. Fix `client.py` default port to `8000`.

## Final Current-State Assessment

The current system is a functional V3 inference environment:

- It can run an OpenEnv server.
- It can reset a flaky-test target.
- It can prompt a model.
- It can parse the model's reasoning and patch.
- It can apply patches.
- It can run pytest repeatedly.
- It can compute a multi-component verifier reward.
- It can terminate when the test becomes stable.

But it is not yet a fully mature RL training platform:

- workspaces are not isolated,
- GRPO integration is only scaffolded,
- reward hacking checks need hardening,
- docs are inconsistent,
- runner naming/deployment story is misleading,
- demos mutate their own seed repos.

The best next architectural move is not to replace V3. It is to wrap V3 in stronger environment infrastructure: isolated workspaces, task catalog/curriculum, verifier stack, and TRL-compatible environment factory.

