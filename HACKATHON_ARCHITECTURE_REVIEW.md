# FlakeForge Hackathon Architecture Review

Date: 2026-04-25

This is a blunt architecture review and upgrade plan for the hackathon submission. The goal is to make FlakeForge feel like an ambitious OpenEnv/RLVR system, not just a prompt loop that patches one toy test.

## Executive Summary

FlakeForge has a strong core idea: train an LLM to repair flaky tests using execution-verified rewards inside an OpenEnv environment. That is a very good hackathon direction because it matches the organizer guidance: step-by-step interaction, programmatic verification, reward engineering, OpenEnv deployment, and GRPO-style post-training.

But the current architecture story is split in two:

- The code has moved toward V3: a unified agent emits `<think>` and `<patch>`, the environment applies search/replace patches, pytest verifies the result, and deterministic rewards replace the LLM judge.
- The docs and some modules still describe V2: analyzer/fixer roles, LoRA adapters, a frozen judge, seven hardcoded actions, Docker-isolated execution, and seed repos that do not appear to exist in the current repo.

That mismatch is the biggest submission risk. Judges will read the README and expect one system, then inspect or run the code and see another.

My recommendation: reposition FlakeForge as an "Adaptive Verifiable Environment for Flaky Test Repair" and make the architecture visibly environment-first:

1. Keep the V3 unified agent path.
2. Add a task/curriculum layer that generates or selects flaky repos by difficulty.
3. Add stronger verifier layers: target test, regression suite, anti-hack checks, diff quality, and holdout replay.
4. Add a proper TRL/OpenEnv `environment_factory` training wrapper.
5. Add isolated per-episode workspaces so GRPO rollouts cannot contaminate each other.
6. Rewrite docs around the real architecture.

If we do those, the project becomes much more ambitious: not "a model edits a flaky test", but "a verifiable RL environment that teaches code agents to debug nondeterministic failures through execution feedback."

## What The Hackathon Materials Emphasize

The three organizer `.txt` files repeatedly push the same themes:

- Build an environment, not only a model wrapper.
- Make success programmatically verifiable.
- Use multiple independent reward checks.
- Avoid reward hacking.
- Start with simple tasks but add curriculum/adaptive difficulty.
- Use OpenEnv as the standardized reset/step/state interface.
- Use TRL/GRPO and optionally Unsloth for efficient RL post-training.
- Inspect outputs and reward components, not just average reward.
- Deploy early and make the demo reproducible.

External references say the same thing:

- TRL's OpenEnv docs recommend using `environment_factory` for interactive GRPO. The trainer creates one environment instance per generation, calls `reset`, exposes typed public methods as tools, and reads reward from environment state.
- TRL docs also warn that simple final-state rewards often work better than over-shaped rewards, and that rewards should be manually tested before training.
- OpenEnv positions environments as deployable, versioned, backend-server artifacts, not ad hoc scripts.
- Recent RLVR/reward-hacking discussions warn that stronger models exploit verifier flaws. So reward hacking audits matter as much as reward design.

Implication for us: FlakeForge should lean harder into "verifiable environment and verifier engineering" than into "we used a clever prompt."

## Current Architecture As Implemented

The actual current code is closest to this:

```text
LLM backend
  -> UnifiedFlakeForgeAgent
       builds observation prompt
       emits <think> + <patch>
  -> FlakeForgeEnvironment.step(action)
       parses/apply search-replace patch
       syntax-checks changed files
       runs target pytest N times
       computes multi-signal reward
       returns observation/state/reward metadata
```

Main modules:

- `agent/unified_agent.py`: prompt, parser, category extraction, action generation.
- `inference.py`: local/remote inference loop and OpenAI/Ollama-compatible backend.
- `server/FlakeForge_environment.py`: OpenEnv environment, reset/step/state, source reading, baseline measurement, reward application.
- `server/patch_applier.py`: search/replace hunk parsing and atomic-ish patch application.
- `server/reward.py`: format, compile, stability, causal proximity, entropy, anti-hack, reasoning-consistency rewards.
- `server/docker_runner.py`: actually runs local pytest subprocesses, despite the name.
- `models.py`: OpenEnv action/observation/state schemas.
- `training/grpo_trainer.py`: early GRPO scaffolding.
- `server/app.py`: OpenEnv/FastAPI app.

The good news: this is a coherent V3 architecture. The bad news: packaging, docs, runner naming, and training integration are behind the code.

## What Is Strong

### 1. The task is a good RLVR/RLVE fit

Flaky test repair has the right shape:

- The agent can act step-by-step.
- Success can be verified by repeated test execution.
- There are meaningful partial signals: compile success, pass-rate delta, failure entropy, regression checks.
- It is hard enough to be interesting but can be made easy with toy repos and curriculum.

This matches the hackathon guidance well.

### 2. V3 removed the weakest V2 idea

Removing the LLM judge from the hot reward path is good. LLM judges are easy to impress and hard to trust under optimization pressure. Execution-verified reward is a stronger story.

The current V3 direction is better than V2:

- One model emits reasoning and patch together.
- Reward comes from execution, not a judge's vibe.
- Search/replace patches are transparent and auditable.
- Observations include run history and static flakiness signals.

### 3. Reward is already multi-signal

`server/reward.py` has the right ingredients:

- Format compliance.
- Patch/syntax success.
- Stability improvement.
- Causal proximity.
- Failure entropy reduction.
- Anti-hack penalty.
- Reasoning/patch consistency.
- Terminal bonus.

That maps well to the organizer advice: multiple independent checks reduce reward hacking risk.

### 4. Observations contain more than raw test output

`FlakeForgeObservation` carries:

- Source under test.
- Test source.
- Run history.
- Failure trace.
- Static deep-flakiness signals.
- Patch history.
- Reward metadata.
- Causal frontier and call-chain fields.

This is good because it lets the environment become richer without changing the basic agent loop.

### 5. Recent debugging improved real reliability

Recent fixes mattered:

- Patch parsing is more tolerant of Markdown/XML wrappers.
- Patch metadata now reaches the inference trajectory.
- Pytest pass detection now trusts return code rather than requiring `"1 passed"` in output.
- The new `multi_step_flaky` demo repo helps show live progress.

Those are practical hackathon improvements.

## What Is Broken Or Weak

### 1. README and code tell different stories

`README.md` still says:

- Two LoRA roles: Analyzer and Fixer.
- Frozen judge model.
- Seven hardcoded actions.
- Docker-isolated runner.
- Seed repos under `seed_repos/`.

But current code says:

- Unified agent.
- No judge.
- Free-form search/replace patches.
- Local pytest subprocess runner.
- Demo repos under `test_repos/`.

This is the highest-priority fix. Judges punish confusion. If they read stale claims, they will assume the project is unfinished or misleading.

### 2. The "DockerTestRunner" is not Docker

`server/docker_runner.py` runs:

```python
subprocess.run(["pytest", test_id, ...], cwd=self.repo_path)
```

That is a local pytest runner. The name and README claim Docker isolation, but the code does not provide per-episode containers.

This matters because code repair agents can mutate files, rely on local state, or contaminate future rollouts. For RL training, isolation is not optional.

Recommended rename or split:

- Rename current class to `LocalPytestRunner`.
- Add `IsolatedWorkspaceRunner` that copies a repo to a temp directory/worktree per episode.
- Later add `DockerPytestRunner` for real container isolation.

### 3. Online GRPO path is not actually production-ready

`training/grpo_trainer.py` is a scaffold, not a strong training architecture.

Problems:

- Execution reward assumes an `env` object is passed through kwargs.
- It does not clearly reset an environment per sample/generation.
- It may mutate the same repo across multiple GRPO completions.
- It does not use TRL's recommended `environment_factory` pattern.
- Offline reward and inference reward are different, creating reward skew.

TRL's OpenEnv integration expects something like:

- An environment wrapper class with no-arg `__init__`.
- `reset(**kwargs)` returning an initial observation string.
- Public typed methods exposed as tools.
- A reward function that reads environment state from the `environments` list.

FlakeForge should add that explicitly.

### 4. The environment mutates the target repo in-place

For inference demos, in-place patching is understandable. For training, it is dangerous.

Current risks:

- Rollout 1 can patch the repo and change rollout 2's baseline.
- Failed patches may leave partial state if a path escapes the intended workspace.
- Re-running demos gives confusing baseline pass rates unless reset scripts are used.

Architecture fix:

```text
TaskSpec -> WorkspaceManager -> isolated repo copy/worktree -> Env episode -> cleanup/archive
```

Every episode should run in an isolated workspace. Store the final diff and logs as artifacts.

### 5. Reward can still be hacked

The anti-hack reward catches some obvious issues:

- Removing asserts.
- Adding sleeps.
- Broad try/except.
- Skip decorators.
- Huge patches.

But missing checks include:

- Editing the target test to weaken it without deleting asserts.
- Modifying `pytest.ini`.
- Modifying the runner or environment files.
- Monkeypatching randomness globally.
- Changing import paths so tests import fake code.
- Returning hardcoded objects only for the target test.
- Removing the flaky behavior but breaking other tests.

The reward should have protected-file rules and regression tests as first-class signals.

### 6. Causal proximity can be weak when frontier is empty

`compute_causal_proximity_reward` returns `0.0` if no failure frontier exists. In practice, many pytest failures or model-created failures will not produce a clean frontier. That means an important reward component often disappears.

Better:

- Infer proximity from patched file vs import graph.
- Use test import graph from `test_file -> source_under_test`.
- Use changed symbol names vs stack trace strings.
- Reward touching source-under-test over unrelated files.

### 7. Client/server behavior has been brittle

You saw repeated WebSocket closure issues. Some were caused by environment bugs, but the client experience is still too fragile.

Needed:

- Better server-side error serialization.
- Include patch diff and patch error in observation always.
- Client should print the server error body if received.
- Add a local mode demo path that does not need WebSocket for judging.

### 8. Default ports are inconsistent

`server/app.py` uses port `8000`.

`openenv.yaml` uses port `8000`.

`client.py` defaults to `http://localhost:8080`.

That should be fixed to `8000` everywhere.

### 9. Demo behavior is too easy for strong models

The new `multi_step_flaky` target helps, but a strong model can patch both gates in one shot. That is not bad for capability, but it undercuts the "live multi-step progress" demo.

If the demo must show several steps, the environment should intentionally reveal only one failure class at a time or use staged tasks:

```text
stage 1: async timeout visible
stage 2: after timeout fix, payload nondeterminism visible
stage 3: after payload fix, order dependence visible
```

That should be represented as a curriculum/task environment, not just a single test file.

### 10. Too many stale docs dilute the story

There are many status docs: `MISSION_ACCOMPLISHED.md`, `V2_*`, `START_HERE.md`, `INDEX.md`, `PROGRESS_REPORT.md`, etc.

For submission, either update them or clearly mark them historical. The judge should have one obvious path:

1. `README.md`
2. `ARCHITECTURE.md`
3. `DEMO.md`
4. `TRAINING.md`

## Recommended New Architecture

I would frame the upgraded system like this:

```text
FlakeForge: Adaptive Verifiable Environment for Flaky Test Repair

Task Bank / Generator
  -> creates flaky repo tasks with difficulty metadata
  -> categories: async_wait, shared_state, order_dependency, resource_leak, mock_residue, nondeterminism

Workspace Manager
  -> copies repo into isolated per-episode workspace
  -> snapshots before/after
  -> archives patch, logs, reward breakdown

Observation Builder
  -> test source + source under test
  -> run history + failure entropy
  -> stack trace + causal frontier
  -> static flakiness signals
  -> previous patch outcome

Agent Interface
  -> unified think+patch for inference
  -> tool-style methods for TRL/OpenEnv GRPO
  -> optional format-SFT warm start

Patch Engine
  -> parse search/replace hunks
  -> apply to isolated workspace
  -> syntax check
  -> produce diff AST summary

Verifier Stack
  -> target test repeated N times
  -> regression tests
  -> protected-file policy
  -> anti-hack scanner
  -> holdout replay
  -> reward breakdown

Curriculum Controller
  -> selects next task difficulty
  -> adjusts num_runs, category mix, compound bugs
  -> tracks success bands

Training Adapter
  -> TRL GRPO environment_factory
  -> one environment per generation
  -> reward read from environment state
  -> logs completions, diffs, reward components

Demo Dashboard / CLI
  -> live episode progress
  -> before/after pass-rate
  -> diff viewer
  -> reward component chart/log
```

This feels much more ambitious because the system is not just "prompt -> patch". It becomes a lab for training flaky-test repair agents.

## Specific Changes To Make

### Priority 0: Fix the story before adding features

Create or rewrite:

- `README.md`: concise V3 story.
- `ARCHITECTURE.md`: real architecture, not V2.
- `DEMO.md`: exact commands for local and remote demos.
- `TRAINING.md`: what works now vs planned GRPO.

Remove or mark stale:

- V2 docs.
- judge-model claims.
- seven-action-space claims.
- missing `seed_repos` claims.

### Priority 1: Add isolated workspaces

Add a module:

```text
server/workspace.py
```

Responsibilities:

- Create temp workspace per episode.
- Copy or git-worktree the task repo.
- Expose `episode_repo_path`.
- Cleanup or archive based on config.
- Prevent patches outside workspace.

Environment change:

- `reset()` should create workspace from original task repo.
- `step()` should patch workspace, not source repo.
- Observation should include artifact paths/diff, not mutate the seed.

This solves baseline confusion and makes GRPO possible.

### Priority 2: Add a task bank and curriculum controller

Add:

```text
server/tasks.py
server/curriculum.py
test_repos/catalog.yaml
```

Task metadata:

```yaml
- id: timing_race_minimal
  repo_path: test_repos/timing_race_minimal
  test_id: tests/test_flaky.py::test_fetch_should_complete
  category: async_wait
  difficulty: 1
  expected_fix_files:
    - source.py
  protected_files:
    - tests/test_flaky.py

- id: multi_step_flaky
  repo_path: test_repos/multi_step_flaky
  test_id: tests/test_flaky.py::test_profile_fetch_should_be_stable
  category: compound
  difficulty: 3
```

Curriculum policy:

- If success rate > 80%, move to harder tasks.
- If success rate < 20%, serve easier tasks or add hints.
- Mix categories so training does not overfit async timeout.

This directly addresses the organizer guidance on adaptive difficulty and curriculum.

### Priority 3: Upgrade reward into a verifier stack

Current reward is okay but should be reorganized into named verifier outputs:

```text
VerifierResult
  target_pass_rate
  regression_pass_rate
  syntax_ok
  patch_applied
  protected_files_touched
  test_weakened
  suspicious_patterns
  causal_locality_score
  diff_size
  failure_entropy_delta
```

Then reward is a projection of verifier results:

```text
reward = outcome_success
       + regression_safety
       + syntax
       + minimal_shaping
       - hack_penalties
```

Recommended reward philosophy:

- Keep final success dominant.
- Use shaping only to break ties.
- Penalize hacks hard.
- Log all reward components.
- Keep a holdout evaluator separate from training reward.

Add checks for:

- Test assertion weakening.
- `pytest.ini` changes.
- runner/environment file changes.
- skip/xfail.
- broad monkeypatching.
- hardcoded target-test detection.
- modifying imports in tests.
- deleting test functions.

### Priority 4: Real TRL/OpenEnv training wrapper

Add:

```text
training/openenv_grpo_adapter.py
```

Shape it around TRL's `environment_factory` pattern:

```python
class FlakeForgeToolEnv:
    def __init__(self):
        self.client = FlakeForgeEnvClient(base_url=ENV_URL)
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str:
        # choose task, reset remote env, return compact observation text
        ...

    def propose_patch(self, root_cause: str, patch: str) -> str:
        """
        Apply a proposed search/replace patch to the current flaky repo.

        Args:
            root_cause: The suspected flaky-test root cause category.
            patch: Search/replace patch hunks.
        """
        ...
```

Why not generic `step(action)`? TRL docs warn generic methods are harder for the model to learn as tools. Typed methods with meaningful names work better.

Reward function:

```python
def reward_func(environments, **kwargs):
    return [env.reward for env in environments]
```

This would be a major architecture lift for the submission.

### Priority 5: Improve the patch engine

Current search/replace is simple and understandable. Keep it, but add:

- File allowlist/denylist.
- Normalized path safety.
- Maximum patch count per step.
- No-op patch detection as neutral/negative.
- AST-aware diff summary.
- Fallback to unified diff parsing only if needed.
- Better error messages returned to the model.

Also fix `_apply_single_hunk` normalized whitespace behavior eventually. It currently normalizes for detection but then calls `original.replace(search.strip(), replace.strip(), 1)`, which can fail if indentation differs.

### Priority 6: Add trace-based observation

Right now observations rely mostly on source, test history, and static scans. More ambitious:

- Run failing test once with instrumentation.
- Capture deepest user frame.
- Capture local variables for safe primitive types.
- Capture async task state.
- Capture timing histogram.
- Feed compact trace into observation.

For flaky tests, runtime evidence is often more valuable than static AST signals.

### Priority 7: Add live demo telemetry

For hackathon impact, show:

- baseline pass rate
- patch diff
- reward components
- pass-rate after patch
- failure modes before/after
- reason for done
- artifact path

This can be a CLI log first. A web UI is nice but not required.

## What To Remove Or De-Emphasize

### Remove the V2 judge story from the main path

Do not present the frozen judge as core. It weakens the RLVR story. If kept, frame it as optional analysis or offline annotation, not reward.

### Remove hardcoded seven-action space from public docs

The current system uses free-form patches. The old action space is no longer the product.

### Do not claim Docker isolation until it exists

Either implement it or rename the runner.

### Do not over-market GRPO until the adapter is real

Say:

- "OpenEnv inference loop works."
- "GRPO scaffold exists."
- "Next step is environment_factory training adapter."

That is more credible than claiming complete training if it is not proven.

## Suggested Submission Narrative

Use this story:

> FlakeForge is an OpenEnv environment for reinforcement learning on flaky test repair. It gives a code model a real failing test, lets it patch the repository, verifies the patch through repeated execution, and returns a multi-signal reward designed to resist reward hacking. Unlike static code datasets, FlakeForge supports adaptive curricula over flaky-test categories and can train agents through environment interaction using GRPO.

Demo arc:

1. Show a flaky repo with pass rate below 1.0.
2. Show the model's diagnosis and patch.
3. Show the environment applying patch and rerunning tests.
4. Show reward breakdown.
5. Show final pass rate and diff.
6. Explain how the same environment can be used by TRL/GRPO.

Key phrase:

> We are not training the model to write prettier patches. We are training it to produce patches that survive execution, regression checks, and anti-hack verifiers.

## Architecture Roadmap

### Day 0 / Immediate

- Rewrite README to V3.
- Fix port defaults.
- Rename `DockerTestRunner` or document honestly.
- Add `DEMO.md`.
- Add a one-command reset/run demo.
- Add tests for current bug fixes.

### Day 1

- Add isolated workspace manager.
- Add task catalog YAML.
- Add protected file policy.
- Add regression-suite reward signal.
- Add better no-op patch penalty.

### Day 2

- Add `FlakeForgeToolEnv` for TRL `environment_factory`.
- Add a tiny GRPO smoke script.
- Log completions and reward components.
- Add curriculum task sampler.

### If More Time

- Real Docker runner.
- Runtime trace instrumentation.
- Holdout verifier.
- HF Space demo UI.
- Unsloth QLoRA training recipe.

## Concrete Code Issues To Fix

### `README.md`

Stale. Rewrite around V3.

### `client.py`

Default URL should be `http://localhost:8000`, not `8080`.

### `server/docker_runner.py`

Rename to `local_pytest_runner.py` or implement real Docker execution.

### `training/grpo_trainer.py`

Convert from "reward function calls env.step" to proper TRL `environment_factory` wrapper.

### `server/FlakeForge_environment.py`

Add workspace isolation and protected-file policy. Avoid mutating seed repos directly.

### `server/reward.py`

Separate verifier results from weighted reward. Add regression and protected-file checks.

### `server/patch_applier.py`

Add path safety and no-op handling. Improve fuzzy replacement.

### `test_repos/`

Keep `timing_race_minimal` and `multi_step_flaky`, but add catalog metadata and reset scripts for every task.

### `pyproject.toml` and `server/requirements.txt`

Unify dependencies. Avoid one dependency list for local and another for Docker unless generated from one source.

## Suggested File Additions

```text
ARCHITECTURE.md
DEMO.md
TRAINING.md
server/workspace.py
server/tasks.py
server/curriculum.py
server/verifiers.py
training/openenv_grpo_adapter.py
test_repos/catalog.yaml
```

## Minimal Architecture Upgrade That Would Impress Judges

If time is short, do this exact package:

1. `ARCHITECTURE.md` with the V3 environment-first story.
2. `server/workspace.py` for per-episode copies.
3. `test_repos/catalog.yaml` with two tasks.
4. `server/verifiers.py` with protected-file and regression checks.
5. `training/openenv_grpo_adapter.py` scaffold following TRL `environment_factory`.
6. `DEMO.md` with a reliable command sequence.

That would make FlakeForge feel like a real RL environment platform instead of a single debugging script.

## Final Recommendation

Do not pivot away from the current V3 core. It is the right direction.

But lift the architecture around it:

- Make tasks adaptive.
- Make workspaces isolated.
- Make verifiers first-class.
- Make GRPO integration match TRL/OpenEnv best practices.
- Make docs match code.
- Make the demo reproducible.

The most compelling final product is:

> FlakeForge: an adaptive OpenEnv benchmark and training environment for flaky-test repair, where LLM agents learn from verifiable execution feedback instead of human-written fixes or LLM judges.

That is ambitious, aligned with the hackathon, and still achievable from the code you already have.

