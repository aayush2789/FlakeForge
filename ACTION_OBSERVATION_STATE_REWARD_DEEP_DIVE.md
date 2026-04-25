# FlakeForge Deep Dive: Action/Observation/State, Flow, Oracle, Patch Validator, Reward System, and Risk Findings

## 1. Action Space

Primary schema lives in `models.py` via `FlakeForgeAction`.

### 1.1 Core action layers

- Raw model transport fields:
  - `raw_response`
  - `think_text`
  - `patch_text`
- Structured fields:
  - `structured_think` (`StructuredThink` with `ThinkClaim[]`)
  - `structured_patch` (`StructuredPatch` with `PatchHunk[]`)
- Scalar guidance fields:
  - `predicted_category`
  - `predicted_confidence`

This gives the system two representations of the same step:
1. Human/debuggable text for logs and backward compatibility.
2. Machine-verifiable structured reasoning and patch hunks.

### 1.2 How actions are produced

In `agent/unified_agent.py`, `UnifiedFlakeForgeAgent.generate()`:
1. Builds a high-context prompt from current observation.
2. Calls model backend with a strict JSON schema contract.
3. Parses returned JSON into structured think/patch.
4. Converts structured patch hunks into search/replace patch text.
5. Fills `FlakeForgeAction` with both structured and fallback fields.

### 1.3 Why this matters

- The environment can reward not only whether code changed, but whether the reasoning claims are consistent with the code diff.
- Structured claims and hunk-to-claim links (`addresses_claim`) make oracle scoring possible.

---

## 2. Observation Space

Primary schema is `FlakeForgeObservation` in `models.py`.

### 2.1 Observation groups

- Episode coordinates:
  - `episode_id`, `test_identifier`, `step`, `steps_remaining`
- Code context:
  - `test_function_source`, `source_under_test`, `relevant_imports`, `file_tree`
- Runtime evidence:
  - `run_history`, `current_pass_rate`, `baseline_pass_rate`
- Preflight classification:
  - `env_type`, `should_train`, `preflight_result`
- Deep flakiness signals:
  - `module_cache_violations`
  - `fixture_scope_risks`
  - `mock_residue_sites`
  - `import_side_effect_files`
  - `async_contamination_alive`
- Causal localization:
  - `failure_frontier`, `call_chain_to_frontier`, `boundary_crossings`
  - `causal_graph`, `causal_hints`
- Step memory and reward context:
  - `last_think_text`, `last_patch_text`, `last_reward`
  - `reward_breakdown`, `patch_result`, `done_reason`
  - `think_history` for diversity penalties and anti-loop prompting

### 2.2 Observation construction

`server/FlakeForge_environment.py` builds observations in `_build_observation()` using internal `EpisodeState` plus latest run/patch/reward outputs.

---

## 3. State Space

There are two state forms by design.

### 3.1 Internal state (authoritative)

`EpisodeState` in `server/state.py` stores full episode internals:
- source snapshots
- run history
- preflight metadata
- patch history
- last reward breakdown
- step think history
- regression flags

### 3.2 OpenEnv/API state (projected)

`FlakeForgeState` in `models.py` is the compact state surfaced through environment interfaces:
- `episode_id`
- `step_count`
- `done`
- `current_pass_rate`
- `baseline_pass_rate`
- `regression_detected`
- `env_type`
- `should_train`

This split keeps transport simple while preserving rich internal bookkeeping.

---

## 4. End-to-End Flow Across Files

## 4.1 Inference orchestration

`inference.py`:
1. Build backend and unified agent.
2. Build environment.
3. Run `run_episode()` loop.

## 4.2 Reset phase

`FlakeForgeEnvironment.reset()`:
1. Apply reset kwargs (repo/test/max_steps/num_runs).
2. Collect source and file tree.
3. Run preflight gate:
   - sanity -> determinism -> flakiness confirm.
4. Compute baseline pass rate and entropy.
5. Build deep signals and causal frontier.
6. Initialize `EpisodeState` and first observation.

## 4.3 Step phase

`FlakeForgeEnvironment.step(action)`:
1. Snapshot pre-sources.
2. Validate patch via `PatchValidator` (no disk write if invalid).
3. If valid, write validated sources.
4. Syntax sanity check, rollback if broken.
5. Run repeated tests.
6. Verify structured claims via oracle.
7. Compute reward via `compute_verifiable_reward`.
8. Update state, think history, patch history.
9. Return final observation for next step.

---

## 5. Oracle Engine (Reasoning Verifier)

Main entry: `server/oracle_engine.py::verify_structured_think()`.

## 5.1 Purpose

Oracle answers: "Do the structured reasoning claims match pre/post code evidence?"

### 5.2 Mechanism

- Each `ThinkClaim` is routed by category to a plugin.
- Plugins inspect AST/libcst patterns in pre and post sources.
- Claim verdicts:
  - confirmed = +1.0
  - inconclusive = +0.2
  - refuted = -1.0
  - unverified = 0.0
- Aggregation:
  - mean(claim_scores) + `format_penalty`, clipped to [-1, 1].

### 5.3 Reward contribution

- `oracle_reasoning_reward` is heavily weighted (x2.5).
- `oracle_gate_penalty` adds extra negative pressure when oracle strongly disagrees.

Net effect: high pass-rate with bad reasoning cannot fully hide from reward shaping.

---

## 6. Patch Validator (Code Safety + Applicability Verifier)

Main entry: `server/patch_validator.py::PatchValidator.validate()`.

## 6.1 Purpose

Validator answers: "Is this patch structurally valid, applicable, safe, and meaningful before touching disk?"

### 6.2 Stages

1. Format checks
2. Hunk parse checks
3. Apply simulation against source snapshot
4. Anti-hack checks (sleep/skip/assert deletion/swallow)
5. Reasoning-action alignment checks
6. Flakiness smell checks
7. Syntax/compile/structure checks
8. Idempotency checks
9. Causal proximity warnings
10. Score production (0..1)

### 6.3 Reward contribution

`compute_verifiable_reward()` includes `patch_validation_signal`:
- invalid patch: -0.3
- valid patch: +0.2 x validation_score

This is why you can see non-zero reward changes even without pass-rate gain.

---

## 7. Reward System (Full)

Main entry: `server/reward.py::compute_verifiable_reward()`.

## 7.1 Signals

- `format_reward`
- `compile_reward`
- `stability_reward`
- `causal_proximity_reward`
- `failure_entropy_reward`
- `anti_hack_penalty`
- `oracle_reasoning_reward` (or fallback `reasoning_consistency_reward`)
- `oracle_gate_penalty`
- `diversity_penalty`
- `claim_novelty_reward`
- `patch_validation_signal`
- `noop_patch_penalty`
- `regression_penalty`
- `terminal_bonus`

### 7.2 Total reward composition

Current weighted sum:

- `format_reward * 0.5`
- `compile_reward * 1.0`
- `stability_reward * 2.0`
- `causal_proximity_reward * 0.5`
- `failure_entropy_reward * 0.5`
- `anti_hack_penalty * 1.5`
- oracle component:
  - oracle path: `oracle_reasoning_reward * 2.5 + oracle_gate_penalty * 1.0`
  - fallback path: `reasoning_consistency_reward * 0.5`
- `diversity_penalty * 1.0`
- `claim_novelty_reward * 1.0`
- `patch_validation_signal * 1.0`
- `noop_patch_penalty * 1.0`
- `regression_penalty * 1.5`
- `terminal_bonus * 1.0`

---

## 8. Why the Logs You Saw Are Internally Consistent

Example pattern you reported:
- patch validation failed (`search_text_not_found_in_*`)
- small positive total reward
- no pass-rate improvement

This is expected under current shaping:
- format can still be positive
- entropy can still be positive/neutral
- validator contributes a negative signal
- compile/stability may be weak or negative

So a tiny positive total can happen even when patch application fails.

---

## 9. Probable Mistakes / Bugs / Redundancies

## 9.1 High-confidence

1. Remote path currently disabled in inference
- `_should_use_remote_env()` returns `False` unconditionally in `inference.py`.
- Effect: ENV_BASE_URL and remote client path are dead from this entrypoint.

2. Runner adapter redundancy in inference
- `runner = _build_default_runner(repo_path)` is computed, but environment is created without that `runner` argument.
- Effect: dead code and potentially misleading logs.

3. Potential runner interface mismatch if adapter is later wired
- Env `_run_tests()` expects `runner.run_test()` returning `RunRecord`-compatible objects.
- Adapter currently emits dicts in `_build_default_runner`.
- Effect: future integration hazard.

## 9.2 Medium-confidence

4. Logging pass-rate before/after is confusing
- Some logs display baseline as the "before" value each step.
- Effect: difficult debugging when trying to inspect step-to-step progression.

5. Preflight infra detection could be too broad
- Message substring checks can classify some non-infra failures as infra.
- Effect: occasional false gate decisions.

## 9.3 Design-level redundancy

6. Parser fallback complexity
- JSON-first design still keeps multiple legacy fallback modes.
- Effect: robustness increases, but behavior variance and debugging complexity also increase.

---

## 10. Suggested Next Refactor Targets

1. Re-enable and harden remote mode selection in `inference.py`.
2. Either remove unused runner adapter or wire it correctly with a strict runner protocol.
3. Normalize runner return types (`RunRecord` vs dict) at one boundary only.
4. Tighten step logging to show true previous pass rate.
5. Add unit tests for:
   - reward component isolation
   - validator reject/accept matrix
   - oracle score behavior under contradictory claims

---

## 11. File Map (Primary Sources)

- `models.py`
- `server/state.py`
- `server/FlakeForge_environment.py`
- `server/reward.py`
- `server/oracle_engine.py`
- `server/patch_validator.py`
- `server/patch_applier.py`
- `agent/unified_agent.py`
- `inference.py`
