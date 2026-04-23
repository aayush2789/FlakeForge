# FlakeForge V2 — RL Training Data Design

## The Core Question: What IS a training sample?

In FlakeForge, the RL agent doesn't learn from a static CSV file. It learns **online** by doing experiments. But you still need a carefully designed corpus to experiment on.

> **One training episode = One flaky seed repository.**
> The agent resets into the repo, runs up to 14 steps of actions (GATHER_EVIDENCE → DIAGNOSE_BOUNDARY → EXTRACT_ASYNC_SCOPE → …), and accumulates reward based on whether the test pass-rate improved, whether chaos stability improved, and whether performance regressed.

---

## Why Ground Truth Labels ARE Needed (And What They Are)

The RL reward signal handles training. But **without ground truth, you cannot answer**:
- Did the agent fix the *right thing*, or just get lucky?
- Is the agent learning `EXTRACT_ASYNC_SCOPE` for async bugs and `ADD_SYNCHRONIZATION` for races? Or is it slapping `time.sleep` on everything and getting rewarded?
- How do we measure accuracy against a held-out eval set?

**Ground truth is a `flake_manifest.json` file placed in every seed repo.**

### `flake_manifest.json` Schema

```json
{
  "repo_name": "async_lock_deadlock",
  "flake_category": "ASYNC_DEADLOCK",
  "root_cause_file": "src/processor.py",
  "root_cause_function": "process_request",
  "root_cause_line": 42,
  "root_cause_description": "threading.Lock() used inside an async def, blocking the event loop when pre-empted under CPU load",
  "correct_actions": ["DIAGNOSE_BOUNDARY", "EXTRACT_ASYNC_SCOPE"],
  "correct_primitives": {
    "from": "threading.Lock",
    "to": "asyncio.Lock"
  },
  "is_infrastructure_sensitive": true,
  "chaos_profile_needed": "CPU",
  "expected_pass_rate_before_fix": 0.35,
  "expected_pass_rate_after_fix": 0.98,
  "solution_diff": "solution/fix.diff"
}
```

### How the Eval Loop Uses Labels

At evaluation time (NOT during training):
1. Run the agent on the eval repo for 14 steps
2. After the episode, compare:
   - `agent.selected_actions` vs `correct_actions` in manifest → **Action Accuracy**
   - `actual_pass_rate_after` vs `expected_pass_rate_after_fix` → **Fix Correctness**
   - `agent.identified_root_cause` vs `root_cause_file + root_cause_function` → **Root Cause Localization Accuracy**
3. An agent that adds `time.sleep(5)` might achieve a 90% pass rate but get 0% on Action Accuracy (it used `ADD_TIMING_GUARD` not `EXTRACT_ASYNC_SCOPE`) and 0% on Root Cause Localization.

This prevents the agent from gaming the reward with lazy fixes.

---

## How Many Repos Do You Need?

### For RL Training to Converge

| Phase | Repos | Episodes per Repo | Total Episodes |
|---|---|---|---|
| **Sanity check** | 3 (current seed repos) | 200 | 600 |
| **Minimum viable** | 20 diverse repos | 150 | 3,000 |
| **Generalization** | 100+ repos | 100 | 10,000+ |
| **Production quality** | 500+ from real open-source | 50+ | 25,000+ |

> **Key insight:** The agent will memorize specific repos if you train on too few. With only 3 repos, after ~200 episodes it will have an optimal policy for *those 3 repos specifically* — not for flakiness in general. You need diversity of language patterns, not just more episodes.

### Types of Repos Needed (Diversity by Category)

| Flake Category | Target % of Dataset | Description |
|---|---|---|
| `ASYNC_DEADLOCK` | 20% | threading.Lock/blocking call inside async context |
| `TIMING_RACE` | 20% | Classic shared-state race with no synchronization |
| `INFRASTRUCTURE_SENSITIVE` | 15% | Only fails under CPU/NET/MEM load |
| `ORDER_DEPENDENCY` | 15% | Tests pass/fail depending on execution order |
| `EXTERNAL_DEPENDENCY` | 15% | Network calls, DB latency, unstable mocks |
| `NONDETERMINISM` | 10% | Random seeds, date-dependent logic |
| `RESOURCE_LEAK` | 5% | File handles, threads, connections not cleaned up |

---

## Where to Source Real Training Data

### Tier 1 — Immediately Available (Free, Labeled)

**IDoFT (International Dataset of Flaky Tests)**
- URL: https://github.com/TestingResearchIllinois/idoft
- Contains: 3,500+ flaky tests from 937 open-source Java (and some Python) projects
- Has labels: each entry includes `Category` (OD, NOD, ASYNC, etc.), `PR` of the fix
- **How to use:** Download the CSV, filter for Python projects, clone each repo at the commit before the fix, create a seed repo wrapper, add `flake_manifest.json` from the CSV metadata

**FlakeFlagger Dataset**
- URL: https://github.com/winglam/flakeflagger
- Contains: 850+ Python test files with known flaky tests
- Has labels: root cause categories and pass/fail patterns
- **How to use:** Each test file → one seed repo template, wrap it with a `conftest.py` and `flake_manifest.json`

**pytest CI Failure Logs from GitHub**
- Search GitHub for `site:github.com "flaky test" "asyncio" fix` to find real-world PRs where developers *describe* the flaky test and the fix in the PR description
- These are high-quality because the developer explains the root cause in prose

### Tier 2 — Synthetic Generation (Programmatic, Fully Labeled)

Write a **seed repo generator** that parameterizes flake patterns:

```python
# Example: generator produces 50 async_deadlock variants
for variant in generate_async_deadlock_variants(n=50):
    # variant = { 'lock_type': 'threading.Lock', 'call_depth': 2, 
    #             'function_name': 'process_batch', ... }
    create_seed_repo(
        template="async_lock_deadlock",
        params=variant,
        output_dir=f"seed_repos/async_deadlock_v{i}"
    )
```

Advantages:
- 100% labeled automatically
- Can control difficulty (shallow vs. deep call chains, simple vs. compound flakes)
- Can generate 1000 variants from 7 templates in minutes

### Tier 3 — Real Production Codebases (Hardest, Most Valuable)

Mine GitHub for repositories that have:
- A commit message like "fix flaky test" / "fix race condition in test"
- The commit changes only the production code or test code (not config/docs)

Use the git diff before/after as your ground truth. The `correct_actions` in `flake_manifest.json` are inferred from what the developer changed.

---

## The `solution/` Directory Convention

Every seed repo should have a `solution/` subdirectory containing the fixed version of the flaky file(s):

```
seed_repos/
  async_lock_deadlock/
    src/
      processor.py          ← broken (training target)
    tests/
      test_flaky.py
      test_benchmark.py
    flake_manifest.json     ← ground truth labels
    solution/
      processor.py          ← fixed version (eval reference)
      fix.diff              ← unified diff of the fix
```

The eval harness checks the agent's final patched file against `solution/processor.py` using a semantic equivalence check (not exact string match — the agent may refactor differently but still correctly).

---

## How the RL Training Loop Uses This Data

```
                    ┌─────────────────────────────────────────┐
                    │        Dataset: 100 seed repos           │
                    │        each with flake_manifest.json     │
                    └────────────────┬────────────────────────┘
                                     │ sample 1 per episode
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FlakeForgeEnvironment.reset(repo)                                  │
│  1. Run test 20× → capture clean pass_rate (e.g. 0.35)             │
│  2. Run test 20× under CPU chaos → capture chaos_pass_rate (0.10)  │
│  3. Capture performance baseline (20 benchmark runs)                │
│  4. Run CrossRepoGraphBuilder → causal graph observation            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Agent step loop    │  (max 14 steps)
                    │                      │
                    │  obs → action →      │
                    │  env.step(action) →  │
                    │  reward → next obs   │
                    └──────────┬──────────┘
                               │ episode ends
                               ▼
              ┌────────────────────────────────────┐
              │  TRAINING reward (no labels needed) │
              │  r_fix + r_chaos + p_perf          │
              └────────────────────────────────────┘

              ┌────────────────────────────────────┐
              │  EVAL (labels used here)           │
              │  action_accuracy vs manifest       │
              │  root_cause_localization vs manifest│
              │  fix_quality (agent vs solution/)  │
              └────────────────────────────────────┘
```

---

## Immediate Next Steps

1. **Create `flake_manifest.json` for the 3 existing seed repos** — this enables the eval harness today.
2. **Download the IDoFT CSV** — filter for Python projects, automate repo cloning + wrapper creation for 20 repos minimum.
3. **Write `scripts/generate_seed_repos.py`** — parameterized generator for the 7 flake categories (creates 50+ synthetic repos automatically).
4. **Create `scripts/eval_agent.py`** — loads `flake_manifest.json`, runs agent on eval split, outputs action accuracy / localization accuracy metrics.
