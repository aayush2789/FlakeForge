---
title: Flakeforge Environment Server
emoji: 🔕
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /
tags:
  - openenv
---

# FlakeForge

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Environment-6366F1?style=for-the-badge&logo=python&logoColor=white)](https://huggingface.co/docs/hub/spaces)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hugging%20Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Hub-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-000?style=for-the-badge&logo=pydantic&logoColor=e92063)](https://docs.pydantic.dev/)
[![uvicorn](https://img.shields.io/badge/uvicorn-ASGI-444?style=for-the-badge)](https://www.uvicorn.org/)

**FlakeForge** is an [OpenEnv](https://pypi.org/project/openenv-core/)–style reinforcement learning **environment** for learning to **repair flaky Python tests**. The world is (mostly) deterministic: repeated `pytest` runs, AST scans, and static checks produce structured observations. The only learned piece is the policy (your code model); reward is **verifiable** from execution and static oracles, not an LLM judge.

---

## Table of contents

1. [Intuition and inspiration](#1-intuition-and-inspiration)
2. [Problem we solve](#2-problem-we-solve)
3. [Environment interface (POMDP view)](#3-environment-interface-pomdp-view)
4. [Observation space](#4-observation-space)
5. [Action space (unified agent)](#5-action-space-unified-agent)
6. [Root-cause categories](#6-root-cause-categories)
7. [Unified agent: what it outputs and what it sees](#7-unified-agent-what-it-outputs-and-what-it-sees)
8. [Core tooling](#8-core-tooling)
   - [Causal graph engine](#causal-graph-engine)
   - [Deep flakiness scanners](#deep-flakiness-scanners)
9. [Verification stack: oracle engine and patch validator](#9-verification-stack-oracle-engine-and-patch-validator)
10. [Verifiable reward system](#10-verifiable-reward-system)
11. [OpenEnv config and commands](#11-openenv-config-and-commands)
12. [Build, run, and test locally](#12-build-run-and-test-locally)
13. [Deploy on Hugging Face Spaces](#13-deploy-on-hugging-face-spaces)
14. [RL training (Unsloth, curriculum)](#14-rl-training-unsloth-curriculum)
15. [Repository layout](#15-repository-layout)
16. [Contributing](#16-contributing)

---

## 1. Intuition and inspiration

Flaky tests are a major source of **wasted CI time** and **wrong signals** for both humans and ML. Classic fixes (bigger timeouts, more retries, blanket skips) often **mask** symptoms while hiding deeper bugs. We wanted a setting where an agent is pushed toward **structural, minimal fixes** that **survive repeated validation**—closer to how a senior engineer reasons (hypothesis → small edit → re-run) than to a one-shot “rewrite the file” code completion task.

**Inspiration** comes from three lines of work: (1) **POMDP / RL** for sequential decision making under partial observability, (2) **software engineering** research on flaky-test patterns (concurrency, order, I/O, shared state, fixtures, mocks, imports), and (3) **verifiable** training signals: what you can *measure* in pytest output and in the repo’s AST beats what you can only *opine* with another LLM.

---

## 2. Problem we solve

**Given** a repository and a test identifier, **stabilize** the failure mode so that the target test (and, ideally, the rest of the suite) passes **repeatably** under the configured runner, without “reward hacking” (weakening assertions, broad `except:`, `skip` markers, or sleeping forever).

The environment therefore:

- Establishes a **baseline** (preflight: sanity, determinism, flakiness) before training credit is spent.
- Surfaces **diagnostics** the policy can lean on: stack-derived failure frontier, call chain, I/O “boundary” hints, a **causal** view of the code path, and **AST-only “deep” flakiness** signals.
- Accepts a **unified** agent turn: structured **diagnosis** + **patch** in one step, then **re-executes** tests and returns **scalar reward** with a full **breakdown** for debugging and GRPO.

---

## 3. Environment interface (POMDP view)

| Concept | In FlakeForge |
|--------|----------------|
| **State** (hidden) | Server-side `FlakeForgeState` + git-like snapshots of the repo, episode counters, pass rates, etc. |
| **Observation** | `FlakeForgeObservation` JSON: sources, trees, preflight, deep signals, causal hints, run history, last reward, and more. |
| **Action** | `FlakeForgeAction`: unified think + patch (structured JSON and/or free-text for parsing). |
| **Transition** | `reset` → (optional) configure repo/test; `step` → validate → apply patch → run tests → compute reward. |
| **Reward** | `compute_verifiable_reward` in `server/reward.py` → `RewardBreakdown` and scalar `total`. |

The HTTP layer is created with **OpenEnv**’s `create_app(FlakeForgeEnvironment, FlakeForgeAction, FlakeForgeObservation)`; see `server/app.py` and `openenv.yaml`.

---

## 4. Observation space

The main payload is `FlakeForgeObservation` in `models.py`. At a high level, each step exposes:

| Block | Role |
|-------|------|
| **Localisation** | `test_function_source`, `source_under_test`, `file_tree`, `relevant_imports`, `async_markers` |
| **Run dynamics** | `run_history` (`RunRecord`: pass, duration, error_type, stderr excerpt), `current_pass_rate`, `baseline_pass_rate` |
| **Preflight** | `env_type`, `should_train`, `preflight_result` (three-stage gate: sanity, determinism, flakiness) |
| **Deep flakiness (AST, fast)** | `module_cache_violations`, `fixture_scope_risks`, `mock_residue_sites`, `import_side_effect_files`, `async_contamination_alive` |
| **Causal** | `failure_frontier` (file:line:func from stack), `call_chain_to_frontier`, `boundary_crossings` (e.g. HTTP/DB hints), `causal_graph` (summary dict), `causal_hints` |
| **Failure shape** | `failing_stack_trace`, `failure_pattern_summary`, `duration_fingerprint` |
| **Probes** | `order_dependency_detected`, `infrastructure_sensitive` (chaos / stress sensitivity) |
| **Policy feedback** | `patches_applied`, `total_diff_lines`, `think_history` (per-step summary for de-duplication and prompts), `last_think_text`, `last_patch_text`, `last_reward`, `reward_breakdown` |
| **Termination** | `done`, `done_reason` |

The observation is the **ground truth** the unified agent is prompted with on each turn (plus recent history) — see `agent/unified_agent.py` for prompt construction.

---

## 5. Action space (unified agent)

V3 uses a **single** high-level action type: **`UNIFIED_PATCH`**. The policy does **not** pick from seven discrete “tool names” in the old Gym sense; it emits a **JSON object** (preferred) with:

- **`think`** — `StructuredThink`: list of `ThinkClaim` items (category, entity, `path::Class.func` `location`, polarity, reason, etc.).
- **`patch`** — `StructuredPatch`: list of `PatchHunk` with `file`, `search`, `replace` (grounded search/replace over real sources).

`FlakeForgeAction` also carries `raw_response`, `think_text`, `patch_text`, and `predicted_category` / `predicted_confidence` for parsers and heuristics.

**Intuition:** the “action space” is **constrained by validation**, not a small enum. Invalid JSON, non-matching `search` blocks, unsafe edits, and hack patterns are rejected or penalized; valid minimal edits that move pass rate and match the **failure frontier** score highest.

---

## 6. Root-cause categories

`ROOT_CAUSE_TYPES` in `models.py` enumerates the taxonomy the think block is supposed to use, for example: `async_wait`, `concurrency`, `test_order_dependency`, `resource_leak`, `shared_state`, `network`, `platform_dependency`, `nondeterminism`, `import_side_effect`, `module_cache_pollution`, `fixture_scope_leak`, `mock_residue`, and `unknown`.

`RELATED_CATEGORIES` defines **soft** relatedness for **reasoning consistency** reward when the model’s stated category and the patch-inferred category differ slightly.

---

## 7. Unified agent: what it outputs and what it sees

**`UnifiedFlakeForgeAgent`** (`agent/unified_agent.py`) is the **JSON-first** interface to your base model. It:

- Builds a long context from the current **`FlakeForgeObservation`** (test code, SUT, signals, run summaries, and history).
- Enforces a **strict JSON** shape (no markdown fences) with **one claim line** in the system prompt; patch hunks use **verifiable** one-line `search` anchors where possible.
- Parses model output into `structured_think` and `structured_patch` when present.

**What the model “sees”** is exactly what you put in the observation: flaky vs stable preflight, pass rates, stack frontier, **deep** flags, and causal summaries — the sort of information you would give a human before asking for a **minimal** fix.

---

## 8. Core tooling

### Causal graph engine

*Implementation: `server/causal_graph.py`.*

**Purpose:** Build a **directed, cross-file** view from the test entry into user code, marking **async**, **call depth**, and **external boundaries** (HTTP, DB, queue, gRPC) where non-determinism or ordering often enters.

- **Input:** repository root, test / entry heuristics, configurable depth.
- **Output:** a `CausalGraph` (nodes, edges, boundary warnings) rendered into `FlakeForgeObservation.causal_graph` and short **`causal_hints`** for the LLM.
- **Why it matters:** Patches that edit files **far** from the **failure frontier** or **causal** neighborhood are easy to down-weight; see **causal_proximity_reward** in `server/reward.py`.

### Deep flakiness scanners

*Implementation: `server/deep_flakiness.py`.*

**Purpose:** Inexpensive **AST-only** checks that mirror real flaky patterns from SE literature:

- Module-level **caches** / mutable defaults / globals (`@lru_cache`, mutable list defaults, etc.)
- **Fixture** scope and yield/teardown issues
- **`mock.patch`** without proper teardown
- **Import-time** side effects
- **Async/thread** “leftover” heuristics

`build_deep_observation_signals(repo_path)` returns a small dict; the environment copies fields onto the observation so the **policy** and **reward** can use them as **hard features** (no model call).

---

## 9. Verification stack: oracle engine and patch validator

### Oracle engine (`server/oracle_engine.py`)

**Question:** *Is the structured **think** consistent with the **code** and the **patch**?*

- Uses **AST** and optional **LibCST** for static checks.
- Resolves `ThinkClaim.location` to real sources; checks things like **sync primitives** for concurrency-style claims, and whether hunks **address** the claimed locus.
- `verify_structured_think` produces a scalar **`oracle_score`** used in reward when available.

This is **not** a “judge model”: it is **code-facing** and **template-aware**, intended to be reproducible and cheap.

### Patch validator (`server/patch_validator.py`)

**Question:** *Is the patch **well-formed**, **applicable**, and **not obviously abusing** the reward?*

**Stages** (from module docstring):

1. **Format** — search/replace blocks, headers, basic hygiene  
2. **Simulate** apply — `search` must match the file (via `simulate_search_replace_patch`)  
3. **Syntax** — `ast.parse` on post-patch text  
4. **Compile** — per-module `compile` check  
5. **Structure** — heuristics (e.g. empty bodies, control-flow red flags)  
6. **Causal proximity (warning)** — far-from-frontier warnings  

**Invalid patches are rejected *before* disk write** in `FlakeForgeEnvironment.step`. The validator can emit a **score** (`validation_score`) that flows into **patch_validation_signal** in reward when the patch actually applies.

---

## 10. Verifiable reward system

`compute_verifiable_reward` in `server/reward.py` assembles a **`RewardBreakdown`**:

| Signal | Role |
|--------|------|
| **format_reward** | Parses structured think/patch; rewards valid claims/hunks; falls back to text markers if needed. |
| **compile_reward** | Patch applied, syntax/compile path; negative if **validator rejected** or not applied. |
| **stability_reward** | **Potential-based** shaping on pass rate: \(\Phi(p)=p^2\), difference vs baseline. |
| **causal_proximity_reward** | Edits **near** `failure_frontier` / call chain; uses boundary metadata. |
| **failure_entropy_reward** | Lower entropy of error **types** after the step is better (more “single failure mode”). |
| **anti_hack_penalty** | Counters assertion weakening, `sleep` insertion, `except:`, `skip` markers, huge diffs, etc. |
| **regression_penalty** | Other tests failing after edit. |
| **reasoning_consistency_reward** / **oracle_reasoning_reward** | Category match (heuristic) vs **oracle** path when `oracle_score` is present. |
| **patch_validation_signal** | From `PatchValidator` **score** (when applied). |
| **noop_patch_penalty** | Trivial or no-op edits that still “succeed” mechanically. |
| **think_history_penalty** | Repeating the same diagnosis without new evidence. |
| **terminal_bonus** | Large bonus when pass rate hits **1.0** or major improvements. |

**Hard short-circuit:** If **format** is too low, **compile** is bad, or **anti_hack** is negative, the total is clamped to a **strong negative** (see the `if breakdown.format_reward < 0.75 or ...` block) so the policy cannot grind partial credit on garbage.

**Total** is a **weighted sum** of the above (see the final `breakdown.total_reward = round(...)` block in `reward.py` for exact coefficients).

**Intuition in one line:** *Reward what **pytest and static checks** can verify, punish **hacks and regressions**, and gently align **stated** diagnosis with **actual** code change.*

---

## 11. OpenEnv config and commands

`openenv.yaml` pins the app entry point:

```yaml
spec_version: 1
name: FlakeForge
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

**Typical commands (from a clone of this repo, repo root = `CWD`):**

| Task | Command |
|------|--------|
| Install deps (pip) | `pip install -r server/requirements.txt` |
| **Run ASGI** (dev) | `uvicorn server.app:app --host 0.0.0.0 --port 8000` |
| **Run** (if the package is installed with `[project.scripts]`) | `uv run server --port 8000` (see `test_repos/SETUP.md`) |
| **Docker** | `docker build -t flakeforge-env:latest -f server/Dockerfile .` then run with published `8000` |
| **HTTP health** | `GET http://localhost:8000/health` |
| **Episode (HTTP API)** | `POST /reset` with JSON body; then `POST /step` with action fields (see `client.py` and OpenEnv’s handler) |

**Hugging Face Hub CLI (optional) — push Space files:**

```bash
hf auth login
hf upload <user>/FlakeForge . --type space
```

(Use your Space repo id. Requires `huggingface_hub` / [`hf` CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli).)

The Space UI in this project serves `templates/index.html` at `/` and adds showcase routes under `/api/*` (see `server/api_showcase.py`).

---

## 12. Build, run, and test locally

**In-process (Python):** with the package installed (`pip install -e .` or your env manager):

```python
from FlakeForge.server.FlakeForge_environment import FlakeForgeEnvironment
from FlakeForge.models import FlakeForgeAction

env = FlakeForgeEnvironment()  # optional: repo_path, test_identifier
obs = env.reset()
# Build FlakeForgeAction with your model or tests; then:
# result = env.step(action)  # return shape from OpenEnv Environment API
```

If you run from a source checkout without installing, set `PYTHONPATH` to the repo root and use `from server.FlakeForge_environment import FlakeForgeEnvironment` (and `from models import ...` as in `tests/`).

**HTTP client:** see `client.py` (`FlakeForgeClient.run_episode_remote`) for `httpx` calls to `POST /reset` and `POST /step` against `ENV_BASE_URL`.

**Image build** uses `server/Dockerfile` (see top of that file for `HEALTHCHECK` and `uvicorn` entrypoint).

---

## 13. Deploy on Hugging Face Spaces

1. **Create** a new Space; set **SDK** to **Docker** (matches this repo’s README front matter: `sdk: docker`).
2. **Port:** `app_port: 8000` (already in the YAML front matter of this `README.md`).
3. **Container:** Hugging Face expects a `Dockerfile` at the **repository root** of the Space. This repository ships the canonical image in **`server/Dockerfile`**. For Spaces, add a **root** `Dockerfile` with the same content (or `COPY` pattern), e.g.:

   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y --no-install-recommends git curl stress-ng iproute2 \
       && rm -rf /var/lib/apt/lists/*
   COPY server/requirements.txt /app/server/requirements.txt
   RUN pip install --no-cache-dir -r /app/server/requirements.txt
   COPY . /app/
   HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
       CMD curl -f http://localhost:8000/health || exit 1
   EXPOSE 8000
   ENTRYPOINT ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

4. **Environment (optional):** set `FF_REPO_PATH` to point at a **bundled** repo in the image (default in code: `test_repos/timing_race_minimal` under the project root) or your checked-in fixture.
5. **Push** via Git (recommended) or `hf upload` as above. Wait for the build; the UI should load at the Space URL.

**Note:** The OpenEnv `create_app` stack exposes `GET /health`, `POST /reset`, and `POST /step` like other OpenEnv HTTP servers. OpenAPI may be at `/docs` if enabled by your `openenv-core` version.

---

## 14. RL training (Unsloth, curriculum)

**Idea:** GRPO rollouts use **`FlakeForgeEnvironment`**, **structured JSON** actions, and **`compute_verifiable_reward`** — no LLM judge for the main training signal.

**Current stack (after latest improvements):**

- **Unsloth GRPOTrainer** (default) — fast path vs vanilla TRL; strong Qwen2.5 support; 4-bit QLoRA
- **Curriculum** with 3 stages (Easy → Medium → Hard)
- **Manifest-grounded** scenarios under `test_repos/synthetic/`, `data/manifests/`, and `data/curriculum_stages/`

**Install training deps and run the driver:**

```bash
uv pip install -r training-requirements.txt
python -m training.train_grpo --model Qwen/Qwen2.5-7B-Instruct --max-episodes 500
```

**Notes:**

- Unsloth is **required** for the default training path; the script fails fast with install hints if it is missing.
- Outputs use the **V3** JSON shape `{"think": {...}, "patch": {...}}`.
- W&B project defaults to `flakeforge-rl` when enabled.
- The notebook `FlakeForge_RL_Training (2).ipynb` is kept in sync with this pipeline conceptually; **`training/train_grpo.py`** is the main entry.

---

## 15. Repository layout (selected)

| Path | Role |
|------|------|
| `models.py` | Schemas: action, observation, state, `RewardBreakdown` |
| `openenv.yaml` | OpenEnv manifest: FastAPI `server.app:app` |
| `server/app.py` | App factory, static UI route, showcase router |
| `server/FlakeForge_environment.py` | `reset` / `step`, preflight, runner, observations |
| `server/reward.py` | Verifiable reward and breakdown |
| `server/oracle_engine.py` | Static oracle over claims and patches |
| `server/patch_validator.py` | Patch simulation and safety |
| `server/causal_graph.py` | Causal graph builder |
| `server/deep_flakiness.py` | AST “deep” signals |
| `server/docker_runner.py` | pytest / Docker test execution |
| `agent/unified_agent.py` | JSON-first agent and prompts |
| `client.py` | Remote HTTP client example |
| `training/` | GRPO and curriculum |


## 16. Contributing

- **Issues & PRs:** Bug reports and small, focused PRs are welcome. Please describe **repro** (repo path, test id, command) and, if possible, a **failing** vs **expected** reward or observation.
- **Scope:** Prefer changes that keep the **environment deterministic** and **reward** tied to **measurable** criteria.
- **Style:** Match existing typing, docstrings, and module boundaries (`models` / `server` / `agent` / `training`).
- **Code of conduct:** Be constructive and specific in review; we optimize for **clarity and reproducibility** over sheer benchmark scores.
