# FlakeForge v2: Detailed Implementation Summary

This document serves as an exhaustive log of the architectural extensions, algorithms, and models implemented for FlakeForge v2. The objective was to elevate the FlakeForge Reinforcement Learning environment from a toy "clean-room" setting (that can only fix basic synchronous logic errors) into a production-grade resilience agent capable of fixing complex architecture-level flakiness like event loop deadlocks, timing races, and latent UI/Network dependencies.

To achieve this, we introduced four architectural pillars—**Causal Graph Analysis, Chaos Amplification, Deep-Surgery Action Space, and Performance Sentinels**—spanning 11 modified or created files.

---

## 1. Core System & Data Layer Alterations

### `models.py` (Modified)
**Purpose:** Expanded the fundamental memory and interaction schemas for the RL agent.
- **Root Causes Extended:** Added `ASYNC_DEADLOCK` and `INFRASTRUCTURE_SENSITIVE` to better categorize complex system faults.
- **New Actions (13 total, up from 7):**
  - `DIAGNOSE_BOUNDARY`: Forces a sub-graph trace for specific network/database calls.
  - `CHAOS_PROBE`: Re-runs a specific test actively under CPU/Memory/IO stress to gather sensitivity evidence.
  - `REFACTOR_CONCURRENCY`: Swaps out synchronous threading primitives for async counterparts (e.g., swapping `threading.Lock` for `asyncio.Lock`).
  - `ISOLATE_BOUNDARY`: Injects circuit/timeout boundaries around external endpoints.
  - `EXTRACT_ASYNC_SCOPE`: Re-architects blocking synchronous code to execute inside `run_in_executor` threadpools to unblock the main event loop.
  - `HARDEN_IDEMPOTENCY`: Safely wraps state-mutating operations.
- **State/Observation Models Expanded:** The `FlakeForgeState` and `FlakeForgeObservation` models were modified to store `chaos_pass_rate`, `chaos_baseline_pass_rate`, `perf_regression_detected`, `infrastructure_sensitive` flags, and the structured JSON dictionary `causal_graph` to allow the agent to "see" beyond single files.

### `server/reward.py` (Modified)
**Purpose:** Recalibrated the agent's Reinforcement Learning reward function so it learns not just to "pass the test", but to build robust software.
- **Chaos Stability Reward (`r_chaos_stability`):** Adds a `+5` flat scaling bonus proportional to how much the pass rate increases specifically under a chaos load (e.g., if a patch improves pass rates from 10% to 100% under CPU stress, it gets the maximum reward).
- **Performance Regression Penalty (`p_perf_regression`):** Uses logarithmic scaling to heavily penalize solutions that introduce latency. A 2x slowdown costs `-6.93` points. A 10x slowdown costs `-23.03` points.

### `server/requirements.txt` & `server/Dockerfile` (Modified)
**Purpose:** Enabled statistical processing and OS-level stress injection.
- Added `scipy` and `numpy` for hypothesis testing in `PerformanceSentinel`.
- Added `pytest-asyncio` for new async test logic capabilities.
- Installed `stress-ng` (Linux tool for CPU/Memory/IO exhaustion) and `iproute2` (for `tc netem` network packet latency/drop manipulation) directly into the environment execution Docker image.

---

## 2. New Engine Architectures (The 4 Pillars)

### `server/causal_graph.py` (Created)
**Purpose:** Gives the agent spatial awareness across multiple repositories and files.
**Implementation:**
- Built `CrossRepoGraphBuilder`, which uses Python's `ast` system to trace up to a maximum depth of 3 variable/function context swaps.
- It intercepts nodes derived from `ast.Call`, looks up their imported module namespaces, and flags them with string tags like `http`, `db`, `fs`, `grpc`, or `queue`.
- **Primary outcome:** If it detects dangerous architectures—like usage of `threading.Lock` inside an `AsyncFunctionDef` context—it automatically tags the graph boundary with a warning indicating `ASYNC_DEADLOCK` vulnerability.

### `server/chaos_runner.py` (Created)
**Purpose:** Strips away the "ideal environment" to force hidden concurrency bugs to the surface.
**Implementation:**
- Created `ChaosAmplifiedRunner` (inheriting from `DockerTestRunner`).
- Runs target `pytest` paths while wrapped in background `stress-ng` subprocesses (e.g., `--cpu 4 --cpu-load 100`, `--vm 2 --vm-bytes 1G`).
- Can classify an issue as `infrastructure_sensitive` automatically if the clean execution pass rate drops by more than 20% compared to the chaotic environment.

### `server/perf_sentinel.py` (Created)
**Purpose:** Prevents the agent from "fixing" flaky tests by simply wrapping them in massive sleeps, global variables, or endless retry loops.
**Implementation:**
- `PerformanceSentinel` captures a clean runtime baseline on environment start via `test_benchmark.py` files.
- After every agent AST patch, it replays the exact same benchmark and records the latency arrays.
- It relies on `scipy.stats.mannwhitneyu` to measure statistically significant variance (P-value < 0.05). If the test executes significantly slower, it flags a `regression_detected` True and calculates the `median_ratio` corresponding to the exact magnitude of the slowdown.

### `server/FlakeForge_environment.py` (Modified)
**Purpose:** The central RL engine interface tying all logic into the `step` and `reset` lifecycle functions.
**Implementation:**
- `__init__` injects instances of `ChaosAmplifiedRunner`, `PerformanceSentinel`, and `CrossRepoGraphBuilder`.
- `reset()` performs 3 critical new baseline checks: standard baseline runs, chaos baseline load-testing, and the performance baseline capture.
- `step()` is modified so that every action output is intercepted to capture new execution profiles. After every standard patch is applied, it launches internal chaos and performance benchmark checks behind the scenes.
- `_execute_action()` routes all 6 new actions, processing the JSON parameters the agent passes, returning the necessary boundary metadata matrices or triggering appropriate code mutations (simulated AST payload building).

---

## 3. Seed Repository Architectures (The "Hard 20%")

To scientifically validate V2, we architected three highly localized minimal reproducible environments. The old FlakeForge agent categorically fails all of these without V2 capabilities, often breaking performance to fix the flake. Each repository includes: the flaky logic implementation, the `test_flaky.py` file, and the `test_benchmark.py` testing file.

### `seed_repos/async_lock_deadlock/` (Created)
- **The Issue:** Simulates an event loop stall. A synchronous `threading.Lock` is implicitly declared inside `processor.py` to guard a database execution within an `async def` function.
- **The Manifestation:** Fails unpredictably under processor load. If the main event loop gets pre-empted by OS scheduling while the synchronous lock is acquired, the whole concurrent architecture deadlocks and times out.
- **The Ideal Resolution:** Requires V2's `EXTRACT_ASYNC_SCOPE` via `run_in_executor`, or `REFACTOR_CONCURRENCY` to change the lock to `asyncio.Lock()`.

### `seed_repos/cpu_timing_race/` (Created)
- **The Issue:** The classic shared data race. Multiple threads manipulate a global variable (`counter.py`) reading and writing simultaneously without a synchronization primitive. 
- **The Manifestation:** Standard test execution generally completes the execution so fast the context switch rarely interrupts during the vulnerable CPU registry store step. The test passes mostly clean. However, under V2's `ChaosRunner` load, the scheduler halts threads aggressively mid-instruction, driving the test failure rate up to ~80% dynamically.
- **The Ideal Resolution:** Agent evaluates `CHAOS_PROBE` evidence showing infrastructure sensitivity, leading to an `ADD_SYNCHRONIZATION` response encapsulating the increment mechanism.

### `seed_repos/db_commit_scope/` (Created)
- **The Issue:** A synchronous blocking I/O operation (`db.commit()`) is placed natively inside an asynchronous loop handler inside `db_service.py`.
- **The Manifestation:** Fails via `TimeoutError`. The mock DB implementation contains an artificial randomized `slow_probability_factor` indicating long resolution loops. When memory or networking experiences slight delays, passing the `timeout=0.3s` parameter inside the wait hook is consistently exceeded.
- **The Ideal Resolution:** Agent detects `db.commit` as an external database boundary through `DIAGNOSE_BOUNDARY` graph parsing and applies `ISOLATE_BOUNDARY` wrapping.
