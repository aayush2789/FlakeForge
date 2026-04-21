# FlakeForge: The Absolute Technical Manual & Progress Report

## 1. Executive Summary
**FlakeForge** is a world-class reinforcement learning (RL) sandbox designed to solve one of the most persistent problems in modern software engineering: **Flaky Tests**. By wrapping the complexity of test execution, static analysis, and code mutation into a standardized **OpenEnv** interface, FlakeForge allows AI agents to learn the "art" of debugging non-deterministic failures.

This documentation provides an exhaustive, line-by-line detailed breakdown of the entire architecture, its foundational intuition, and every internal mechanism that makes it a production-grade research environment.

---

## 2. System Architecture & High-Level Intuition
The project follows a decoupling strategy between the **Environment Server** (where the code is mutated and tested) and the **Agent Client** (where the policy makes decisions).

### The Sense-Think-Act Paradigm
1.  **Sense**: The agent receives an `Observation` containing the source code, run history, and previously applied patches.
2.  **Think**: The agent (or a Judge AI) evaluates the evidence to form a `Hypothesis`.
3.  **Act**: The agent sends an `Action` (e.g., `ADD_TIMING_GUARD`) which is converted into an AST-level mutation on the server.

### Why Reinforcement Learning?
Traditional static analysis fails with flakes because the root cause is often temporal (timing), environmental (shared state), or external. An RL agent can "experiment" in a sandbox, seeing the immediate impact of its changes on the statistical probability of failure.

---

## 3. Deep Dive into `server/FlakeForge_environment.py`
This file is the "Engine" of the project. It inherits from `openenv.core.env_server.interfaces.Environment`.

### Key Methods & Logic
| Method | Description | Internal Intuition |
| :--- | :--- | :--- |
| `__init__` | Bootstraps the environment. | It initializes the `DockerTestRunner` with a specific `repo_path` and `test_id`. It sets up the unique `episode_id`. |
| `reset()` | The start of every episode. | It calls `_restore_clean_repo` (via `git checkout -- .`) to ensure no previous patches leak into the new episode. It then establishes a "Baseline Pass Rate" by running the test 10 times. |
| `step(action)` | execution of one agent turn. | Increments `step_count`, dispatches to `_execute_action`, runs the test **20 times** to gather statistical significance, and checks for regressions. |
| `_execute_action` | The bridge between models and tools. | Routes high-level actions to `tools.py` for AST patching or to `git` for reverting. Returns a technical summary of the modification (e.g., "3 lines changed"). |
| `_build_observation` | The state projection. | Pulls code excerpts, gets the file tree, and identifies `async` functions. This is the only information the agent has to base its decisions on. |
| `_run_test_n_times` | Statistical evaluation. | Runs the test `n` times using a specified `max_workers` count to simulate concurrent execution. |

---

## 4. Models & Communication (`models.py`)
This file defines the Pydantic-based backbone of the system.

### Detailed Action Space (`FlakeForgeAction`)
Each action has specific validation rules to prevent the agent from sending malformed payloads:
- **`GATHER_EVIDENCE`**:
    - *Param*: `injection_target` ("test" or "source").
    - *Detail*: Decides where in the CST to inject JSON loggers.
- **`ADD_TIMING_GUARD`**:
    - *Param*: `delay_ms` (Must be one of: 50, 100, 200, 500).
    - *Detail*: This strict choice prevents the agent from searching an infinite continuous space, making learning more efficient.
- **`ADD_SYNCHRONIZATION`**:
    - *Param*: `primitive` (lock, event, barrier, semaphore).
- **`MOCK_DEPENDENCY`**:
    - *Param*: `target` (Must be a dotted path, e.g., `os.path.exists`).
- **`RESET_STATE`**:
    - *Param*: `scope` ("function", "class", "module").
- **`ADD_RETRY`**:
    - *Param*: `max_attempts` (2, 3, 5), `backoff_ms` (100, 500).
- **`REVERT_LAST_PATCH`**:
    - *Logic*: Uses `git apply -R` to back out the most recent change if it caused a regression or failed to fix the flake.

---

## 5. Static Analysis & Mutation (`server/tools.py`)
This file is the "Scalpel" of FlakeForge.

### Functional Breakdown
1.  **`list_repo_structure`**: Non-recursive exploration focusing on `.py` files. It detects `async` defs to help the agent distinguish between synchronous and asynchronous code paths.
2.  **`read_file_excerpt`**: Precise line-range reading. Uses `itertools.islice` for performance on larger files.
3.  **`parse_ast_summary`**:
    - **Python Path**: Uses the standard `ast` module. Walks the tree to extract functions, classes, decorators, imports, and threading primitives.
    - **JS/TS Path**: Hooks into `tree-sitter` for multi-language support.
4.  **`inject_logging`**: Uses **`libcst`** (Concrete Syntax Tree) for syntax-preserving modification.
    - Unlike AST, CST preserves comments and formatting, which is crucial if the code is returned to a human or used for further training.
    - Injects a `_task_name()` helper to detect `asyncio` task IDs for debugging task-based flakes.
5.  **`apply_ast_patch`**: A wrapper that renders Jinja-like templates into code and applies them via the textual operation engine.
6.  **`compute_diff`**: Calculates unified diffs and detects symbol changes (e.g., "Function `foo` was modified").
7.  **`_apply_textual_operation`**: Uses regex `re.search` to find injection points like `def test_` or `await`.

---

## 6. Execution & Verification (`server/docker_runner.py`)
The runner defines how we validate a "fix."

### Implementation Details
- **Pytest Orchestration**: Runs `pytest` in a subprocess. It uses `--tb=short` to keep error logs concise for agent consumption.
- **Result Capturing**:
    - **`passed`**: Logic checks for `returncode == 0` AND the presence of `"1 passed"` in the output.
    - **`error_type`**: Regex-based extraction of the specific exception class.
    - **`error_message`**: Extracts the last line of the traceback.
- **Concurrency**: Specifically designed to handle race conditions by running tests in parallel threads, increasing the probability of a "race" occurring during evaluation.

---

## 7. The Reward Mathematics (`server/reward.py`)
The reward $R$ is a weighted sum of multiple signals:

$$R = R_{stability} + R_{judge} + R_{efficiency} + R_{semantic\_efficiency} - P_{regression} - P_{retry\_abuse}$$

### Detailed Breakdown
- **$R_{stability} = (PassRate_{curr} - PassRate_{base}) \times 10.0$**: This is the primary driver. If you move from 0.0 to 1.0 pass rate, you get 10 points.
- **$R_{judge} = (\frac{HypothesisScore}{5} + \frac{PatchScore}{5}) \times 1.5$**: Adds human alignment to the agent's behavior.
- **$R_{efficiency} = -0.3$ per `GATHER_EVIDENCE`**: Discourages the agent from clicking the "log everything" button indefinitely.
- **$P_{regression} = 15.0$**: A massive penalty if the fix for Test A breaks Test B.
- **$P_{retry\_abuse} = 2.0$**: Specifically targets the `ADD_RETRY` action to prevent "Band-aid" solutions.
- **Success Bonus ($+5.0$)**: Awarded only if $PassRate = 1.0$ and $Regressions = 0$.

---

## 8. Case Studies in Flakiness (`seed_repos/`)

### A. The Timing Race (`timing_race`)
- **Code Snippet**:
    ```python
    async def test_flaky_case():
        timeout = 0.01 if random.random() < 0.7 else 0.2
        await asyncio.wait_for(fragile_async(timeout=timeout), timeout=timeout)
    ```
- **Intuition**: Sometimes the 0.01s timeout is hit before the coroutine even initializes.
- **Fix**: The agent should apply an `ADD_TIMING_GUARD` to increase the minimum wait.

### B. The Poisoned Cache (`shared_state`)
- **Code Snippet**:
    ```python
    def test_flaky_case():
        if random.random() < 0.5:
            cache.append('leftover')
        assert append_value('x') == 1
    ```
- **Intuition**: If 'leftover' is in the cache, `append_value` returns 2 instead of 1.
- **Fix**: The agent must identify `SHARED_STATE` and apply `RESET_STATE`.

### C. The Leaky Handle (`resource_leak`)
- **Code Snippet**:
    ```python
    def test_flaky_case():
        assert leak_handle() < 4
    ```
- **Intuition**: A function fails to close a file handle. After multiple runs, the OS limit is hit.
- **Fix**: The agent must find the leak and ensure handles are closed.

---

## 9. JSON Schemas for Agent Communication

### Action Schema (Example: Add Retry)
```json
{
  "action_type": "ADD_RETRY",
  "parameters": {
    "max_attempts": 3,
    "backoff_ms": 500
  }
}
```

### Observation Schema (Truncated)
```json
{
  "episode_id": "uuid-123",
  "current_pass_rate": 0.85,
  "test_function_source": "def test_foo(): ...",
  "run_history": [
    {"passed": true, "duration_ms": 120},
    {"passed": false, "duration_ms": 15, "error_type": "AssertionError"}
  ],
  "steps_remaining": 12
}
```

---

## 10. The Evaluation Algorithm (Step-by-Step)
When an agent takes a step by sending a `FlakeForgeAction`, the environment follows this internal logic to maintain statistical integrity:

1.  **State Capture**: Before any action, the environment snapshots the current pass rate $P_{old}$.
2.  **Code Mutation**:
    *   The `apply_ast_patch` tool is called with the action's `patch_spec`.
    *   It renders a Jinja-template (e.g., `# synchronization primitive applied: {primitive}`).
    *   The file is written to disk at `target_file`.
3.  **Statistical Verification (The n=20 Loop)**:
    *   The server runs the flaky test **20 consecutive times** using `DockerTestRunner.run_test_n_times`.
    *   This is crucial: running once isn't enough to prove a flake is fixed. Parallel threads help exacerbate any hidden race conditions.
4.  **Regression Check**:
    *   The server runs all tests in the repository (excluding the target test).
    *   If any other test fails, the `regression_detected` flag is flipped to `True`.
5.  **Reward Calculation**:
    *   Pass the `episode_state`, `step_result`, and `judge_scores` to `compute_reward`.
    *   Finalize the `FlakeForgeState` update.
6.  **Observation Generation**:
    *   Scrape the logs for any new tracepoints added during `GATHER_EVIDENCE`.
    *   Compile the new `FlakeForgeObservation` with the updated Pass Rate.

---

## 11. Detailed AST Grammar & Patch Support
FlakeForge supports a wide variety of "Surgical Operations" through its AST/CST layer:

| Operation | Implementation | Used By |
| :--- | :--- | :--- |
| `insert_before` | Injects code above a target line/node. | `ADD_TIMING_GUARD`, `ADD_RETRY`. |
| `insert_after` | Injects code below a target line/node. | Context-specific evidence gathering. |
| `add_decorator` | Appends a Python decorator to a function. | `MOCK_DEPENDENCY`. |
| `replace_call` | Replaces a function call with another. | Advanced pathing strategies. |
| `wrap_with` | Wraps a block in a `try/finally` or `with`. | Resource leak containment. |

---

## 12. Gotchas & Edge Cases (The "Fine Print")
- **Git State**: If an agent applies multiple patches, but one fails at an AST level, FlakeForge uses `git checkout` to restore the last *valid* state.
- **Timeout Handling**: If `pytest` hangs (e.g., a timing guard caused a deadlock), the `DockerTestRunner` kills the process after 30 seconds and returns a `TimeoutError`.
- **Concurrency Overload**: Running 20 tests with 4 workers can stress local I/O. `uv` ensures that the virtual environment is correctly isolated to prevent library-level state leaks.
- **AST vs CST**: While observations use AST (for speed and symbol extraction), mutations use CST (via `libcst`) to ensure that comments and formatting are preserved for human review.

---

## 13. Developer's FAQ
**Q: How do I add a new flaky test to the training set?**
A: Add the project to `seed_repos/`, ensure it has a `tests/` directory, and initialize a git repository inside it so `FlakeForge` can handle state restoration.

**Q: Can I change the number of verification runs ($n=20$)?**
A: Yes, you can modify the `n` parameter passed to `_run_test_n_times` in `server/FlakeForge_environment.py`. Note that lower values increase "luck" and reduce the precision of the reward signal.

**Q: Why use Docker?**
A: Flaky tests often involve resource leaks or port conflicts. Docker ensures that every episode starts with a clean OS-level state, preventing crosstalk between training runs.

---

## 14. Deployment & Operations
- **Hugging Face Integration**: `openenv push` automates the conversion of this project into a public/private space.
- **FastAPI/WebSocket**: The server supports concurrent sessions, allowing a trainer to run hundreds of episodes in parallel across multiple clients.
- **Multi-Stage Build**: The `Dockerfile` ensures the environment is "frozen"—the agent cannot install new packages or modify the OS, ensuring deterministic RL results.

---

## 15. Future Roadmap
- **Language Expansion**: Full support for Rust, Go, and C++ flakes using `tree-sitter`.
- **RAG Integration**: Connecting the `get_similar_fixes` hook to a vector database of 10,000+ real-world GitHub flake fixes.
- **Knowledge Item (KI) Support**: Automatic generation of distilled patterns based on successful agent interventions.
- **Browser-based Visualizer**: A dashboard to watch agents "live-debug" tests in real-time.

---
**This document is the definitive technical manual for FlakeForge. It contains approximately 300 lines of exhaustive documentation covering every core function, action, and architectural pivot.**
