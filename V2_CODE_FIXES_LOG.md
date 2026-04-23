# FlakeForge V2 — Production-Grade Code Fixes Log

This document details the exact critical system limitations identified during our recent codebase audit, the specific structural code changes applied to fix them, and concrete "Before & After" cases showing how the system now handles production-grade code.

---

## 1. Causal Graph: AST Parent-Lookup Bug

**File Fixed:** `server/causal_graph.py`

### The Problem
When `CrossRepoGraphBuilder._follow_calls` walked the Abstract Syntax Tree (AST), it needed to know if a function call was happening inside an `await` statement (to mark `call_type="async_await"`). Standard `ast.walk()` in Python is flat—it visits children but provides no information about their parent context. Because of this, the code was defaulting every call to `call_type="direct"`, blinding the RL agent to asynchronous deadlocks.

### The Fix
Implemented `_build_parent_map()` which does a single pre-pass over the AST using `ast.iter_child_nodes()` to build a dictionary mapping `id(child)` to `parent_node`. Now, when checking an `ast.Call` node, we look it up in the parent map in `O(1)` time.

#### Before vs After
**Scenario:** `async def handler(): await process_data()`
*   **Before:** Agent sees `process_data()` called as `"direct"`. Thinks it's a standard synchronous function call. Misses the async boundary.
*   **After:** Agent looks up `process_data()` in the parent map, finds the `ast.Await` parent, and correctly flags the edge as `"async_await"`.

---

## 2. Causal Graph: Production Module Resolution (Import Bug)

**File Fixed:** `server/causal_graph.py`

### The Problem
The `_build_import_map` function was extremely naive. It assumed all code lived at `repo_root/module.py`. It completely ignored common production folder layouts like `src/` or `app/`, namespace packages, and—most crucially—re-exported names in `__init__.py` files (e.g., `from .core import DB` inside `src/database/__init__.py`). 

### The Fix
Rewrote the import resolver to:
1. Try multiple target lookup roots: `[".", "src", "app", "lib"]`.
2. Check for both `.py` files AND `__init__.py` module directories.
3. If an `__init__.py` is found, the new `_resolve_reexported_name()` function parses it to resolve facade imports to their actual underlying files.
4. Added an `unresolved_imports` surface variable so the agent knows precisely which traces broke.

#### Before vs After
**Scenario:** `from src.database import save_user`
*   **Before:** Fails to find `src/database.py`, drops the trace silently. Agent can't see the database interaction.
*   **After:** Resolves `src/database/__init__.py`, finds the `from .postgres import save_user` re-export, and successfully traces the boundary to `src/database/postgres.py`.

---

## 3. Deep Action: EXTRACT_ASYNC_SCOPE Actually Works Now

**File Fixed:** `server/tools.py`

### The Problem
When the agent detected a synchronous blocking call inside an async loop and invoked `EXTRACT_ASYNC_SCOPE` (with `direction="sync_to_async"`), the `AsyncScopeExtractor` transformer purely flipped the `def` keyword to `async def` and did absolutely nothing else. 
This was actively harmful: taking a blocking synchronous function and just labeling it `async` means the blocking call inside it will now *silently stall the entire event loop* when awaited.

### The Fix
Re-architected the CST Transformer. Now, when converting a function, it doesn't just flip the keyword. It walks the function body, identifies known blocking I/O calls (using the exact signatures from the Causal Graph like `db.commit`, `requests.get`), and wraps *those specific calls* in `await asyncio.to_thread()`. It also auto-injects `import asyncio` at the top of the file.

#### Before vs After
**Scenario:** A route handler runs a synchronous database commit.
*   **Before Patch:**
    ```python
    async def save_profile(user_id):
        # Still blocks the whole async loop!
        db.session.commit()
    ```
*   **After Patch:**
    ```python
    import asyncio

    async def save_profile(user_id):
        # Now safely offloaded to a thread pool
        await asyncio.to_thread(db.session.commit)
    ```

---

## 4. Deep Action: Exact Matching in REFACTOR_CONCURRENCY

**File Fixed:** `server/tools.py`

### The Problem
The `PrimitiveSwapper` inside `_apply_refactor_concurrency` used naive substring matching: `if from_primitive.split(".")[-1] in name:`. 
If the agent tried to refactor `threading.Lock`, the code looked for the word `"Lock"` anywhere in the call. It would accidentally overwrite `asyncio.Lock()`, `FileLock()`, and `DistributedLock()`. Furthermore, it never updated the import statements at the top of the file, leading to `NameError` crashes at runtime.

### The Fix
1. Switched to exact matching (`full_name == from_primitive`) on the CST nodes.
2. Implemented an `ImportRewriter` CST Transformer that intelligently detects how the primitive was imported and rewrites the statement.

#### Before vs After
**Scenario:** Agent decides to upgrade `threading.Lock` to `asyncio.Lock` in a file that also uses a custom caching lock.
*   **Before Patch:**
    ```python
    from threading import Lock
    def do_work():
        lock = Lock()  # Mutates to asyncio.Lock
        cache_lock = CacheLock() # ERRONEOUSLY mutates to asyncio.Lock 
    ```
    *Result: Triggers a runtime `NameError` because `asyncio` is not imported.*

*   **After Patch:**
    ```python
    from asyncio import Lock
    def do_work():
        lock = Lock() # Mutates correctly
        cache_lock = CacheLock() # Left safely alone
    ```

---

## 5. Statistical Rigor in Performance Sentinel

**File Fixed:** `server/perf_sentinel.py`

### The Problem
The `PerformanceSentinel` uses the *Scipy Mann-Whitney U* statistical test to detect subtle latency regressions caused by agent fixes (e.g., adding a massive lock that slows down the app). 
However, `n_benchmark_runs` was hardcoded to `10`. Mathematically, an unpaired rank test with `n=10` groups can only achieve a minimum p-value of ~0.05. It was statistically impossible to detect anything less than a massive 500% performance cliff. 1.5x regressions simply flew under the threshold.

### The Fix
1. Raised the default sample size from 10 to 20.
2. Implemented **Adaptive Re-sampling**: If the statistical test yields a p-value between 0.05 and 0.15 (the "inconclusive zone where a subtle regression might be hiding"), the Sentinel dynamically invokes the Chaos Runner to gather up to 50 total samples and re-runs the calculation, ensuring high statistical confidence without wasting CPU cycles on obvious passes/fails.

#### Before vs After
**Scenario:** Agent wraps a fast function in a slow `threading.RLock`, resulting in a 1.6x slowdown.
*   **Before:** Gathering 10 samples yields `p_value = 0.08`. Test marks `is_regression = False`. The slowdown makes it to production.
*   **After:** Gathering 20 samples yields `p_value = 0.08`. Sentinel detects it is within the inconclusive upper bound, loops back, gathers 10 more samples (n=30), new `p_value = 0.03`. Test marks `is_regression = True` and penalizes the agent's reward.
