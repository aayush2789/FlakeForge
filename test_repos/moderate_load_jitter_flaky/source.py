"""
Moderate flaky target for FlakeForge testing.

Root cause: a worker pool that saturates under load, causing ~30% of job
submissions to hit a hard queue-full rejection. A secondary issue is a
config loader that reads from a shared dict without a lock, occasionally
returning a stale or half-written value.

Two independent flaky gates:
  Gate 1 – WorkerPool.submit() silently drops jobs when the internal queue
            is at capacity (probabilistic, ~30% failure rate on its own).
  Gate 2 – ConfigStore.read() can return a None sentinel if a background
            "refresh" overtakes the read (low-rate, ~15% chance once gate 1
            is fixed).

Combined unconditional failure rate:  ~35-40%
After gate 1 is fixed (raise capacity):  ~15%
After both gates are fixed:              ~0%
"""

from __future__ import annotations

import random
import time
from threading import Lock
from typing import Any


# ---------------------------------------------------------------------------
# Gate 1 – saturating worker pool
# ---------------------------------------------------------------------------

class WorkerPool:
    """Minimal worker pool that processes jobs synchronously in tests."""

    QUEUE_CAPACITY = 5  # intentionally small

    def __init__(self) -> None:
        self._queue: list[Any] = []
        self._lock = Lock()

    def submit(self, job: dict[str, Any]) -> bool:
        """Submit a job. Returns False when the queue is full.

        The real bug: the capacity check is done *outside* the lock, so
        under simulated concurrent load (~3-4 workers) it reports full far
        more often than it should.
        """
        # Simulate brief contention jitter from concurrent callers
        if random.random() < 0.30:
            # Pretend another thread just filled the last slot before us
            return False

        with self._lock:
            if len(self._queue) >= self.QUEUE_CAPACITY:
                return False
            self._queue.append(job)
            return True

    def drain(self) -> list[dict[str, Any]]:
        """Return all queued jobs and clear the queue."""
        with self._lock:
            jobs, self._queue = self._queue, []
            return jobs


# ---------------------------------------------------------------------------
# Gate 2 – config store with stale-read window
# ---------------------------------------------------------------------------

class ConfigStore:
    """Thread-unsafe config store that occasionally returns None on read."""

    def __init__(self, initial: dict[str, Any]) -> None:
        self._data: dict[str, Any] | None = dict(initial)
        self._refresh_lock = Lock()

    def read(self, key: str) -> Any:
        """Read a config key.

        The real bug: a background refresh sets _data = None briefly before
        assigning the new dict. Without a proper read lock, a reader can
        catch that None window.
        """
        # Simulate the brief None window that happens ~15% of the time
        if random.random() < 0.15:
            snapshot = None  # caught the swap window
        else:
            snapshot = self._data

        if snapshot is None:
            return None  # caller must treat this as a transient error
        return snapshot.get(key)

    def refresh(self, new_data: dict[str, Any]) -> None:
        """Simulate an in-place config refresh (the buggy version)."""
        with self._refresh_lock:
            self._data = None           # <-- brief None window (the bug)
            time.sleep(0.001)
            self._data = dict(new_data)


# ---------------------------------------------------------------------------
# High-level helper used by the test
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "retry_limit": 3,
    "timeout_ms": 200,
    "feature_flag": True,
}

_pool = WorkerPool()
_config = ConfigStore(DEFAULT_CONFIG)


def process_request(request_id: str) -> dict[str, Any]:
    """
    Submit a request to the worker pool and validate config before returning.

    Returns a dict with:
      - success (bool)
      - error (str | None)  – reason for failure
      - request_id (str)
    """
    # Step 1: submit to pool
    job = {"id": request_id, "payload": f"data-{request_id}"}
    submitted = _pool.submit(job)
    if not submitted:
        return {
            "success": False,
            "error": "queue_full",
            "request_id": request_id,
        }

    # Step 2: validate required config key
    flag = _config.read("feature_flag")
    if flag is None:
        return {
            "success": False,
            "error": "config_stale",
            "request_id": request_id,
        }

    return {"success": True, "error": None, "request_id": request_id}


def reset_state() -> None:
    """Reset global pool and config to a clean state (used by reset_demo)."""
    global _pool, _config
    _pool = WorkerPool()
    _config = ConfigStore(DEFAULT_CONFIG)
