"""
Moderate flaky target for FlakeForge testing.

Root cause: a worker pool that reports queue-full under simulated concurrent
load before it takes its lock. The primary test is intentionally moderate:
about 30% flaky at baseline, stable once the queue submission bug is fixed.
"""

from __future__ import annotations

import random
from threading import Lock
from typing import Any


# ---------------------------------------------------------------------------
# Gate 1 - saturating worker pool
# ---------------------------------------------------------------------------

class WorkerPool:
    """Minimal worker pool that processes jobs synchronously in tests."""

    QUEUE_CAPACITY = 5  # intentionally small

    def __init__(self) -> None:
        self._queue: list[Any] = []
        self._lock = Lock()

    def submit(self, job: dict[str, Any]) -> bool:
        """Submit a job. Returns False when the queue is full."""
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
# Stable config store
# ---------------------------------------------------------------------------

class ConfigStore:
    """Small config store used by the request pipeline."""

    def __init__(self, initial: dict[str, Any]) -> None:
        self._data: dict[str, Any] = dict(initial)
        self._refresh_lock = Lock()

    def read(self, key: str) -> Any:
        """Read a config key."""
        snapshot = self._data
        if snapshot is None:
            return None
        return snapshot.get(key)

    def refresh(self, new_data: dict[str, Any]) -> None:
        """Replace config atomically."""
        with self._refresh_lock:
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
      - error (str | None) - reason for failure
      - request_id (str)
    """
    job = {"id": request_id, "payload": f"data-{request_id}"}
    submitted = _pool.submit(job)
    if not submitted:
        return {
            "success": False,
            "error": "queue_full",
            "request_id": request_id,
        }

    flag = _config.read("feature_flag")
    if flag is None:
        return {
            "success": False,
            "error": "config_stale",
            "request_id": request_id,
        }

    return {"success": True, "error": None, "request_id": request_id}


def reset_state() -> None:
    """Reset global pool and config to a clean state."""
    global _pool, _config
    _pool = WorkerPool()
    _config = ConfigStore(DEFAULT_CONFIG)
