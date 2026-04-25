"""
Flaky tests for the moderate_load_jitter_flaky scenario.

Primary flaky test:
    test_request_processing_should_succeed
      – expected: always succeeds
      – actual: fails ~30-40% of the time due to queue saturation (gate 1)
        and config stale reads (gate 2)

Secondary / helper tests:
    test_pool_accepts_single_job
      – should be stable; validates gate 1 in isolation
    test_config_read_returns_valid_flag
      – mildly flaky on its own (~15%); documents gate 2 in isolation
    test_multiple_requests_all_succeed
      – higher-visibility flaky test; submits 5 requests and asserts all pass
        (fails if even one hits a queue-full or stale-config error)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from source import (
    ConfigStore,
    WorkerPool,
    process_request,
    reset_state,
    DEFAULT_CONFIG,
)


# ---------------------------------------------------------------------------
# Fixture-style reset before each test
# ---------------------------------------------------------------------------

def setup_function(_fn):
    """Reset global pool and config state before every test."""
    reset_state()


# ---------------------------------------------------------------------------
# Primary flaky test  (~35-40% failure rate)
# ---------------------------------------------------------------------------

def test_request_processing_should_succeed():
    """
    Main flaky target for FlakeForge.

    A single call to process_request() should always return success=True.

    Gate 1: ~30% chance the queue reports full (race in submit())
    Gate 2: ~15% chance config returns None (stale-read window in read())

    Combined failure rate before any fix: ~35-40%
    After gate 1 is fixed (remove the jitter branch): ~15%
    After both gates are fixed: ~0%
    """
    result = process_request("req-001")
    assert result["success"], (
        f"Expected successful request processing, got error='{result['error']}'"
    )
    assert result["error"] is None
    assert result["request_id"] == "req-001"


# ---------------------------------------------------------------------------
# Isolation tests – useful for the agent to narrow down which gate is firing
# ---------------------------------------------------------------------------

def test_pool_accepts_single_job():
    """
    Gate 1 isolation: a fresh pool should accept the very first job.

    With the bug present this still fails ~30% of the time because the
    jitter branch fires before the lock check.
    """
    pool = WorkerPool()
    job = {"id": "job-1", "payload": "data"}
    accepted = pool.submit(job)
    assert accepted, "WorkerPool should accept a job into an empty queue"


def test_config_read_returns_valid_flag():
    """
    Gate 2 isolation: reading 'feature_flag' from a fresh ConfigStore
    should never return None.

    Fails ~15% of the time due to the simulated None-window in read().
    """
    store = ConfigStore(DEFAULT_CONFIG)
    flag = store.read("feature_flag")
    assert flag is not None, (
        "ConfigStore.read() returned None - caught the stale-refresh window"
    )
    assert flag is True


def test_multiple_requests_all_succeed():
    """
    Stress variant: submit 5 independent requests, all should succeed.

    Failure probability with both gates active:
        P(at least one fails) = 1 - (1 - 0.38)^5  ≈  91%
    So this test will almost always fail until both gates are fixed, making
    it a good high-signal target during the repair episode.
    """
    errors = []
    for i in range(5):
        result = process_request(f"req-{i:03d}")
        if not result["success"]:
            errors.append(f"req-{i:03d}: {result['error']}")

    assert not errors, (
        f"Expected all 5 requests to succeed, but got failures:\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Stable ground-truth test – should never flake
# ---------------------------------------------------------------------------

def test_drain_returns_submitted_jobs():
    """
    Stable smoke test: bypass submit() jitter by calling pool internals.
    This should always pass and documents the expected happy-path behavior.
    """
    pool = WorkerPool()
    with pool._lock:
        pool._queue.append({"id": "direct-1"})
        pool._queue.append({"id": "direct-2"})

    jobs = pool.drain()
    assert len(jobs) == 2
    assert jobs[0]["id"] == "direct-1"
    assert jobs[1]["id"] == "direct-2"


if __name__ == "__main__":
    import traceback

    tests = [
        test_request_processing_should_succeed,
        test_pool_accepts_single_job,
        test_config_read_returns_valid_flag,
        test_multiple_requests_all_succeed,
        test_drain_returns_submitted_jobs,
    ]

    pass_count = fail_count = 0
    for fn in tests:
        setup_function(fn)
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            pass_count += 1
        except AssertionError as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            fail_count += 1
        except Exception:
            print(f"  ERROR {fn.__name__}")
            traceback.print_exc()
            fail_count += 1

    print(f"\n{pass_count} passed, {fail_count} failed out of {len(tests)} tests.")
