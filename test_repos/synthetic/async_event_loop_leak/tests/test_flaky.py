"""Tests where async tasks leak across event loop runs.

test_no_pending_tasks is FLAKY because _background_tasks set may retain
tasks from prior tests if they haven't completed yet.
"""
import asyncio
from source import run_and_collect, pending_count, schedule_task, slow_operation

def test_run_operations():
    results = asyncio.run(run_and_collect(["a", "b", "c"]))
    assert len(results) == 3

def test_no_pending_tasks():
    """FLAKY — background tasks from prior test may still be in the set."""
    assert pending_count() == 0

def test_schedule_and_await():
    async def _test():
        task = await schedule_task(slow_operation("x", delay=0.01))
        result = await task
        return result
    result = asyncio.run(_test())
    assert result == "done:x"

def test_slow_operation():
    result = asyncio.run(slow_operation("test", delay=0.01))
    assert result == "done:test"
