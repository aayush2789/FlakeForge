"""Tests for worker pool submission.

test_all_submitted is FLAKY because the capacity check in submit() is
outside the lock, causing race conditions under concurrent access.
"""
from source import WorkerPool, submit_batch

def test_all_submitted():
    """FLAKY — race condition in submit() causes some jobs to be lost."""
    pool = WorkerPool()
    accepted = submit_batch(pool, 8, workers=4)
    assert accepted == 8

def test_single_submit():
    pool = WorkerPool()
    assert pool.submit({"id": 1}) is True

def test_process_clears():
    pool = WorkerPool()
    pool.submit({"id": 1})
    pool.submit({"id": 2})
    count = pool.process_all()
    assert count == 2
    assert pool.pending() == 0
