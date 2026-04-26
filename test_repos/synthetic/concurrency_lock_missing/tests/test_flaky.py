"""Tests for thread-safe counter.

test_parallel_count_exact is FLAKY because Counter.increment has a
read-modify-write race without a lock.
"""
from source import Counter, parallel_increment

def test_parallel_count_exact():
    """FLAKY — race condition causes lost increments."""
    c = Counter()
    parallel_increment(c, 1000, workers=4)
    assert c.get() == 1000

def test_single_thread_works():
    c = Counter()
    for _ in range(100):
        c.increment()
    assert c.get() == 100

def test_reset_works():
    c = Counter()
    c.increment()
    c.reset()
    assert c.get() == 0
