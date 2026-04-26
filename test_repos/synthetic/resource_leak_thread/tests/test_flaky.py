"""Tests that start background workers without stopping them.

test_worker_count is FLAKY because timing of background thread affects
how many results accumulate.
"""
import time
from source import BackgroundWorker

def test_worker_produces_results():
    w = BackgroundWorker()
    w.start(lambda: "tick")
    time.sleep(0.15)
    results = w.get_results()
    assert len(results) > 0
    # Bug: no w.stop()

def test_worker_count():
    """FLAKY — timing-dependent: expects exactly 2 results in 0.1s."""
    w = BackgroundWorker()
    w.start(lambda: "x", interval=0.05)
    time.sleep(0.1)
    assert len(w.get_results()) == 2  # may be 1 or 3 due to timing

def test_worker_stops():
    w = BackgroundWorker()
    w.start(lambda: "y")
    w.stop()
    assert not w.is_alive()
