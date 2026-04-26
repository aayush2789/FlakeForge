"""Tests for concurrent result collection.

test_parallel_total is FLAKY because ResultCollector.record has a TOCTOU race.
"""
from source import ResultCollector, parallel_record

def test_parallel_total():
    """FLAKY — concurrent record() calls lose updates."""
    rc = ResultCollector()
    parallel_record(rc, "hits", 1000, workers=8)
    assert rc.get("hits") == 1000

def test_single_record():
    rc = ResultCollector()
    rc.record("x", 5)
    assert rc.get("x") == 5

def test_total():
    rc = ResultCollector()
    rc.record("a", 10)
    rc.record("b", 20)
    assert rc.total() == 30
