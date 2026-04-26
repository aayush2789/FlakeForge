"""Tests that share a global counter without resetting it.

The flaky test is test_increment_from_zero: it assumes the counter starts
at 0, but prior tests may have incremented it.
"""
import source

def test_increment_from_zero():
    """FLAKY — assumes counter is at 0 but other tests increment it."""
    source.increment()
    assert source.get_count() == 1

def test_increment_twice():
    source.increment()
    source.increment()
    assert source.get_count() >= 2

def test_counter_positive():
    source.increment()
    assert source.get_count() > 0
