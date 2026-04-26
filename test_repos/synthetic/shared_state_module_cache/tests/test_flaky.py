"""Tests that share a module-level cache.

test_cache_empty_initially is FLAKY because prior tests populate _cache.
test_compute_called_once depends on cache being empty.
"""
from source import get_or_compute, cache_size, invalidate

_call_count = 0

def _expensive():
    global _call_count
    _call_count += 1
    return 42

def test_cache_empty_initially():
    """FLAKY — cache retains entries from prior tests."""
    assert cache_size() == 0

def test_compute_called_once():
    """FLAKY — if 'answer' is already cached, _expensive is not called."""
    global _call_count
    _call_count = 0
    get_or_compute("answer", _expensive)
    get_or_compute("answer", _expensive)
    assert _call_count == 1  # may be 0 if cached from prior test run

def test_get_value():
    result = get_or_compute("val", lambda: 99)
    assert result == 99

def test_invalidate():
    get_or_compute("temp", lambda: 1)
    invalidate("temp")
    assert cache_size() >= 0  # always true
