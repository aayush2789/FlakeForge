"""Tests that rely on temp file state leaking between runs.

test_cache_starts_empty is FLAKY because a prior test's write_cache
leaves _cache_path set.
"""
import source

def test_cache_starts_empty():
    """FLAKY — _cache_path may be set from a previous test."""
    assert source.read_cache() == ""

def test_write_and_read():
    path = source.write_cache("hello")
    assert source.read_cache() == "hello"

def test_write_overwrites():
    source.write_cache("first")
    source.write_cache("second")
    assert source.read_cache() == "second"
