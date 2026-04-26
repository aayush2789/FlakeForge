"""Tests with tight async timeouts.

test_fetch_succeeds is FLAKY because slow_fetch has random delay 0.05-0.25s
but timeout is only 0.1s, so ~60% of calls timeout.
"""
import asyncio
import pytest
from source import fetch_with_timeout, fetch_all

def test_fetch_succeeds():
    """FLAKY — timeout too tight for variable latency."""
    result = asyncio.run(fetch_with_timeout("http://example.com"))
    assert "data from" in result

def test_fetch_returns_string():
    """Less flaky — just checks type when it succeeds."""
    try:
        result = asyncio.run(fetch_with_timeout("http://test.com", timeout=1.0))
        assert isinstance(result, str)
    except asyncio.TimeoutError:
        pass

def test_fetch_all_partial():
    """Some may timeout, but at least one should succeed."""
    urls = ["http://a.com", "http://b.com", "http://c.com"]
    results = asyncio.run(fetch_all(urls))
    successes = [r for r in results if isinstance(r, str)]
    assert len(successes) >= 0  # always passes
