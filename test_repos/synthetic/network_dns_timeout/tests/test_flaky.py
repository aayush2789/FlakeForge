"""Tests for DNS resolution with tight timeout.

test_resolve_known_host is FLAKY because the random delay (0.01-0.2s)
can exceed the default 0.1s timeout.
"""
from source import resolve, resolve_all

def test_resolve_known_host():
    """FLAKY — random delay may exceed 0.1s timeout."""
    ip = resolve("api.example.com")
    assert ip == "10.0.0.1"

def test_resolve_unknown_host():
    """Less flaky — still affected by timeout but checks default."""
    try:
        ip = resolve("unknown.com")
        assert ip == "0.0.0.0"
    except TimeoutError:
        pass

def test_resolve_all_partial():
    hosts = ["api.example.com", "db.example.com"]
    results = resolve_all(hosts)
    assert len(results) == 2
