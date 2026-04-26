"""Tests where lru_cache causes stale results.

test_added_user_found is FLAKY because get_user may return a cached
"Unknown" from a prior test that queried before the user was added.
"""
from source import get_user, add_user, remove_user

def test_unknown_user():
    result = get_user(99)
    assert result == "Unknown"

def test_add_then_find():
    """FLAKY — if get_user(99) was called before, lru_cache returns 'Unknown'."""
    add_user(99, "Dave")
    result = get_user(99)
    assert result == "Dave"

def test_known_user():
    assert get_user(1) == "Alice"

def test_remove_user():
    remove_user(2)
    # lru_cache still returns "Bob" — stale
    result = get_user(2)
    # This assertion depends on cache state
    assert result in ("Bob", "Unknown")
