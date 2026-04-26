"""Tests that assume dict state from other tests.

test_get_returns_setup_value is FLAKY because it assumes 'config' key
was set by a prior test.
"""
import source

def test_put_and_get():
    source.put("config", "enabled")
    assert source.get("config") == "enabled"

def test_get_returns_setup_value():
    """FLAKY — assumes 'config' was set by test_put_and_get running first."""
    assert source.get("config") == "enabled"

def test_has_key():
    source.put("flag", True)
    assert source.has("flag")

def test_delete_key():
    source.put("temp", 42)
    source.delete("temp")
    assert not source.has("temp")
