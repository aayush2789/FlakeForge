"""Tests that create 'fresh' ConfigManager but get the singleton.

test_new_manager_is_empty is FLAKY because the singleton carries state
from previous tests.
"""
from source import ConfigManager


def test_new_manager_is_empty():
    """FLAKY — singleton instance retains settings from prior tests."""
    mgr = ConfigManager()
    assert mgr.all_settings() == {}


def test_set_and_get():
    mgr = ConfigManager()
    mgr.set("debug", True)
    assert mgr.get("debug") is True


def test_set_multiple():
    mgr = ConfigManager()
    mgr.set("host", "localhost")
    mgr.set("port", 8080)
    assert mgr.get("host") == "localhost"
