"""Tests where os.environ is patched without cleanup.

test_default_debug_off is FLAKY because a prior test sets APP_DEBUG=true
in os.environ without restoring it.
"""
import os
import source

def test_debug_mode_when_set():
    """Patches os.environ but never restores. Bug: leaks env var."""
    os.environ["APP_DEBUG"] = "true"
    assert source.get_debug_mode() is True
    # Bug: never does del os.environ["APP_DEBUG"]

def test_default_debug_off():
    """FLAKY — APP_DEBUG may still be 'true' from prior test."""
    assert source.get_debug_mode() is False

def test_app_name_default():
    assert source.get_app_name() == "MyApp"

def test_log_level_default():
    assert source.get_log_level() == "INFO"
