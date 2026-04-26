"""Tests where one test creates a flag file another test depends on.

test_flag_was_set is FLAKY because it depends on test_set_flag running first.
"""
import source


def test_set_flag():
    source.set_flag("ready")
    assert source.check_flag("ready")


def test_flag_was_set():
    """FLAKY — depends on test_set_flag having run first."""
    assert source.check_flag("ready")


def test_clear_flag():
    source.set_flag("temp")
    source.clear_flag("temp")
    assert not source.check_flag("temp")


def test_flag_not_exists():
    assert not source.check_flag("nonexistent")
