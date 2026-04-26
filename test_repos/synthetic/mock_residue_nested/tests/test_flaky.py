"""Tests with nested mock.patch that leaks.

test_real_version is FLAKY because test_mock_config patches multiple
functions with start() but only stops one of them.
"""
from unittest import mock
import source

def test_mock_config():
    """Patches get_config_path and load_config but only stops one."""
    p1 = mock.patch("source.get_config_path", return_value="/tmp/test.json")
    p2 = mock.patch("source.load_config", return_value={"version": "2.0", "debug": True})
    p1.start()
    p2.start()
    assert source.get_version() == "2.0"
    assert source.is_debug() is True
    p1.stop()  # Bug: p2 is never stopped

def test_real_version():
    """FLAKY — load_config may still be mocked from test_mock_config."""
    version = source.get_version()
    assert version == "1.0"

def test_debug_off_by_default():
    """FLAKY — same leak issue."""
    assert source.is_debug() is False

def test_config_path_default():
    path = source.get_config_path()
    assert isinstance(path, str)
