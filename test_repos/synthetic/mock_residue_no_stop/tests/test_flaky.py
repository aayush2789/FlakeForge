"""Tests where a mock.patch.start() is never stopped.

test_timestamp_is_real is FLAKY because a prior test patches time.time
with start() but never calls stop(), so the mock leaks.
"""
from unittest import mock
import source


def test_mock_time_for_testing():
    """Patches time.time but forgets to stop — leaks mock."""
    patcher = mock.patch("source.time.time", return_value=1000.0)
    patcher.start()  # Bug: no corresponding patcher.stop()
    assert source.current_timestamp() == 1000.0


def test_timestamp_is_real():
    """FLAKY — if mock leaked, this returns 1000.0 instead of real time."""
    import time
    now = time.time()
    ts = source.current_timestamp()
    assert abs(ts - now) < 2.0


def test_is_recent():
    ts = source.current_timestamp()
    assert source.is_recent(ts)
