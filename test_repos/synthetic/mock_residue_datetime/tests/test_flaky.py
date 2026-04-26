"""Tests where datetime.now is mocked without cleanup.

test_timestamp_is_today is FLAKY because a prior test mocks datetime.now
with patch.start() and never calls stop().
"""
from unittest import mock
from datetime import datetime
import source

def test_morning_greeting():
    """Mocks datetime but leaks it."""
    fake_dt = mock.MagicMock()
    fake_dt.now.return_value = datetime(2025, 1, 1, 9, 0, 0)
    fake_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
    patcher = mock.patch("source.datetime", fake_dt)
    patcher.start()  # Bug: no patcher.stop()
    assert source.greeting() == "Good morning"

def test_timestamp_is_today():
    """FLAKY — if mock leaked, datetime.now() returns mocked 2025-01-01."""
    label = source.timestamp_label()
    today = datetime.now().strftime("%Y-%m-%d")
    assert label == today

def test_greeting_is_string():
    result = source.greeting()
    assert isinstance(result, str)
