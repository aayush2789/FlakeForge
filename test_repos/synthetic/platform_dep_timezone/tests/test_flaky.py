"""Tests that assume local time == UTC.

test_today_matches_utc is FLAKY because near midnight the local date
and UTC date may differ depending on timezone.
"""
from datetime import datetime, timezone
from source import today_label, utc_label, is_same_day_utc

def test_today_matches_utc():
    """FLAKY — fails when local timezone differs from UTC near midnight."""
    assert today_label() == utc_label()

def test_same_day_check():
    """FLAKY — same root cause as test_today_matches_utc."""
    assert is_same_day_utc()

def test_label_format():
    label = today_label()
    parts = label.split("-")
    assert len(parts) == 3
