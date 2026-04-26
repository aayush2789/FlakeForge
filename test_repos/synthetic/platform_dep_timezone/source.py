"""Date utilities that assume local timezone == UTC."""
from datetime import datetime, timezone

def today_label() -> str:
    """Return today's date label. Bug: uses local time, not UTC."""
    return datetime.now().strftime("%Y-%m-%d")

def utc_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def is_same_day_utc() -> bool:
    """Check if local date matches UTC date.
    Bug: returns False near midnight when local != UTC.
    """
    local = datetime.now().strftime("%Y-%m-%d")
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return local == utc
