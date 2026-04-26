"""Tests where import-time env read conflicts with runtime env changes.

test_mode_can_change is FLAKY because APP_MODE is frozen at import time.
test_default_is_production is FLAKY if another test sets APP_MODE before
source is imported.
"""
import os

def test_set_debug_mode():
    """Sets env var, but source.APP_MODE is already frozen."""
    os.environ["APP_MODE"] = "debug"
    import source
    # This may or may not see "debug" depending on import order

def test_default_is_production():
    """FLAKY — depends on whether APP_MODE was set before import."""
    import source
    assert source.is_production()

def test_mode_is_string():
    import source
    assert isinstance(source.get_mode(), str)

def test_get_mode_not_empty():
    import source
    assert len(source.get_mode()) > 0
