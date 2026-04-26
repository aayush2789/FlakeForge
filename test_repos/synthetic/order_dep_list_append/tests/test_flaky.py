"""Tests that depend on execution order via shared _events list.

test_first_event_is_start is FLAKY because it assumes "start" is the
first event, but prior tests may have emitted other events.
"""
import source

def test_first_event_is_start():
    """FLAKY — assumes events list starts empty or with 'start' first."""
    source.emit("start")
    assert source.get_events()[0] == "start"

def test_emit_and_count():
    source.emit("ping")
    source.emit("pong")
    assert len(source.get_events()) >= 2

def test_events_contain_item():
    source.emit("heartbeat")
    assert "heartbeat" in source.get_events()
