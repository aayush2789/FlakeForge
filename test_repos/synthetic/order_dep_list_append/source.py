"""Simple event log — events accumulate globally."""

_events = []

def emit(event: str):
    _events.append(event)

def get_events() -> list:
    return list(_events)

def clear():
    _events.clear()
