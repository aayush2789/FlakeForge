"""Shared counter service — the counter is a module-level global that is never reset."""

_counter = 0

def increment():
    global _counter
    _counter += 1

def get_count():
    return _counter

def reset():
    """Proper reset — but nobody calls it between tests."""
    global _counter
    _counter = 0
