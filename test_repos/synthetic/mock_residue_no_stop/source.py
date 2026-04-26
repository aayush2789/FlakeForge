"""Simple time utility wrapping time.time()."""
import time


def current_timestamp() -> float:
    return time.time()


def is_recent(ts: float, window: float = 5.0) -> bool:
    return (time.time() - ts) < window
