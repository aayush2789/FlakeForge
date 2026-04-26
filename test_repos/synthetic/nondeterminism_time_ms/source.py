"""ID generator using timestamps."""
import time

def generate_id() -> str:
    """Generate a unique ID from current millisecond timestamp.
    Bug: two calls in the same millisecond produce the same ID.
    """
    ms = int(time.time() * 1000)
    return f"id_{ms}"

def generate_pair() -> tuple:
    """Generate two IDs. Bug: may collide if called within same ms."""
    a = generate_id()
    b = generate_id()
    return a, b
