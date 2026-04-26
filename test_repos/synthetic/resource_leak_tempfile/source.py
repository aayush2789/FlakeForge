"""Temp-file based cache that never cleans up."""
import tempfile
import os

_cache_path = None

def write_cache(data: str) -> str:
    """Write data to a temp file. Bug: never deletes old file."""
    global _cache_path
    fd, _cache_path = tempfile.mkstemp(prefix="cache_", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(data)
    return _cache_path

def read_cache() -> str:
    if _cache_path is None or not os.path.exists(_cache_path):
        return ""
    with open(_cache_path) as f:
        return f.read()

def cleanup():
    """Proper cleanup — but tests don't call this."""
    global _cache_path
    if _cache_path and os.path.exists(_cache_path):
        os.unlink(_cache_path)
    _cache_path = None
