"""Module that monkey-patches os.path.join on import."""
import os

_original_join = os.path.join

def _patched_join(*args):
    """Monkey-patched join that normalizes to forward slashes.
    Bug: this patch is applied at import time and affects all code.
    """
    result = _original_join(*args)
    return result.replace("\\", "/")

# Bug: monkey-patches os.path.join at import time
os.path.join = _patched_join

def join_paths(*parts) -> str:
    return os.path.join(*parts)

def restore():
    """Undo the monkey-patch."""
    os.path.join = _original_join
