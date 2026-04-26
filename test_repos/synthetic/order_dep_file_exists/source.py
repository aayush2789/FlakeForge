"""File-based flag system."""
import os
import tempfile

_flag_dir = tempfile.mkdtemp(prefix="flags_")


def set_flag(name: str):
    """Create a flag file."""
    path = os.path.join(_flag_dir, name)
    with open(path, "w") as f:
        f.write("1")


def check_flag(name: str) -> bool:
    """Check if flag exists."""
    return os.path.exists(os.path.join(_flag_dir, name))


def clear_flag(name: str):
    path = os.path.join(_flag_dir, name)
    if os.path.exists(path):
        os.unlink(path)
