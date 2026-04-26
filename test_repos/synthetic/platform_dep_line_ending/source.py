"""Text file writer with line-ending assumption."""
import os
import tempfile


def write_lines(lines: list, path: str = None) -> str:
    """Write lines joined by newline to a file."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def count_bytes(path: str) -> int:
    """Count raw bytes in file."""
    with open(path, "rb") as f:
        return len(f.read())
