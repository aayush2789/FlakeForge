"""Tests that assume Unix line endings in byte counts.

test_byte_count_matches is FLAKY on Windows because 'w' mode translates
'\\n' to '\\r\\n', adding extra bytes.
"""
from source import write_lines, count_bytes


def test_byte_count_matches():
    """FLAKY on Windows — expects Unix line endings (\\n) but gets \\r\\n."""
    lines = ["hello", "world"]
    path = write_lines(lines)
    # "hello\nworld" = 11 bytes on Unix, 12 bytes on Windows (\r\n)
    assert count_bytes(path) == 11


def test_write_creates_file():
    import os
    path = write_lines(["test"])
    assert os.path.exists(path)


def test_write_multiple_lines():
    path = write_lines(["a", "b", "c"])
    assert count_bytes(path) > 0
