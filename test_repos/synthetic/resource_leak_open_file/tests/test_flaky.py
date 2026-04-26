"""Tests that create FileLogger without closing it.

test_empty_log_on_creation is FLAKY on Windows because unclosed handles
prevent file operations in subsequent tests. On any platform the leaked
handle keeps the file open, which can cause issues.
"""
import os
from source import FileLogger

def test_empty_log_on_creation():
    """FLAKY — previous test's logger may have written to the same tmp file."""
    logger = FileLogger()
    assert logger.read_all() == ""

def test_log_message():
    logger = FileLogger()
    logger.log("hello")
    assert "hello" in logger.read_all()

def test_log_multiple():
    logger = FileLogger()
    logger.log("a")
    logger.log("b")
    content = logger.read_all()
    assert "a" in content and "b" in content
