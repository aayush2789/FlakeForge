"""Tests where importing source monkey-patches os.path.join.

test_os_join_native is FLAKY because importing source changes the behavior
of os.path.join globally.
"""
import os

def test_os_join_native():
    """FLAKY — os.path.join may be monkey-patched from source import."""
    result = os.path.join("a", "b", "c")
    # On Windows, native join gives 'a\\b\\c'; patched gives 'a/b/c'
    assert "\\" in result or os.sep == "/"

def test_import_source():
    import source
    result = source.join_paths("x", "y")
    assert "x" in result and "y" in result

def test_join_has_separator():
    import source
    result = source.join_paths("a", "b")
    assert "/" in result or "\\" in result
