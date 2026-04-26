"""Tests that assume stable set iteration order.

test_unique_preserves_order is FLAKY because set() does not preserve
insertion order.
"""
from source import unique_tags, first_tag

def test_unique_preserves_order():
    """FLAKY — assumes set iteration matches insertion order."""
    result = unique_tags(["banana", "apple", "cherry", "banana"])
    assert result == ["banana", "apple", "cherry"]

def test_unique_removes_duplicates():
    result = unique_tags(["a", "b", "a", "c", "b"])
    assert len(result) == 3

def test_first_tag_in_set():
    result = first_tag(["x", "y", "z"])
    assert result in {"x", "y", "z"}
