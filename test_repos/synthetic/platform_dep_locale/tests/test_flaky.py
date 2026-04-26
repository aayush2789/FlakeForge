"""Tests that assume specific locale sorting behavior.

test_sort_order is FLAKY because locale.strxfrm produces different
orderings on different platforms and locale settings.
"""
from source import sorted_names, compare_names

def test_sort_order():
    """FLAKY — locale-dependent sorting may differ from ASCII sort."""
    names = ["Zebra", "apple", "Banana", "cherry"]
    result = sorted_names(names)
    assert result == ["apple", "Banana", "cherry", "Zebra"]

def test_compare_same():
    assert compare_names("hello", "hello") == 0

def test_sort_returns_all():
    names = ["c", "a", "b"]
    result = sorted_names(names)
    assert set(result) == {"a", "b", "c"}

def test_normalize():
    from source import normalize_for_sort
    assert normalize_for_sort("  Hello  ") == "hello"
