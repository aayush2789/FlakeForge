"""Tests that create fresh Registry instances but share the class-level list.

test_fresh_registry_is_empty is FLAKY because _items is shared across all
instances via the class variable.
"""
from source import Registry

def test_fresh_registry_is_empty():
    """FLAKY — new instance still sees items from other tests."""
    reg = Registry()
    assert reg.count() == 0

def test_add_one_item():
    reg = Registry()
    reg.add("alpha")
    assert "alpha" in reg.get_items()

def test_add_two_items():
    reg = Registry()
    reg.add("beta")
    reg.add("gamma")
    assert reg.count() >= 2
