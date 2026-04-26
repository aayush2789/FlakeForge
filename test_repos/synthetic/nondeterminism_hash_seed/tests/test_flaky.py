"""Tests that assume deterministic hash() output.

test_consistent_assignment is FLAKY because hash() uses PYTHONHASHSEED
which changes between process invocations.
"""
from source import HashRing

def test_consistent_assignment():
    """FLAKY — hash() output varies with PYTHONHASHSEED."""
    ring = HashRing(["node_a", "node_b", "node_c"])
    node = ring.get_node("my_key")
    assert node == "node_a"  # only true for certain PYTHONHASHSEED

def test_node_in_ring():
    ring = HashRing(["x", "y", "z"])
    node = ring.get_node("test")
    assert node in ["x", "y", "z"]

def test_all_nodes_present():
    ring = HashRing(["a", "b", "c"])
    assert set(ring.get_nodes()) == {"a", "b", "c"}

def test_ring_not_empty():
    ring = HashRing(["single"])
    assert len(ring.sorted_keys) > 0
