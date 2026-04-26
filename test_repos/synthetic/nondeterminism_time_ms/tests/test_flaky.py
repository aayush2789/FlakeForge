"""Tests that assume timestamp-based IDs are unique.

test_pair_unique is FLAKY because both IDs may be generated in the
same millisecond.
"""
from source import generate_id, generate_pair

def test_pair_unique():
    """FLAKY — both IDs generated in same millisecond are identical."""
    a, b = generate_pair()
    assert a != b

def test_id_format():
    id_val = generate_id()
    assert id_val.startswith("id_")

def test_id_changes_over_time():
    import time
    a = generate_id()
    time.sleep(0.01)
    b = generate_id()
    assert a != b
