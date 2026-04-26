"""Tests using exact float comparison.

test_tax_exact and test_split_roundtrip are FLAKY because floating-point
arithmetic is not exact.
"""
from source import total_with_tax, split_bill, verify_split

def test_tax_exact():
    """FLAKY — float accumulation makes this fail for certain price lists."""
    prices = [0.1, 0.2, 0.3]
    result = total_with_tax(prices, tax_rate=0.1)
    assert result == 0.66

def test_split_roundtrip():
    """FLAKY — float division * multiplication doesn't round-trip exactly."""
    assert verify_split(100.0, 3)

def test_split_positive():
    result = split_bill(100.0, 4)
    assert result > 0

def test_tax_positive():
    result = total_with_tax([10.0, 20.0])
    assert result > 30.0
