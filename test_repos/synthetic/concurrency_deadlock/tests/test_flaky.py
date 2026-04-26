"""Tests for bank transfer system.

test_parallel_transfers_complete is FLAKY because concurrent transfers
in opposite directions can deadlock due to lock ordering.
"""
from source import Account, transfer, parallel_transfers

def test_simple_transfer():
    a = Account("A", 100)
    b = Account("B", 50)
    assert transfer(a, b, 30) is True
    assert a.balance == 70
    assert b.balance == 80

def test_parallel_transfers_complete():
    """FLAKY — deadlock can occur with opposite-direction transfers."""
    a = Account("A", 10000)
    b = Account("B", 10000)
    count = parallel_transfers(a, b, 1, 100)
    assert count > 0
    assert a.balance + b.balance == 20000

def test_insufficient_funds():
    a = Account("A", 10)
    b = Account("B", 0)
    assert transfer(a, b, 20) is False
