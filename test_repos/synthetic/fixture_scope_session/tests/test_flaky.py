"""Tests using a session-scoped fixture that returns a mutable object.

test_cart_starts_empty is FLAKY because the session-scoped fixture
returns the same Cart instance across all tests, and prior tests add items.
"""
import pytest
from source import Cart

@pytest.fixture(scope="session")
def cart():
    """Bug: session scope means same Cart shared across ALL tests."""
    return Cart()

def test_cart_starts_empty(cart):
    """FLAKY — cart may have items from prior tests in the session."""
    assert cart.count() == 0

def test_add_item(cart):
    cart.add("apple")
    assert "apple" in cart.contents()

def test_add_multiple(cart):
    cart.add("banana")
    cart.add("cherry")
    assert cart.count() >= 2
