"""Tests using a module-scoped connection pool fixture.

test_pool_is_fresh is FLAKY because the module fixture shares the pool,
and prior tests acquire connections without releasing.
"""
import pytest
from source import ConnectionPool

@pytest.fixture(scope="module")
def pool():
    """Bug: module scope means pool is shared across all tests in module."""
    return ConnectionPool(max_size=3)

def test_pool_is_fresh(pool):
    """FLAKY — pool may already have connections from prior tests."""
    assert pool.active_count() == 0

def test_acquire_one(pool):
    pool.acquire()
    assert pool.active_count() >= 1

def test_acquire_two(pool):
    pool.acquire()
    pool.acquire()
    assert pool.active_count() >= 2

def test_pool_not_exhausted(pool):
    """FLAKY — may hit max_size if prior tests didn't release."""
    conn = pool.acquire()
    assert conn is not None
