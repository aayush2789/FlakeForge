"""Tests using a generator fixture that doesn't finalize.

test_no_open_connections is FLAKY because the fixture yields
but never reaches the cleanup code after yield.
"""
import pytest
from source import Connection

@pytest.fixture
def db_conn():
    """Bug: generator fixture — but if test errors before yield cleanup,
    the connection is never closed."""
    conn = Connection("test://db")
    conn.open()
    yield conn
    # This cleanup code may not run if the fixture is session/module scoped
    # or if the test framework has issues

@pytest.fixture(scope="module")
def module_conn():
    """Bug: module-scoped generator fixture — cleanup delayed until module end."""
    conn = Connection("test://module_db")
    conn.open()
    yield conn
    conn.close()

def test_use_connection(module_conn):
    txn = module_conn.begin_transaction()
    assert txn == "txn_0"

def test_use_connection_again(module_conn):
    txn = module_conn.begin_transaction()
    module_conn.commit()

def test_no_open_connections():
    """FLAKY — module_conn is still open until all tests in module complete."""
    assert Connection.open_count() == 0

def test_fresh_connection(db_conn):
    assert db_conn.is_open
