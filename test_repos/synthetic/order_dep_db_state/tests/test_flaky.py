"""Tests that share a module-level DB instance.

test_users_table_has_row is FLAKY because it depends on test_insert_user
having run first to create the table and insert a row.
"""
from source import db

def test_create_users_table():
    db.create_table("users")
    assert db.count("users") == 0

def test_insert_user():
    db.insert("users", {"name": "Alice", "age": 30})
    assert db.count("users") >= 1

def test_users_table_has_row():
    """FLAKY — depends on prior tests having created table and inserted data."""
    rows = db.select("users")
    assert any(r["name"] == "Alice" for r in rows)

def test_create_orders_table():
    db.create_table("orders")
    db.insert("orders", {"item": "widget", "qty": 5})
    assert db.count("orders") == 1
