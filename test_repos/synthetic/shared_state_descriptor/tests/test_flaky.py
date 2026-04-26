"""Tests where TrackedField._all_fields accumulates across tests.

test_only_model_fields is FLAKY because _all_fields retains entries
from classes defined in other tests.
"""
from source import TrackedField

def test_define_user_model():
    class User:
        name = TrackedField()
        email = TrackedField()
    u = User()
    u.name = "Alice"
    assert u.name == "Alice"

def test_define_product_model():
    class Product:
        title = TrackedField()
        price = TrackedField()
    p = Product()
    p.title = "Widget"
    assert p.title == "Widget"

def test_only_model_fields():
    """FLAKY — expects only Order fields but sees User/Product fields too."""
    class Order:
        item = TrackedField()
        qty = TrackedField()
    fields = TrackedField.get_all_fields()
    order_fields = [k for k in fields if k.startswith("Order.")]
    assert len(fields) == len(order_fields)

def test_field_access():
    class Thing:
        val = TrackedField()
    t = Thing()
    t.val = 42
    assert t.val == 42
