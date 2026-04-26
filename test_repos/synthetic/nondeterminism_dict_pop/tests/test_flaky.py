"""Tests that assume popitem() returns items in a specific order.

test_schedule_order is FLAKY because it assumes FIFO but popitem() is LIFO
in CPython 3.7+.
"""
from source import schedule, first_task


def test_schedule_order():
    """FLAKY — assumes FIFO order but popitem() is LIFO."""
    tasks = {"build": 1, "test": 2, "deploy": 3}
    result = schedule(tasks)
    assert result == ["build", "test", "deploy"]


def test_all_tasks_processed():
    tasks = {"a": 1, "b": 2, "c": 3}
    result = schedule(tasks)
    assert set(result) == {"a", "b", "c"}


def test_first_task_exists():
    tasks = {"x": 10, "y": 20}
    assert first_task(tasks) in {"x", "y"}
