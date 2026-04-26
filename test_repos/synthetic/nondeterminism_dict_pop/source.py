"""Task scheduler that pops items from a dict."""


def schedule(tasks: dict) -> list:
    """Process tasks by popping them one at a time.
    Bug: dict.popitem() order is insertion order in 3.7+ but the test
    assumes a specific order that may differ across Python versions
    or if dict is built differently.
    """
    order = []
    work = dict(tasks)
    while work:
        key, val = work.popitem()  # LIFO in CPython 3.7+
        order.append(key)
    return order


def first_task(tasks: dict) -> str:
    """Return the first task to be scheduled."""
    result = schedule(tasks)
    return result[0] if result else ""
