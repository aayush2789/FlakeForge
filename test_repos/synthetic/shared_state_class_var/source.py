"""Registry that uses a class-level list — items accumulate across instances."""

class Registry:
    _items = []  # Bug: class-level mutable default shared across all instances

    def add(self, item):
        self._items.append(item)

    def get_items(self):
        return list(self._items)

    def count(self):
        return len(self._items)
