"""Shopping cart implementation."""

class Cart:
    def __init__(self):
        self.items = []

    def add(self, item: str):
        self.items.append(item)

    def count(self) -> int:
        return len(self.items)

    def clear(self):
        self.items.clear()

    def contents(self) -> list:
        return list(self.items)
