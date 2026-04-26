"""Consistent hashing ring implementation."""

class HashRing:
    def __init__(self, nodes: list, replicas: int = 3):
        self.ring = {}
        self.sorted_keys = []
        for node in nodes:
            for i in range(replicas):
                key = hash(f"{node}:{i}")
                self.ring[key] = node
        self.sorted_keys = sorted(self.ring.keys())

    def get_node(self, item: str) -> str:
        """Get the node responsible for an item.
        Bug: hash() is randomized per process (PYTHONHASHSEED), so
        node assignment changes between runs."""
        h = hash(item)
        for key in self.sorted_keys:
            if h <= key:
                return self.ring[key]
        return self.ring[self.sorted_keys[0]]

    def get_nodes(self) -> list:
        return list(set(self.ring.values()))
