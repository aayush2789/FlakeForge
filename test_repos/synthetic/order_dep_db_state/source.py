"""In-memory database with basic CRUD operations."""

class InMemoryDB:
    def __init__(self):
        self._tables = {}

    def create_table(self, name: str):
        self._tables[name] = []

    def insert(self, table: str, row: dict):
        if table not in self._tables:
            raise KeyError(f"Table '{table}' does not exist")
        self._tables[table].append(row)

    def select(self, table: str) -> list:
        if table not in self._tables:
            raise KeyError(f"Table '{table}' does not exist")
        return list(self._tables[table])

    def count(self, table: str) -> int:
        return len(self.select(table))

    def drop_table(self, name: str):
        self._tables.pop(name, None)

# Module-level singleton — shared across tests
db = InMemoryDB()
