"""Database connection manager with context management."""
import threading

class Connection:
    _open_connections = 0
    _lock = threading.Lock()

    def __init__(self, dsn: str = "test://localhost"):
        self.dsn = dsn
        self.is_open = False
        self._transactions = []

    def open(self):
        self.is_open = True
        with Connection._lock:
            Connection._open_connections += 1

    def close(self):
        if self.is_open:
            self.is_open = False
            with Connection._lock:
                Connection._open_connections -= 1

    def begin_transaction(self) -> str:
        txn_id = f"txn_{len(self._transactions)}"
        self._transactions.append(txn_id)
        return txn_id

    def commit(self):
        if self._transactions:
            self._transactions.pop()

    @classmethod
    def open_count(cls) -> int:
        return cls._open_connections
