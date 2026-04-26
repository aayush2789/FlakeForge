"""Connection pool tracker."""

class ConnectionPool:
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._connections = []

    def acquire(self) -> str:
        if len(self._connections) >= self.max_size:
            raise RuntimeError("Pool exhausted")
        conn_id = f"conn_{len(self._connections)}"
        self._connections.append(conn_id)
        return conn_id

    def release(self, conn_id: str):
        if conn_id in self._connections:
            self._connections.remove(conn_id)

    def active_count(self) -> int:
        return len(self._connections)
