"""Connection pool that exhausts under concurrent use."""
import socket
import threading
import time

class ConnectionPool:
    def __init__(self, host: str, port: int, max_size: int = 3):
        self.host = host
        self.port = port
        self.max_size = max_size
        self._pool = []
        self._in_use = []
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 0.5) -> socket.socket:
        """Acquire a connection. Bug: no blocking wait — raises immediately
        if pool is empty and max_size reached."""
        with self._lock:
            if self._pool:
                conn = self._pool.pop()
                self._in_use.append(conn)
                return conn
            if len(self._in_use) < self.max_size:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(timeout)
                try:
                    conn.connect((self.host, self.port))
                except (ConnectionRefusedError, OSError):
                    pass
                self._in_use.append(conn)
                return conn
            raise RuntimeError("Connection pool exhausted")

    def release(self, conn: socket.socket):
        with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                self._pool.append(conn)

    def close_all(self):
        with self._lock:
            for c in self._pool + self._in_use:
                try:
                    c.close()
                except Exception:
                    pass
            self._pool.clear()
            self._in_use.clear()

    def available(self) -> int:
        return self.max_size - len(self._in_use)
