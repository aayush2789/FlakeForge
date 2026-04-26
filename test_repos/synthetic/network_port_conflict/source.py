"""HTTP-like server that binds to a specific port."""
import socket
import threading

class SimpleServer:
    def __init__(self, port: int):
        self.port = port
        self.sock = None
        self._thread = None
        self._running = False

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.listen(1)
        self._running = True
        def _serve():
            while self._running:
                try:
                    self.sock.settimeout(0.1)
                    conn, _ = self.sock.accept()
                    conn.sendall(b"OK")
                    conn.close()
                except socket.timeout:
                    pass
                except OSError:
                    break
        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self.sock:
            self.sock.close()
        if self._thread:
            self._thread.join(timeout=1)

    def is_running(self) -> bool:
        return self._running
