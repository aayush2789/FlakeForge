"""Simple TCP echo server that leaks the socket."""
import socket
import threading


class EchoServer:
    def __init__(self, port=0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self._thread = None

    def start(self):
        def _serve():
            try:
                conn, _ = self.sock.accept()
                data = conn.recv(1024)
                conn.sendall(data)
                conn.close()
            except OSError:
                pass
        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()

    def stop(self):
        """Proper cleanup — but tests don't always call it."""
        self.sock.close()
        if self._thread:
            self._thread.join(timeout=1)
