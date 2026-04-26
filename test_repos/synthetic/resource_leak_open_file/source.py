"""File logger that leaks file handles."""
import os
import tempfile

class FileLogger:
    def __init__(self, path=None):
        if path is None:
            fd, path = tempfile.mkstemp(prefix="log_", suffix=".log")
            os.close(fd)
        self.path = path
        self._handle = open(path, "a")  # Bug: never closed

    def log(self, msg: str):
        self._handle.write(msg + "\n")
        self._handle.flush()

    def read_all(self) -> str:
        with open(self.path) as f:
            return f.read()

    def close(self):
        """Proper close — but not always called."""
        self._handle.close()
