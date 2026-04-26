"""Background worker that runs in a daemon thread."""
import threading
import time

class BackgroundWorker:
    def __init__(self):
        self._running = False
        self._thread = None
        self._results = []

    def start(self, task_fn, interval: float = 0.05):
        """Start background task. Bug: thread is never joined/stopped."""
        self._running = True
        def _loop():
            while self._running:
                result = task_fn()
                self._results.append(result)
                time.sleep(interval)
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def get_results(self) -> list:
        return list(self._results)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
