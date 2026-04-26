"""Thread-safe counter — except the lock is missing."""
import threading

class Counter:
    def __init__(self):
        self._value = 0
        # Bug: no lock protecting _value

    def increment(self):
        """Increment counter. Not thread-safe without a lock."""
        current = self._value
        self._value = current + 1

    def get(self) -> int:
        return self._value

    def reset(self):
        self._value = 0


def parallel_increment(counter: Counter, n: int, workers: int = 4):
    """Increment counter n times across multiple threads."""
    def _worker(count):
        for _ in range(count):
            counter.increment()

    per_worker = n // workers
    threads = [threading.Thread(target=_worker, args=(per_worker,)) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
