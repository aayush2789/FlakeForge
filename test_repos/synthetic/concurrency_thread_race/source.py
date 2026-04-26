"""Shared results dict written by concurrent threads."""
import threading

class ResultCollector:
    def __init__(self):
        self.results = {}

    def record(self, key: str, value: int):
        """Record a result. Bug: no lock, concurrent writes can interleave."""
        current = self.results.get(key, 0)
        self.results[key] = current + value

    def get(self, key: str) -> int:
        return self.results.get(key, 0)

    def total(self) -> int:
        return sum(self.results.values())


def parallel_record(collector: ResultCollector, key: str, n: int, workers: int = 4):
    """Record value n times across workers."""
    def _worker(count):
        for _ in range(count):
            collector.record(key, 1)

    per_worker = n // workers
    threads = [threading.Thread(target=_worker, args=(per_worker,)) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
