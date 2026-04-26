"""Worker pool with capacity-limited queue."""
import threading
import random

class WorkerPool:
    QUEUE_CAPACITY = 10

    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()

    def submit(self, job: dict) -> bool:
        """Submit a job. Bug: capacity check outside lock causes races."""
        if len(self._queue) >= self.QUEUE_CAPACITY:
            return False
        # Bug: another thread can fill queue between check and append
        with self._lock:
            self._queue.append(job)
            return True

    def process_all(self) -> int:
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    def pending(self) -> int:
        return len(self._queue)


def submit_batch(pool: WorkerPool, n: int, workers: int = 4) -> int:
    """Submit n jobs across workers. Returns number accepted."""
    accepted = [0]
    lock = threading.Lock()

    def _worker(count):
        for i in range(count):
            if pool.submit({"id": i}):
                with lock:
                    accepted[0] += 1

    per = n // workers
    threads = [threading.Thread(target=_worker, args=(per,)) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return accepted[0]
