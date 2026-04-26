"""Bank transfer system with potential deadlock."""
import threading

class Account:
    def __init__(self, name: str, balance: float):
        self.name = name
        self.balance = balance
        self.lock = threading.Lock()

    def deposit(self, amount: float):
        with self.lock:
            self.balance += amount

    def withdraw(self, amount: float) -> bool:
        with self.lock:
            if self.balance >= amount:
                self.balance -= amount
                return True
            return False


def transfer(source: Account, dest: Account, amount: float) -> bool:
    """Transfer money between accounts.
    Bug: acquires locks in arbitrary order — can deadlock when two
    concurrent transfers go in opposite directions.
    """
    with source.lock:
        if source.balance < amount:
            return False
        with dest.lock:
            source.balance -= amount
            dest.balance += amount
            return True


def parallel_transfers(a: Account, b: Account, amount: float, count: int) -> int:
    """Run transfers in both directions concurrently."""
    success = [0]
    lock = threading.Lock()

    def _ab(n):
        for _ in range(n):
            if transfer(a, b, amount):
                with lock:
                    success[0] += 1

    def _ba(n):
        for _ in range(n):
            if transfer(b, a, amount):
                with lock:
                    success[0] += 1

    half = count // 2
    t1 = threading.Thread(target=_ab, args=(half,))
    t2 = threading.Thread(target=_ba, args=(half,))
    t1.start()
    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)

    if t1.is_alive() or t2.is_alive():
        raise RuntimeError("Deadlock detected — threads did not complete")
    return success[0]
