"""Tests for connection pool.

test_concurrent_acquire is FLAKY because pool exhaustion depends on
timing of acquire/release across threads.
"""
import threading
import socket
from source import ConnectionPool

def _make_pool():
    """Create a pool pointing to a local server."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    port = srv.getsockname()[1]

    def _accept():
        while True:
            try:
                conn, _ = srv.accept()
                conn.close()
            except OSError:
                break

    t = threading.Thread(target=_accept, daemon=True)
    t.start()
    return ConnectionPool("127.0.0.1", port, max_size=2), srv

def test_acquire_one():
    pool, srv = _make_pool()
    conn = pool.acquire()
    assert conn is not None
    pool.close_all()
    srv.close()

def test_concurrent_acquire():
    """FLAKY — 4 threads competing for max_size=2 pool."""
    pool, srv = _make_pool()
    errors = []

    def _worker():
        try:
            c = pool.acquire(timeout=0.2)
            import time; time.sleep(0.1)
            pool.release(c)
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pool.close_all()
    srv.close()
    assert len(errors) == 0

def test_release_returns_to_pool():
    pool, srv = _make_pool()
    c = pool.acquire()
    pool.release(c)
    assert pool.available() == 2
    pool.close_all()
    srv.close()
