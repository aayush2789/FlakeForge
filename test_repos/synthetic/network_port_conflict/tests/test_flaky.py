"""Tests that bind to the same port without cleanup.

test_second_server is FLAKY because test_first_server binds port 18234
and doesn't always stop before the next test tries to bind the same port.
"""
from source import SimpleServer

PORT = 18234

def test_first_server():
    srv = SimpleServer(PORT)
    srv.start()
    assert srv.is_running()
    # Bug: no srv.stop()

def test_second_server():
    """FLAKY — port may still be bound by test_first_server."""
    srv = SimpleServer(PORT)
    srv.start()
    assert srv.is_running()
    srv.stop()

def test_server_stop():
    srv = SimpleServer(PORT + 1)
    srv.start()
    srv.stop()
    assert not srv.is_running()
