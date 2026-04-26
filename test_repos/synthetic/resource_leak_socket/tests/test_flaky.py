"""Tests that create EchoServer without calling stop().

test_server_binds_free_port is FLAKY because leaked sockets from prior
tests may still hold ports or cause resource exhaustion.
"""
import socket
from source import EchoServer


def test_server_binds_free_port():
    """FLAKY — resource exhaustion from leaked sockets can cause bind failures."""
    server = EchoServer()
    assert server.port > 0
    # Bug: no server.stop()


def test_echo_works():
    server = EchoServer()
    server.start()
    with socket.socket() as s:
        s.connect(("127.0.0.1", server.port))
        s.sendall(b"hello")
        assert s.recv(1024) == b"hello"
    # Bug: no server.stop()


def test_server_starts():
    server = EchoServer()
    server.start()
    assert server._thread is not None
    # Bug: no server.stop()
