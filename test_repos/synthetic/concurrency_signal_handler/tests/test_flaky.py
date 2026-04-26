"""Tests where signal handler installation leaks.

test_default_sigint is FLAKY because a prior test installs a custom
SIGINT handler without restoring the original.
"""
import signal
import source

def test_install_handler():
    source.install_handler()
    assert source.is_shutdown_requested() is False
    # Bug: no source.restore_handler()

def test_default_sigint():
    """FLAKY — SIGINT handler may be the custom one from test_install_handler."""
    handler = signal.getsignal(signal.SIGINT)
    assert handler == signal.default_int_handler

def test_shutdown_flag_default():
    """FLAKY — _shutdown_requested may be True from prior test."""
    assert source.is_shutdown_requested() is False

def test_restore_works():
    source.install_handler()
    source.restore_handler()
    handler = signal.getsignal(signal.SIGINT)
    assert handler != source.install_handler
