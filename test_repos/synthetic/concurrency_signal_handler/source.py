"""Service with custom signal handling."""
import signal
import threading

_shutdown_requested = False
_original_handler = None

def install_handler():
    """Install custom SIGINT handler. Bug: replaces original handler
    and doesn't restore it."""
    global _original_handler, _shutdown_requested
    _shutdown_requested = False
    _original_handler = signal.getsignal(signal.SIGINT)
    def _handler(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
    signal.signal(signal.SIGINT, _handler)

def is_shutdown_requested() -> bool:
    return _shutdown_requested

def restore_handler():
    """Restore original handler."""
    global _original_handler
    if _original_handler is not None:
        signal.signal(signal.SIGINT, _original_handler)
        _original_handler = None

def reset():
    global _shutdown_requested
    _shutdown_requested = False
    restore_handler()
