"""Process manager that spawns subprocesses."""
import subprocess
import sys

_processes = []

def run_task(script: str) -> subprocess.Popen:
    """Run a Python script in a subprocess.
    Bug: process is added to list but never waited on or cleaned up.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _processes.append(proc)
    return proc

def active_count() -> int:
    """Count processes that haven't been waited on."""
    return sum(1 for p in _processes if p.poll() is None)

def collect_output(proc: subprocess.Popen) -> str:
    """Get output. Bug: doesn't call communicate(), may hang on large output."""
    stdout = proc.stdout.read()
    return stdout.decode() if stdout else ""

def cleanup():
    """Proper cleanup."""
    for p in _processes:
        p.terminate()
        p.wait()
    _processes.clear()
