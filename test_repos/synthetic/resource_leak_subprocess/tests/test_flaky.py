"""Tests that leak subprocesses.

test_no_active_processes is FLAKY because prior tests spawn processes
without cleaning them up.
"""
import source

def test_run_simple():
    proc = source.run_task("print('hello')")
    output = source.collect_output(proc)
    assert "hello" in output

def test_run_slow():
    proc = source.run_task("import time; time.sleep(0.5); print('done')")
    # Bug: doesn't wait for completion

def test_no_active_processes():
    """FLAKY — processes from prior tests may still be running."""
    assert source.active_count() == 0

def test_run_returns_proc():
    proc = source.run_task("pass")
    assert proc is not None
