"""Tests where metaclass registry accumulates across tests.

test_registry_only_has_declared is FLAKY because classes defined in
other tests get auto-registered by the metaclass.
"""
from source import Plugin, PluginMeta

def test_define_and_find():
    class Greeter(Plugin):
        def execute(self):
            return "hello"
    assert "Greeter" in PluginMeta.get_plugins()

def test_define_another():
    class Logger(Plugin):
        def execute(self):
            return "log"
    assert "Logger" in PluginMeta.get_plugins()

def test_registry_only_has_declared():
    """FLAKY — registry may contain classes from other tests."""
    class Worker(Plugin):
        def execute(self):
            return "work"
    plugins = PluginMeta.get_plugins()
    assert list(plugins.keys()) == ["Worker"]

def test_plugin_executes():
    class Runner(Plugin):
        def execute(self):
            return "run"
    assert Runner().execute() == "run"
