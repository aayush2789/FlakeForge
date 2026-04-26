"""Tests where the import side-effect causes accumulation.

test_only_default_plugin is FLAKY because _plugins accumulates
'default_plugin' entries across module reloads and test runs.
"""
import source


def test_only_default_plugin():
    """FLAKY — may have multiple 'default_plugin' entries from prior imports."""
    plugins = source.get_plugins()
    assert plugins == ["default_plugin"]


def test_register_adds_plugin():
    source.register("my_plugin")
    assert "my_plugin" in source.get_plugins()


def test_plugins_not_empty():
    assert len(source.get_plugins()) > 0
