"""Tests where importlib.reload corrupts module state.

test_version_after_reload is FLAKY because reloading the module resets
_initialized but other tests may have cached references to old module state.
"""
import importlib
import source

def test_custom_init():
    source.initialize({"mode": "testing", "verbose": True})
    config = source.get_config()
    assert config["mode"] == "testing"

def test_reload_module():
    """Reloads source, resetting module state."""
    importlib.reload(source)
    # After reload, _initialized is False again

def test_version_after_reload():
    """FLAKY — module reference may be stale after reload."""
    assert source.get_version() == "1.0.0"

def test_config_default():
    """FLAKY — depends on init state after reload."""
    config = source.get_config()
    assert config.get("mode") == "production"
