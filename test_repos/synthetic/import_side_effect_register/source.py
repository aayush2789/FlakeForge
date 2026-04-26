"""Plugin system with import-time registration side effect."""

_plugins = []


def _auto_register():
    """Called at import time — appends a default plugin."""
    _plugins.append("default_plugin")


_auto_register()  # Bug: runs every time module is imported/reloaded


def get_plugins() -> list:
    return list(_plugins)


def register(name: str):
    _plugins.append(name)


def clear_plugins():
    _plugins.clear()
