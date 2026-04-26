"""Plugin system using a metaclass registry."""

class PluginMeta(type):
    """Metaclass that auto-registers all subclasses."""
    _registry = {}

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if bases:  # skip the base Plugin class itself
            PluginMeta._registry[name] = cls

    @classmethod
    def get_plugins(mcs) -> dict:
        return dict(mcs._registry)

    @classmethod
    def clear(mcs):
        mcs._registry.clear()


class Plugin(metaclass=PluginMeta):
    """Base plugin class."""
    def execute(self) -> str:
        return "base"
