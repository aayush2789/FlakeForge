"""Descriptor-based field system with a class-level registry."""

class TrackedField:
    """A descriptor that registers itself in a class-level registry."""
    _all_fields = {}  # Bug: class-level dict shared across ALL descriptors

    def __set_name__(self, owner, name):
        key = f"{owner.__name__}.{name}"
        TrackedField._all_fields[key] = self
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)

    @classmethod
    def get_all_fields(cls) -> dict:
        return dict(cls._all_fields)

    @classmethod
    def clear_registry(cls):
        cls._all_fields.clear()
