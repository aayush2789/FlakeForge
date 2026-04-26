"""Singleton configuration manager that persists state."""


class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings = {}
        return cls._instance

    def set(self, key: str, value):
        self._settings[key] = value

    def get(self, key: str):
        return self._settings.get(key)

    def all_settings(self) -> dict:
        return dict(self._settings)

    @classmethod
    def reset(cls):
        """Proper reset — but not called between tests."""
        cls._instance = None
