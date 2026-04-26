"""Configuration module that uses module-level state."""

VERSION = "1.0.0"
_initialized = False
_config = {}

def initialize(settings: dict = None):
    global _initialized, _config
    if _initialized:
        return
    _config = settings or {"mode": "production"}
    _initialized = True

def get_config() -> dict:
    if not _initialized:
        initialize()
    return dict(_config)

def get_version() -> str:
    return VERSION

def reset():
    global _initialized, _config
    _initialized = False
    _config = {}
