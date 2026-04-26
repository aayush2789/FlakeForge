"""Service layer with nested dependencies."""
import os
import json

def get_config_path() -> str:
    return os.environ.get("CONFIG_PATH", "/etc/app/config.json")

def load_config() -> dict:
    path = get_config_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"version": "1.0", "debug": False}

def get_version() -> str:
    config = load_config()
    return config.get("version", "unknown")

def is_debug() -> bool:
    config = load_config()
    return config.get("debug", False)
