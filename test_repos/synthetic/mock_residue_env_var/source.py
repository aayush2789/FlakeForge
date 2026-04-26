"""App configuration that reads from environment variables."""
import os

def get_debug_mode() -> bool:
    return os.environ.get("APP_DEBUG", "false").lower() == "true"

def get_app_name() -> str:
    return os.environ.get("APP_NAME", "MyApp")

def get_log_level() -> str:
    return os.environ.get("LOG_LEVEL", "INFO")
