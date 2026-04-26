"""Configuration loaded once at import time from os.environ."""
import os

# Bug: reads env var at import time — frozen for entire process
APP_MODE = os.environ.get("APP_MODE", "production")

def get_mode() -> str:
    return APP_MODE

def is_debug() -> bool:
    return APP_MODE == "debug"

def is_production() -> bool:
    return APP_MODE == "production"
