"""Path utility that hardcodes forward-slash separator."""
import os


def build_path(*parts: str) -> str:
    """Join path parts. Bug: hardcodes '/' instead of os.sep."""
    return "/".join(parts)


def config_path(app_name: str) -> str:
    return build_path("etc", app_name, "config.ini")
