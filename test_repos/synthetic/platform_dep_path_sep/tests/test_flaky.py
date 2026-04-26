"""Tests that compare paths with os.path.join.

test_config_path_matches_os is FLAKY on Windows because build_path uses
'/' while os.path.join uses '\\'.
"""
import os
from source import build_path, config_path


def test_config_path_matches_os():
    """FLAKY on Windows — hardcoded '/' vs os.sep backslash."""
    expected = os.path.join("etc", "myapp", "config.ini")
    assert config_path("myapp") == expected


def test_build_path_has_parts():
    result = build_path("a", "b", "c")
    assert "a" in result and "b" in result and "c" in result


def test_config_path_ends_with_ini():
    assert config_path("app").endswith("config.ini")
