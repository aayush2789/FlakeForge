"""Flaky tests that usually require multiple repair steps."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from source import build_payload, fetch_profile


def test_profile_fetch_should_be_stable():
    """A single user-facing flow with two independent flaky gates."""
    result = asyncio.run(fetch_profile())
    assert result["success"], f"Expected stable profile fetch, got: {result}"
    assert result["payload"]["request_id"] == "stable-request"
    assert result["payload"]["data"] == "completed"


def test_profile_fetch_with_explicit_timeout_still_exposes_payload_flake():
    """Ground-truth helper: bypasses the timeout but leaves payload flakiness."""
    result = asyncio.run(fetch_profile(timeout_override=0.5))
    assert result["success"], f"Payload should be deterministic too, got: {result}"


def test_payload_builder_should_be_deterministic():
    """Directly documents the second issue once the timeout is gone."""
    payloads = [build_payload() for _ in range(8)]
    assert {payload["request_id"] for payload in payloads} == {"stable-request"}
