"""Reset the multi-step demo repo back to its flaky starting state."""

from pathlib import Path


SOURCE = '''"""Multi-step flaky target for FlakeForge live progress demos."""

from __future__ import annotations

import asyncio
import random
from typing import Any


async def fetch_profile(timeout_override: float | None = None) -> dict[str, Any]:
    """Fetch a profile through two independent flaky gates.

    Gate 1 is an async timeout race. Once fixed, gate 2 remains: the payload
    builder still returns nondeterministic data.
    """
    if timeout_override is not None:
        timeout = timeout_override
    else:
        timeout = 0.03 if random.random() < 0.75 else 0.5

    try:
        payload = await asyncio.wait_for(_network_payload(), timeout=timeout)
    except asyncio.TimeoutError:
        return {"success": False, "error": "timeout"}

    if payload["request_id"] != "stable-request":
        return {
            "success": False,
            "error": "unstable_request_id",
            "payload": payload,
        }

    return {"success": True, "payload": payload}


async def _network_payload() -> dict[str, Any]:
    """Pretend the network layer needs stable startup plus processing time."""
    await asyncio.sleep(0.06)
    await asyncio.sleep(0.04)
    return build_payload()


def build_payload() -> dict[str, Any]:
    """Build the response payload with nondeterministic state."""
    request_id = random.choice(["stable-request", "stale-request"])
    return {
        "request_id": request_id,
        "data": "completed",
    }
'''


def main() -> None:
    path = Path(__file__).with_name("source.py")
    path.write_text(SOURCE, encoding="utf-8")
    print(f"Reset {path}")


if __name__ == "__main__":
    main()
