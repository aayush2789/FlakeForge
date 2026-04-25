"""Simple async function with timing race condition."""

import asyncio
import random
from typing import Any


async def fetch_data_with_race(timeout_override: float | None = None) -> dict[str, Any]:
    """
    Simulates an async operation that has a timing race condition.
    
    80% of the time it uses a very tight timeout (0.05s), which can cause
    the operation to timeout before it even starts.
    
    Args:
        timeout_override: Override the computed timeout for testing.
        
    Returns:
        A dictionary with the result.
    """
    if timeout_override is not None:
        timeout = timeout_override
    else:
        timeout = 0.5
    
    # Simulate an async operation that takes some time to initialize
    try:
        result = await asyncio.wait_for(_slow_async_op(), timeout=timeout)
        return {"success": True, "data": result}
    except asyncio.TimeoutError:
        return {"success": False, "error": "timeout"}


async def _slow_async_op() -> str:
    """Simulate a slow async operation that initializes over ~0.1s."""
    await asyncio.sleep(0.1)  # Initialization delay
    await asyncio.sleep(0.05)  # Processing
    return "completed"


def sync_fetch_with_cache() -> dict[str, Any]:
    """
    Simulates a sync operation with a shared cache that can cause flakiness.
    """
    import time
    
    # Simulate reading from a potentially stale cache
    if random.random() < 0.6:
        # Cache is stale this run
        time.sleep(0.01)
    
    return {"from_cache": random.random() < 0.5}
