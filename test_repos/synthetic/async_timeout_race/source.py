"""Async task runner with tight timeouts."""
import asyncio
import random

async def slow_fetch(url: str) -> str:
    """Simulate a network fetch with variable latency."""
    delay = random.uniform(0.05, 0.25)
    await asyncio.sleep(delay)
    return f"data from {url}"

async def fetch_with_timeout(url: str, timeout: float = 0.1) -> str:
    """Fetch with timeout. Bug: timeout is too tight for the variable latency."""
    return await asyncio.wait_for(slow_fetch(url), timeout=timeout)

async def fetch_all(urls: list) -> list:
    """Fetch all URLs concurrently with timeouts."""
    tasks = [fetch_with_timeout(u) for u in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
