"""Async pipeline with partial cancellation issues."""
import asyncio
import random

async def stage_one(data: str) -> str:
    delay = random.uniform(0.01, 0.15)
    await asyncio.sleep(delay)
    return f"s1:{data}"

async def stage_two(data: str) -> str:
    delay = random.uniform(0.01, 0.15)
    await asyncio.sleep(delay)
    if random.random() < 0.3:
        raise ValueError(f"stage_two failed for {data}")
    return f"s2:{data}"

async def pipeline(items: list) -> list:
    """Run two-stage pipeline. Bug: if any stage_two fails with
    return_exceptions=False (the default for gather), all tasks
    are NOT cancelled — they just raise.
    """
    s1_tasks = [stage_one(item) for item in items]
    s1_results = await asyncio.gather(*s1_tasks)

    s2_tasks = [stage_two(r) for r in s1_results]
    # Bug: return_exceptions=False means first exception cancels gather
    s2_results = await asyncio.gather(*s2_tasks)
    return s2_results

async def safe_pipeline(items: list) -> list:
    """Same pipeline but with return_exceptions=True."""
    s1_tasks = [stage_one(item) for item in items]
    s1_results = await asyncio.gather(*s1_tasks)
    s2_tasks = [stage_two(r) for r in s1_results]
    s2_results = await asyncio.gather(*s2_tasks, return_exceptions=True)
    return [r for r in s2_results if isinstance(r, str)]
