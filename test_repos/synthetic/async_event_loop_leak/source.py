"""Async task manager that leaks tasks across event loop runs."""
import asyncio

_background_tasks = set()

async def schedule_task(coro):
    """Schedule a background task. Bug: tasks stored in module-level set
    survive across event loop runs."""
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

async def slow_operation(name: str, delay: float = 0.1) -> str:
    await asyncio.sleep(delay)
    return f"done:{name}"

async def run_and_collect(names: list) -> list:
    """Run operations and collect results."""
    tasks = [await schedule_task(slow_operation(n)) for n in names]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, str)]

def pending_count() -> int:
    return len(_background_tasks)

def clear_tasks():
    _background_tasks.clear()
