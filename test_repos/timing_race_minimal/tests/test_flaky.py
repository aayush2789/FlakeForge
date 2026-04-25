"""Flaky test cases for timing race detection."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
# sys.path.insert(0, str(Path(__file__).parent.parent))

from source import fetch_data_with_race, sync_fetch_with_cache


import asyncio

@pytest.mark.asyncio
async def test_fetch_should_complete():
    """
    This test is flaky because fetch_data_with_race has an 80% chance
    of using a 0.05s timeout, which is shorter than the 0.15s operation time.
    
    Expected: Should pass every time.
    Actual: Fails ~30% of the time due to timeout race.
    """
    result = await fetch_data_with_race(timeout_override=1.0)
    assert result["success"], f"Expected success, got: {result}"
    assert result["data"] == "completed"


def test_fetch_with_explicit_timeout():
    """
    This test should be stable since we override the timeout.
    This is the "ground truth" version.
    """
    asyncio.run(asyncio.sleep(0.2))  # Ensure no race condition with previous tests
    result = asyncio.run(fetch_data_with_race(timeout_override=1.0))
    assert result["success"], f"Expected success with explicit timeout, got: {result}"
    assert result["data"] == "completed"


def test_sync_cache_state():
    """
    This test is mildly flaky because the cache state is not reset
    between test runs, and the function behavior depends on stale cache.
    
    Expected: Should always succeed.
    Actual: May fail if cache is in an unexpected state.
    """
    result = sync_fetch_with_cache()
    assert isinstance(result, dict)
    assert "from_cache" in result


# Simple non-async versions for testing without pytest-asyncio
def test_flaky_simple():
    """Simple synchronous flaky test."""
    # Simulate a flaky condition: 70% pass, 30% fail
    import random
    
    # This should pass but fails 30% of the time randomly
    assert random.random() > 0.3, "Random flaky failure"


async def async_test_wrapper():
    """Wrapper to run async test without pytest."""
    result = await fetch_data_with_race()
    assert result["success"], f"Expected success, got: {result}"
    assert result["data"] == "completed"


if __name__ == "__main__":
    # Manual test runner
    print("Running simple flaky test...")
    try:
        test_flaky_simple()
        print("✓ test_flaky_simple passed")
    except AssertionError as e:
        print(f"✗ test_flaky_simple failed: {e}")
    
    print("Running async test...")
    try:
        asyncio.run(async_test_wrapper())
        print("✓ async_test_wrapper passed")
    except Exception as e:
        print(f"✗ async_test_wrapper failed: {e}")
