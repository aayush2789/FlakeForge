import asyncio
import os
import json
from models import FlakeForgeAction, Hypothesis
from server.FlakeForge_environment import FlakeForgeEnvironment
import logging

logging.basicConfig(level=logging.INFO)

async def test_env():
    env = FlakeForgeEnvironment(
        repo_path="seed_repos/async_lock_deadlock",
        test_id="tests/test_flaky.py::test_flaky_case",
        max_steps=5
    )
    obs = env.reset()
    print("Baseline pass rate:", obs.baseline_pass_rate)

    # Let's see what happens with action_type="add_sleep"
    action = FlakeForgeAction(
        action_type="add_sleep",
        parameters={"delay_ms": 100},
        hypothesis={
            "root_cause_category": "race",
            "confidence": 0.9,
            "evidence": ["counter.py:10"],
            "suggested_action": "add_sleep",
            "reasoning_steps": []
        }
    )
    
    try:
        new_obs = env.step(action)
        print("Done step! Metadata error:", new_obs.metadata.get("error"))
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_env())
