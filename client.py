"""V3 FlakeForge Client — OpenEnv client for the unified agent architecture.

Changes from V2:
- No judge calls
- No hypothesis gating
- Unified observe → generate → step loop
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

try:
    from models import FlakeForgeAction, FlakeForgeObservation
    from agent.unified_agent import (
        UnifiedFlakeForgeAgent,
        extract_think,
        extract_patch,
        extract_category_from_think,
        extract_confidence_from_think,
    )
except ImportError:
    from .models import FlakeForgeAction, FlakeForgeObservation
    from .agent.unified_agent import (
        UnifiedFlakeForgeAgent,
        extract_think,
        extract_patch,
        extract_category_from_think,
        extract_confidence_from_think,
    )

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from .utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__)


class FlakeForgeClient:
    """V3 client for communicating with the FlakeForge environment.

    Wraps the unified agent and environment interaction into a
    simple interface for both inference and training.
    """

    def __init__(
        self,
        agent: Optional[UnifiedFlakeForgeAgent] = None,
        env_url: Optional[str] = None,
    ) -> None:
        self.agent = agent
        self.env_url = env_url or os.environ.get("ENV_BASE_URL", "http://localhost:8080")

    def generate_action(self, observation: FlakeForgeObservation) -> FlakeForgeAction:
        """Generate a unified action from an observation."""
        if self.agent is None:
            raise RuntimeError("No agent configured. Pass agent to constructor.")
        return self.agent.generate(observation)

    def parse_raw_response(self, raw_response: str) -> FlakeForgeAction:
        """Parse a raw model response into a FlakeForgeAction."""
        think = extract_think(raw_response)
        patch = extract_patch(raw_response)

        return FlakeForgeAction(
            raw_response=raw_response,
            think_text=think,
            patch_text=patch,
            predicted_category=extract_category_from_think(think),
            predicted_confidence=extract_confidence_from_think(think),
        )

    async def run_episode_remote(
        self,
        test_identifier: str,
        repo_path: str,
        max_steps: int = 8,
    ) -> Dict[str, Any]:
        """Run an episode against a remote FlakeForge environment server."""
        try:
            import httpx
        except ImportError:
            raise ImportError("Remote client requires: pip install httpx")

        async with httpx.AsyncClient(
            base_url=self.env_url, timeout=120.0
        ) as client:
            # Reset
            reset_response = await client.post(
                "/reset",
                json={"test_identifier": test_identifier, "repo_path": repo_path},
            )
            reset_data = reset_response.json()
            observation = FlakeForgeObservation(**reset_data["observation"])

            trajectory = []
            total_reward = 0.0

            for step in range(max_steps):
                if reset_data.get("done", False) and step > 0:
                    break

                # Generate action
                action = self.generate_action(observation)

                # Send to environment
                step_response = await client.post(
                    "/step",
                    json={
                        "raw_response": action.raw_response,
                        "think_text": action.think_text,
                        "patch_text": action.patch_text,
                        "predicted_category": action.predicted_category,
                        "predicted_confidence": action.predicted_confidence,
                    },
                )
                step_data = step_response.json()

                observation = FlakeForgeObservation(**step_data["observation"])
                reward = step_data.get("reward", 0.0)
                total_reward += reward

                trajectory.append({
                    "step": step + 1,
                    "category": action.predicted_category,
                    "confidence": action.predicted_confidence,
                    "reward": reward,
                    "pass_rate": step_data.get("state", {}).get("current_pass_rate", 0.0),
                    "done": step_data.get("done", False),
                })

                if step_data.get("done", False):
                    break

            return {
                "trajectory": trajectory,
                "total_reward": total_reward,
                "steps": len(trajectory),
                "final_pass_rate": observation.current_pass_rate,
            }
