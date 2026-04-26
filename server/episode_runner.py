"""Episode runner — orchestrates agent + environment loop for API and WebSocket use."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from .api_models import StepResult, RunEpisodeResponse

logger = logging.getLogger(__name__)


def _get_model_backend():
    """Build a model backend from environment configuration."""
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model_name = os.environ.get("MODEL_NAME", "nvidia/llama-3.1-nemotron-nano-8b-v1")
    temperature = float(os.environ.get("TEMPERATURE", "0.1"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "900"))
    timeout = float(os.environ.get("REQUEST_TIMEOUT_S", "45"))

    if not api_key:
        logger.warning("No API key configured; episode runner will use mock backend")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; using mock backend")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    class NvidiaBackend:
        def generate(self, prompt: str, *, system_prompt: str) -> str:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                logger.error("Model backend call failed: %s", exc)
                return ""

    return NvidiaBackend()


class EpisodeRunner:
    """Runs a complete episode: reset env, loop agent steps, collect trajectory."""

    def __init__(self) -> None:
        self._active_episodes: Dict[str, Dict[str, Any]] = {}
        self._completed_episodes: Dict[str, RunEpisodeResponse] = {}

    @property
    def active_episode_ids(self) -> List[str]:
        return list(self._active_episodes.keys())

    def get_status(self, episode_id: str) -> Optional[Dict[str, Any]]:
        if episode_id in self._active_episodes:
            return self._active_episodes[episode_id]
        if episode_id in self._completed_episodes:
            resp = self._completed_episodes[episode_id]
            return {
                "episode_id": episode_id,
                "status": resp.status,
                "current_step": len(resp.steps),
                "max_steps": resp.steps[-1].step if resp.steps else 0,
                "pass_rate": resp.final_pass_rate,
                "total_reward": resp.total_reward,
                "done": True,
            }
        return None

    def get_result(self, episode_id: str) -> Optional[RunEpisodeResponse]:
        return self._completed_episodes.get(episode_id)

    def run_episode_sync(
        self,
        repo_path: str = "",
        test_identifier: str = "",
        max_steps: int = 8,
        num_runs: int = 10,
        on_step: Optional[Callable[[StepResult], None]] = None,
    ) -> RunEpisodeResponse:
        """Run a full episode synchronously. Returns completed response."""
        from .FlakeForge_environment import FlakeForgeEnvironment
        from .docker_runner import DockerTestRunner

        project_root = Path(__file__).parents[1]
        default_repo = os.environ.get(
            "FF_REPO_PATH",
            str(project_root / "test_repos" / "timing_race_minimal"),
        )
        repo_path = repo_path or default_repo
        test_identifier = test_identifier or os.environ.get(
            "FF_TEST_ID", "tests/test_flaky.py::test_fetch_should_complete"
        )

        episode_id = str(uuid.uuid4())[:8]
        self._active_episodes[episode_id] = {
            "episode_id": episode_id,
            "status": "initializing",
            "current_step": 0,
            "max_steps": max_steps,
            "pass_rate": 0.0,
            "total_reward": 0.0,
            "done": False,
        }

        try:
            runner = DockerTestRunner(repo_path)
            env = FlakeForgeEnvironment(
                repo_path=repo_path,
                test_identifier=test_identifier,
                max_steps=max_steps,
                num_runs=num_runs,
                runner=runner,
            )

            observation = env.reset(episode_id=episode_id)
            baseline_pass_rate = observation.baseline_pass_rate

            self._active_episodes[episode_id].update({
                "status": "running",
                "pass_rate": baseline_pass_rate,
            })

            backend = _get_model_backend()
            agent = None
            if backend is not None:
                try:
                    import sys
                    proj = str(Path(__file__).parents[1])
                    if proj not in sys.path:
                        sys.path.insert(0, proj)
                    from agent.unified_agent import UnifiedFlakeForgeAgent
                    agent = UnifiedFlakeForgeAgent(backend)
                except ImportError:
                    logger.warning("Could not import UnifiedFlakeForgeAgent")

            steps: List[StepResult] = []
            total_reward = 0.0
            causal_graph = observation.causal_graph

            for step_num in range(max_steps):
                if observation.done:
                    break

                pass_rate_before = observation.current_pass_rate

                if agent is not None:
                    action = agent.generate(observation)
                else:
                    from models import FlakeForgeAction
                    action = FlakeForgeAction(
                        raw_response="",
                        think_text='{"claims":[{"category":"unknown","entity":"","location":"","polarity":"present","reason":"no model backend configured"}],"confidence":0.1}',
                        patch_text="",
                        predicted_category="unknown",
                        predicted_confidence=0.1,
                    )

                observation = env.step(action)
                reward = observation.reward
                total_reward += reward

                step_result = StepResult(
                    step=step_num + 1,
                    action=action.action_type,
                    category=action.predicted_category,
                    confidence=action.predicted_confidence,
                    reward=round(reward, 4),
                    reward_breakdown=observation.reward_breakdown,
                    pass_rate_before=round(pass_rate_before, 4),
                    pass_rate_after=round(observation.current_pass_rate, 4),
                    patch_applied=bool(observation.patch_result.get("success")),
                    patch_files=observation.patch_result.get("files_modified", []),
                    think_summary=action.think_text[:200] if action.think_text else "",
                    done=observation.done,
                    done_reason=observation.done_reason,
                )
                steps.append(step_result)

                self._active_episodes[episode_id].update({
                    "current_step": step_num + 1,
                    "pass_rate": observation.current_pass_rate,
                    "total_reward": round(total_reward, 4),
                })

                if on_step:
                    on_step(step_result)

                if observation.done:
                    break

            result = RunEpisodeResponse(
                episode_id=episode_id,
                status="completed",
                steps=steps,
                total_reward=round(total_reward, 4),
                final_pass_rate=observation.current_pass_rate,
                baseline_pass_rate=baseline_pass_rate,
                done_reason=observation.done_reason or "max_steps_reached",
                causal_graph=causal_graph,
            )

        except Exception as exc:
            logger.error("Episode %s failed: %s", episode_id, exc, exc_info=True)
            result = RunEpisodeResponse(
                episode_id=episode_id,
                status="error",
                steps=[],
                total_reward=0.0,
                final_pass_rate=0.0,
                baseline_pass_rate=0.0,
                done_reason=f"error: {exc}",
            )

        self._completed_episodes[episode_id] = result
        self._active_episodes.pop(episode_id, None)
        return result

    async def run_episode_async(
        self,
        repo_path: str = "",
        test_identifier: str = "",
        max_steps: int = 8,
        num_runs: int = 10,
    ) -> RunEpisodeResponse:
        """Run episode in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.run_episode_sync(
                repo_path=repo_path,
                test_identifier=test_identifier,
                max_steps=max_steps,
                num_runs=num_runs,
            ),
        )

    async def stream_episode(
        self,
        repo_path: str = "",
        test_identifier: str = "",
        max_steps: int = 8,
        num_runs: int = 10,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield step-by-step results as they happen, for WebSocket streaming."""
        step_queue: asyncio.Queue[Optional[StepResult]] = asyncio.Queue()

        def on_step(result: StepResult) -> None:
            step_queue.put_nowait(result)

        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(
            None,
            lambda: self.run_episode_sync(
                repo_path=repo_path,
                test_identifier=test_identifier,
                max_steps=max_steps,
                num_runs=num_runs,
                on_step=on_step,
            ),
        )

        completed = False
        while not completed:
            try:
                result = await asyncio.wait_for(step_queue.get(), timeout=120.0)
                if result is None:
                    break
                yield {"type": "step", "data": result.model_dump()}
                if result.done:
                    completed = True
            except asyncio.TimeoutError:
                yield {"type": "timeout", "data": {"message": "Step timed out"}}
                completed = True

        try:
            episode_result = await task
            yield {"type": "complete", "data": episode_result.model_dump()}
        except Exception as exc:
            yield {"type": "error", "data": {"message": str(exc)}}


# Singleton instance
_runner: Optional[EpisodeRunner] = None


def get_episode_runner() -> EpisodeRunner:
    global _runner
    if _runner is None:
        _runner = EpisodeRunner()
    return _runner
