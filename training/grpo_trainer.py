from __future__ import annotations

from typing import Any, Dict, List

from agent.judge import FrozenJudge
from agent.roles import FlakeForgeAgentPipeline
from models import FlakeForgeAction, FlakeForgeObservation
from server.reward import compute_reward


def run_episode(
    env: Any,
    agent_pipeline: FlakeForgeAgentPipeline,
    judge: FrozenJudge,
    max_steps: int = 14,
) -> List[Dict[str, Any]]:
    """Run one episode rollout and return trajectory records."""

    obs_result = env.reset()
    obs: FlakeForgeObservation = obs_result if isinstance(obs_result, FlakeForgeObservation) else obs_result.observation

    trajectory: List[Dict[str, Any]] = []
    pending_judge_scores: Dict[str, int] = {}

    for _ in range(max_steps):
        hypothesis, action = agent_pipeline.run_step(obs)
        if pending_judge_scores:
            action.judge_feedback = dict(pending_judge_scores)

        step_result = env.step(action)
        next_obs: FlakeForgeObservation = step_result if isinstance(step_result, FlakeForgeObservation) else step_result.observation

        step_info = getattr(next_obs, "metadata", {}) or {}
        patch_diff = step_info.get("diff", "")

        hyp_score = judge.score_hypothesis(obs, hypothesis)
        patch_score = judge.score_patch(obs, hypothesis, action, patch_diff)

        judge_scores = {
            "judge_hypothesis_score": int(hyp_score.get("score", 0)),
            "judge_patch_score": int(patch_score.get("score", 0)),
        }
        pending_judge_scores = dict(judge_scores)

        reward_input = step_info.get("step_result", {
            "current_pass_rate": next_obs.current_pass_rate,
            "regression_detected": False,
            "action_taken": action.action_type,
            "done": bool(next_obs.done),
            "timed_out": False,
            "ast_diff": step_info.get("ast_diff", {}),
        })

        full_reward, reward_breakdown = compute_reward(env.state, reward_input, judge_scores)

        trajectory.append(
            {
                "obs": obs,
                "hypothesis": hypothesis,
                "action": action,
                "reward": full_reward,
                "reward_breakdown": reward_breakdown,
                "judge_scores": judge_scores,
                "done": bool(next_obs.done),
            }
        )

        obs = next_obs
        if bool(next_obs.done):
            break

    return trajectory


def build_grpo_batch(episodes: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Prepare a lightweight GRPO-ready batch payload from episode trajectories."""

    flat = [step for episode in episodes for step in episode]
    rewards = [float(step["reward"]) for step in flat]
    if not rewards:
        return {"samples": [], "reward_mean": 0.0, "reward_std": 0.0}

    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = variance ** 0.5

    return {
        "samples": flat,
        "reward_mean": mean,
        "reward_std": std,
    }
