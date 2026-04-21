"""Training-side rollout and GRPO integration helpers."""

from .grpo_trainer import run_episode, build_grpo_batch

__all__ = ["run_episode", "build_grpo_batch"]
