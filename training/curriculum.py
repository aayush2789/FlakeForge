"""V3 Curriculum Scheduler for FlakeForge GRPO Training.

Three-stage curriculum: easy → medium → hard flakiness patterns.

Stage 1 (easy):   timing_sensitive, random_seed, shared_state
Stage 2 (medium): concurrency, async_wait, fixture_scope_leak, mock_residue
Stage 3 (hard):   network, platform_dependency, import_side_effect, module_cache_pollution

Advancement: rolling mean of last K episode rewards exceeds stage threshold.
Repos are drawn from test_repos/synthetic/ grouped by the 'difficulty' field in
flake_manifest.json, which allows the scheduler to work without any extra data
directory setup.

Research basis:
- Bengio ICLR 2009: Curriculum Learning (easy-to-hard improves convergence)
- Power et al. 2022: Grokking (hard tasks need warm started representations)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from utils.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda n, **kw: logging.getLogger(n)

logger = get_logger(__name__)


# ── Stage definitions ────────────────────────────────────────────────────────

@dataclass
class CurriculumStage:
    """One stage of the curriculum."""
    name: str
    difficulty: str
    min_episodes: int
    advance_threshold: float
    allowed_categories: set = field(default_factory=set)

    # Populated by the scheduler after loading repos.
    cases: List[Dict[str, Any]] = field(default_factory=list)

    def sample_case(self) -> Optional[Dict[str, Any]]:
        """Return a randomly selected case from this stage, or None if empty."""
        if not self.cases:
            return None
        return random.choice(self.cases)


_DEFAULT_STAGES: List[CurriculumStage] = [
    CurriculumStage(
        name="Stage1_Easy",
        difficulty="easy",
        min_episodes=30,
        advance_threshold=3.0,
        allowed_categories={
            "nondeterminism", "shared_state", "test_order_dependency",
            "resource_leak",
        },
    ),
    CurriculumStage(
        name="Stage2_Medium",
        difficulty="medium",
        min_episodes=35,
        advance_threshold=2.0,
        allowed_categories={
            "concurrency", "async_wait", "fixture_scope_leak",
            "mock_residue", "import_side_effect",
        },
    ),
    CurriculumStage(
        name="Stage3_Hard",
        difficulty="hard",
        min_episodes=35,
        advance_threshold=1.5,
        allowed_categories={
            "network", "platform_dependency", "module_cache_pollution",
        },
    ),
]


# ── Scheduler ────────────────────────────────────────────────────────────────

class CurriculumScheduler:
    """Manage progression through curriculum stages.

    Usage::

        scheduler = CurriculumScheduler("test_repos/synthetic")
        for episode in range(max_episodes):
            case = scheduler.sample()      # dict with repo_path, manifest, …
            reward = run_episode(case)
            scheduler.record(reward)
            if scheduler.should_save_checkpoint:
                save_checkpoint()
    """

    def __init__(
        self,
        synthetic_root: str = "test_repos/synthetic",
        stages: Optional[List[CurriculumStage]] = None,
        reward_window: int = 10,
    ) -> None:
        self.synthetic_root = Path(synthetic_root)
        self.stages = stages or [
            CurriculumStage(
                name=s.name,
                difficulty=s.difficulty,
                min_episodes=s.min_episodes,
                advance_threshold=s.advance_threshold,
                allowed_categories=set(s.allowed_categories),
            )
            for s in _DEFAULT_STAGES
        ]
        self.reward_window = reward_window

        self._stage_idx: int = 0
        self._episode_count: int = 0
        self._recent_rewards: List[float] = []
        self._all_cases: List[Dict[str, Any]] = []

        self._load_cases()

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self._stage_idx]

    @property
    def current_stage_index(self) -> int:
        return self._stage_idx

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def recent_mean_reward(self) -> float:
        if not self._recent_rewards:
            return 0.0
        tail = self._recent_rewards[-self.reward_window:]
        return sum(tail) / len(tail)

    def sample(self) -> Optional[Dict[str, Any]]:
        """Sample a case from the current stage.

        Falls back to the full pool if the current stage has no cases
        (graceful degradation when running without synthetic repos).
        """
        case = self.current_stage.sample_case()
        if case is None and self._all_cases:
            logger.warning(
                "[CURRICULUM] Stage %s has no cases; sampling from full pool.",
                self.current_stage.name,
            )
            case = random.choice(self._all_cases)
        return case

    def record(self, reward: float) -> None:
        """Record the result of one episode and advance the stage if ready."""
        self._episode_count += 1
        self._recent_rewards.append(reward)

        if len(self._recent_rewards) < self.reward_window:
            return

        stage = self.current_stage
        mean_reward = self.recent_mean_reward
        next_idx = self._stage_idx + 1

        if (
            self._episode_count >= stage.min_episodes
            and mean_reward >= stage.advance_threshold
            and next_idx < len(self.stages)
        ):
            self._stage_idx = next_idx
            logger.info(
                "[CURRICULUM] Advanced to %s after %d episodes (mean_reward=%.3f >= threshold=%.1f)",
                self.current_stage.name,
                self._episode_count,
                mean_reward,
                stage.advance_threshold,
            )

    def state_dict(self) -> Dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "stage_idx": self._stage_idx,
            "episode_count": self._episode_count,
            "recent_rewards": list(self._recent_rewards),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self._stage_idx = int(state.get("stage_idx", 0))
        self._episode_count = int(state.get("episode_count", 0))
        self._recent_rewards = list(state.get("recent_rewards", []))
        logger.info(
            "[CURRICULUM] Loaded state: stage=%s episode=%d",
            self.current_stage.name,
            self._episode_count,
        )

    def summary(self) -> str:
        """Human-readable one-line status string."""
        return (
            f"stage={self.current_stage.name} "
            f"episode={self._episode_count} "
            f"mean_reward={self.recent_mean_reward:.3f}"
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _load_cases(self) -> None:
        """Discover all synthetic cases and assign them to stages."""
        if not self.synthetic_root.exists():
            logger.warning(
                "[CURRICULUM] synthetic_root does not exist: %s — scheduler will use empty case pools.",
                self.synthetic_root,
            )
            return

        for manifest_path in sorted(self.synthetic_root.glob("*/flake_manifest.json")):
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception as exc:
                logger.warning("[CURRICULUM] Failed to read %s: %s", manifest_path, exc)
                continue

            case_id = manifest_path.parent.name
            difficulty = manifest.get("difficulty", "medium").lower()
            category = manifest.get("category", manifest.get("flake_category", "unknown"))
            test_id = manifest.get("test_identifier", manifest.get("test_id", "tests/test_flaky.py"))

            case = {
                "case_id": case_id,
                "repo_path": str(manifest_path.parent),
                "manifest_path": str(manifest_path),
                "manifest": manifest,
                "difficulty": difficulty,
                "category": category,
                "test_identifier": test_id,
            }
            self._all_cases.append(case)

            # Assign to stage by difficulty field
            assigned = False
            for stage in self.stages:
                if stage.difficulty == difficulty:
                    stage.cases.append(case)
                    assigned = True
                    break

            if not assigned:
                # Fallback: assign by category membership
                for stage in self.stages:
                    if category in stage.allowed_categories:
                        stage.cases.append(case)
                        assigned = True
                        break

            if not assigned:
                # Last resort: put in hardest stage
                self.stages[-1].cases.append(case)

        total = sum(len(s.cases) for s in self.stages)
        logger.info(
            "[CURRICULUM] Loaded %d cases into %d stages: %s",
            total,
            len(self.stages),
            {s.name: len(s.cases) for s in self.stages},
        )
