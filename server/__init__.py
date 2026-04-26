"""FlakeForge V3 Server — environment, reward, detection, and API."""

from .FlakeForge_environment import FlakeForgeEnvironment, create_flakeforge_environment
from .reward import compute_verifiable_reward
from .patch_applier import apply_search_replace_patch, parse_search_replace_hunks
from .deep_flakiness import build_deep_observation_signals, extract_failure_frontier
from .state import EpisodeState

__all__ = [
    "FlakeForgeEnvironment",
    "create_flakeforge_environment",
    "compute_verifiable_reward",
    "apply_search_replace_patch",
    "parse_search_replace_hunks",
    "build_deep_observation_signals",
    "extract_failure_frontier",
    "EpisodeState",
]
