"""FlakeForge V3 Server — environment, reward, detection, and API."""

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


def __getattr__(name: str):
    """Lazy imports so modules that don't need openenv can still load."""
    if name in ("FlakeForgeEnvironment", "create_flakeforge_environment"):
        from .FlakeForge_environment import FlakeForgeEnvironment, create_flakeforge_environment
        return FlakeForgeEnvironment if name == "FlakeForgeEnvironment" else create_flakeforge_environment
    if name == "compute_verifiable_reward":
        from .reward import compute_verifiable_reward
        return compute_verifiable_reward
    if name in ("apply_search_replace_patch", "parse_search_replace_hunks"):
        from .patch_applier import apply_search_replace_patch, parse_search_replace_hunks
        return apply_search_replace_patch if name == "apply_search_replace_patch" else parse_search_replace_hunks
    if name in ("build_deep_observation_signals", "extract_failure_frontier"):
        from .deep_flakiness import build_deep_observation_signals, extract_failure_frontier
        return build_deep_observation_signals if name == "build_deep_observation_signals" else extract_failure_frontier
    if name == "EpisodeState":
        from .state import EpisodeState
        return EpisodeState
    raise AttributeError(f"module 'server' has no attribute {name!r}")
