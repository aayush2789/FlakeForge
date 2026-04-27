"""FlakeForge V3 — Unified Agent Architecture for Flaky Test Repair.

V3 Key Changes:
- Unified agent: single model generates <think> + <patch> in one pass
- No LLM judge: 6-signal verifiable reward from execution outcomes
- Deep flakiness detection: AST-based module cache, fixture, mock, import, async patterns
- Free-form patches: search/replace hunks instead of hardcoded 7-action space
- GRPO training: group-relative policy optimization with verifiable reward
"""

from .models import (
    FlakeForgeAction,
    FlakeForgeObservation,
    FlakeForgeState,
    RunRecord,
    PatchRecord,
    RewardBreakdown,
    ROOT_CAUSE_TYPES,
    failure_mode_entropy,
    # Backward compatible aliases
    FlakeforgeAction,
    FlakeforgeObservation,
)

from .agent.unified_agent import (
    ToolAugmentedFlakeForgeAgent,
    UnifiedFlakeForgeAgent,
    build_minimal_agent_prompt,
    build_unified_prompt,
    extract_category_from_think,
    extract_patch,
    extract_think,
)

from .server.FlakeForge_environment import (
    FlakeForgeEnvironment,
    create_flakeforge_environment,
)

from .server.reward import compute_verifiable_reward
from .server.patch_applier import apply_search_replace_patch, parse_search_replace_hunks
from .server.deep_flakiness import build_deep_observation_signals

from .client import FlakeForgeClient

__all__ = [
    # Models
    "FlakeForgeAction",
    "FlakeForgeObservation",
    "FlakeForgeState",
    "RunRecord",
    "PatchRecord",
    "RewardBreakdown",
    "ROOT_CAUSE_TYPES",
    "failure_mode_entropy",
    "FlakeforgeAction",
    "FlakeforgeObservation",
    # Agent
    "ToolAugmentedFlakeForgeAgent",
    "UnifiedFlakeForgeAgent",
    "build_minimal_agent_prompt",
    "build_unified_prompt",
    "extract_think",
    "extract_patch",
    "extract_category_from_think",
    # Environment
    "FlakeForgeEnvironment",
    "create_flakeforge_environment",
    # Reward
    "compute_verifiable_reward",
    # Patch
    "apply_search_replace_patch",
    "parse_search_replace_hunks",
    # Detection
    "build_deep_observation_signals",
    # Client
    "FlakeForgeClient",
]
