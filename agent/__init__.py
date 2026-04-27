"""FlakeForge V3 Agent — unified agent architecture."""

from .tool_loop import (
    TOOL_MANIFEST_TEXT,
    ToolContext,
    ToolExecutor,
    ToolTraceEntry,
    build_default_tool_executor,
    format_tool_trace_for_prompt,
)
from .unified_agent import (
    TOOL_AUGMENTED_SYSTEM_PROMPT,
    ToolAugmentedFlakeForgeAgent,
    UnifiedFlakeForgeAgent,
    build_minimal_agent_prompt,
    build_unified_prompt,
    extract_category_from_think,
    extract_confidence_from_think,
    extract_patch,
    extract_think,
    infer_category_from_patch,
)

from .observation_utils import build_observation_from_state, summarize_observation

__all__ = [
    "TOOL_AUGMENTED_SYSTEM_PROMPT",
    "TOOL_MANIFEST_TEXT",
    "ToolAugmentedFlakeForgeAgent",
    "ToolContext",
    "ToolExecutor",
    "ToolTraceEntry",
    "UnifiedFlakeForgeAgent",
    "build_default_tool_executor",
    "build_minimal_agent_prompt",
    "build_unified_prompt",
    "extract_category_from_think",
    "extract_confidence_from_think",
    "extract_patch",
    "extract_think",
    "format_tool_trace_for_prompt",
    "infer_category_from_patch",
    "build_observation_from_state",
    "summarize_observation",
]
