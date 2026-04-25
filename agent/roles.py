"""DEPRECATED — V3 removed the Analyzer/Fixer split.

The AnalyzerRole and FixerRole classes have been replaced by the
UnifiedFlakeForgeAgent in agent/unified_agent.py.

Import from agent.unified_agent instead:
    from agent.unified_agent import UnifiedFlakeForgeAgent
"""

import warnings

warnings.warn(
    "agent.roles is DEPRECATED in FlakeForge V3. "
    "Use agent.unified_agent.UnifiedFlakeForgeAgent instead. "
    "The two-head Analyzer/Fixer model has been replaced by a single "
    "unified agent that generates <think> + <patch> in one forward pass.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the unified agent for backward compatibility
from .unified_agent import UnifiedFlakeForgeAgent

# Legacy aliases (will raise warnings if used)
AnalyzerRole = UnifiedFlakeForgeAgent
FixerRole = UnifiedFlakeForgeAgent
