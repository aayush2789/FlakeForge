"""DEPRECATED — V3 removed the LLM Judge.

The FrozenJudge has been replaced by a six-signal verifiable reward
architecture in server/reward.py. Rewards are now deterministic and
derived from execution outcomes.

Import from server.reward instead:
    from server.reward import compute_verifiable_reward
"""

import warnings

warnings.warn(
    "agent.judge is DEPRECATED in FlakeForge V3. "
    "The LLM Judge has been replaced by deterministic verifiable rewards. "
    "Use server.reward.compute_verifiable_reward instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Legacy alias
FrozenJudge = None
