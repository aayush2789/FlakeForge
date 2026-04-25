"""V3 Episode State — enhanced for unified agent architecture.

Changes from V2:
- Removed hypothesis_stack (no hypothesis gating)
- Removed judge/teacher fields
- Added deep flakiness signals cache
- Added patch history tracking
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from models import RunRecord, PatchRecord, RewardBreakdown
except ImportError:
    from ..models import RunRecord, PatchRecord, RewardBreakdown


class EpisodeState(BaseModel):
    """V3 server-side episode state."""

    # Identity
    episode_id: str
    test_identifier: str
    repo_path: str

    # Step tracking
    step_count: int = 0
    max_steps: int = 8
    done: bool = False

    # Code snapshots
    original_test_source: str = ""
    original_source_under_test: str = ""
    current_test_source: str = ""
    current_source_under_test: str = ""

    # Run data
    run_history: List[RunRecord] = Field(default_factory=list)
    baseline_pass_rate: float = 0.0
    current_pass_rate: float = 0.0
    baseline_entropy: float = 0.0

    # Patch tracking
    patches_applied: List[PatchRecord] = Field(default_factory=list)
    total_diff_lines: int = 0

    # Deep flakiness signals (cached from initial detection)
    module_cache_violations: List[str] = Field(default_factory=list)
    fixture_scope_risks: List[str] = Field(default_factory=list)
    mock_residue_sites: List[str] = Field(default_factory=list)
    import_side_effect_files: List[str] = Field(default_factory=list)
    async_contamination_alive: bool = False

    # Causal frontier
    failure_frontier: str = ""
    call_chain_to_frontier: List[str] = Field(default_factory=list)
    boundary_crossings: List[str] = Field(default_factory=list)

    # iDFlakies signals
    order_dependency_detected: bool = False
    infrastructure_sensitive: bool = False

    # Stack trace
    failing_stack_trace: str = ""
    last_error_type: Optional[str] = None

    # Causal graph
    causal_graph: Optional[Dict[str, Any]] = None
    causal_hints: List[str] = Field(default_factory=list)

    # Last action tracking (for multi-step episodes)
    last_think_text: str = ""
    last_patch_text: str = ""
    last_reward: float = 0.0
    last_reward_breakdown: Dict[str, float] = Field(default_factory=dict)

    # File tree
    file_tree: List[str] = Field(default_factory=list)

    # Regression tracking
    regression_detected: bool = False

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.step_count)

    @property
    def is_terminal(self) -> bool:
        return self.done or self.step_count >= self.max_steps

    @property
    def pass_rate_delta(self) -> float:
        return round(self.current_pass_rate - self.baseline_pass_rate, 4)
