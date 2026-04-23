import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from ..models import Hypothesis
except ImportError:
    from models import Hypothesis

@dataclass
class EpisodeState:
    episode_id: str
    target_function_source: str
    source_under_test: str
    step: int = 0
    max_steps: int = 10
    total_reward: float = 0.0

    baseline_pass_rate: float = 0.0
    current_pass_rate: float = 0.0

    last_actions: List[str] = field(default_factory=list)
    last_outcomes: List[float] = field(default_factory=list)
    prediction_error_history: List[float] = field(default_factory=list)
    actions_tried: List[str] = field(default_factory=list)

    current_hypothesis: Optional[Hypothesis] = None
    secondary_hypothesis: Optional[Hypothesis] = None
    reflection: Optional[str] = None
    run_history: List[Any] = field(default_factory=list)
    log_snippets: List[str] = field(default_factory=list)

    # ── V2 Dynamic Features
    causal_graph_dict: Optional[Dict[str, Any]] = None
    boundary_warnings: List[str] = field(default_factory=list)
    infrastructure_sensitive: bool = False
    duration_fingerprint: Optional[Dict[str, float]] = None

    failure_pattern_summary: str = ""
    causal_hints: List[str] = field(default_factory=list)
    async_markers: List[str] = field(default_factory=list)
    
    start_time: float = field(default_factory=time.time)

    def is_done(self) -> bool:
        return self.step >= self.max_steps or self.current_pass_rate >= 0.95
