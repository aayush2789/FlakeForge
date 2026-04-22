from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class FullTraceLogger:
    """Episode-level structured logger for reasoning, actions, rewards and errors."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        root = Path(__file__).resolve().parent.parent
        self.output_dir = output_dir or (root / "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_enabled = os.getenv("ENV_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
        self.trace: Dict[str, Any] = {}
        self.trace_file: Optional[Path] = None

    def start_episode(
        self,
        episode_id: str,
        test_identifier: str,
        max_steps: int,
        baseline_pass_rate: float,
    ) -> None:
        self.trace = {
            "episode_id": episode_id,
            "test_identifier": test_identifier,
            "max_steps": max_steps,
            "baseline_pass_rate": baseline_pass_rate,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "steps": [],
            "errors": [],
            "reward_history": [],
            "prediction_error_history": [],
            "summary": {},
        }
        self.trace_file = self.output_dir / f"full_trace_{episode_id}.json"
        self._flush()

    def log_step(self, payload: Dict[str, Any]) -> None:
        self.trace.setdefault("steps", []).append(payload)

        reward = payload.get("reward_breakdown")
        if reward:
            self.trace.setdefault("reward_history", []).append(reward)

        learning = payload.get("learning_signals") or {}
        pred_err = learning.get("prediction_error")
        if isinstance(pred_err, (int, float)):
            self.trace.setdefault("prediction_error_history", []).append(float(pred_err))

        if self.debug_enabled:
            self._debug_step(payload)
        self._flush()

    def log_error(self, payload: Dict[str, Any]) -> None:
        self.trace.setdefault("errors", []).append(payload)
        if self.debug_enabled:
            print(f"[ENV_DEBUG][ERROR] {json.dumps(payload, ensure_ascii=True)}", flush=True)
        self._flush()

    def set_summary(self, summary: Dict[str, Any]) -> None:
        self.trace["summary"] = summary
        self.trace["ended_at"] = datetime.utcnow().isoformat() + "Z"
        self._flush()

    def _debug_step(self, payload: Dict[str, Any]) -> None:
        reasoning = payload.get("reasoning") or {}
        action = payload.get("action") or {}
        reward = payload.get("reward_breakdown") or {}
        learning = payload.get("learning_signals") or {}
        print(
            "[ENV_DEBUG][REASONING]"
            f" step={payload.get('step')}"
            f" category={reasoning.get('root_cause_category')}"
            f" confidence={reasoning.get('confidence')}",
            flush=True,
        )
        print(
            "[ENV_DEBUG][ACTION]"
            f" action={action.get('action_type')}"
            f" predicted={action.get('predicted_pass_rate_after')}"
            f" actual={payload.get('execution', {}).get('pass_rate_after')}",
            flush=True,
        )
        print(f"[ENV_DEBUG][REWARD] {json.dumps(reward, ensure_ascii=True)}", flush=True)
        print(f"[ENV_DEBUG][LEARNING] {json.dumps(learning, ensure_ascii=True)}", flush=True)

    def _flush(self) -> None:
        if not self.trace_file:
            return
        self.trace_file.write_text(json.dumps(self.trace, indent=2), encoding="utf-8")
