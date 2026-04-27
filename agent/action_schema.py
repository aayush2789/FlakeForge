"""Strict JSON action schema for tool-augmented vs patch-only agent steps."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field


class ToolCallActionModel(BaseModel):
    """Model chose to invoke a registered analysis tool."""

    action: Literal["tool_call"] = "tool_call"
    tool: str = Field(min_length=1, description="Registered tool name")
    args: Dict[str, Any] = Field(default_factory=dict)


class PatchActionModel(BaseModel):
    """Model chose to emit a final patch (same nested shape as legacy unified JSON)."""

    action: Literal["patch"] = "patch"
    think: Any = Field(default_factory=dict)
    patch: Any = Field(default_factory=dict)


AgentStepModel = Union[ToolCallActionModel, PatchActionModel]


def parse_agent_step_json(data: Optional[Dict[str, Any]]) -> Optional[AgentStepModel]:
    """Validate a parsed JSON object as tool_call, patch, or legacy think/patch."""
    if not data or not isinstance(data, dict):
        return None

    action = data.get("action")
    if action == "tool_call":
        try:
            return ToolCallActionModel.model_validate(data)
        except Exception:
            return None
    if action == "patch":
        try:
            return PatchActionModel.model_validate(data)
        except Exception:
            return None

    # Legacy unified JSON: top-level think and/or patch without discriminator.
    if isinstance(data.get("think"), dict) or isinstance(data.get("patch"), dict):
        try:
            return PatchActionModel(
                think=data.get("think", {}),
                patch=data.get("patch", {}),
            )
        except Exception:
            return None

    return None


class AgentJSONParseError(Exception):
    """Raised when model output is not valid agent JSON."""


def classify_raw_agent_json(raw: str, load_json_object: Any) -> Optional[AgentStepModel]:
    """Parse model text into a step model using the caller's JSON loader."""
    data = load_json_object(raw)
    return parse_agent_step_json(data)
