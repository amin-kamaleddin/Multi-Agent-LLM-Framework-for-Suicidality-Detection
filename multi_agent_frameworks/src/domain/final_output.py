"""Final judge output and JSON parsing contracts shared by all workflows."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .labels import canonicalize_label


class JudgeAnswer(BaseModel):
    """Validated final judge output."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(description="One of Indicator, Ideation, Behavior, Attempt, Unsupportive.")

    @field_validator("label_name")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        return canonicalize_label(value)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract and parse one JSON object from raw model text."""
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        parsed, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
