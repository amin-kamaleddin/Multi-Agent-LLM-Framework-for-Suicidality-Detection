"""Final judge output and JSON parsing contracts shared by all workflows."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .labels import canonicalize_label, label_to_suicidal_intent


class JudgeAnswer(BaseModel):
    """Validated final judge output."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(description="One of Indicator, Ideation, Behavior, Attempt, Unsupportive.")
    suicidal_intent: StrictBool = Field(
        description=(
            "Dataset positive-class membership: true for Indicator, Ideation, Behavior, or Attempt; "
            "false for Unsupportive."
        )
    )

    @field_validator("label_name")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        return canonicalize_label(value)

    @model_validator(mode="after")
    def _validate_suicidal_intent(self) -> "JudgeAnswer":
        expected = label_to_suicidal_intent(self.label_name)
        if self.suicidal_intent is not expected:
            raise ValueError(
                f"suicidal_intent must be {expected} when label_name is {self.label_name}."
            )
        return self


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
