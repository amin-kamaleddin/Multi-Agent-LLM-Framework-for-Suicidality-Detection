"""Pydantic schemas for LLM JSON contracts."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from .labels import canonicalize_label


class SpecialistAnswer(BaseModel):
    """Validated specialist answer returned by one binary specialist agent."""

    model_config = ConfigDict(extra="forbid")

    answer: StrictBool = Field(description="Whether the text matches this specialist's assigned label.")


class JudgeAnswer(BaseModel):
    """Validated final judge output."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(description="One of Indicator, Ideation, Behavior, Attempt.")

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
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
