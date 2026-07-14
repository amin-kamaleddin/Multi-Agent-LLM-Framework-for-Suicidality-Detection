"""Pydantic schemas for LLM JSON contracts."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator

from .labels import canonicalize_label, label_to_suicidal_intent


class SpecialistAnswer(BaseModel):
    """Validated specialist answer returned by one binary specialist agent."""

    model_config = ConfigDict(extra="forbid")

    answer: StrictBool = Field(description="Whether the text matches this specialist's assigned label.")


class SpecialistReasonedAnswer(BaseModel):
    """Validated specialist answer with a concise rationale."""

    model_config = ConfigDict(extra="forbid")

    answer: StrictBool = Field(description="Whether the text matches this specialist's assigned label.")
    reasoning: StrictStr = Field(
        min_length=1,
        description="One to three sentence rationale explaining why the assigned label does or does not match.",
    )

    @field_validator("reasoning")
    @classmethod
    def _validate_reasoning(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reasoning must not be blank")
        return stripped


class JudgeAnswer(BaseModel):
    """Validated final judge output."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(description="One of Indicator, Ideation, Behavior, Attempt, Unsupportive.")
    suicidal_intent: StrictBool = Field(
        description="True for Indicator, Ideation, Behavior, or Attempt; false for Unsupportive."
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
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
