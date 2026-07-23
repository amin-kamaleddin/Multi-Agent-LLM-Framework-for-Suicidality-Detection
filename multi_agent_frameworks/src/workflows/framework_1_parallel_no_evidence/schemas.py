"""Intermediate specialist contracts for framework 1."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator


class SpecialistAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: StrictBool = Field(description="Whether the text matches the assigned label.")


class SpecialistReasonedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: StrictBool = Field(description="Whether the text matches the assigned label.")
    reasoning: StrictStr = Field(min_length=1)

    @field_validator("reasoning")
    @classmethod
    def _validate_reasoning(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reasoning must not be blank")
        return stripped
