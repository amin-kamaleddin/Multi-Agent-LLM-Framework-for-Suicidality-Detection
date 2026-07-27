"""Intermediate specialist contracts for framework 1."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class SpecialistAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: StrictBool = Field(description="Whether the text matches the assigned label.")
