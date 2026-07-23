"""Intermediate judge and reviewer decisions for framework 4."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from domain.labels import FINAL_LABELS, canonicalize_label
from workflows.shared.hierarchical.schemas import (
    BehaviorAttemptDecision,
    IdeationActionDecision,
    PerspectiveDecision,
)


class JudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    competing_label: str
    competing_confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("label_name", "competing_label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        canonical = canonicalize_label(value)
        if canonical not in FINAL_LABELS:
            raise ValueError(f"Invalid final label={value!r}. Expected one of {FINAL_LABELS}.")
        return canonical

    @model_validator(mode="after")
    def labels_must_differ(self) -> "JudgeDecision":
        if self.label_name == self.competing_label:
            raise ValueError("label_name and competing_label must be different.")
        return self


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    escalation_flag: bool
    reasoning: str

    @field_validator("label_name")
    @classmethod
    def validate_label(cls, value: str) -> str:
        canonical = canonicalize_label(value)
        if canonical not in FINAL_LABELS:
            raise ValueError(f"Invalid final label={value!r}. Expected one of {FINAL_LABELS}.")
        return canonical
