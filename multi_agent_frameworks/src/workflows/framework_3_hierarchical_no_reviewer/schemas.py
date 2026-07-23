"""Intermediate decisions for framework 3's hierarchy."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domain.labels import FINAL_LABELS
from workflows.shared.hierarchical.schemas import (
    BehaviorAttemptDecision,
    IdeationActionDecision,
    PerspectiveDecision,
)


class EvaluatorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    escalation_flag: bool
    reasoning: str

    @field_validator("label_name")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in FINAL_LABELS:
            raise ValueError(f"Invalid label_name={value!r}. Expected one of {FINAL_LABELS}.")
        return value
