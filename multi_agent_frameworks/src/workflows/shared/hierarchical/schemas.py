"""Routing decisions common to both hierarchical frameworks."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PerspectiveDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["SELF_RISK", "INDICATOR", "UNSUPPORTIVE"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class IdeationActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["IDEATION", "ACTION_PATHWAY"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class BehaviorAttemptDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["BEHAVIOR", "ATTEMPT"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
