"""Pydantic schemas for LLM JSON contracts."""

from __future__ import annotations

import json
from typing import Any, Literal

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


EvidenceType = Literal[
    "third_party",
    "self_denial",
    "general_suicide_context",
    "death_wish",
    "suicidal_thought",
    "hypothetical_method",
    "suicide_plan",
    "preparation",
    "self_harm",
    "attempt_report",
    "suicidal_action",
    "attempt_aftermath",
    "unrelated_topic",
    "benign_language",
    "non_suicidal_distress",
]


class EvidenceItem(BaseModel):
    """One compact, text-grounded observation from an evidence collector."""

    model_config = ConfigDict(extra="forbid")

    quote: StrictStr = Field(min_length=1, description="Verbatim input span of at most 100 words.")
    type: EvidenceType = Field(description="Collector-specific evidence type.")
    subject: Literal["author", "other", "general", "unclear"]
    time: Literal["past", "current", "future", "unclear", "na"]
    observation: StrictStr = Field(
        min_length=1,
        description="Factual explanation of what the quote shows, using at most 100 words.",
    )

    @field_validator("quote")
    @classmethod
    def _validate_quote(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("quote must not be blank")
        if len(stripped.split()) > 100:
            raise ValueError("quote must contain at most 100 words")
        return stripped

    @field_validator("observation")
    @classmethod
    def _validate_observation(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("observation must not be blank")
        if len(stripped.split()) > 100:
            raise ValueError("observation must contain at most 100 words")
        return stripped


class EvidenceCollectorAnswer(BaseModel):
    """Validated output from one non-decision-making evidence collector."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[EvidenceItem] = Field(
        max_length=3,
        description="Zero to three distinct evidence observations.",
    )


def validate_evidence_answer(
    answer: EvidenceCollectorAnswer,
    *,
    text: str,
    allowed_types: set[str],
) -> None:
    """Validate collector-specific types and verbatim grounding against the input."""
    for item in answer.evidence:
        if item.type not in allowed_types:
            raise ValueError(
                f"Evidence type {item.type!r} is not allowed for this collector; "
                f"expected one of {sorted(allowed_types)}"
            )
        if item.quote not in text:
            raise ValueError(f"Evidence quote is not an exact substring of the input: {item.quote!r}")


class JudgeAnswer(BaseModel):
    """Validated final judge output."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(description="One of Indicator, Ideation, Behavior, Attempt, Unsupportive.")
    suicidal_intent: StrictBool = Field(
        description=(
            "Dataset positive-class membership: true for Indicator, Ideation, Behavior, or Attempt; "
            "false for Unsupportive. This is not by itself a clinical claim that the author expresses intent."
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
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
