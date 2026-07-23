"""Shared evidence contracts used by evidence-producing workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


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

    quote: StrictStr = Field(
        min_length=1,
        description="Relevant excerpt or concise paraphrase of at most 100 words.",
    )
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
    allowed_types: set[str],
) -> None:
    """Ensure every item uses an evidence type assigned to its collector."""
    for item in answer.evidence:
        if item.type not in allowed_types:
            raise ValueError(
                f"Evidence type {item.type!r} is not allowed for this collector; "
                f"expected one of {sorted(allowed_types)}"
            )
