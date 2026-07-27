"""Canonical labels and definitions for the suicide-risk task."""

from __future__ import annotations

from typing import Final


VALID_LABELS: Final[tuple[str, ...]] = ("Indicator", "Ideation", "Behavior", "Attempt")
UNSUPPORTIVE_LABEL: Final[str] = "Unsupportive"
FINAL_LABELS: Final[tuple[str, ...]] = (*VALID_LABELS, UNSUPPORTIVE_LABEL)

LABEL_DEFINITIONS: Final[dict[str, str]] = {
    "Indicator": (
        "The post does not indicate the author's own suicidal risk. It may discuss "
        "another person's suicidality, general suicide-related topics, concern for "
        "someone else, or explicit denial of the author's own suicidal intent."
    ),
    "Ideation": (
        "The author expresses suicidal thoughts, wishes, or desire to die, but does "
        "not describe concrete self-harm or suicide action."
    ),
    "Behavior": (
        "The author describes preparatory or self-harm behavior beyond verbal "
        "thoughts, such as gathering means, writing notes, preparing for death, or "
        "self-inflicted harm without an explicit fatal attempt."
    ),
    "Attempt": (
        "The post mentions a suicide attempt, current or past. A past attempt should "
        "still be labeled Attempt."
    ),
    "Unsupportive": (
        "The text does not concern suicide, self-harm, suicidal "
        "thinking, suicidal behavior, or suicide attempts. "
        "Do not infer from negative emotions alone without any clear evidence of suicidal thoughts."
    ),
}


def canonicalize_label(value: str) -> str:
    """Return the canonical label spelling or raise ValueError."""
    normalized = str(value).strip().lower()
    for label in FINAL_LABELS:
        if normalized == label.lower():
            return label
    raise ValueError(f"Invalid label_name={value!r}. Expected one of {', '.join(FINAL_LABELS)}.")


def label_to_suicidal_intent(value: str) -> bool:
    """Map the final label to the required binary suicidal-intent flag."""
    return canonicalize_label(value) != UNSUPPORTIVE_LABEL
