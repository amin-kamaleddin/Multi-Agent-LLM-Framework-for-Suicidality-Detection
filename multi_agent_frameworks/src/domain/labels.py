"""Canonical labels and definitions for the suicide-risk task."""

from __future__ import annotations

from typing import Final


VALID_LABELS: Final[tuple[str, ...]] = ("Indicator", "Ideation", "Behavior", "Attempt")
UNSUPPORTIVE_LABEL: Final[str] = "Unsupportive"
FINAL_LABELS: Final[tuple[str, ...]] = (*VALID_LABELS, UNSUPPORTIVE_LABEL)

LABEL_DEFINITIONS: Final[dict[str, str]] = {
    "Indicator": (
        "The post does NOT indicate the author's own suicidal risk, but might be a "
        "window into their inner thoughts related to suicide. Includes: discussion "
        "of another person's suicidality, general discussion/news, or explicit "
        "denial of own suicidal intent. Person does not seem at any risk, but suicide "
        "was mentioned or discussed, even if in an explicit denial, so it should be "
        "flagged accordingly. Also includes concern for someone else at risk."
    ),
    "Ideation": (
        "The author expresses suicidal thoughts, wishes, or desire to die/end life. "
        "Includes passive or active suicidal thinking, including hypothetical or "
        "unrealistic methods. No concrete self-harm/suicide action has been taken in "
        "the post."
    ),
    "Behavior": (
        "The author describes preparatory or self-harm behavior beyond verbal "
        "thoughts. Includes planning/preparation actions (for example, gathering "
        "means, writing notes, preparing for death). Includes non-suicidal self-harm "
        "when self-inflicted harm is described, even without explicit intent to die."
    ),
    "Attempt": (
        "The post mentions a suicide attempt (current or past). A past attempt should "
        "still be labeled Attempt, even if the current post does not express ongoing "
        "suicidal thoughts."
    ),
    "Unsupportive": (
        "The post does NOT indicate the author's own suicidal risk or any connection "
        "to suicidal ideation. Includes: discussion of topics unrelated to suicidal "
        "ideation or self harm, such as hobbies, nature, and any other topic. However, "
        "also includes discussions about different emotions, including negative ones "
        "such as anger, sadness and even depression, as long as the author does not "
        "indicate at all to be thinking or discussing suicide or self-harm."
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
