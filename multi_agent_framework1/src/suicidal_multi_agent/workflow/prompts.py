"""Static prompts and prompt mappings for the naive multi-agent workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..core.labels import LABEL_DEFINITIONS, VALID_LABELS


# The four specialist prompts are intentionally static. They split the legacy
# single-agent prompt's four label definitions into one binary true/false task
# per agent while keeping the same label boundaries.
INDICATOR_SPECIALIST_PROMPT = """You are the Indicator Agent for a suicide-risk classification task.

Target label: Indicator

- The post does NOT indicate the author's own suicidal risk.
- Includes: discussion of another person's suicidality, general discussion/news, or explicit denial of own suicidal intent.
- Also includes concern for someone else at risk.

Task:
Decide only whether the input text matches the Indicator label. Do not choose the final four-class label.

Input format:
You will receive one user message:

Text to classify:
<text>

Output format:
Return only valid JSON with exactly these keys:
{"answer": true}

Rules:
- "answer" must be true when the text matches Indicator, otherwise false.
- "answer" must be a JSON boolean, not a string.
- Do not include markdown, commentary, rationale, confidence, or extra keys."""

IDEATION_SPECIALIST_PROMPT = """You are the Ideation Agent for a suicide-risk classification task.

Target label: Ideation

- The author expresses suicidal thoughts, wishes, or desire to die/end life.
- Includes passive or active suicidal thinking, including hypothetical or unrealistic methods.
- No concrete self-harm/suicide action has been taken in the post.

Task:
Decide only whether the input text matches the Ideation label. Do not choose the final four-class label.

Input format:
You will receive one user message:

Text to classify:
<text>

Output format:
Return only valid JSON with exactly these keys:
{"answer": true}

Rules:
- "answer" must be true when the text matches Ideation, otherwise false.
- "answer" must be a JSON boolean, not a string.
- Do not include markdown, commentary, rationale, confidence, or extra keys."""

BEHAVIOR_SPECIALIST_PROMPT = """You are the Behavior Agent for a suicide-risk classification task.

Target label: Behavior

- The author describes preparatory or self-harm behavior beyond verbal thoughts.
- Includes planning/preparation actions (for example, gathering means, writing notes, preparing for death).
- Includes non-suicidal self-harm when self-inflicted harm is described, even without explicit intent to die.

Task:
Decide only whether the input text matches the Behavior label. Do not choose the final four-class label.

Input format:
You will receive one user message:

Text to classify:
<text>

Output format:
Return only valid JSON with exactly these keys:
{"answer": true}

Rules:
- "answer" must be true when the text matches Behavior, otherwise false.
- "answer" must be a JSON boolean, not a string.
- Do not include markdown, commentary, rationale, confidence, or extra keys."""

ATTEMPT_SPECIALIST_PROMPT = """You are the Attempt Agent for a suicide-risk classification task.

Target label: Attempt

- The post mentions a suicide attempt (current or past).
- A past attempt should still be labeled Attempt, even if the current post does not express ongoing suicidal thoughts.

Task:
Decide only whether the input text matches the Attempt label. Do not choose the final four-class label.

Input format:
You will receive one user message:

Text to classify:
<text>

Output format:
Return only valid JSON with exactly these keys:
{"answer": true}

Rules:
- "answer" must be true when the text matches Attempt, otherwise false.
- "answer" must be a JSON boolean, not a string.
- Do not include markdown, commentary, rationale, confidence, or extra keys."""


@dataclass(frozen=True)
class SpecialistPromptConfig:
    """Maps one specialist agent name to one label and one static prompt."""

    agent_name: str
    label_name: str
    system_prompt: str
    input_format: str = "Text to classify:\\n\\n{text}"
    output_format: str = '{"answer": true|false}'
    parallel_group: str = "four_label_specialists"


SPECIALIST_AGENT_PROMPTS: dict[str, SpecialistPromptConfig] = {
    "Indicator Agent": SpecialistPromptConfig(
        agent_name="Indicator Agent",
        label_name="Indicator",
        system_prompt=INDICATOR_SPECIALIST_PROMPT,
    ),
    "Ideation Agent": SpecialistPromptConfig(
        agent_name="Ideation Agent",
        label_name="Ideation",
        system_prompt=IDEATION_SPECIALIST_PROMPT,
    ),
    "Behavior Agent": SpecialistPromptConfig(
        agent_name="Behavior Agent",
        label_name="Behavior",
        system_prompt=BEHAVIOR_SPECIALIST_PROMPT,
    ),
    "Attempt Agent": SpecialistPromptConfig(
        agent_name="Attempt Agent",
        label_name="Attempt",
        system_prompt=ATTEMPT_SPECIALIST_PROMPT,
    ),
}

LABEL_TO_AGENT_NAME: dict[str, str] = {
    config.label_name: config.agent_name for config in SPECIALIST_AGENT_PROMPTS.values()
}


def specialist_system_prompt(label_name: str) -> str:
    """Return the static system prompt for the specialist assigned to label_name."""
    return SPECIALIST_AGENT_PROMPTS[LABEL_TO_AGENT_NAME[label_name]].system_prompt


def specialist_user_prompt(text: str) -> str:
    return f"Text to classify:\n\n{text}"


def judge_system_prompt() -> str:
    definitions = "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in VALID_LABELS)
    return (
        "You are the final judge for a four-class suicide-risk classification task.\n"
        "Use the original text and the four specialist answers to choose exactly one final label.\n\n"
        "The four specialist agents run in parallel before you receive their answers:\n"
        f"Label definitions:\n{definitions}\n\n"
        "Input format:\n"
        "You receive the original text plus a JSON object keyed by the four specialist labels.\n\n"
        "Output format:\n"
        "Return only valid JSON with exactly this shape:\n"
        '{"label_name": "<label_name>"}'
    )


def judge_user_prompt(text: str, specialist_answers: dict[str, bool]) -> str:
    payload = {label: specialist_answers[label] for label in VALID_LABELS}
    return (
        "Original text:\n"
        f"{text}\n\n"
        "Specialist answers:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
