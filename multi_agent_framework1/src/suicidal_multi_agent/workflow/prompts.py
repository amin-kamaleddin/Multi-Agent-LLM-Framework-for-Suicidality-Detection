"""Static prompts and prompt mappings for the naive multi-agent workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..core.labels import FINAL_LABELS, LABEL_DEFINITIONS, VALID_LABELS
from ..core.schemas import JudgeAnswer, SpecialistAnswer, SpecialistReasonedAnswer


# The four specialist prompts split the shared label definitions into one
# binary true/false task per agent while keeping labels centralized in
# core.labels.LABEL_DEFINITIONS.
def _make_specialist_prompt(*, label_name: str, include_reasoning: bool) -> str:
    output_shape = (
        '{"answer": true, "reasoning": "1-3 sentence explanation including any supporting evidence"}'
        if include_reasoning
        else '{"answer": true}'
    )
    reasoning_rules = (
        '\n- "reasoning" must be one to three sentences max.'
        '\n- Include any supporting evidence inside "reasoning"; do not output a separate evidence key.'
        if include_reasoning
        else "\n- Do not include markdown, commentary, rationale, confidence, or extra keys."
    )
    return f"""You are the {label_name} Agent for a suicide-risk classification task.

Target label: {label_name}

Label definition:
{LABEL_DEFINITIONS[label_name]}

Task:
Decide only whether the input text matches the {label_name} label. Do not choose the final label.

Input format:
You will receive one user message:

Text to classify:
<text>

Output format:
Return only valid JSON with exactly these keys:
{output_shape}

Rules:
- "answer" must be true when the text matches {label_name}, otherwise false.
- "answer" must be a JSON boolean, not a string.{reasoning_rules}"""


INDICATOR_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Indicator", include_reasoning=False)
IDEATION_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Ideation", include_reasoning=False)
BEHAVIOR_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Behavior", include_reasoning=False)
ATTEMPT_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Attempt", include_reasoning=False)

INDICATOR_REASONED_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Indicator", include_reasoning=True)
IDEATION_REASONED_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Ideation", include_reasoning=True)
BEHAVIOR_REASONED_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Behavior", include_reasoning=True)
ATTEMPT_REASONED_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Attempt", include_reasoning=True)


@dataclass(frozen=True)
class SpecialistPromptConfig:
    """Maps one specialist agent name to one label and one static prompt."""

    agent_name: str
    label_name: str
    system_prompt: str
    input_format: str = "Text to classify:\\n\\n{text}"
    output_format: str = '{"answer": true|false}'
    parallel_group: str = "four_label_specialists"


@dataclass(frozen=True)
class WorkflowModeConfig:
    """A reusable prompt/schema set for the same four-specialist graph."""

    name: str
    specialist_prompts: dict[str, SpecialistPromptConfig]
    specialist_schema: type[Any]
    judge_schema: type[Any]
    judge_system_prompt: Callable[[], str]
    judge_user_prompt: Callable[[str, dict[str, dict[str, Any]]], str]


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

REASONED_SPECIALIST_AGENT_PROMPTS: dict[str, SpecialistPromptConfig] = {
    "Indicator Agent": SpecialistPromptConfig(
        agent_name="Indicator Agent",
        label_name="Indicator",
        system_prompt=INDICATOR_REASONED_SPECIALIST_PROMPT,
        output_format='{"answer": true|false, "reasoning": string}',
    ),
    "Ideation Agent": SpecialistPromptConfig(
        agent_name="Ideation Agent",
        label_name="Ideation",
        system_prompt=IDEATION_REASONED_SPECIALIST_PROMPT,
        output_format='{"answer": true|false, "reasoning": string}',
    ),
    "Behavior Agent": SpecialistPromptConfig(
        agent_name="Behavior Agent",
        label_name="Behavior",
        system_prompt=BEHAVIOR_REASONED_SPECIALIST_PROMPT,
        output_format='{"answer": true|false, "reasoning": string}',
    ),
    "Attempt Agent": SpecialistPromptConfig(
        agent_name="Attempt Agent",
        label_name="Attempt",
        system_prompt=ATTEMPT_REASONED_SPECIALIST_PROMPT,
        output_format='{"answer": true|false, "reasoning": string}',
    ),
}


def specialist_system_prompt(label_name: str) -> str:
    """Return the static system prompt for the specialist assigned to label_name."""
    return SPECIALIST_AGENT_PROMPTS[LABEL_TO_AGENT_NAME[label_name]].system_prompt


def specialist_user_prompt(text: str) -> str:
    return f"Text to classify:\n\n{text}"


def judge_system_prompt() -> str:
    definitions = "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in FINAL_LABELS)
    return (
        "You are the final judge for a five-label suicide-risk classification task.\n"
        "Use the original text and the four specialist answers to choose exactly one final label.\n\n"
        "The four specialist agents run in parallel before you receive their answers:\n"
        f"Label definitions:\n{definitions}\n\n"
        "Input format:\n"
        "You receive the original text plus a JSON object keyed by the four specialist labels.\n\n"
        "Output format:\n"
        "Return only valid JSON with exactly this shape:\n"
        '{"label_name": "<label_name>", "suicidal_intent": true}\n'
        "Set suicidal_intent to true for Indicator, Ideation, Behavior, or Attempt, "
        "and false for Unsupportive. Treat this boolean as membership in the dataset's four-label "
        "positive class, not as a separate clinical inference about the author's intent."
    )


def judge_user_prompt(text: str, specialist_answers: dict[str, bool]) -> str:
    payload = {label: specialist_answers[label] for label in VALID_LABELS}
    return (
        "Original text:\n"
        f"{text}\n\n"
        "Specialist answers:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def judge_user_prompt_from_outputs(text: str, specialist_outputs: dict[str, dict[str, Any]]) -> str:
    answers = {label: bool(specialist_outputs[label]["answer"]) for label in VALID_LABELS}
    return judge_user_prompt(text, answers)


def reasoned_judge_system_prompt() -> str:
    definitions = "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in FINAL_LABELS)
    return (
        "You are the final judge for a five-label suicide-risk classification task.\n"
        "Use the original text plus the four specialist answers and rationales to choose exactly one final label.\n\n"
        "The four specialist agents run in parallel before you receive their outputs:\n"
        f"Label definitions:\n{definitions}\n\n"
        "Input format:\n"
        "You receive the original text plus a JSON object keyed by the four specialist labels. "
        "Each specialist object contains answer and reasoning.\n\n"
        "Output format:\n"
        "Return only valid JSON with exactly this shape:\n"
        '{"label_name": "<label_name>", "suicidal_intent": true}\n'
        "Set suicidal_intent to true for Indicator, Ideation, Behavior, or Attempt, "
        "and false for Unsupportive. Treat this boolean as membership in the dataset's four-label "
        "positive class, not as a separate clinical inference about the author's intent."
    )


def reasoned_judge_user_prompt(text: str, specialist_outputs: dict[str, dict[str, Any]]) -> str:
    payload = {label: specialist_outputs[label] for label in VALID_LABELS}
    return (
        "Original text:\n"
        f"{text}\n\n"
        "Specialist outputs:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


WORKFLOW_MODES: dict[str, WorkflowModeConfig] = {
    "bool": WorkflowModeConfig(
        name="bool",
        specialist_prompts=SPECIALIST_AGENT_PROMPTS,
        specialist_schema=SpecialistAnswer,
        judge_schema=JudgeAnswer,
        judge_system_prompt=judge_system_prompt,
        judge_user_prompt=judge_user_prompt_from_outputs,
    ),
    "reasoned": WorkflowModeConfig(
        name="reasoned",
        specialist_prompts=REASONED_SPECIALIST_AGENT_PROMPTS,
        specialist_schema=SpecialistReasonedAnswer,
        judge_schema=JudgeAnswer,
        judge_system_prompt=reasoned_judge_system_prompt,
        judge_user_prompt=reasoned_judge_user_prompt,
    ),
}

WORKFLOW_MODE_NAMES = tuple(WORKFLOW_MODES)


def get_workflow_mode(mode: str | WorkflowModeConfig) -> WorkflowModeConfig:
    if isinstance(mode, WorkflowModeConfig):
        return mode
    try:
        return WORKFLOW_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow mode {mode!r}. Expected one of: {', '.join(WORKFLOW_MODE_NAMES)}") from exc
