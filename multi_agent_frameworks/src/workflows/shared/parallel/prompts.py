"""Static prompts and prompt mappings for the shared parallel workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from domain.evidence import EvidenceCollectorAnswer
from domain.final_output import JudgeAnswer
from domain.labels import FINAL_LABELS, LABEL_DEFINITIONS
from workflows.framework_1_parallel_no_evidence.schemas import SpecialistAnswer


# The five specialist prompts split the shared label definitions into one
# binary true/false task per agent while keeping labels centralized in
# core.labels.LABEL_DEFINITIONS.
def _make_specialist_prompt(*, label_name: str) -> str:
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
{{"answer": true}}

Rules:
- "answer" must be true when the text matches {label_name}, otherwise false.
- "answer" must be a JSON boolean, not a string.
- Do not include markdown, commentary, rationale, confidence, or extra keys."""


INDICATOR_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Indicator")
IDEATION_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Ideation")
BEHAVIOR_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Behavior")
ATTEMPT_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Attempt")
UNSUPPORTIVE_SPECIALIST_PROMPT = _make_specialist_prompt(label_name="Unsupportive")


EVIDENCE_TYPE_DEFINITIONS: dict[str, dict[str, str]] = {
    "Indicator": {
        "third_party": "Another person is suicidal, or the author is concerned about another person.",
        "self_denial": "The author explicitly denies their own suicidality.",
        "general_suicide_context": "Suicide is meaningfully discussed without personal thoughts or actions.",
    },
    "Ideation": {
        "death_wish": "A passive wish to die, disappear, or not wake up.",
        "suicidal_thought": "Active, self-directed suicidal thinking or desire.",
        "hypothetical_method": "A suicide method is mentioned without concrete action or preparation.",
    },
    "Behavior": {
        "suicide_plan": "A concrete method, place, time, or planned suicidal act.",
        "preparation": "Acquiring means, writing a note, or preparing for death.",
        "self_harm": "Self-inflicted harm without an established fatal attempt.",
    },
    "Attempt": {
        "attempt_report": "An explicit current or historical suicide attempt.",
        "suicidal_action": "An action carried out with stated suicidal intent.",
        "attempt_aftermath": "Survival, rescue, hospitalization, or treatment after an attempt.",
    },
    "Unsupportive": {
        "unrelated_topic": "An ordinary topic with no suicide or self-harm meaning.",
        "benign_language": "A figurative or harmless use of words such as die or kill.",
        "non_suicidal_distress": "Distress is expressed without suicide or self-harm meaning.",
    },
}


def _make_evidence_collector_prompt(label_name: str) -> str:
    type_lines = "\n".join(
        f"- {evidence_type}: {description}"
        for evidence_type, description in EVIDENCE_TYPE_DEFINITIONS[label_name].items()
    )
    allowed_types = " | ".join(EVIDENCE_TYPE_DEFINITIONS[label_name])
    return f"""You are the {label_name} Evidence Collector for a suicide-risk classification task.

Evidence family: {label_name}

Relevant label description:
{LABEL_DEFINITIONS[label_name]}

Task:
Collect only explicit text evidence that could help a later judge distinguish this evidence family from the other labels. You do not classify the text, vote for a label, or make a final decision.

Evidence types and how to interpret them:
{type_lines}

Input format:
You will receive one user message:

Text to classify:
<text>

Output format:
Return only valid JSON with exactly this shape:
{{"evidence":[{{"quote":"relevant excerpt or concise paraphrase","type":"{allowed_types}","subject":"author | other | general | unclear","time":"past | current | future | unclear | na","observation":"what the quote factually shows"}}]}}

Rules:
- Return zero to three evidence items. If none is supported, return {{"evidence":[]}}.
- Each quote must be a relevant excerpt or concise paraphrase of at most 100 words; exact substring matching is not required.
- Use only the three evidence types defined for this collector.
- subject must be author, other, general, or unclear.
- time must be past, current, future, unclear, or na.
- observation must explain how to interpret the quote in at most 100 words.
- Do not output answer, label_name, prediction, confidence, citations, review status, or extra keys.
- Do not infer unstated facts or recommend a final label."""


INDICATOR_EVIDENCE_COLLECTOR_PROMPT = _make_evidence_collector_prompt("Indicator")
IDEATION_EVIDENCE_COLLECTOR_PROMPT = _make_evidence_collector_prompt("Ideation")
BEHAVIOR_EVIDENCE_COLLECTOR_PROMPT = _make_evidence_collector_prompt("Behavior")
ATTEMPT_EVIDENCE_COLLECTOR_PROMPT = _make_evidence_collector_prompt("Attempt")
UNSUPPORTIVE_EVIDENCE_COLLECTOR_PROMPT = _make_evidence_collector_prompt("Unsupportive")


@dataclass(frozen=True)
class SpecialistPromptConfig:
    """Maps one specialist agent name to one label and one static prompt."""

    agent_name: str
    label_name: str
    system_prompt: str
    input_format: str = "Text to classify:\\n\\n{text}"
    output_format: str = '{"answer": true|false}'
    parallel_group: str = "five_label_specialists"
    role_suffix: str = "specialist"
    produces_answer: bool = True


@dataclass(frozen=True)
class WorkflowModeConfig:
    """A reusable prompt/schema set for the same parallel graph."""

    name: str
    specialist_prompts: dict[str, SpecialistPromptConfig]
    specialist_schema: type[Any]
    judge_schema: type[Any]
    judge_system_prompt: Callable[[], str]
    judge_user_prompt: Callable[[str, dict[str, dict[str, Any]]], str]
    agent_labels: tuple[str, ...] = FINAL_LABELS
    evidence_mode: bool = False


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
    "Unsupportive Agent": SpecialistPromptConfig(
        agent_name="Unsupportive Agent",
        label_name="Unsupportive",
        system_prompt=UNSUPPORTIVE_SPECIALIST_PROMPT,
    ),
}

LABEL_TO_AGENT_NAME: dict[str, str] = {
    config.label_name: config.agent_name for config in SPECIALIST_AGENT_PROMPTS.values()
}

EVIDENCE_COLLECTOR_PROMPTS: dict[str, SpecialistPromptConfig] = {
    f"{label} Evidence Collector": SpecialistPromptConfig(
        agent_name=f"{label} Evidence Collector",
        label_name=label,
        system_prompt=_make_evidence_collector_prompt(label),
        output_format=(
            '{"evidence":[{"quote":string,"type":string,"subject":string,'
            '"time":string,"observation":string}]}'
        ),
        parallel_group="five_evidence_collectors",
        role_suffix="collector",
        produces_answer=False,
    )
    for label in FINAL_LABELS
}

EVIDENCE_LABEL_TO_AGENT_NAME: dict[str, str] = {
    config.label_name: config.agent_name for config in EVIDENCE_COLLECTOR_PROMPTS.values()
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
        "Use the original text and the five specialist answers to choose exactly one final label.\n\n"
        "The five specialist agents run in parallel before you receive their answers:\n"
        f"Label definitions:\n{definitions}\n\n"
        "Input format:\n"
        "You receive the original text plus a JSON object keyed by all five specialist labels.\n\n"
        "Output format:\n"
        "Return only valid JSON with exactly this shape:\n"
        '{"label_name": "<label_name>"}\n'
    )


def judge_user_prompt(text: str, specialist_answers: dict[str, bool]) -> str:
    payload = {label: specialist_answers[label] for label in FINAL_LABELS}
    return (
        "Original text:\n"
        f"{text}\n\n"
        "Specialist answers:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def judge_user_prompt_from_outputs(text: str, specialist_outputs: dict[str, dict[str, Any]]) -> str:
    answers = {label: bool(specialist_outputs[label]["answer"]) for label in FINAL_LABELS}
    return judge_user_prompt(text, answers)


def evidence_judge_system_prompt() -> str:
    definitions = "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in FINAL_LABELS)
    evidence_guide = "\n".join(
        f"{label} collector:\n"
        + "\n".join(
            f"- {evidence_type}: {description}"
            for evidence_type, description in EVIDENCE_TYPE_DEFINITIONS[label].items()
        )
        for label in FINAL_LABELS
    )
    return (
        "You are the final judge for a five-label suicide-risk classification task.\n"
        "Use the original text and the validated outputs from five parallel evidence collectors "
        "to choose exactly one final label.\n\n"
        f"Label descriptions:\n{definitions}\n\n"
        f"Evidence types and how to interpret them:\n{evidence_guide}\n\n"
        "Decision rules:\n"
        "- The original text is authoritative; collector observations are compact aids grounded in it.\n"
        "- Collectors do not predict, vote, or assign confidence. A packet is not a vote for its label.\n"
        "- Empty evidence means that collector found no supported item; it does not prove a label false.\n"
        "- Repeated evidence across packets is still one underlying text fact, not multiple votes.\n"
        "- Distinguish the subject, timing, thought, preparation/self-harm, and attempt status.\n"
        "- Choose Unsupportive only when the text does not meaningfully concern suicide, self-harm, "
        "suicidal thinking, suicidal behavior, or suicide attempts.\n"
        "- Suicide-related discussion, another person's suicidality, or an explicit denial of the "
        "author's own suicidal intent supports Indicator rather than Unsupportive.\n\n"
        "Input format:\n"
        "You receive the original text plus a JSON object keyed by Indicator, Ideation, Behavior, "
        "Attempt, and Unsupportive. Each value contains only an evidence array.\n\n"
        "Output format:\n"
        "Return only valid JSON with exactly this shape:\n"
        '{"label_name": "<label_name>"}\n'
    )


def evidence_judge_user_prompt(text: str, collector_outputs: dict[str, dict[str, Any]]) -> str:
    payload = {label: collector_outputs[label] for label in FINAL_LABELS}
    return (
        "Original text:\n"
        f"{text}\n\n"
        "Validated evidence packets:\n"
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
    "evidence": WorkflowModeConfig(
        name="evidence",
        specialist_prompts=EVIDENCE_COLLECTOR_PROMPTS,
        specialist_schema=EvidenceCollectorAnswer,
        judge_schema=JudgeAnswer,
        judge_system_prompt=evidence_judge_system_prompt,
        judge_user_prompt=evidence_judge_user_prompt,
        agent_labels=FINAL_LABELS,
        evidence_mode=True,
    ),
}

WORKFLOW_MODE_NAMES = tuple(WORKFLOW_MODES)
FRAMEWORK_1_PARALLEL_NO_EVIDENCE_MODE_NAMES = ("bool",)


def get_workflow_mode(mode: str | WorkflowModeConfig) -> WorkflowModeConfig:
    if isinstance(mode, WorkflowModeConfig):
        return mode
    try:
        return WORKFLOW_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow mode {mode!r}. Expected one of: {', '.join(WORKFLOW_MODE_NAMES)}") from exc
