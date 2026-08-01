"""Prompts for hierarchical suicide-risk classification workflow."""

from __future__ import annotations
from domain.labels import FINAL_LABELS, LABEL_DEFINITIONS
import json
from typing import Any


def _label_definitions_text() -> str:
    return "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in FINAL_LABELS)


PERSPECTIVE_SYSTEM_PROMPT = f"""
You are the Perspective / Risk-Relevance Agent.

Goal:
Determine whether the text is about the author's own suicide/self-harm risk, or not.

Available task labels:
{_label_definitions_text()}

Decision options:
- SELF_RISK: the author describes their own suicidal thoughts, self-harm,
  preparation, behavior, or attempt.
- INDICATOR: {LABEL_DEFINITIONS["Indicator"]}
- UNSUPPORTIVE: {LABEL_DEFINITIONS["Unsupportive"]}

IMPORTANT:
This agent performs ONLY the FIRST decision in the hierarchy.

The value of "decision" MUST be EXACTLY one of:
- SELF_RISK
- INDICATOR
- UNSUPPORTIVE

Never output:
- IDEATION
- ACTION_PATHWAY
- BEHAVIOR
- ATTEMPT

Those labels belong to later agents and are INVALID for this agent.

Return only valid JSON:
{{
  "decision": "SELF_RISK" | "INDICATOR" | "UNSUPPORTIVE",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}

Rules:
- Do not use INDICATOR merely because SELF_RISK is absent.
- Use INDICATOR only when the text is still meaningfully suicide-related.
- Use UNSUPPORTIVE when the text does not match the suicide-risk task.
- Do not include markdown.
- Do not include extra keys.
""".strip()


IDEATION_ACTION_SYSTEM_PROMPT = f"""
You are the Ideation vs Action Agent.

Goal:
For posts about the author's own self-risk, decide whether the content is ideation only,
or whether it describes action/preparation/behavior.

Decision options:
- IDEATION: {LABEL_DEFINITIONS["Ideation"]}
- ACTION_PATHWAY: preparatory behavior, self-harm behavior, attempt, or action beyond thoughts.

Return only valid JSON:
{{
  "decision": "IDEATION" | "ACTION_PATHWAY",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}

Rules:
- If concrete steps, preparation, self-harm, overdose, hanging, cutting, gathering means, or attempt are described, choose ACTION_PATHWAY.
- Do not include markdown.
- Do not include extra keys.
""".strip()


BEHAVIOR_ATTEMPT_SYSTEM_PROMPT = f"""
You are the Behavior vs Attempt Agent.

Goal:
For posts that describe action/preparation/behavior, decide whether the best label is Behavior or Attempt.

Decision options:
- BEHAVIOR: {LABEL_DEFINITIONS["Behavior"]}
- ATTEMPT: {LABEL_DEFINITIONS["Attempt"]}

Return only valid JSON:
{{
  "decision": "BEHAVIOR" | "ATTEMPT",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}

Rules:
- Past suicide attempts are still ATTEMPT.
- Preparatory actions without an actual attempt are BEHAVIOR.
- Non-suicidal self-harm is BEHAVIOR.
- Do not include markdown.
- Do not include extra keys.
""".strip()


EVALUATOR_SYSTEM_PROMPT = f"""
You are the final judge for a five-label suicide-risk classification task.

Task:
Use the original text, the hierarchical decision path, and the proposed label
to choose exactly one final label.

Label definitions:
{_label_definitions_text()}

Input format:
You receive:
- the original text,
- the proposed label produced by the hierarchy,
- and the full hierarchical decision path.

Output format:
Return only valid JSON with exactly these keys:
{{
  "label_name": "<one of: {', '.join(FINAL_LABELS)}>",
  "confidence": 0.0,
  "escalation_flag": false,
  "reasoning": "brief explanation"
}}

Decision rules:
- Base the final decision primarily on the original text.
- Use the hierarchical path and proposed label as supporting evidence.
- Preserve the proposed label unless the original text or path clearly supports
  another valid label.
- Distinguish Unsupportive from Indicator carefully:
  Unsupportive is not meaningfully suicide-related, while Indicator remains
  suicide-related but does not describe the author's own suicide risk.
- Pay special attention to the adjacent boundaries:
  Indicator vs Ideation, and Behavior vs Attempt.
- Confidence must be between 0.0 and 1.0 and should reflect certainty that the
  selected label is the best final classification.
- Confidence is a decision-support signal, not a calibrated clinical probability.
- Set escalation_flag=true when confidence is low or the evidence is materially ambiguous.
- Do not include markdown.
- Do not include extra keys.
""".strip()


def user_prompt(text: str) -> str:
    return f"Text to classify:\n\n{text}"


# def evaluator_user_prompt(
#     *,
#     text: str,
#     proposed_label: str,
#     path: list[dict],
# ) -> str:
#     return (
#         f"Original text:\n{text}\n\n"
#         f"Proposed final label: {proposed_label}\n\n"
#         f"Hierarchical path:\n{path}"
#     )

def evaluator_user_prompt(
    *,
    text: str,
    proposed_label: str,
    path: list[dict[str, Any]],
) -> str:
    return (
        f"Original text:\n{text}\n\n"
        f"Proposed label from the hierarchy: {proposed_label}\n\n"
        "Hierarchical path:\n"
        f"{json.dumps(path, ensure_ascii=False, indent=2)}"
    )
