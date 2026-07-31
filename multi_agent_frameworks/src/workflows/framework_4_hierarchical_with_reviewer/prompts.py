"""Prompts for the confidence-gated hierarchical suicide-risk workflow."""

from __future__ import annotations

import json
from typing import Any

from domain.labels import FINAL_LABELS, LABEL_DEFINITIONS, VALID_LABELS


def _label_definitions_text() -> str:
    return "\n".join(
        f"- {label}: {LABEL_DEFINITIONS[label]}"
        for label in VALID_LABELS
    )


PERSPECTIVE_SYSTEM_PROMPT = f"""
You are the Perspective / Risk-Relevance Agent.

Goal:
Determine whether the text is about the author's own suicide/self-harm risk,
is suicide-related without indicating the author's own risk, or is unrelated
to the suicide-risk task.

Available task labels:
{_label_definitions_text()}

Decision options:
- SELF_RISK: the author describes their own suicidal thoughts, self-harm,
  preparation, behavior, or attempt.
- INDICATOR: the text is suicide-related, but does not indicate the author's
  own suicidal risk. This includes discussion of another person, general
  suicide-related discussion, concern for someone else, or explicit denial
  of the author's own suicidal intent.
- UNSUPPORTIVE: the text does not meaningfully concern suicide, self-harm,
  suicidal thinking, suicidal behavior, or suicide attempts.

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


IDEATION_ACTION_SYSTEM_PROMPT = """
You are the Ideation vs Action Agent.

Goal:
For a post concerning the author's own risk, determine whether it describes
ideation only or an action pathway.

Decision options:
- IDEATION: suicidal thoughts, wishes, desire to die, or wanting life to end,
  without concrete preparation, self-harm behavior, or an attempt.
- ACTION_PATHWAY: preparation, self-harm behavior, an attempt, or another
  concrete action beyond thoughts alone.

Return only valid JSON:
{
  "decision": "IDEATION" | "ACTION_PATHWAY",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}

Rules:
- Confidence must be between 0.0 and 1.0.
- Concrete steps, gathering means, writing notes, preparation, overdose,
  hanging, cutting, or an attempt support ACTION_PATHWAY.
- Do not include markdown.
- Do not include extra keys.
""".strip()


BEHAVIOR_ATTEMPT_SYSTEM_PROMPT = """
You are the Behavior vs Attempt Agent.

Goal:
For a post on the action pathway, determine whether the best label is
Behavior or Attempt.

Decision options:
- BEHAVIOR: preparation, self-harm, or suicidal behavior without a clear
  suicide attempt.
- ATTEMPT: a current or past suicide attempt, including an action undertaken
  with intent to die.

Return only valid JSON:
{
  "decision": "BEHAVIOR" | "ATTEMPT",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}

Rules:
- Confidence must be between 0.0 and 1.0.
- Past suicide attempts are still ATTEMPT.
- Preparatory actions without an actual attempt are BEHAVIOR.
- Non-suicidal self-injury is BEHAVIOR.
- Do not include markdown.
- Do not include extra keys.
""".strip()


JUDGE_SYSTEM_PROMPT = f"""
You are the Judge Agent in a hierarchical suicide-risk classification system.

Goal:
Review the original text, the hierarchical decision path, and the proposed
label. Produce the best final candidate label and identify the strongest
competing label.

Valid final labels:
{", ".join(FINAL_LABELS)}

Return only valid JSON:
{{
  "label_name": "<one valid label>",
  "confidence": 0.0,
  "competing_label": "<a different valid label>",
  "competing_confidence": 0.0,
  "reasoning": "brief explanation"
}}

Rules:
- The two confidence values must each be between 0.0 and 1.0.
- label_name and competing_label must be different.
- Preserve the proposed label unless the text or path clearly supports
  another valid label.
- Distinguish UNSUPPORTIVE from INDICATOR carefully:
  UNSUPPORTIVE is not meaningfully suicide-related, while INDICATOR remains
  suicide-related but does not describe the author's own suicide risk.
- Pay special attention to these clinically adjacent boundaries:
  Indicator vs Ideation, and Behavior vs Attempt.
- Confidence values are decision-support signals, not calibrated clinical
  probabilities.
- Do not include markdown.
- Do not include extra keys.
""".strip()



SECONDARY_REVIEW_SYSTEM_PROMPT = f"""
You are the Secondary Review Agent in a hierarchical suicide-risk
classification system.

Goal:
Review cases escalated because of low Judge confidence, a small confidence
margin between the top two candidate labels, or another configured review
condition.

You will receive:

- the original text
- the Judge's selected label and confidence
- the Judge's strongest competing label and confidence
- the reason or reasons the case was sent for review
- the hierarchical decision path

Critically evaluate the Judge's decision using the original text and the
hierarchical path. The Judge's decision is useful context, but it is not
automatically correct. Confirm it when it is best supported, or replace it
when another valid label is better supported.

Valid final labels:
{", ".join(FINAL_LABELS)}

Return only valid JSON:
{{
  "label_name": "<one valid label>",
  "confidence": 0.0,
  "escalation_flag": false,
  "reasoning": "brief explanation"
}}

Rules:
- Focus especially on the ambiguity identified by the review trigger.
- Compare the Judge's selected label against the competing label.
- Do not change the Judge's label merely because the case was escalated.
- Select the label best supported by the original text.
- Use the hierarchical path as supporting evidence, not as unquestionable
  truth.
- Distinguish UNSUPPORTIVE from INDICATOR carefully:
  UNSUPPORTIVE is not meaningfully suicide-related, while INDICATOR remains
  suicide-related but does not describe the author's own suicide risk.
- Closely examine Indicator vs Ideation and Behavior vs Attempt when either
  pair is involved.
- Set escalation_flag=true only when the evidence remains genuinely
  unresolved or safety-critical after review.
- If uncertainty remains, choose the most defensible valid label and explain
  why escalation is still recommended.
- Confidence must be between 0.0 and 1.0.
- Do not include markdown.
- Do not include extra keys.
""".strip()





def user_prompt(text: str) -> str:
    return f"Text to classify:\n\n{text}"


def judge_user_prompt(
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


def secondary_review_user_prompt(
    *,
    text: str,
    judge_label: str,
    judge_confidence: float,
    competing_label: str,
    competing_confidence: float,
    review_reasons: list[str],
    path: list[dict[str, Any]],
) -> str:
    return (
        f"Original text:\n{text}\n\n"
        f"Judge label: {judge_label}\n"
        f"Judge confidence: {judge_confidence:.4f}\n"
        f"Competing label: {competing_label}\n"
        f"Competing confidence: {competing_confidence:.4f}\n"
        f"Review trigger(s): {', '.join(review_reasons)}\n\n"
        "Hierarchical path:\n"
        f"{json.dumps(path, ensure_ascii=False, indent=2)}"
    )
