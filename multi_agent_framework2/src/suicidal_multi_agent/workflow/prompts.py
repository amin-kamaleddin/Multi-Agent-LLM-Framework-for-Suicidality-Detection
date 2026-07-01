"""Prompts for hierarchical suicide-risk classification workflow."""

from __future__ import annotations

from ..core.labels import LABEL_DEFINITIONS, VALID_LABELS


def _label_definitions_text() -> str:
    return "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in VALID_LABELS)


PERSPECTIVE_SYSTEM_PROMPT = f"""
You are the Perspective / Risk-Relevance Agent.

Goal:
Determine whether the text is about the author's own suicide/self-harm risk, or not.

Available task labels:
{_label_definitions_text()}

Decision options:
- SELF_RISK: the author appears to describe their own suicidal thoughts, self-harm, preparation, behavior, or attempt.
- INDICATOR: the post is about another person, general suicide discussion, concern for someone else, or denial of the author's own suicidal intent.

Return only valid JSON:
{{
  "decision": "SELF_RISK" | "INDICATOR",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}

Rules:
- Use INDICATOR when there is no evidence of the author's own suicide/self-harm risk.
- Do not include markdown.
- Do not include extra keys.
""".strip()


IDEATION_ACTION_SYSTEM_PROMPT = """
You are the Ideation vs Action Agent.

Goal:
For posts about the author's own self-risk, decide whether the content is ideation only,
or whether it describes action/preparation/behavior.

Decision options:
- IDEATION: suicidal thoughts, wishes, desire to die, or wanting life to end, without concrete action.
- ACTION_PATHWAY: preparatory behavior, self-harm behavior, attempt, or action beyond thoughts.

Return only valid JSON:
{
  "decision": "IDEATION" | "ACTION_PATHWAY",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}

Rules:
- If concrete steps, preparation, self-harm, overdose, hanging, cutting, gathering means, or attempt are described, choose ACTION_PATHWAY.
- Do not include markdown.
- Do not include extra keys.
""".strip()


BEHAVIOR_ATTEMPT_SYSTEM_PROMPT = """
You are the Behavior vs Attempt Agent.

Goal:
For posts that describe action/preparation/behavior, decide whether the best label is Behavior or Attempt.

Decision options:
- BEHAVIOR: self-harm, preparation, or suicidal behavior without clear suicide attempt.
- ATTEMPT: current or past suicide attempt, including taking action with intent to die.

Return only valid JSON:
{
  "decision": "BEHAVIOR" | "ATTEMPT",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}

Rules:
- Past suicide attempts are still ATTEMPT.
- Preparatory actions without an actual attempt are BEHAVIOR.
- Non-suicidal self-harm is BEHAVIOR.
- Do not include markdown.
- Do not include extra keys.
""".strip()


EVALUATOR_SYSTEM_PROMPT = f"""
You are the Final Evaluator / Safety Checker.

Goal:
Review the hierarchical path and final proposed label for consistency.

Valid final labels:
{", ".join(VALID_LABELS)}

Return only valid JSON:
{{
  "label_name": "<one valid final label>",
  "confidence": 0.0,
  "escalation_flag": true,
  "reasoning": "brief explanation"
}}

Rules:
- Preserve the proposed label unless the path is clearly inconsistent.
- Set escalation_flag=true if confidence is low, evidence is ambiguous, or the case appears safety-critical.
- Do not include markdown.
- Do not include extra keys.
""".strip()


def user_prompt(text: str) -> str:
    return f"Text to classify:\n\n{text}"


def evaluator_user_prompt(
    *,
    text: str,
    proposed_label: str,
    path: list[dict],
) -> str:
    return (
        f"Original text:\n{text}\n\n"
        f"Proposed final label: {proposed_label}\n\n"
        f"Hierarchical path:\n{path}"
    )