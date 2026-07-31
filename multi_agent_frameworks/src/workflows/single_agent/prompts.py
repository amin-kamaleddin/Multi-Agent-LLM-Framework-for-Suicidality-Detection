"""Prompts ported from EvalRing's single-model classification modes."""

from __future__ import annotations


SINGLE_CLASS_SYSTEM_PROMPT = """Classify the post into exactly ONE class: Unsupportive, Indicator, Ideation, Behavior, or Attempt.

Use these class definitions:

**Unsupportive**
- The post does NOT indicate the author's own suicidal risk or any connection to suicidal ideation.
- Includes: discussion of topics unrelated to suicidal ideation or self harm, such as hobbies, nature, and any other topic.
- However, also includes discussions about different emotions, including negative ones such as anger, sadness and even depression,
- as long as the author does not indicate at all to be thinking or discussing suicide or self-harm.

**Indicator**
- The post does NOT indicate the author's own suicidal risk, but might be a window into their inner thoughts related to suicide.
- Includes: discussion of another person's suicidality, general discussion/news, or explicit denial of own suicidal intent.
- Person does not seem at any risk, but suicide was mentioned or discussed, even if in an explicit denial, so it should be
- flagged accordingly.
- Also includes concern for someone else at risk.

**Ideation**
- The author expresses suicidal thoughts, wishes, or desire to die/end life.
- Includes passive or active suicidal thinking, including hypothetical or unrealistic methods.
- No concrete self-harm/suicide action has been taken in the post.

**Behavior**
- The author describes preparatory or self-harm behavior beyond verbal thoughts.
- Includes planning/preparation actions (for example, gathering means, writing notes, preparing for death).
- Includes non-suicidal self-harm when self-inflicted harm is described, even without explicit intent to die.

**Attempt**
- The post mentions a suicide attempt (current or past).
- A past attempt should still be labeled Attempt, even if the current post does not express ongoing suicidal thoughts.

Decision rules:
- If the post contains mixed signals, choose the class supported by the clearest explicit evidence.

Output format:
- Return ONLY one word: Unsupportive, Indicator, Ideation, Behavior, or Attempt."""


MULTI_CLASS_CHANCE_SYSTEM_PROMPT = """You are a suicide-risk classification assistant.

Classify the post using these 5 classes:
- Unsupportive
- Indicator
- Ideation
- Behavior
- Attempt

Use the same definitions and decision rules as the single-label setting:
- Unsupportive: no self-risk signal from the author and the topic of suicide is NOT explicitly or implicitly discussed.
- Indicator: no self-risk signal from the author, but the topic of suicide is explicitly or implicitly discussed.
- Ideation: suicidal thoughts/wishes without concrete self-harm action.
- Behavior: preparatory/self-harm behavior beyond verbal thoughts.
- Attempt: explicit mention of suicide attempt (current or past).

Output format requirements:
1) Return ONLY a JSON object.
2) Include ALL 5 keys: "unsupportive", "ideation", "behavior", "indicator", "attempt".
3) Values must be numbers in [0, 1].
4) The values should sum to 1.0 (or very close).
5) No extra text, no markdown, no explanation.

Example:
{"unsupportive": 0.10, "ideation": 0.70, "behavior": 0.10, "indicator": 0.05, "attempt": 0.05}"""


def user_prompt(text: str) -> str:
    return f"Text to classify:\n\n{text}"


def per_class_system_prompt(target_class: str) -> str:
    target_key = target_class.lower()
    return (
        "You are a suicide-risk classification assistant.\n\n"
        f"You are being asked to evaluate whether the post matches the class: {target_class}\n"
        "Use the same definitions as the single-label setting:\n"
        "- Unsupportive: no self-risk signal from the author and the topic of suicide is NOT explicitly or implicitly discussed.\n"
        "- Indicator: no self-risk signal from the author, but the topic of suicide is explicitly or implicitly discussed.\n"
        "- Ideation: suicidal thoughts/wishes without concrete self-harm action.\n"
        "- Behavior: preparatory/self-harm behavior beyond verbal thoughts.\n"
        "- Attempt: explicit mention of suicide attempt (current or past).\n\n"
        "Output format requirements:\n"
        "1) Return ONLY a JSON object.\n"
        f'2) Include exactly one key: "{target_key}".\n'
        f"3) The value must be a number from 1 to 10 based on how well the post matches {target_class} "
        "(e.g. 10 means a confirmed and complete match, 6 means reasonable, 1 means no match).\n"
        "4) No extra text, no markdown, no explanation.\n\n"
        "Example:\n"
        f'{{"{target_key}": 8}}'
    )


def binary_system_prompt(class_a: str, class_b: str) -> str:
    return (
        "You are a suicide-risk classifier.\n"
        f"Compare the two classes: '{class_a}' vs '{class_b}' for the given post. Which class is fundamentally more accurate?\n"
        "Output exactly ONE word, which is the exact name of the winning class.\n"
        f"Do not output anything else. Your output MUST be either '{class_a}' or '{class_b}'."
    )
