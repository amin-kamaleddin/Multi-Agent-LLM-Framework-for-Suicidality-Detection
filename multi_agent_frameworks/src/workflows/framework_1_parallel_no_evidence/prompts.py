"""Framework 1 prompt API."""

from workflows.shared.parallel.prompts import (
    ATTEMPT_SPECIALIST_PROMPT,
    BEHAVIOR_SPECIALIST_PROMPT,
    FRAMEWORK_1_PARALLEL_NO_EVIDENCE_MODE_NAMES,
    INDICATOR_SPECIALIST_PROMPT,
    IDEATION_SPECIALIST_PROMPT,
    LABEL_TO_AGENT_NAME,
    SPECIALIST_AGENT_PROMPTS,
    SpecialistPromptConfig,
    UNSUPPORTIVE_SPECIALIST_PROMPT,
    judge_system_prompt,
    judge_user_prompt,
    specialist_system_prompt,
    specialist_user_prompt,
)

__all__ = [name for name in globals() if not name.startswith("_")]
