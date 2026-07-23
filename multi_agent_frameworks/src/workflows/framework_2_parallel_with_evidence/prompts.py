"""Framework 2 evidence-collector and judge prompt API."""

from workflows.shared.parallel.prompts import (
    ATTEMPT_EVIDENCE_COLLECTOR_PROMPT,
    BEHAVIOR_EVIDENCE_COLLECTOR_PROMPT,
    EVIDENCE_COLLECTOR_PROMPTS,
    EVIDENCE_LABEL_TO_AGENT_NAME,
    EVIDENCE_TYPE_DEFINITIONS,
    INDICATOR_EVIDENCE_COLLECTOR_PROMPT,
    IDEATION_EVIDENCE_COLLECTOR_PROMPT,
    UNSUPPORTIVE_EVIDENCE_COLLECTOR_PROMPT,
    evidence_judge_system_prompt,
    evidence_judge_user_prompt,
)

__all__ = [name for name in globals() if not name.startswith("_")]
