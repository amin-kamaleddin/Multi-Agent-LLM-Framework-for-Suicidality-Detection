"""Configuration contract for framework 2."""

from workflows.framework_1_parallel_no_evidence.config import AGENT_CONFIG_KEYS

WORKFLOW_NAME = "framework_2_parallel_with_evidence"
DEFAULT_MODE = "evidence"
CONFIG_KEYS = (
    "model",
    "temperature",
    "max_completion_tokens",
    "max_retries",
    "mode",
    "service_tier",
    "fake_llm",
    "agents",
)
