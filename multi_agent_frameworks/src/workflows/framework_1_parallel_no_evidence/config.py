"""Configuration contract for framework 1."""

WORKFLOW_NAME = "framework_1_parallel_no_evidence"
DEFAULT_MODE = "bool"
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
AGENT_CONFIG_KEYS = ("agent_name", "model", "temperature", "max_completion_tokens")
