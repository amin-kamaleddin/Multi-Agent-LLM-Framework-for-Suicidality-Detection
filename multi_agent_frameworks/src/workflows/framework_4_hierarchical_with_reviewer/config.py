"""Configuration contract for framework 4."""

WORKFLOW_NAME = "framework_4_hierarchical_with_reviewer"
DEFAULT_MODE = "bool"
MODE_NAMES = ("bool", "reasoned")
CONFIG_KEYS = (
    "model",
    "temperature",
    "max_completion_tokens",
    "max_retries",
    "mode",
    "service_tier",
    "fake_llm",
)
