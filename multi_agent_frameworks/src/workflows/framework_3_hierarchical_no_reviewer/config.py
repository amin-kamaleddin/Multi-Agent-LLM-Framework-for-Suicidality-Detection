"""Configuration contract for framework 3."""

WORKFLOW_NAME = "framework_3_hierarchical_no_reviewer"
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
