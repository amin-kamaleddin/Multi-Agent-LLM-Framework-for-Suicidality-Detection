"""Configuration contract for the single-agent workflow."""

WORKFLOW_NAME = "single_agent"
DEFAULT_MODE = "single-class"
MODE_NAMES = (
    "single-class",
    "multi-class-chance",
    "per-class-score",
    "base-vs-rest-binary",
)
CONFIG_KEYS = (
    "model",
    "temperature",
    "max_completion_tokens",
    "max_retries",
    "mode",
    "service_tier",
    "fake_llm",
    "base_class",
)
