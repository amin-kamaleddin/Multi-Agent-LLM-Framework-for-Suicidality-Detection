"""Configuration contract for framework 4."""

WORKFLOW_NAME = "framework_4_hierarchical_with_reviewer"
DEFAULT_MODE = "bool"
MODE_NAMES = ("bool", "reasoned")
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_MARGIN_THRESHOLD = 0.25
DEFAULT_REVIEW_KNOWN_BOUNDARIES_ONLY = False
CONFIG_KEYS = (
    "model",
    "temperature",
    "max_completion_tokens",
    "max_retries",
    "mode",
    "service_tier",
    "fake_llm",
    "confidence_threshold",
    "margin_threshold",
    "review_known_boundaries_only",
)
