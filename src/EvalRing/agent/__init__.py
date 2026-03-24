"""
Agent module for EvalRing.
"""

from .base import BaseAgent, AgentResponse
from .classification import (
    ClassificationPrediction,
    aggregate_base_vs_rest_probabilities,
    normalize_probability_distribution,
    parse_json_object,
    resolve_classification_prediction,
)
from .implementations import MockAgent, RuleBasedAgent, OpenAIAgent

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "ClassificationPrediction",
    "aggregate_base_vs_rest_probabilities",
    "normalize_probability_distribution",
    "parse_json_object",
    "resolve_classification_prediction",
    "MockAgent",
    "RuleBasedAgent",
    "OpenAIAgent",
]
