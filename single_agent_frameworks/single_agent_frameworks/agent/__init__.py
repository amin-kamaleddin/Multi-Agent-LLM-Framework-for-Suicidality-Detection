"""
Agent module for single_agent_frameworks.
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
from .multi_role import MultiRoleHostOrchestrator, RoleConfig

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "ClassificationPrediction",
    "aggregate_base_vs_rest_probabilities",
    "normalize_probability_distribution",
    "parse_json_object",
    "resolve_classification_prediction",
    "MultiRoleHostOrchestrator",
    "RoleConfig",
    "MockAgent",
    "RuleBasedAgent",
    "OpenAIAgent",
]
