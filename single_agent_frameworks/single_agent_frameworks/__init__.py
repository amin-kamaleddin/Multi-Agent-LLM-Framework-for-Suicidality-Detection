"""
single_agent_frameworks: a unified evaluation framework for agents across LLMs and agent versions.

This package provides a comprehensive framework for evaluating AI agents with
standardized metrics, datasets, and evaluation protocols.
"""

__version__ = "0.1.0"

from .dataset import BaseDataset, DataSample, JSONDataset, CSVDataset, DataFrameDataset
from .agent import BaseAgent, AgentResponse, MockAgent, RuleBasedAgent, OpenAIAgent
from .evaluator import (
    BaseEvaluator, EvaluationResult, EvaluationMetrics, ClassificationEvaluator,
    Rubric, RubricLevel, ScoringCriteria,
    JudgeVerdict, EvalSteps, JudgeTemplate,
    JudgeMetric, LLMJudge, OpenAIJudge, LLMJudgeEvaluator,
)

__all__ = [
    "BaseDataset",
    "DataSample",
    "JSONDataset",
    "CSVDataset",
    "DataFrameDataset",
    "BaseAgent",
    "AgentResponse",
    "MockAgent",
    "RuleBasedAgent",
    "OpenAIAgent",
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "ClassificationEvaluator",
    # LLM-as-a-Judge
    "Rubric",
    "RubricLevel",
    "ScoringCriteria",
    "JudgeVerdict",
    "EvalSteps",
    "JudgeTemplate",
    "JudgeMetric",
    "LLMJudge",
    "OpenAIJudge",
    "LLMJudgeEvaluator",
]
