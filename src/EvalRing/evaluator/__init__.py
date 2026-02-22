"""
Evaluator module for EvalRing.
"""

from .base import BaseEvaluator, EvaluationResult, EvaluationMetrics
from .implementations import ClassificationEvaluator
from .llm_judge import (
    Rubric,
    RubricLevel,
    ScoringCriteria,
    JudgeVerdict,
    EvalSteps,
    JudgeTemplate,
    JudgeMetric,
    LLMJudge,
    OpenAIJudge,
    LLMJudgeEvaluator,
)

__all__ = [
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
