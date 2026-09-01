"""
LLM-as-a-Judge evaluation module for single_agent_frameworks.

Inspired by deepeval's GEval architecture, this module provides a generic
framework for using LLMs as evaluators/judges of agent responses.

Key components:
- JudgeMetric: Define what to measure (criteria → CoT steps → score)
- Rubric: Define scoring anchors and grade bands
- LLMJudge: The LLM backend that performs the judging
- LLMJudgeEvaluator: Orchestrates evaluation over datasets
"""

from .rubric import Rubric, RubricLevel, ScoringCriteria
from .schema import JudgeVerdict, EvalSteps
from .template import JudgeTemplate
from .metric import JudgeMetric
from .judge import LLMJudge, OpenAIJudge
from .evaluator import LLMJudgeEvaluator

__all__ = [
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
