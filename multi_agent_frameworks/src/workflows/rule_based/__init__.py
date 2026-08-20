"""Deterministic rule-based suicide-related post classification workflow."""

from .config import WORKFLOW_NAME
from .workflow import RuleBasedWorkflow

__all__ = ["RuleBasedWorkflow", "WORKFLOW_NAME"]
