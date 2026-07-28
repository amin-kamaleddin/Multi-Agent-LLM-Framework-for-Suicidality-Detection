"""Single-model suicide-risk classification workflow."""

from .config import WORKFLOW_NAME
from .workflow import SingleAgentWorkflow

__all__ = ["SingleAgentWorkflow", "WORKFLOW_NAME"]
