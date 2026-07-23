"""Framework 1: parallel decision-making specialists and a final judge."""

from .config import WORKFLOW_NAME
from .graph import MultiAgentWorkflow

__all__ = ["MultiAgentWorkflow", "WORKFLOW_NAME"]
