"""Framework 3: hierarchical classification without a secondary reviewer."""

from .config import WORKFLOW_NAME
from .graph import MultiAgentWorkflow

__all__ = ["MultiAgentWorkflow", "WORKFLOW_NAME"]
