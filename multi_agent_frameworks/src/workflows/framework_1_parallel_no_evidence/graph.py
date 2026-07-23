"""Public graph entry point for parallel decision-making specialists."""

from __future__ import annotations

from typing import Any

from workflows.shared.parallel import MultiAgentWorkflow as _ParallelWorkflow


class MultiAgentWorkflow(_ParallelWorkflow):
    """Five parallel label specialists followed by one final judge."""

    def __init__(self, *args: Any, mode: str = "bool", **kwargs: Any) -> None:
        if mode not in {"bool", "reasoned"}:
            raise ValueError("framework_1_parallel_no_evidence mode must be 'bool' or 'reasoned'")
        super().__init__(*args, mode=mode, **kwargs)


ParallelNoEvidenceWorkflow = MultiAgentWorkflow
