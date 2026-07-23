"""Public graph entry point for parallel evidence collection."""

from __future__ import annotations

from typing import Any

from workflows.shared.parallel import MultiAgentWorkflow as _ParallelWorkflow


class MultiAgentWorkflow(_ParallelWorkflow):
    """Five parallel evidence collectors followed by one final judge."""

    def __init__(self, *args: Any, mode: str = "evidence", **kwargs: Any) -> None:
        if mode != "evidence":
            raise ValueError("framework_2_parallel_with_evidence mode must be 'evidence'")
        super().__init__(*args, mode=mode, **kwargs)


ParallelWithEvidenceWorkflow = MultiAgentWorkflow
