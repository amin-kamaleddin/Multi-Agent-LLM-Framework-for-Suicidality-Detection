"""Architecture-neutral workflow contracts consumed by evaluation code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd


class CaseResult(Protocol):
    sample_id: str
    final_label: str
    suicidal_intent: bool | None
    errors: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]


class Workflow(Protocol):
    def run_case(self, *, sample_id: str, text: str) -> CaseResult:
        """Evaluate one sample and return the shared result fields."""


@dataclass(frozen=True)
class WorkflowRunRequest:
    """Common inputs passed from the CLI to a registered workflow runner."""

    sample_df: pd.DataFrame
    config: Any
    workflow_config: dict[str, Any]
    fake_llm: bool
    run_id: str | None
    artifact_base_dir: str | Path


WorkflowRunner = Callable[[WorkflowRunRequest], Any]


@dataclass(frozen=True)
class WorkflowRegistration:
    """One CLI-selectable workflow and its configuration contract."""

    name: str
    description: str
    runner: WorkflowRunner
    config_keys: tuple[str, ...] = ()
    default_mode: str = "bool"
