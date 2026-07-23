"""Evaluation adapter for framework 3 routing outputs."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.evaluator import EvaluationConfig, EvaluationResult, run_evaluation as run_shared_evaluation
from llm.protocol import InvokableModel
from .graph import MultiAgentWorkflow


def run_evaluation(
    *,
    sample_df: pd.DataFrame,
    model: InvokableModel,
    config: EvaluationConfig,
    run_id: str | None = None,
    artifact_base_dir: str | Path = "logs/runs",
    dataset_audit: dict[str, Any] | None = None,
) -> EvaluationResult:
    workflow = MultiAgentWorkflow(
        model=model,
        model_name=config.model_name,
        max_retries=config.max_retries,
        mode=config.mode,
    )
    resolved_config = EvaluationConfig(
        **{
            **asdict(config),
            "workflow_name": "framework_3_hierarchical_no_reviewer",
            "agent_configuration": config.agent_configuration or {},
        }
    )
    return run_shared_evaluation(
        sample_df=sample_df,
        workflow=workflow,
        config=resolved_config,
        run_id=run_id,
        artifact_base_dir=artifact_base_dir,
        dataset_audit=dataset_audit,
    )
