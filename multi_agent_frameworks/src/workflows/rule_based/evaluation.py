"""Shared-evaluator adapter for the deterministic rule-based workflow."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.evaluator import EvaluationConfig, EvaluationResult, run_evaluation as run_shared_evaluation

from .config import WORKFLOW_NAME
from .workflow import RuleBasedWorkflow


def run_evaluation(
    *,
    sample_df: pd.DataFrame,
    config: EvaluationConfig,
    run_id: str | None = None,
    artifact_base_dir: str | Path = "logs/runs",
    dataset_audit: dict[str, Any] | None = None,
    resume_run_dir: str | Path | None = None,
) -> EvaluationResult:
    resolved_config = EvaluationConfig(
        **{
            **asdict(config),
            "workflow_name": WORKFLOW_NAME,
            "agent_configuration": {
                "rule_based_classifier": {
                    "agent_name": "Rule-Based Classifier",
                    "model_name": "none",
                    "temperature": 0.0,
                }
            },
        }
    )
    return run_shared_evaluation(
        sample_df=sample_df,
        workflow=RuleBasedWorkflow(),
        config=resolved_config,
        run_id=run_id,
        artifact_base_dir=artifact_base_dir,
        dataset_audit=dataset_audit,
        resume_run_dir=resume_run_dir,
    )
