"""Evaluation adapter for the single-agent workflow."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.evaluator import EvaluationConfig, EvaluationResult, run_evaluation as run_shared_evaluation
from llm.limiter import LlmConcurrencyLimiter
from llm.protocol import InvokableModel

from .config import WORKFLOW_NAME
from .workflow import SingleAgentWorkflow


def run_evaluation(
    *,
    sample_df: pd.DataFrame,
    model: InvokableModel,
    config: EvaluationConfig,
    base_class: str = "Unsupportive",
    run_id: str | None = None,
    artifact_base_dir: str | Path = "logs/runs",
    dataset_audit: dict[str, Any] | None = None,
    resume_run_dir: str | Path | None = None,
) -> EvaluationResult:
    limiter = LlmConcurrencyLimiter(config.max_in_flight_llm)
    workflow = SingleAgentWorkflow(
        model=limiter.wrap(model),
        model_name=config.model_name,
        max_retries=config.max_retries,
        mode=config.mode,
        base_class=base_class,
    )
    resolved_config = EvaluationConfig(
        **{
            **asdict(config),
            "workflow_name": WORKFLOW_NAME,
            "agent_configuration": {
                "single_agent": {
                    "agent_name": "Single Agent",
                    "model_name": config.model_name,
                    "temperature": config.temperature,
                }
            },
        }
    )
    return run_shared_evaluation(
        sample_df=sample_df,
        workflow=workflow,
        config=resolved_config,
        run_id=run_id,
        artifact_base_dir=artifact_base_dir,
        dataset_audit=dataset_audit,
        resume_run_dir=resume_run_dir,
    )
