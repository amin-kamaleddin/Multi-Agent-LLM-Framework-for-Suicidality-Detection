"""Evaluation adapter for framework 2 evidence packets."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    _collector_prediction_fields,
    _framework_1_parallel_no_evidence_agent_configuration,
    _validate_framework_1_parallel_no_evidence_agent_overrides,
    run_evaluation as run_shared_evaluation,
)
from llm.protocol import InvokableModel
from llm.limiter import LlmConcurrencyLimiter
from .graph import MultiAgentWorkflow


def run_evaluation(
    *,
    sample_df: pd.DataFrame,
    model: InvokableModel,
    config: EvaluationConfig,
    run_id: str | None = None,
    artifact_base_dir: str | Path = "logs/runs",
    agent_models: dict[str, InvokableModel] | None = None,
    dataset_audit: dict[str, Any] | None = None,
    resume_run_dir: str | Path | None = None,
) -> EvaluationResult:
    agent_configuration = _framework_1_parallel_no_evidence_agent_configuration(config)
    _validate_framework_1_parallel_no_evidence_agent_overrides(
        config, agent_models, agent_configuration
    )
    limiter = LlmConcurrencyLimiter(config.max_in_flight_llm)
    limited_agent_models = {
        role: limiter.wrap(role_model)
        for role, role_model in (agent_models or {}).items()
    }
    workflow = MultiAgentWorkflow(
        model=limiter.wrap(model),
        model_name=config.model_name,
        max_retries=config.max_retries,
        mode=config.mode,
        models_by_role=limited_agent_models or None,
        model_names_by_role={
            role: str(details.get("model_name") or config.model_name)
            for role, details in agent_configuration.items()
        },
        use_create_agent=False,
    )
    resolved_config = EvaluationConfig(
        **{
            **asdict(config),
            "workflow_name": "framework_2_parallel_with_evidence",
            "agent_configuration": agent_configuration,
        }
    )
    return run_shared_evaluation(
        sample_df=sample_df,
        workflow=workflow,
        config=resolved_config,
        run_id=run_id,
        artifact_base_dir=artifact_base_dir,
        dataset_audit=dataset_audit,
        prediction_enricher=_collector_prediction_fields,
        resume_run_dir=resume_run_dir,
    )
