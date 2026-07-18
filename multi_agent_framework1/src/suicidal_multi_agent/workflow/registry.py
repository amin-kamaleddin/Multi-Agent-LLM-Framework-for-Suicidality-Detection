"""Registry of workflow runners available to the shared evaluation CLI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..evaluation.evaluator import (
    FRAMEWORK1_AGENT_ROLES,
    EvaluationConfig,
    EvaluationResult,
    run_workflow_evaluation,
    run_workflow2_hierarchical_evaluation,
    run_workflow3_hierarchical_reviewer_evaluation,
)

from .models import (
    DeterministicFakeModel,
    build_real_model,
    openai_base_url,
    resolve_model_name_for_base_url,
)
from .prompts import WORKFLOW_MODE_NAMES


@dataclass(frozen=True)
class WorkflowRunRequest:
    """Common inputs passed from the shared CLI to a registered runner."""

    sample_df: pd.DataFrame
    config: EvaluationConfig
    workflow_config: dict[str, Any]
    fake_llm: bool
    run_id: str | None
    artifact_base_dir: str | Path


WorkflowRunner = Callable[[WorkflowRunRequest], EvaluationResult]


@dataclass(frozen=True)
class WorkflowRegistration:
    """One CLI-selectable workflow and its configuration contract."""

    name: str
    description: str
    runner: WorkflowRunner
    config_keys: tuple[str, ...] = ()


class WorkflowRegistry:
    """Map stable CLI names to workflow runners."""

    def __init__(self) -> None:
        self._registrations: dict[str, WorkflowRegistration] = {}

    def register(self, registration: WorkflowRegistration) -> None:
        if not registration.name or registration.name != registration.name.lower():
            raise ValueError("Workflow names must be non-empty lowercase strings")
        if registration.name in self._registrations:
            raise ValueError(f"Workflow {registration.name!r} is already registered")
        self._registrations[registration.name] = registration

    def get(self, name: str) -> WorkflowRegistration:
        try:
            return self._registrations[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown workflow {name!r}. Available workflows: {list(self.names())}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": registration.name,
                "description": registration.description,
                "config_keys": list(registration.config_keys),
            }
            for registration in (self._registrations[name] for name in self.names())
        ]


FRAMEWORK1_CONFIG_KEYS = (
    "model",
    "temperature",
    "max_completion_tokens",
    "max_retries",
    "mode",
    "fake_llm",
    "agents",
)
FRAMEWORK1_AGENT_CONFIG_KEYS = (
    "agent_name",
    "model",
    "temperature",
    "max_completion_tokens",
)

WORKFLOW2_HIERARCHICAL_CONFIG_KEYS = (
    "model",
    "temperature",
    "max_completion_tokens",
    "max_retries",
    "mode",
    "fake_llm",
)

WORKFLOW3_HIERARCHICAL_REVIEWER_CONFIG_KEYS = (
    "model",
    "temperature",
    "max_completion_tokens",
    "max_retries",
    "mode",
    "fake_llm",
)


def _run_framework1(request: WorkflowRunRequest) -> EvaluationResult:
    unknown_keys = sorted(set(request.workflow_config) - set(FRAMEWORK1_CONFIG_KEYS))
    if unknown_keys:
        raise ValueError(f"Unknown framework1 workflow-config keys: {unknown_keys}")
    if request.config.mode not in WORKFLOW_MODE_NAMES:
        raise ValueError(
            f"framework1 mode must be one of {list(WORKFLOW_MODE_NAMES)}; "
            f"got {request.config.mode!r}"
        )

    model = (
        DeterministicFakeModel()
        if request.fake_llm
        else build_real_model(
            request.config.requested_model_name,
            request.config.temperature,
            request.config.max_completion_tokens,
        )
    )
    agent_configuration, agent_models = _framework1_agent_overrides(request)
    resolved_config = replace(
        request.config,
        agent_configuration=agent_configuration or None,
    )
    return run_workflow_evaluation(
        sample_df=request.sample_df,
        model=model,
        config=resolved_config,
        run_id=request.run_id,
        artifact_base_dir=request.artifact_base_dir,
        agent_models=agent_models or None,
    )

def _run_workflow2_hierarchical(
    request: WorkflowRunRequest,
) -> EvaluationResult:
    unknown_keys = sorted(
        set(request.workflow_config)
        - set(WORKFLOW2_HIERARCHICAL_CONFIG_KEYS)
    )
    if unknown_keys:
        raise ValueError(
            "Unknown workflow2_hierarchical workflow-config keys: "
            f"{unknown_keys}"
        )

    if request.config.mode not in WORKFLOW_MODE_NAMES:
        raise ValueError(
            f"workflow2_hierarchical mode must be one of "
            f"{list(WORKFLOW_MODE_NAMES)}; got {request.config.mode!r}"
        )

    model = (
        DeterministicFakeModel()
        if request.fake_llm
        else build_real_model(
            request.config.requested_model_name,
            request.config.temperature,
            request.config.max_completion_tokens,
        )
    )


    return run_workflow2_hierarchical_evaluation(
        sample_df=request.sample_df,
        model=model,
        config=request.config,
        run_id=request.run_id,
        artifact_base_dir=request.artifact_base_dir,
    )

    
def _run_workflow3_hierarchical_reviewer(
    request: WorkflowRunRequest,
) -> EvaluationResult:
    unknown_keys = sorted(
        set(request.workflow_config)
        - set(WORKFLOW3_HIERARCHICAL_REVIEWER_CONFIG_KEYS)
    )
    if unknown_keys:
        raise ValueError(
            "Unknown workflow3_hierarchical_reviewer workflow-config keys: "
            f"{unknown_keys}"
        )

    if request.config.mode not in WORKFLOW_MODE_NAMES:
        raise ValueError(
            f"workflow3_hierarchical_reviewer mode must be one of "
            f"{list(WORKFLOW_MODE_NAMES)}; got {request.config.mode!r}"
        )

    model = (
        DeterministicFakeModel()
        if request.fake_llm
        else build_real_model(
            request.config.requested_model_name,
            request.config.temperature,
            request.config.max_completion_tokens,
        )
    )

     
    return run_workflow3_hierarchical_reviewer_evaluation(
        sample_df=request.sample_df,
        model=model,
        config=request.config,
        run_id=request.run_id,
        artifact_base_dir=request.artifact_base_dir,
    )




def _framework1_agent_overrides(
    request: WorkflowRunRequest,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    agent_specs = request.workflow_config.get("agents") or {}
    if not isinstance(agent_specs, dict):
        raise ValueError("framework1 workflow-config 'agents' must be a JSON object")

    unknown_roles = sorted(set(agent_specs) - set(FRAMEWORK1_AGENT_ROLES))
    if unknown_roles:
        raise ValueError(
            f"Unknown framework1 agent roles: {unknown_roles}. "
            f"Expected roles: {list(FRAMEWORK1_AGENT_ROLES)}"
        )

    agent_configuration: dict[str, dict[str, Any]] = {}
    agent_models: dict[str, Any] = {}
    for role, raw_spec in agent_specs.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"framework1 agent configuration for {role!r} must be a JSON object")
        unknown_keys = sorted(set(raw_spec) - set(FRAMEWORK1_AGENT_CONFIG_KEYS))
        if unknown_keys:
            raise ValueError(f"Unknown framework1 agent config keys for {role!r}: {unknown_keys}")

        requested_model = str(raw_spec.get("model") or request.config.requested_model_name)
        temperature = float(raw_spec.get("temperature", request.config.temperature))
        max_completion_tokens = int(
            raw_spec.get("max_completion_tokens", request.config.max_completion_tokens)
        )
        if max_completion_tokens <= 0:
            raise ValueError(f"max_completion_tokens for {role!r} must be greater than zero")
        resolved_model = (
            requested_model
            if request.fake_llm
            else resolve_model_name_for_base_url(requested_model, openai_base_url())
        )
        details: dict[str, Any] = {
            "model_name": resolved_model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        if raw_spec.get("agent_name") is not None:
            details["agent_name"] = str(raw_spec["agent_name"])
        agent_configuration[role] = details
        agent_models[role] = (
            DeterministicFakeModel()
            if request.fake_llm
            else build_real_model(requested_model, temperature, max_completion_tokens)
        )
    return agent_configuration, agent_models


WORKFLOW_REGISTRY = WorkflowRegistry()
WORKFLOW_REGISTRY.register(
    WorkflowRegistration(
        name="framework1",
        description="Four parallel suicide-risk specialists followed by a final judge.",
        runner=_run_framework1,
        config_keys=FRAMEWORK1_CONFIG_KEYS,
    )
)

WORKFLOW_REGISTRY.register(
    WorkflowRegistration(
        name="workflow2_hierarchical",
        description=(
            "Hierarchical suicide-risk workflow with coordinated "
            "specialist analysis and final decision."
        ),
        runner=_run_workflow2_hierarchical,
        config_keys=WORKFLOW2_HIERARCHICAL_CONFIG_KEYS,
    )
)

WORKFLOW_REGISTRY.register(
    WorkflowRegistration(
        name="workflow3_hierarchical_reviewer",
        description=(
            "Hierarchical suicide-risk workflow with confidence-gated "
            "secondary reviewer."
        ),
        runner=_run_workflow3_hierarchical_reviewer,
        config_keys=WORKFLOW3_HIERARCHICAL_REVIEWER_CONFIG_KEYS,
    )
)
