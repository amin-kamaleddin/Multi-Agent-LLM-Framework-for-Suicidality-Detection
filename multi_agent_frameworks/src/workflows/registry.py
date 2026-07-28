"""Registry of workflow runners available to the shared evaluation CLI."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from evaluation.evaluator import (
    FRAMEWORK_1_PARALLEL_NO_EVIDENCE_AGENT_ROLES,
    FRAMEWORK_2_PARALLEL_WITH_EVIDENCE_AGENT_ROLES,
    EvaluationConfig,
    EvaluationResult,
)
from llm.factory import (
    build_real_model,
    openai_base_url,
    resolve_model_name_for_base_url,
)
from llm.fakes import DeterministicFakeModel
from workflows.contracts import WorkflowRegistration, WorkflowRunRequest
from workflows.framework_1_parallel_no_evidence.config import (
    AGENT_CONFIG_KEYS as FRAMEWORK_1_PARALLEL_NO_EVIDENCE_AGENT_CONFIG_KEYS,
    CONFIG_KEYS as FRAMEWORK_1_PARALLEL_NO_EVIDENCE_CONFIG_KEYS,
)
from workflows.framework_1_parallel_no_evidence.prompts import FRAMEWORK_1_PARALLEL_NO_EVIDENCE_MODE_NAMES
from workflows.framework_1_parallel_no_evidence.evaluation import run_evaluation as run_framework_1_evaluation
from workflows.framework_2_parallel_with_evidence.config import CONFIG_KEYS as FRAMEWORK_2_PARALLEL_WITH_EVIDENCE_CONFIG_KEYS
from workflows.framework_2_parallel_with_evidence.evaluation import run_evaluation as run_framework_2_evaluation
from workflows.framework_3_hierarchical_no_reviewer.config import CONFIG_KEYS as FRAMEWORK_3_HIERARCHICAL_NO_REVIEWER_CONFIG_KEYS
from workflows.framework_3_hierarchical_no_reviewer.config import MODE_NAMES as FRAMEWORK_3_HIERARCHICAL_NO_REVIEWER_MODE_NAMES
from workflows.framework_3_hierarchical_no_reviewer.evaluation import run_evaluation as run_framework_3_evaluation
from workflows.framework_4_hierarchical_with_reviewer.config import CONFIG_KEYS as FRAMEWORK_4_HIERARCHICAL_WITH_REVIEWER_CONFIG_KEYS
from workflows.framework_4_hierarchical_with_reviewer.config import MODE_NAMES as FRAMEWORK_4_HIERARCHICAL_WITH_REVIEWER_MODE_NAMES
from workflows.framework_4_hierarchical_with_reviewer.evaluation import run_evaluation as run_framework_4_evaluation
from workflows.single_agent.config import (
    CONFIG_KEYS as SINGLE_AGENT_CONFIG_KEYS,
    MODE_NAMES as SINGLE_AGENT_MODE_NAMES,
)
from workflows.single_agent.evaluation import run_evaluation as run_single_agent_evaluation


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


def _run_single_agent(request: WorkflowRunRequest) -> EvaluationResult:
    unknown_keys = sorted(set(request.workflow_config) - set(SINGLE_AGENT_CONFIG_KEYS))
    if unknown_keys:
        raise ValueError(f"Unknown single_agent workflow-config keys: {unknown_keys}")
    if request.config.mode not in SINGLE_AGENT_MODE_NAMES:
        raise ValueError(
            f"single_agent mode must be one of {list(SINGLE_AGENT_MODE_NAMES)}; "
            f"got {request.config.mode!r}"
        )

    model = (
        DeterministicFakeModel()
        if request.fake_llm
        else build_real_model(
            request.config.requested_model_name,
            request.config.temperature,
            request.config.max_completion_tokens,
            service_tier=request.config.service_tier,
        )
    )
    return run_single_agent_evaluation(
        sample_df=request.sample_df,
        model=model,
        config=request.config,
        base_class=str(request.workflow_config.get("base_class", "Unsupportive")),
        run_id=request.run_id,
        artifact_base_dir=request.artifact_base_dir,
        resume_run_dir=request.resume_run_dir,
    )


def _run_framework_1_parallel_no_evidence(request: WorkflowRunRequest) -> EvaluationResult:
    unknown_keys = sorted(set(request.workflow_config) - set(FRAMEWORK_1_PARALLEL_NO_EVIDENCE_CONFIG_KEYS))
    if unknown_keys:
        raise ValueError(f"Unknown framework_1_parallel_no_evidence workflow-config keys: {unknown_keys}")
    if request.config.mode not in FRAMEWORK_1_PARALLEL_NO_EVIDENCE_MODE_NAMES:
        raise ValueError(
            f"framework_1_parallel_no_evidence mode must be one of {list(FRAMEWORK_1_PARALLEL_NO_EVIDENCE_MODE_NAMES)}; "
            f"got {request.config.mode!r}"
        )

    model = (
        DeterministicFakeModel()
        if request.fake_llm
        else build_real_model(
            request.config.requested_model_name,
            request.config.temperature,
            request.config.max_completion_tokens,
            service_tier=request.config.service_tier,
        )
    )
    agent_configuration, agent_models = _agent_overrides(
        request,
        config=request.config,
        allowed_roles=FRAMEWORK_1_PARALLEL_NO_EVIDENCE_AGENT_ROLES,
        framework_name="framework_1_parallel_no_evidence",
    )
    resolved_config = replace(
        request.config,
        agent_configuration=agent_configuration or None,
    )
    return run_framework_1_evaluation(
        sample_df=request.sample_df,
        model=model,
        config=resolved_config,
        run_id=request.run_id,
        artifact_base_dir=request.artifact_base_dir,
        agent_models=agent_models or None,
        resume_run_dir=request.resume_run_dir,
    )

def _run_framework_3_hierarchical_no_reviewer(
    request: WorkflowRunRequest,
) -> EvaluationResult:
    unknown_keys = sorted(
        set(request.workflow_config)
        - set(FRAMEWORK_3_HIERARCHICAL_NO_REVIEWER_CONFIG_KEYS)
    )
    if unknown_keys:
        raise ValueError(
            "Unknown framework_3_hierarchical_no_reviewer workflow-config keys: "
            f"{unknown_keys}"
        )

    if request.config.mode not in FRAMEWORK_3_HIERARCHICAL_NO_REVIEWER_MODE_NAMES:
        raise ValueError(
            f"framework_3_hierarchical_no_reviewer mode must be one of "
            f"{list(FRAMEWORK_3_HIERARCHICAL_NO_REVIEWER_MODE_NAMES)}; got {request.config.mode!r}"
        )

    model = (
        DeterministicFakeModel()
        if request.fake_llm
        else build_real_model(
            request.config.requested_model_name,
            request.config.temperature,
            request.config.max_completion_tokens,
            service_tier=request.config.service_tier,
        )
    )
    return run_framework_3_evaluation(
        sample_df=request.sample_df,
        model=model,
        config=request.config,
        run_id=request.run_id,
        artifact_base_dir=request.artifact_base_dir,
        resume_run_dir=request.resume_run_dir,
    )



def _run_framework_4_hierarchical_with_reviewer(
    request: WorkflowRunRequest,
) -> EvaluationResult:
    unknown_keys = sorted(
        set(request.workflow_config)
        - set(FRAMEWORK_4_HIERARCHICAL_WITH_REVIEWER_CONFIG_KEYS)
    )
    if unknown_keys:
        raise ValueError(
            "Unknown framework_4_hierarchical_with_reviewer workflow-config keys: "
            f"{unknown_keys}"
        )

    if request.config.mode not in FRAMEWORK_4_HIERARCHICAL_WITH_REVIEWER_MODE_NAMES:
        raise ValueError(
            f"framework_4_hierarchical_with_reviewer mode must be one of "
            f"{list(FRAMEWORK_4_HIERARCHICAL_WITH_REVIEWER_MODE_NAMES)}; got {request.config.mode!r}"
        )

    model = (
        DeterministicFakeModel()
        if request.fake_llm
        else build_real_model(
            request.config.requested_model_name,
            request.config.temperature,
            request.config.max_completion_tokens,
            service_tier=request.config.service_tier,
        )
    )
    return run_framework_4_evaluation(
        sample_df=request.sample_df,
        model=model,
        config=request.config,
        run_id=request.run_id,
        artifact_base_dir=request.artifact_base_dir,
        resume_run_dir=request.resume_run_dir,
    )
def _run_framework_2_parallel_with_evidence(request: WorkflowRunRequest) -> EvaluationResult:
    unknown_keys = sorted(set(request.workflow_config) - set(FRAMEWORK_2_PARALLEL_WITH_EVIDENCE_CONFIG_KEYS))
    if unknown_keys:
        raise ValueError(f"Unknown framework_2_parallel_with_evidence workflow-config keys: {unknown_keys}")
    if request.config.mode != "evidence":
        raise ValueError("framework_2_parallel_with_evidence mode must be 'evidence'")

    model = (
        DeterministicFakeModel()
        if request.fake_llm
        else build_real_model(
            request.config.requested_model_name,
            request.config.temperature,
            request.config.max_completion_tokens,
            service_tier=request.config.service_tier,
        )
    )
    agent_configuration, agent_models = _agent_overrides(
        request,
        config=request.config,
        allowed_roles=FRAMEWORK_2_PARALLEL_WITH_EVIDENCE_AGENT_ROLES,
        framework_name="framework_2_parallel_with_evidence",
    )
    resolved_config = replace(
        request.config,
        agent_configuration=agent_configuration or None,
    )
    return run_framework_2_evaluation(
        sample_df=request.sample_df,
        model=model,
        config=resolved_config,
        run_id=request.run_id,
        artifact_base_dir=request.artifact_base_dir,
        agent_models=agent_models or None,
        resume_run_dir=request.resume_run_dir,
    )


def _agent_overrides(
    request: WorkflowRunRequest,
    *,
    config: EvaluationConfig,
    allowed_roles: tuple[str, ...],
    framework_name: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    agent_specs = request.workflow_config.get("agents") or {}
    if not isinstance(agent_specs, dict):
        raise ValueError(f"{framework_name} workflow-config 'agents' must be a JSON object")

    unknown_roles = sorted(set(agent_specs) - set(allowed_roles))
    if unknown_roles:
        raise ValueError(
            f"Unknown {framework_name} agent roles: {unknown_roles}. "
            f"Expected roles: {list(allowed_roles)}"
        )

    agent_configuration: dict[str, dict[str, Any]] = {}
    agent_models: dict[str, Any] = {}
    for role, raw_spec in agent_specs.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(
                f"{framework_name} agent configuration for {role!r} must be a JSON object"
            )
        unknown_keys = sorted(set(raw_spec) - set(FRAMEWORK_1_PARALLEL_NO_EVIDENCE_AGENT_CONFIG_KEYS))
        if unknown_keys:
            raise ValueError(
                f"Unknown {framework_name} agent config keys for {role!r}: {unknown_keys}"
            )

        requested_model = str(raw_spec.get("model") or config.requested_model_name)
        temperature = float(raw_spec.get("temperature", config.temperature))
        max_completion_tokens = int(
            raw_spec.get("max_completion_tokens", config.max_completion_tokens)
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
            else build_real_model(
                requested_model,
                temperature,
                max_completion_tokens,
                service_tier=config.service_tier,
            )
        )
    return agent_configuration, agent_models


WORKFLOW_REGISTRY = WorkflowRegistry()
WORKFLOW_REGISTRY.register(
    WorkflowRegistration(
        name="single_agent",
        description="One model using an EvalRing-compatible classification mode.",
        runner=_run_single_agent,
        config_keys=SINGLE_AGENT_CONFIG_KEYS,
        default_mode="single-class",
    )
)
WORKFLOW_REGISTRY.register(
    WorkflowRegistration(
        name="framework_1_parallel_no_evidence",
        description="Five parallel label specialists followed by a final judge.",
        runner=_run_framework_1_parallel_no_evidence,
        config_keys=FRAMEWORK_1_PARALLEL_NO_EVIDENCE_CONFIG_KEYS,
    )
)
WORKFLOW_REGISTRY.register(
    WorkflowRegistration(
        name="framework_2_parallel_with_evidence",
        description="Five parallel evidence collectors followed by a final judge.",
        runner=_run_framework_2_parallel_with_evidence,
        config_keys=FRAMEWORK_2_PARALLEL_WITH_EVIDENCE_CONFIG_KEYS,
        default_mode="evidence",
    )
)

WORKFLOW_REGISTRY.register(
    WorkflowRegistration(
        name="framework_3_hierarchical_no_reviewer",
        description=(
            "Hierarchical suicide-risk workflow with coordinated "
            "specialist analysis and final decision."
        ),
        runner=_run_framework_3_hierarchical_no_reviewer,
        config_keys=FRAMEWORK_3_HIERARCHICAL_NO_REVIEWER_CONFIG_KEYS,
    )
)

WORKFLOW_REGISTRY.register(
    WorkflowRegistration(
        name="framework_4_hierarchical_with_reviewer",
        description=(
            "Hierarchical suicide-risk workflow with confidence-gated "
            "secondary reviewer."
        ),
        runner=_run_framework_4_hierarchical_with_reviewer,
        config_keys=FRAMEWORK_4_HIERARCHICAL_WITH_REVIEWER_CONFIG_KEYS,
    )
)
