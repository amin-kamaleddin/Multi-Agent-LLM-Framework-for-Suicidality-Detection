import pytest

from workflows.registry import (
    WORKFLOW_REGISTRY,
    WorkflowRegistration,
    WorkflowRegistry,
)


def test_builtin_registry_exposes_workflow_metadata():
    assert WORKFLOW_REGISTRY.names() == (
        "framework_1_parallel_no_evidence",
        "framework_2_parallel_with_evidence",
        "framework_3_hierarchical_no_reviewer",
        "framework_4_hierarchical_with_reviewer",
    )
    assert WORKFLOW_REGISTRY.get("framework_1_parallel_no_evidence").name == "framework_1_parallel_no_evidence"
    assert WORKFLOW_REGISTRY.describe()[0]["config_keys"] == [
        "model",
        "temperature",
        "max_completion_tokens",
        "max_retries",
        "mode",
        "service_tier",
        "fake_llm",
        "agents",
    ]
    assert WORKFLOW_REGISTRY.get("framework_2_parallel_with_evidence").default_mode == "evidence"
    assert WORKFLOW_REGISTRY.get("framework_3_hierarchical_no_reviewer").default_mode == "bool"
    assert WORKFLOW_REGISTRY.get("framework_4_hierarchical_with_reviewer").default_mode == "bool"


def test_registry_accepts_new_named_runner_and_rejects_invalid_names():
    registry = WorkflowRegistry()
    registration = WorkflowRegistration(
        name="hierarchical",
        description="test workflow",
        runner=lambda request: request,
    )

    registry.register(registration)

    assert registry.names() == ("hierarchical",)
    assert registry.get("hierarchical") is registration
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registration)
    with pytest.raises(ValueError, match="lowercase"):
        registry.register(
            WorkflowRegistration(
                name="Not-Lowercase",
                description="invalid",
                runner=lambda request: request,
            )
        )
    with pytest.raises(ValueError, match="Available workflows"):
        registry.get("missing")
