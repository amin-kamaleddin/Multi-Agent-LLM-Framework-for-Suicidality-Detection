import pytest

from suicidal_multi_agent.workflow.registry import (
    WORKFLOW_REGISTRY,
    WorkflowRegistration,
    WorkflowRegistry,
)


def test_builtin_registry_exposes_framework1_metadata():
    assert WORKFLOW_REGISTRY.names() == ("framework1",)
    assert WORKFLOW_REGISTRY.get("framework1").name == "framework1"
    assert WORKFLOW_REGISTRY.describe()[0]["config_keys"] == [
        "model",
        "temperature",
        "max_completion_tokens",
        "max_retries",
        "mode",
        "fake_llm",
        "agents",
    ]


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
