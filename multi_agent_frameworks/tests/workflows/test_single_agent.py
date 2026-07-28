import json

import pytest

from workflows.single_agent.workflow import SingleAgentWorkflow


class SingleAgentModeModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        system = messages[0]["content"]
        if system.startswith("Classify the post into exactly ONE class"):
            return "Ideation"
        if "Include ALL 5 keys" in system:
            return json.dumps(
                {
                    "unsupportive": 0.05,
                    "ideation": 0.8,
                    "behavior": 0.05,
                    "indicator": 0.05,
                    "attempt": 0.05,
                }
            )
        if "evaluate whether the post matches the class" in system:
            target = system.split("matches the class: ", 1)[1].splitlines()[0]
            return json.dumps({target.lower(): 10 if target == "Ideation" else 1})
        if "Compare the two classes" in system:
            options = system.split("Compare the two classes: '", 1)[1]
            class_a, class_b = options.split("' vs '", 1)
            class_b = class_b.split("'", 1)[0]
            return "Ideation" if "Ideation" in (class_a, class_b) else class_a
        raise AssertionError("Unexpected prompt")


@pytest.mark.parametrize(
    ("mode", "expected_calls"),
    [
        ("single-class", 1),
        ("multi-class-chance", 1),
        ("per-class-score", 5),
        ("base-vs-rest-binary", 4),
    ],
)
def test_single_agent_modes_return_shared_contract(mode, expected_calls):
    model = SingleAgentModeModel()
    workflow = SingleAgentWorkflow(model=model, mode=mode, max_retries=0)

    result = workflow.run_case(sample_id="case-1", text="I do not want to live.")

    assert result.sample_id == "case-1"
    assert result.final_label == "Ideation"
    assert result.suicidal_intent is True
    assert result.errors == []
    assert len(result.llm_calls) == expected_calls
    assert len(model.calls) == expected_calls
    assert result.traces[0]["event"] == "single_agent_complete"
    assert result.traces[0]["mode"] == mode
    assert all(call["_input_messages"] for call in result.llm_calls)


def test_single_agent_unsupportive_maps_to_false():
    class UnsupportiveModel:
        def invoke(self, messages):
            return "Unsupportive"

    result = SingleAgentWorkflow(
        model=UnsupportiveModel(),
        mode="single-class",
        max_retries=0,
    ).run_case(sample_id="negative", text="I went for a walk.")

    assert result.final_label == "Unsupportive"
    assert result.suicidal_intent is False


def test_single_agent_retries_invalid_output_and_returns_error():
    class InvalidModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return "unknown"

    model = InvalidModel()
    result = SingleAgentWorkflow(
        model=model,
        mode="single-class",
        max_retries=1,
    ).run_case(sample_id="bad", text="test")

    assert result.final_label == "Error"
    assert result.suicidal_intent is None
    assert model.calls == 2
    assert len(result.llm_calls) == 2
    assert len(result.errors) == 1
    assert result.errors[0]["error_type"] == "ValueError"


def test_single_agent_rejects_host_and_unknown_modes():
    with pytest.raises(ValueError, match="single_agent mode"):
        SingleAgentWorkflow(model=SingleAgentModeModel(), mode="multi-agent-host")
