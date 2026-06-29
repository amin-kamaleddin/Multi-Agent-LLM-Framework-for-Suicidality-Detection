import json
import time

from suicidal_multi_agent.core.labels import VALID_LABELS
from suicidal_multi_agent.core.schemas import JudgeAnswer, SpecialistAnswer
from suicidal_multi_agent.workflow.graph import MultiAgentWorkflow


class FakeGraphModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        system = messages[0]["content"]
        if "final judge" in system.lower():
            assert "Specialist answers" in messages[1]["content"]
            for label in VALID_LABELS:
                assert label in messages[1]["content"]
            return json.dumps({"label_name": "Attempt"})

        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                return json.dumps({"answer": label == "Attempt"})
        raise AssertionError("Unexpected prompt")


def test_graph_runs_four_specialists_and_judge():
    model = FakeGraphModel()
    workflow = MultiAgentWorkflow(model=model, max_retries=1)

    result = workflow.run_case(sample_id="case-1", text="test text")

    assert result.final_label == "Attempt"
    assert result.specialist_answers == {
        "Indicator": False,
        "Ideation": False,
        "Behavior": False,
        "Attempt": True,
    }
    assert len(result.errors) == 0
    assert len(model.calls) == 5
    assert [call["role"] for call in result.llm_calls].count("judge") == 1
    specialist_traces = [trace for trace in result.traces if trace["event"] == "specialist_complete"]
    assert len(specialist_traces) == 4
    assert {trace["parallel_group"] for trace in specialist_traces} == {"four_label_specialists"}
    assert all(trace["node_duration_s"] >= 0 for trace in specialist_traces)


class SlowSpecialistModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            return json.dumps({"label_name": "Indicator"})
        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                time.sleep(0.2)
                return json.dumps({"answer": label == "Indicator"})
        raise AssertionError("Unexpected prompt")


def test_graph_runs_specialists_in_parallel_with_overlapping_node_timings():
    workflow = MultiAgentWorkflow(model=SlowSpecialistModel(), max_retries=0)

    result = workflow.run_case(sample_id="case-parallel", text="test text")

    specialist_traces = [trace for trace in result.traces if trace["event"] == "specialist_complete"]
    assert len(specialist_traces) == 4
    latest_specialist_start = max(trace["node_started_at_s"] for trace in specialist_traces)
    earliest_specialist_finish = min(trace["node_finished_at_s"] for trace in specialist_traces)
    assert latest_specialist_start < earliest_specialist_finish

    total_specialist_duration = sum(trace["node_duration_s"] for trace in specialist_traces)
    specialist_wall_span = (
        max(trace["node_finished_at_s"] for trace in specialist_traces)
        - min(trace["node_started_at_s"] for trace in specialist_traces)
    )
    assert specialist_wall_span < total_specialist_duration

    judge_trace = next(trace for trace in result.traces if trace["event"] == "judge_complete")
    assert judge_trace["node_started_at_s"] >= max(trace["node_finished_at_s"] for trace in specialist_traces)


class MalformedThenValidModel:
    def __init__(self):
        self.calls_by_role = {}

    def invoke(self, messages):
        system = messages[0]["content"]
        role = "judge" if "final judge" in system.lower() else next(
            label for label in VALID_LABELS if f"Target label: {label}" in system
        )
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        if role == "Indicator" and self.calls_by_role[role] == 1:
            return "not json"
        if role == "judge":
            return json.dumps({"label_name": "Indicator"})
        return json.dumps({"answer": role == "Indicator"})


def test_graph_retries_malformed_json_once():
    model = MalformedThenValidModel()
    workflow = MultiAgentWorkflow(model=model, max_retries=1)

    result = workflow.run_case(sample_id="case-2", text="test text")

    assert result.final_label == "Indicator"
    assert model.calls_by_role["Indicator"] == 2
    assert len(result.errors) == 0
    indicator_calls = [call for call in result.llm_calls if call["role"] == "Indicator_specialist"]
    assert len(indicator_calls) == 2
    assert indicator_calls[0]["error"]["attempt"] == 1


class AlwaysMalformedModel:
    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return "not json"


def test_graph_marks_sample_error_after_retries_exhausted():
    model = AlwaysMalformedModel()
    workflow = MultiAgentWorkflow(model=model, max_retries=3)

    result = workflow.run_case(sample_id="case-3", text="test text")

    assert result.final_label == "Error"
    assert result.errors
    assert max(error["attempt"] for error in result.errors) == 4
    assert result.traces[-1] == {"event": "judge_skipped", "reason": "specialist_error"}


class UsageResponse:
    def __init__(self, content):
        self.content = content

    usage_metadata = {
        "input_tokens": 12,
        "output_tokens": 5,
        "total_tokens": 17,
        "cost": 0.0012,
        "input_token_details": {"cache_read": 3},
    }


class UsageModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            return UsageResponse('{"label_name": "Indicator"}')
        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                return UsageResponse(json.dumps({"answer": label == "Indicator"}))
        raise AssertionError("Unexpected prompt")


def test_graph_records_provider_usage_cost_and_cached_token_details():
    model = UsageModel()
    workflow = MultiAgentWorkflow(model=model, max_retries=0)

    result = workflow.run_case(sample_id="case-4", text="test text")

    first_call = result.llm_calls[0]
    assert first_call["prompt_tokens"] == 12
    assert first_call["completion_tokens"] == 5
    assert first_call["total_tokens"] == 17
    assert first_call["cost"] == 0.0012
    assert first_call["prompt_tokens_details"] == {"cache_read": 3}


class OpenRouterCostResponse:
    def __init__(self, content):
        self.content = content

    usage_metadata = {
        "input_tokens": 12,
        "output_tokens": 5,
        "total_tokens": 17,
        "input_token_details": {"cache_read": 0},
    }
    response_metadata = {
        "id": "gen-test-123",
        "latency": 1250,
        "generation_time": 1000,
        "moderation_latency": 50,
        "provider_name": "OpenAI",
        "upstream_id": "chatcmpl-test",
        "headers": {
            "x-generation-id": "gen-header-456",
            "x-request-id": "req-header-789",
            "authorization": "redacted-by-snapshot-filter",
        },
        "token_usage": {
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
            "cost": 0.0042,
            "cost_details": {"upstream_inference_cost": 0.0042},
            "prompt_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 1},
        }
    }


class OpenRouterCostModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            return OpenRouterCostResponse('{"label_name": "Indicator"}')
        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                return OpenRouterCostResponse(json.dumps({"answer": label == "Indicator"}))
        raise AssertionError("Unexpected prompt")


def test_graph_merges_openrouter_cost_from_response_metadata():
    model = OpenRouterCostModel()
    workflow = MultiAgentWorkflow(model=model, max_retries=0)

    result = workflow.run_case(sample_id="case-4b", text="test text")

    first_call = result.llm_calls[0]
    assert first_call["cost"] == 0.0042
    assert first_call["cost_details"] == {"upstream_inference_cost": 0.0042}
    assert first_call["prompt_tokens_details"] == {"cached_tokens": 2, "cache_write_tokens": 1}
    assert first_call["raw_usage"]["token_usage"]["cost"] == 0.0042
    assert first_call["generation_id"] == "gen-header-456"
    assert first_call["request_id"] == "req-header-789"
    assert first_call["provider_name"] == "OpenAI"
    assert first_call["upstream_id"] == "chatcmpl-test"
    assert first_call["provider_latency_s"] == 1.25
    assert first_call["generation_time_s"] == 1.0
    assert first_call["moderation_latency_s"] == 0.05
    assert first_call["completion_tps"] == 5.0
    assert first_call["e2e_completion_tps"] is not None
    assert first_call["raw_usage"]["response_metadata"]["headers"] == {
        "x-generation-id": "gen-header-456",
        "x-request-id": "req-header-789",
    }


class TransportFailureModel:
    def invoke(self, messages):
        raise RuntimeError("network unavailable: {'user_id': 'secret-user'}")


def test_graph_records_failed_transport_attempt_as_llm_call():
    model = TransportFailureModel()
    workflow = MultiAgentWorkflow(model=model, model_name="failing-model", max_retries=0)

    result = workflow.run_case(sample_id="case-5", text="test text")

    assert result.final_label == "Error"
    assert result.llm_calls
    first_call = result.llm_calls[0]
    assert first_call["model_name"] == "failing-model"
    assert first_call["attempt"] == 1
    assert first_call["prompt_tokens"] == 0
    assert first_call["error"]["error_type"] == "RuntimeError"
    assert "secret-user" not in first_call["error"]["error_message"]
    assert "[redacted]" in first_call["error"]["error_message"]


class BindableStructuredModel:
    def bind_tools(self, *args, **kwargs):
        raise AssertionError("bind_tools is owned by LangChain create_agent in this test")


class FakeStructuredAgent:
    def __init__(self, schema, invocations):
        self.schema = schema
        self.invocations = invocations

    def invoke(self, payload):
        self.invocations.append((self.schema, payload))
        if self.schema is SpecialistAnswer:
            return {"messages": [], "structured_response": {"answer": True}}
        if self.schema is JudgeAnswer:
            return {"messages": [], "structured_response": {"label_name": "Indicator"}}
        raise AssertionError(f"Unexpected schema {self.schema}")


def test_graph_uses_create_agent_path_for_bindable_models(monkeypatch):
    invocations = []

    def fake_get_structured_agent(self, *, system_prompt, schema):
        assert system_prompt
        return FakeStructuredAgent(schema, invocations)

    monkeypatch.setattr(MultiAgentWorkflow, "_get_structured_agent", fake_get_structured_agent)

    workflow = MultiAgentWorkflow(model=BindableStructuredModel(), model_name="structured-model", max_retries=0)

    result = workflow.run_case(sample_id="case-6", text="test text")

    assert result.final_label == "Indicator"
    assert len(result.errors) == 0
    assert len(invocations) == 5
    assert [schema for schema, _ in invocations].count(SpecialistAnswer) == 4
    assert [schema for schema, _ in invocations].count(JudgeAnswer) == 1
    assert all(call["structured_output_strategy"] == "create_agent_tool_strategy" for call in result.llm_calls)
