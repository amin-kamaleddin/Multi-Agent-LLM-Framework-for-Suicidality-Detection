"""LangGraph workflow for parallel specialists or evidence collectors plus a judge."""

from __future__ import annotations

import operator
import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypedDict

from domain.evidence import validate_evidence_answer
from domain.final_output import extract_json_object
from llm.protocol import InvokableModel
from .prompts import (
    EVIDENCE_TYPE_DEFINITIONS,
    WorkflowModeConfig,
    get_workflow_mode,
    specialist_user_prompt,
)


class WorkflowState(TypedDict):
    sample_id: str
    text: str
    specialist_answers: Annotated[dict[str, bool], operator.or_]
    specialist_outputs: Annotated[dict[str, dict[str, Any]], operator.or_]
    collector_outputs: Annotated[dict[str, dict[str, Any]], operator.or_]
    final_label: str
    suicidal_intent: bool | None
    errors: Annotated[list[dict[str, Any]], operator.add]
    traces: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class CaseResult:
    sample_id: str
    text: str
    final_label: str
    suicidal_intent: bool | None
    specialist_answers: dict[str, bool]
    specialist_outputs: dict[str, dict[str, Any]]
    collector_outputs: dict[str, dict[str, Any]]
    errors: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    agent_outputs: dict[str, Any]


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(part) for part in content)
    return str(content)


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    token_usage = metadata.get("token_usage") or {}
    token_usage = token_usage if isinstance(token_usage, dict) else {}
    if usage:
        prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
            "cost": usage.get("cost") or token_usage.get("cost"),
            "cost_details": token_usage.get("cost_details") or {},
            "prompt_tokens_details": (
                token_usage.get("prompt_tokens_details")
                or usage.get("input_token_details")
                or usage.get("prompt_tokens_details")
                or {}
            ),
            "completion_tokens_details": (
                token_usage.get("completion_tokens_details")
                or usage.get("output_token_details")
                or usage.get("completion_tokens_details")
                or {}
            ),
            **_provider_timing(metadata=metadata, token_usage=token_usage, usage=usage, completion_tokens=completion_tokens),
            "raw_usage": {
                "usage_metadata": dict(usage),
                "token_usage": dict(token_usage),
                "response_metadata": _metadata_snapshot(metadata),
            },
        }

    prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
    completion_tokens = int(token_usage.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(token_usage.get("total_tokens") or prompt_tokens + completion_tokens),
        "cost": token_usage.get("cost"),
        "cost_details": token_usage.get("cost_details") or {},
        "prompt_tokens_details": token_usage.get("prompt_tokens_details") or {},
        "completion_tokens_details": token_usage.get("completion_tokens_details") or {},
        **_provider_timing(metadata=metadata, token_usage=token_usage, usage=usage, completion_tokens=completion_tokens),
        "raw_usage": {
            "token_usage": dict(token_usage),
            "response_metadata": _metadata_snapshot(metadata),
        },
    }


def _provider_timing(
    *,
    metadata: dict[str, Any],
    token_usage: dict[str, Any],
    usage: dict[str, Any],
    completion_tokens: int,
) -> dict[str, Any]:
    """Extract provider timing only when LangChain/provider metadata exposes it."""
    sources = [metadata, token_usage, usage]
    generation_time_s = _seconds_from_sources(
        sources,
        millisecond_keys=("generation_time", "generation_time_ms"),
        second_keys=("generation_time_s",),
    )
    provider_latency_s = _seconds_from_sources(
        sources,
        millisecond_keys=("latency", "latency_ms"),
        second_keys=("latency_s", "provider_latency_s"),
    )
    moderation_latency_s = _seconds_from_sources(
        sources,
        millisecond_keys=("moderation_latency", "moderation_latency_ms"),
        second_keys=("moderation_latency_s",),
    )
    ttft_s = _seconds_from_sources(
        sources,
        millisecond_keys=("ttft_ms", "time_to_first_token_ms", "first_token_latency_ms"),
        second_keys=("ttft", "ttft_s", "time_to_first_token", "time_to_first_token_s", "first_token_latency_s"),
    )
    completion_tps = _float_from_sources(
        sources,
        ("completion_tps", "tokens_per_second", "tps", "throughput"),
    )
    if completion_tps is None and generation_time_s:
        completion_tps = completion_tokens / generation_time_s

    headers = _response_headers(metadata)
    generation_id = (
        _header_value(headers, "x-generation-id")
        or _string_from_sources(sources, ("generation_id", "openrouter_generation_id"))
    )
    metadata_id = _string_from_sources(sources, ("id",))
    if generation_id is None and metadata_id and metadata_id.startswith("gen-"):
        generation_id = metadata_id

    return {
        "generation_id": generation_id,
        "request_id": _header_value(headers, "x-request-id") or _string_from_sources(sources, ("request_id",)),
        "upstream_id": _string_from_sources(sources, ("upstream_id",)),
        "provider_name": _string_from_sources(sources, ("provider_name",)),
        "provider_latency_s": provider_latency_s,
        "generation_time_s": generation_time_s,
        "moderation_latency_s": moderation_latency_s,
        "ttft_s": ttft_s,
        "completion_tps": completion_tps,
    }


def _seconds_from_sources(
    sources: list[dict[str, Any]],
    *,
    millisecond_keys: tuple[str, ...],
    second_keys: tuple[str, ...],
) -> float | None:
    for source in sources:
        for key in millisecond_keys:
            if key in source:
                value = _to_float(source.get(key))
                return value / 1000.0 if value is not None else None
        for key in second_keys:
            if key in source:
                return _to_float(source.get(key))
    return None


def _float_from_sources(sources: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    for source in sources:
        for key in keys:
            if key in source:
                return _to_float(source.get(key))
    return None


def _string_from_sources(sources: list[dict[str, Any]], keys: tuple[str, ...]) -> str | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                text = str(value)
                if text:
                    return text
    return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _response_headers(metadata: dict[str, Any]) -> dict[str, Any]:
    headers = metadata.get("headers") or metadata.get("response_headers") or {}
    return dict(headers) if isinstance(headers, dict) else {}


def _header_value(headers: dict[str, Any], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target and value is not None:
            return str(value)
    return None


def _metadata_snapshot(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    snapshot = {
        key: _json_safe(value)
        for key, value in metadata.items()
        if key not in {"headers", "response_headers"}
    }
    headers = _response_headers(metadata)
    public_headers = {
        key: _json_safe(value)
        for key, value in headers.items()
        if str(key).lower() in {"x-generation-id", "x-request-id"}
    }
    if public_headers:
        snapshot["headers"] = public_headers
    return snapshot


def _json_safe(value: Any) -> Any:
    try:
        import json

        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _sanitize_error_message(message: str) -> str:
    """Remove provider-side user identifiers from stored error strings."""
    sanitized = re.sub(r"('user_id'\s*:\s*)'[^']+'", r"\1'[redacted]'", message)
    sanitized = re.sub(r'("user_id"\s*:\s*)"[^"]+"', r'\1"[redacted]"', sanitized)
    return sanitized


def _agent_call_records(
    *,
    result: dict[str, Any],
    role: str,
    model_name: str,
    outer_attempt: int,
    latency_s: float,
    parsed: Any,
) -> list[dict[str, Any]]:
    """Convert LangChain agent messages into local LLM-call telemetry."""
    messages = result.get("messages") or []
    ai_messages = [
        message
        for message in messages
        if getattr(message, "type", None) == "ai" or message.__class__.__name__ == "AIMessage"
    ]
    if not ai_messages:
        return [
            {
                "role": role,
                "attempt": outer_attempt,
                "model_name": model_name,
                "latency_s": latency_s,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": None,
                "cost_details": {},
                "prompt_tokens_details": {},
                "completion_tokens_details": {},
                "raw_usage": {},
                "generation_id": None,
                "request_id": None,
                "upstream_id": None,
                "provider_name": None,
                "provider_latency_s": None,
                "generation_time_s": None,
                "moderation_latency_s": None,
                "ttft_s": None,
                "completion_tps": None,
                "e2e_completion_tps": None,
                "raw_output": "",
                "structured_output_strategy": "create_agent_tool_strategy",
                "parsed": parsed.model_dump(),
            }
        ]

    per_message_latency = latency_s / len(ai_messages)
    records = []
    for index, message in enumerate(ai_messages, start=1):
        usage = _usage(message)
        record = {
            "role": role,
            "attempt": index if outer_attempt == 1 else ((outer_attempt - 1) * len(ai_messages)) + index,
            "model_name": model_name,
            "latency_s": per_message_latency,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "cost": usage.get("cost"),
            "cost_details": usage.get("cost_details") or {},
            "prompt_tokens_details": usage.get("prompt_tokens_details") or {},
            "completion_tokens_details": usage.get("completion_tokens_details") or {},
            "raw_usage": usage.get("raw_usage") or {},
            "generation_id": usage.get("generation_id"),
            "request_id": usage.get("request_id"),
            "upstream_id": usage.get("upstream_id"),
            "provider_name": usage.get("provider_name"),
            "provider_latency_s": usage.get("provider_latency_s"),
            "generation_time_s": usage.get("generation_time_s"),
            "moderation_latency_s": usage.get("moderation_latency_s"),
            "ttft_s": usage.get("ttft_s"),
            "completion_tps": usage.get("completion_tps"),
            "e2e_completion_tps": _e2e_completion_tps(usage["completion_tokens"], per_message_latency),
            "raw_output": _message_raw_output(message),
            "structured_output_strategy": "create_agent_tool_strategy",
        }
        if index == len(ai_messages):
            record["parsed"] = parsed.model_dump()
        records.append(record)
    return records


def _message_raw_output(message: Any) -> str:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return json_dumps_safe(tool_calls)
    return _response_text(message)


def json_dumps_safe(payload: Any) -> str:
    try:
        import json

        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)


def _e2e_completion_tps(completion_tokens: int, latency_s: float) -> float | None:
    if completion_tokens <= 0 or latency_s <= 0:
        return None
    return completion_tokens / latency_s


class MultiAgentWorkflow:
    """Run the naive multi-agent graph for one text sample."""

    def __init__(
        self,
        *,
        model: InvokableModel,
        model_name: str = "unknown",
        max_retries: int = 3,
        max_json_retries: int | None = None,
        mode: str | WorkflowModeConfig = "bool",
        models_by_role: dict[str, InvokableModel] | None = None,
        model_names_by_role: dict[str, str] | None = None,
        use_create_agent: bool = True,
    ):
        self.model = model
        self.model_name = model_name
        self.models_by_role = models_by_role or {}
        self.model_names_by_role = model_names_by_role or {}
        self.max_retries = max(0, int(max_json_retries if max_json_retries is not None else max_retries))
        self.use_create_agent = use_create_agent
        self.mode_config = get_workflow_mode(mode)
        self.mode_name = self.mode_config.name
        self._structured_agent_cache: dict[tuple[int, str, str], Any] = {}
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise ImportError("LangGraph is required. Install with: pip install -e .") from exc

        builder = StateGraph(WorkflowState)
        node_names = []
        prompt_by_label = {
            prompt_config.label_name: prompt_config
            for prompt_config in self.mode_config.specialist_prompts.values()
        }
        for label in self.mode_config.agent_labels:
            prompt_config = prompt_by_label[label]
            node_name = f"{label.lower()}_{prompt_config.role_suffix}"
            node_names.append(node_name)
            builder.add_node(node_name, self._make_intermediate_node(prompt_config))
            builder.add_edge(START, node_name)
        builder.add_node("judge", self._judge_node)
        builder.add_edge(node_names, "judge")
        builder.add_edge("judge", END)
        return builder.compile()

    def _can_use_create_agent(self, model: InvokableModel) -> bool:
        return hasattr(model, "bind_tools")

    def _get_structured_agent(
        self,
        *,
        model: InvokableModel,
        system_prompt: str,
        schema: type[Any],
    ) -> Any:
        cache_key = (id(model), schema.__name__, system_prompt)
        if cache_key not in self._structured_agent_cache:
            try:
                from langchain.agents import create_agent
                from langchain.agents.structured_output import ToolStrategy
            except ImportError as exc:
                raise ImportError("langchain is required for create_agent structured output.") from exc

            self._structured_agent_cache[cache_key] = create_agent(
                model=model,
                tools=[],
                system_prompt=system_prompt,
                # The evaluator owns retry limits and telemetry. Let one malformed
                # structured response escape so the outer loop records exactly one
                # request per configured attempt.
                response_format=ToolStrategy(schema, handle_errors=False),
            )
        return self._structured_agent_cache[cache_key]

    def _call_and_parse(
        self,
        *,
        role: str,
        messages: list[dict[str, str]],
        schema: type[Any],
        validator: Callable[[Any], None] | None = None,
    ) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
        model = self.models_by_role.get(role, self.model)
        model_name = self.model_names_by_role.get(role, self.model_name)
        if (
            self.use_create_agent
            and self._can_use_create_agent(model)
            and not self.mode_config.evidence_mode
        ):
            parsed, calls, errors = self._call_with_create_agent(
                role=role,
                messages=messages,
                schema=schema,
                model=model,
                model_name=model_name,
                validator=validator,
            )
        else:
            parsed, calls, errors = self._call_and_parse_manually(
                role=role,
                messages=messages,
                schema=schema,
                model=model,
                model_name=model_name,
                validator=validator,
            )
        for call in calls:
            call["_input_messages"] = [dict(message) for message in messages]
        return parsed, calls, errors

    def _call_with_create_agent(
        self,
        *,
        role: str,
        messages: list[dict[str, str]],
        schema: type[Any],
        model: InvokableModel,
        model_name: str,
        validator: Callable[[Any], None] | None = None,
    ) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []
        final_errors: list[dict[str, Any]] = []
        attempts = self.max_retries + 1
        system_prompt = next((message["content"] for message in messages if message["role"] == "system"), "")
        agent_messages = [message for message in messages if message["role"] != "system"]
        agent = self._get_structured_agent(model=model, system_prompt=system_prompt, schema=schema)

        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                result = agent.invoke({"messages": agent_messages})
                latency_s = time.perf_counter() - started
                structured = result.get("structured_response")
                parsed = structured if isinstance(structured, schema) else schema.model_validate(structured)
                if validator is not None:
                    validator(parsed)
                calls.extend(
                    _agent_call_records(
                        result=result,
                        role=role,
                        model_name=model_name,
                        outer_attempt=attempt,
                        latency_s=latency_s,
                        parsed=parsed,
                    )
                )
                return parsed, calls, final_errors
            except Exception as exc:
                latency_s = time.perf_counter() - started
                failed_message = getattr(exc, "ai_message", None)
                usage = _usage(failed_message)
                error = {
                    "role": role,
                    "attempt": attempt,
                    "latency_s": latency_s,
                    "error_type": exc.__class__.__name__,
                    "error_message": _sanitize_error_message(str(exc)),
                }
                calls.append(
                    {
                        "role": role,
                        "attempt": attempt,
                        "model_name": model_name,
                        "latency_s": latency_s,
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "cost": usage.get("cost"),
                        "cost_details": usage.get("cost_details") or {},
                        "prompt_tokens_details": usage.get("prompt_tokens_details") or {},
                        "completion_tokens_details": usage.get("completion_tokens_details") or {},
                        "raw_usage": usage.get("raw_usage") or {},
                        "generation_id": usage.get("generation_id"),
                        "request_id": usage.get("request_id"),
                        "upstream_id": usage.get("upstream_id"),
                        "provider_name": usage.get("provider_name"),
                        "provider_latency_s": usage.get("provider_latency_s"),
                        "generation_time_s": usage.get("generation_time_s"),
                        "moderation_latency_s": usage.get("moderation_latency_s"),
                        "ttft_s": usage.get("ttft_s"),
                        "completion_tps": usage.get("completion_tps"),
                        "e2e_completion_tps": _e2e_completion_tps(
                            usage["completion_tokens"], latency_s
                        ),
                        "raw_output": (
                            _message_raw_output(failed_message)
                            if failed_message is not None
                            else ""
                        ),
                        "structured_output_strategy": "create_agent_tool_strategy",
                        "error": error,
                    }
                )
                if attempt >= attempts:
                    final_errors.append(error)
                    return None, calls, final_errors
        return None, calls, final_errors

    def _call_and_parse_manually(
        self,
        *,
        role: str,
        messages: list[dict[str, str]],
        schema: type[Any],
        model: InvokableModel,
        model_name: str,
        validator: Callable[[Any], None] | None = None,
    ) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []
        final_errors: list[dict[str, Any]] = []
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                response = model.invoke(messages)
                latency_s = time.perf_counter() - started
                raw = _response_text(response)
                usage = _usage(response)
                call_record = {
                    "role": role,
                    "attempt": attempt,
                    "model_name": model_name,
                    "latency_s": latency_s,
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "cost": usage.get("cost"),
                    "cost_details": usage.get("cost_details") or {},
                    "prompt_tokens_details": usage.get("prompt_tokens_details") or {},
                    "completion_tokens_details": usage.get("completion_tokens_details") or {},
                    "raw_usage": usage.get("raw_usage") or {},
                    "generation_id": usage.get("generation_id"),
                    "request_id": usage.get("request_id"),
                    "upstream_id": usage.get("upstream_id"),
                    "provider_name": usage.get("provider_name"),
                    "provider_latency_s": usage.get("provider_latency_s"),
                    "generation_time_s": usage.get("generation_time_s"),
                    "moderation_latency_s": usage.get("moderation_latency_s"),
                    "ttft_s": usage.get("ttft_s"),
                    "completion_tps": usage.get("completion_tps"),
                    "e2e_completion_tps": _e2e_completion_tps(usage["completion_tokens"], latency_s),
                    "raw_output": raw,
                }
                calls.append(call_record)
                parsed = schema.model_validate(extract_json_object(raw))
                if validator is not None:
                    validator(parsed)
                call_record["parsed"] = parsed.model_dump()
                return parsed, calls, final_errors
            except Exception as exc:
                latency_s = time.perf_counter() - started
                error = {
                    "role": role,
                    "attempt": attempt,
                    "latency_s": latency_s,
                    "error_type": exc.__class__.__name__,
                    "error_message": _sanitize_error_message(str(exc)),
                }
                if calls and calls[-1].get("role") == role and calls[-1].get("attempt") == attempt:
                    calls[-1]["error"] = error
                else:
                    calls.append(
                        {
                            "role": role,
                            "attempt": attempt,
                            "model_name": model_name,
                            "latency_s": latency_s,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "cost": None,
                            "cost_details": {},
                            "prompt_tokens_details": {},
                            "completion_tokens_details": {},
                            "raw_usage": {},
                            "generation_id": None,
                            "request_id": None,
                            "upstream_id": None,
                            "provider_name": None,
                            "provider_latency_s": None,
                            "generation_time_s": None,
                            "moderation_latency_s": None,
                            "ttft_s": None,
                            "completion_tps": None,
                            "e2e_completion_tps": None,
                            "raw_output": "",
                            "error": error,
                        }
                    )
                if attempt >= attempts:
                    final_errors.append(error)
                    return None, calls, final_errors
        return None, calls, final_errors

    def _make_intermediate_node(self, prompt_config):
        def node(state: WorkflowState) -> dict[str, Any]:
            messages = [
                {"role": "system", "content": prompt_config.system_prompt},
                {"role": "user", "content": specialist_user_prompt(state["text"])},
            ]
            validator = None
            if not prompt_config.produces_answer:
                allowed_types = set(EVIDENCE_TYPE_DEFINITIONS[prompt_config.label_name])
                validator = lambda answer: validate_evidence_answer(
                    answer,
                    allowed_types=allowed_types,
                )
            node_started_at_s = time.perf_counter()
            parsed, calls, errors = self._call_and_parse(
                role=f"{prompt_config.label_name}_{prompt_config.role_suffix}",
                messages=messages,
                schema=self.mode_config.specialist_schema,
                validator=validator,
            )
            node_finished_at_s = time.perf_counter()
            output = parsed.model_dump() if parsed is not None else None
            specialist_answers = (
                {prompt_config.label_name: output["answer"]}
                if output and prompt_config.produces_answer
                else {}
            )
            specialist_outputs = (
                {prompt_config.label_name: output}
                if output and prompt_config.produces_answer
                else {}
            )
            collector_outputs = (
                {prompt_config.label_name: output}
                if output and not prompt_config.produces_answer
                else {}
            )
            traces = [
                {
                    "event": f"{prompt_config.role_suffix}_complete",
                    "agent_name": prompt_config.agent_name,
                    "label_name": prompt_config.label_name,
                    "parallel_group": prompt_config.parallel_group,
                    "ok": parsed is not None,
                    "node_started_at_s": node_started_at_s,
                    "node_finished_at_s": node_finished_at_s,
                    "node_duration_s": node_finished_at_s - node_started_at_s,
                }
            ]
            return {
                "specialist_answers": specialist_answers,
                "specialist_outputs": specialist_outputs,
                "collector_outputs": collector_outputs,
                "errors": errors,
                "traces": traces,
                "llm_calls": calls,
            }

        return node

    def _judge_node(self, state: WorkflowState) -> dict[str, Any]:
        if state.get("errors"):
            agent_kind = "collector" if self.mode_config.evidence_mode else "specialist"
            return {
                "final_label": "Error",
                "suicidal_intent": None,
                "traces": [{"event": "judge_skipped", "reason": f"{agent_kind}_error"}],
            }

        expected_labels = self.mode_config.agent_labels
        intermediate_outputs = (
            state.get("collector_outputs", {})
            if self.mode_config.evidence_mode
            else state.get("specialist_outputs", {})
        )
        missing_output_labels = [label for label in expected_labels if label not in intermediate_outputs]
        extra_output_labels = sorted(set(intermediate_outputs) - set(expected_labels))
        answers = state.get("specialist_answers", {})
        missing_labels = (
            []
            if self.mode_config.evidence_mode
            else [label for label in expected_labels if label not in answers]
        )
        extra_labels = (
            []
            if self.mode_config.evidence_mode
            else sorted(set(answers) - set(expected_labels))
        )
        if missing_labels or extra_labels or missing_output_labels or extra_output_labels:
            agent_kind = "collector" if self.mode_config.evidence_mode else "specialist"
            return {
                "final_label": "Error",
                "suicidal_intent": None,
                "errors": [
                    {
                        "role": "judge",
                        "error_type": f"Missing{agent_kind.title()}Output",
                        "error_message": (
                            f"Expected one {agent_kind} output for each label; "
                            f"missing={missing_labels}, extra={extra_labels}, "
                            f"missing_outputs={missing_output_labels}, extra_outputs={extra_output_labels}"
                        ),
                    }
                ],
                "traces": [{"event": "judge_skipped", "reason": f"missing_{agent_kind}_output"}],
            }

        messages = [
            {"role": "system", "content": self.mode_config.judge_system_prompt()},
            {
                "role": "user",
                "content": self.mode_config.judge_user_prompt(state["text"], intermediate_outputs),
            },
        ]
        node_started_at_s = time.perf_counter()
        parsed, calls, errors = self._call_and_parse(
            role="judge",
            messages=messages,
            schema=self.mode_config.judge_schema,
        )
        node_finished_at_s = time.perf_counter()
        final_label = parsed.label_name if parsed is not None else "Error"
        suicidal_intent = parsed.suicidal_intent if parsed is not None else None
        return {
            "final_label": final_label,
            "suicidal_intent": suicidal_intent,
            "errors": errors,
            "traces": [
                {
                    "event": "judge_complete",
                    "ok": parsed is not None,
                    "final_label": final_label,
                    "suicidal_intent": suicidal_intent,
                    "node_started_at_s": node_started_at_s,
                    "node_finished_at_s": node_finished_at_s,
                    "node_duration_s": node_finished_at_s - node_started_at_s,
                }
            ],
            "llm_calls": calls,
        }

    def run_case(self, *, sample_id: str, text: str) -> CaseResult:
        initial_state: WorkflowState = {
            "sample_id": sample_id,
            "text": text,
            "specialist_answers": {},
            "specialist_outputs": {},
            "collector_outputs": {},
            "final_label": "",
            "suicidal_intent": None,
            "errors": [],
            "traces": [],
            "llm_calls": [],
        }
        result = self.graph.invoke(initial_state)
        specialist_answers = {
            label: result.get("specialist_answers", {})[label]
            for label in self.mode_config.agent_labels
            if label in result.get("specialist_answers", {})
        }
        specialist_outputs = {
            label: result.get("specialist_outputs", {})[label]
            for label in self.mode_config.agent_labels
            if label in result.get("specialist_outputs", {})
        }
        collector_outputs = {
            label: result.get("collector_outputs", {})[label]
            for label in self.mode_config.agent_labels
            if label in result.get("collector_outputs", {})
        }
        agent_outputs = {
            f"{label}_specialist": output
            for label, output in specialist_outputs.items()
        }
        agent_outputs.update(
            {
                f"{label}_collector": output
                for label, output in collector_outputs.items()
            }
        )
        if result.get("final_label"):
            agent_outputs["judge"] = {
                "label_name": result.get("final_label") or "Error",
                "suicidal_intent": result.get("suicidal_intent"),
            }
        return CaseResult(
            sample_id=sample_id,
            text=text,
            final_label=result.get("final_label") or "Error",
            suicidal_intent=result.get("suicidal_intent"),
            specialist_answers=specialist_answers,
            specialist_outputs=specialist_outputs,
            collector_outputs=collector_outputs,
            errors=result.get("errors", []),
            traces=result.get("traces", []),
            llm_calls=result.get("llm_calls", []),
            agent_outputs=agent_outputs,
        )
