"""LangGraph workflow for hierarchical suicide-risk classification."""

from __future__ import annotations

import time
from dataclasses import dataclass
import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel

from langgraph.graph import END, START, StateGraph

from domain.labels import (
    FINAL_LABELS,
    VALID_LABELS,
    canonicalize_label,
    label_to_suicidal_intent,
)

from domain.final_output import extract_json_object
from llm.invoker import response_text as _response_text, usage_summary as _usage
from llm.protocol import InvokableModel
from .prompts import (
    BEHAVIOR_ATTEMPT_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    IDEATION_ACTION_SYSTEM_PROMPT,
    PERSPECTIVE_SYSTEM_PROMPT,
    evaluator_user_prompt,
    user_prompt,
)
from .schemas import (
    BehaviorAttemptDecision,
    EvaluatorDecision,
    IdeationActionDecision,
    PerspectiveDecision,
)


class WorkflowState(TypedDict):
    sample_id: str
    text: str
    final_label: str
    proposed_label: str
    specialist_answers: dict[str, bool]
    specialist_outputs: dict[str, Any]
    path: list[dict[str, Any]]
    errors: Annotated[list[dict[str, Any]], operator.add]
    traces: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class CaseResult:
    sample_id: str
    text: str
    final_label: str
    suicidal_intent: bool
    # specialist_answers: dict[str, bool]
    # specialist_outputs: dict[str, dict[str, Any] | None]
    errors: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]


class MultiAgentWorkflow:
    """Hierarchical decision-tree multi-agent workflow."""

    def __init__(
    self,
    *,
    model: InvokableModel,
    model_name: str = "unknown",
    max_retries: int = 2,
    confidence_threshold: float = 0.65,
    mode: str = "bool",
    models_by_role: dict[str, InvokableModel] | None = None,
    model_names_by_role: dict[str, str] | None = None,
    ):
        self.model = model
        self.model_name = model_name
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold
        self.mode = mode
        self.models_by_role = models_by_role or {}
        self.model_names_by_role = model_names_by_role or {}
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkflowState)

        builder.add_node("perspective", self._perspective_node)
        builder.add_node("ideation_action", self._ideation_action_node)
        builder.add_node("behavior_attempt", self._behavior_attempt_node)
        builder.add_node("evaluator", self._evaluator_node)

        builder.add_edge(START, "perspective")

        builder.add_conditional_edges(
            "perspective",
            self._route_after_perspective,
            {
                "ideation_action": "ideation_action",
                "evaluator": "evaluator",
            },
        )

        builder.add_conditional_edges(
            "ideation_action",
            self._route_after_ideation_action,
            {
                "behavior_attempt": "behavior_attempt",
                "evaluator": "evaluator",
            },
        )

        builder.add_edge("behavior_attempt", "evaluator")
        builder.add_edge("evaluator", END)

        return builder.compile()

    def _call_llm(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt_text: str,
        schema: type[BaseModel],
    ) -> tuple[Any | None, list[dict[str, Any]], list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_text},
        ]

        for attempt in range(1, self.max_retries + 2):
            started = time.perf_counter()

            raw = ""
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": None,
            }

            try:
                response = self.model.invoke(messages)
                latency_s = time.perf_counter() - started
                raw = _response_text(response)
                usage = _usage(response)
                parsed = schema.model_validate(extract_json_object(raw))

                calls.append(
                    {
                        "role": role,
                        "attempt": attempt,
                        "model_name": self.model_name,
                        "latency_s": latency_s,
                        "raw_output": raw,
                        "parsed": parsed.model_dump(),
                        **usage,
                    }
                )
                return parsed, calls, errors

            except Exception as exc:
                latency_s = time.perf_counter() - started
                error = {
                    "role": role,
                    "attempt": attempt,
                    "latency_s": latency_s,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
                calls.append(
                    {
                        "role": role,
                        "attempt": attempt,
                        "model_name": self.model_name,
                        "latency_s": latency_s,
                        "raw_output": raw,
                        **usage,
                        "error": error,
                    }
                )
                if attempt >= self.max_retries + 1:
                    errors.append(error)
                    return None, calls, errors

        return None, calls, errors

    def _perspective_node(self, state: WorkflowState) -> dict[str, Any]:
        parsed, calls, errors = self._call_llm(
            role="perspective_agent",
            system_prompt=PERSPECTIVE_SYSTEM_PROMPT,
            user_prompt_text=user_prompt(state["text"]),
            schema=PerspectiveDecision,
        )

        if parsed is None:
            return {"final_label": "Error", "errors": errors, "llm_calls": calls}

        decision = parsed.decision.strip().upper()
        path_item = {
            "agent": "perspective_agent",
            "decision": decision,
            "confidence": parsed.confidence,
            "reasoning": parsed.reasoning,
        }

        if decision == "SELF_RISK":
            proposed_label = ""
            specialist_answers = {}

        elif decision == "INDICATOR":
            proposed_label = "Indicator"
            specialist_answers = {"Indicator": True}

        elif decision == "UNSUPPORTIVE":
            proposed_label = "Unsupportive"
            specialist_answers = {}

        else:
            raise ValueError(f"Unexpected perspective decision: {decision}")

        return {
            "proposed_label": proposed_label,
            "specialist_answers": specialist_answers,
            "path": [path_item],
            "traces": [{"event": "perspective_complete", **path_item}],
            "errors": errors,
            "llm_calls": calls,
        }

    def _route_after_perspective(self, state: WorkflowState) -> str:
        return "evaluator" if state.get("proposed_label") else "ideation_action"

    def _ideation_action_node(self, state: WorkflowState) -> dict[str, Any]:
        parsed, calls, errors = self._call_llm(
            role="ideation_action_agent",
            system_prompt=IDEATION_ACTION_SYSTEM_PROMPT,
            user_prompt_text=user_prompt(state["text"]),
            schema=IdeationActionDecision,
        )

        if parsed is None:
            return {"final_label": "Error", "errors": errors, "llm_calls": calls}

        decision = parsed.decision.strip().upper()
        path_item = {
            "agent": "ideation_action_agent",
            "decision": decision,
            "confidence": parsed.confidence,
            "reasoning": parsed.reasoning,
        }

        if decision == "IDEATION":
            proposed_label = "Ideation"
            specialist_answers = {"Ideation": True}
        else:
            proposed_label = ""
            specialist_answers = {}

        return {
            "proposed_label": proposed_label,
            "specialist_answers": specialist_answers,
            "path": state.get("path", []) + [path_item],
            "traces": [{"event": "ideation_action_complete", **path_item}],
            "errors": errors,
            "llm_calls": calls,
        }

    def _route_after_ideation_action(self, state: WorkflowState) -> str:
        return "evaluator" if state.get("proposed_label") else "behavior_attempt"

    def _behavior_attempt_node(self, state: WorkflowState) -> dict[str, Any]:
        parsed, calls, errors = self._call_llm(
            role="behavior_attempt_agent",
            system_prompt=BEHAVIOR_ATTEMPT_SYSTEM_PROMPT,
            user_prompt_text=user_prompt(state["text"]),
            schema=BehaviorAttemptDecision,
        )

        if parsed is None:
            return {"final_label": "Error", "errors": errors, "llm_calls": calls}

        decision = parsed.decision.strip().upper()
        path_item = {
            "agent": "behavior_attempt_agent",
            "decision": decision,
            "confidence": parsed.confidence,
            "reasoning": parsed.reasoning,
        }

        proposed_label = "Attempt" if decision == "ATTEMPT" else "Behavior"
        specialist_answers = {proposed_label: True}

        return {
            "proposed_label": proposed_label,
            "specialist_answers": specialist_answers,
            "path": state.get("path", []) + [path_item],
            "traces": [{"event": "behavior_attempt_complete", **path_item}],
            "errors": errors,
            "llm_calls": calls,
        }

    def _evaluator_node(self, state: WorkflowState) -> dict[str, Any]:
        proposed_label = state.get("proposed_label") or state.get("final_label") or "Error"

        if proposed_label == "Error":
            return {
                "final_label": "Error",
                "traces": [{"event": "evaluator_skipped", "reason": "previous_error"}],
            }

        parsed, calls, errors = self._call_llm(
            role="evaluator_agent",
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            user_prompt_text=evaluator_user_prompt(
                text=state["text"],
                proposed_label=proposed_label,
                path=state.get("path", []),
            ),
            schema=EvaluatorDecision,
        )

        if parsed is None:
            return {"final_label": "Error", "errors": errors, "llm_calls": calls}

        final_label = parsed.label_name
        trace = {
            "event": "evaluator_complete",
            "proposed_label": proposed_label,
            "final_label": final_label,
            "confidence": parsed.confidence,
            "escalation_flag": parsed.escalation_flag,
            "reasoning": parsed.reasoning,
        }

        return {
            "final_label": final_label,
            "traces": [trace],
            "errors": errors,
            "llm_calls": calls,
        }

    def run_case(self, *, sample_id: str, text: str) -> CaseResult:
        initial_state: WorkflowState = {
            "sample_id": sample_id,
            "text": text,
            "final_label": "",
            "proposed_label": "",
            "specialist_answers": {},
            "specialist_outputs": {},
            "path": [],
            "errors": [],
            "traces": [],
            "llm_calls": [],
        }

        result = self.graph.invoke(initial_state)
        final_label = result.get("final_label") or "Error"

        # # Ming's evaluator expects one boolean answer for each canonical label.
        # # These are compatibility fields, not claims that four independent
        # # specialists were executed.
        # specialist_answers = {
        #     label: final_label == label
        #     for label in VALID_LABELS
        # }

        # # Find the evaluator's final reasoning, when available.
        # final_reasoning: str | None = None
        # for trace in reversed(result.get("traces", [])):
        #     if trace.get("event") == "evaluator_complete":
        #         final_reasoning = trace.get("reasoning")
        #         break

        # # Ming's evaluator also expects an output dictionary for each label.
        # specialist_outputs = {
        #     label: {
        #         "answer": specialist_answers[label],
        #         "reasoning": final_reasoning if specialist_answers[label] else None,
        #     }
        #     for label in VALID_LABELS
        # }

        suicidal_intent = (
            label_to_suicidal_intent(final_label)
            if final_label in FINAL_LABELS
            else False
        )

        return CaseResult(
            sample_id=sample_id,
            text=text,
            final_label=final_label,
            suicidal_intent=suicidal_intent,
            errors=result.get("errors", []),
            traces=result.get("traces", []),
            llm_calls=result.get("llm_calls", []),
        )
