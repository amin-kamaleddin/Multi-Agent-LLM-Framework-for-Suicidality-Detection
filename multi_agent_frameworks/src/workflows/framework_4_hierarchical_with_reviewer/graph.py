# """Confidence-gated hierarchical LangGraph workflow.

# The workflow preserves the same public input/output contract as the existing
# framework while adding:

# 1. a Judge Agent that returns a candidate label and competing label;
# 2. a deterministic uncertainty gate;
# 3. a Secondary Review Agent invoked only for ambiguous cases.
# """

# from __future__ import annotations

# import re
# import time
# from dataclasses import dataclass
# from typing import Annotated, Any, TypedDict

# from langgraph.graph import END, START, StateGraph
# from pydantic import BaseModel

# from domain.labels import (
#     FINAL_LABELS,
#     VALID_LABELS,
#     canonicalize_label,
#     label_to_suicidal_intent,
# )

# from domain.final_output import extract_json_object
# from llm.invoker import response_text as _response_text, usage_summary as _usage
# from llm.protocol import InvokableModel
# from .prompts import (
#     BEHAVIOR_ATTEMPT_SYSTEM_PROMPT,
#     IDEATION_ACTION_SYSTEM_PROMPT,
#     JUDGE_SYSTEM_PROMPT,
#     PERSPECTIVE_SYSTEM_PROMPT,
#     SECONDARY_REVIEW_SYSTEM_PROMPT,
#     judge_user_prompt,
#     secondary_review_user_prompt,
#     user_prompt,
# )

# from .schemas import (
#     BehaviorAttemptDecision,
#     IdeationActionDecision,
#     JudgeDecision,
#     PerspectiveDecision,
#     ReviewDecision,
# )


# class WorkflowState(TypedDict):
#     sample_id: str
#     text: str

#     proposed_label: str
#     final_label: str

#     judge_label: str
#     judge_confidence: float
#     competing_label: str
#     competing_confidence: float
#     review_reasons: list[str]
#     escalation_flag: bool

#     specialist_answers: dict[str, bool]
#     path: list[dict[str, Any]]
#     errors: list[dict[str, Any]]
#     traces: list[dict[str, Any]]
#     llm_calls: list[dict[str, Any]]



# @dataclass
# class CaseResult:
#     """Public output contract expected by the shared evaluator."""

#     sample_id: str
#     text: str
#     final_label: str
#     suicidal_intent: bool
#     errors: list[dict[str, Any]]
#     traces: list[dict[str, Any]]
#     llm_calls: list[dict[str, Any]]





# def _sanitize_error_message(message: str) -> str:
#     """Remove provider-side user identifiers from stored errors."""

#     sanitized = re.sub(
#         r"('user_id'\s*:\s*)'[^']+'",
#         r"\1'[redacted]'",
#         message,
#     )
#     sanitized = re.sub(
#         r'("user_id"\s*:\s*)"[^"]+"',
#         r'\1"[redacted]"',
#         sanitized,
#     )
#     return sanitized


# class MultiAgentWorkflow:
#     """Confidence-gated hierarchical multi-agent workflow."""

#     def __init__(
#         self,
#         *,
#         model: InvokableModel,
#         model_name: str = "unknown",
#         max_retries: int = 2,
#         confidence_threshold: float = 0.85,
#         margin_threshold: float = 0.25,
#         review_known_boundaries_only: bool = False,
#         mode: str = "bool",
#         models_by_role: dict[str, InvokableModel] | None = None,
#         model_names_by_role: dict[str, str] | None = None,
#     ):
#         self.model = model
#         self.model_name = model_name
#         self.max_retries = max(0, int(max_retries))
#         self.confidence_threshold = _validate_unit_interval(
#             confidence_threshold,
#             "confidence_threshold",
#         )
#         self.margin_threshold = _validate_unit_interval(
#             margin_threshold,
#             "margin_threshold",
#         )
#         self.review_known_boundaries_only = bool(review_known_boundaries_only)
#         self.mode = mode
#         self.models_by_role = models_by_role or {}
#         self.model_names_by_role = model_names_by_role or {}
#         self.graph = self._build_graph()

#     def _build_graph(self):
#         builder = StateGraph(WorkflowState)

#         builder.add_node("perspective", self._perspective_node)
#         builder.add_node("ideation_action", self._ideation_action_node)
#         builder.add_node("behavior_attempt", self._behavior_attempt_node)
#         builder.add_node("judge", self._judge_node)
#         builder.add_node("secondary_review", self._secondary_review_node)
#         builder.add_node("finalize", self._finalize_node)

#         builder.add_edge(START, "perspective")

#         builder.add_conditional_edges(
#             "perspective",
#             self._route_after_perspective,
#             {
#                 "ideation_action": "ideation_action",
#                 "judge": "judge",
#             },
#         )

#         builder.add_conditional_edges(
#             "ideation_action",
#             self._route_after_ideation_action,
#             {
#                 "behavior_attempt": "behavior_attempt",
#                 "judge": "judge",
#             },
#         )

#         builder.add_edge("behavior_attempt", "judge")

#         builder.add_conditional_edges(
#             "judge",
#             self._route_after_judge,
#             {
#                 "secondary_review": "secondary_review",
#                 "finalize": "finalize",
#             },
#         )

#         builder.add_edge("secondary_review", "finalize")
#         builder.add_edge("finalize", END)

#         return builder.compile()

#     def _call_llm(
#         self,
#         *,
#         role: str,
#         system_prompt: str,
#         user_prompt_text: str,
#         schema: type[BaseModel],
#     ) -> tuple[Any | None, list[dict[str, Any]], list[dict[str, Any]]]:
#         calls: list[dict[str, Any]] = []
#         final_errors: list[dict[str, Any]] = []

#         selected_model = self.models_by_role.get(role, self.model)
#         selected_model_name = self.model_names_by_role.get(role, self.model_name)

#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt_text},
#         ]

#         attempts = self.max_retries + 1

#         for attempt in range(1, self.max_retries + 2):
#             started = time.perf_counter()

#             raw = ""
#             usage = {
#                 "prompt_tokens": 0,
#                 "completion_tokens": 0,
#                 "total_tokens": 0,
#                 "cost": None,
#             }

#             try:
#                 response = selected_model.invoke(messages)
#                 latency_s = time.perf_counter() - started
#                 raw = _response_text(response)
#                 usage = _usage(response)
#                 parsed = schema.model_validate(extract_json_object(raw))

#                 calls.append(
#                     {
#                         "role": role,
#                         "attempt": attempt,
#                         "model_name": selected_model_name,
#                         "_input_messages": [dict(message) for message in messages],
#                         "latency_s": latency_s,
#                         "raw_output": raw,
#                         "parsed": parsed.model_dump(),
#                         **usage,
#                     }
#                 )
#                 return parsed, calls, final_errors

#             except Exception as exc:
#                 latency_s = time.perf_counter() - started
#                 error = {
#                     "role": role,
#                     "attempt": attempt,
#                     "latency_s": latency_s,
#                     "error_type": exc.__class__.__name__,
#                     "error_message": _sanitize_error_message(str(exc)),
#                 }

#                 calls.append(
#                     {
#                         "role": role,
#                         "attempt": attempt,
#                         "model_name": selected_model_name,
#                         "_input_messages": [dict(message) for message in messages],
#                         "latency_s": latency_s,
#                         "raw_output": raw,
#                         **usage,
#                         "error": error,
#                     }
#                 )

#                 if attempt >= attempts:
#                     final_errors.append(error)
#                     return None, calls, final_errors

#         return None, calls, final_errors

#     def _perspective_node(self, state: WorkflowState) -> dict[str, Any]:
#         parsed, calls, errors = self._call_llm(
#             role="perspective_agent",
#             system_prompt=PERSPECTIVE_SYSTEM_PROMPT,
#             user_prompt_text=user_prompt(state["text"]),
#             schema=PerspectiveDecision,
#         )

#         if parsed is None:
#             return self._failed_node_update(
#                 state,
#                 role="perspective_agent",
#                 calls=calls,
#                 errors=errors,
#             )

#         decision = parsed.decision.strip().upper()
#         if decision not in {"SELF_RISK", "INDICATOR", "UNSUPPORTIVE"}:
#             return self._invalid_decision_update(
#                 state,
#                 role="perspective_agent",
#                 decision=decision,
#                 calls=calls,
#             )

#         path_item = {
#             "agent": "perspective_agent",
#             "decision": decision,
#             "confidence": parsed.confidence,
#             "reasoning": parsed.reasoning,
#         }

#         if decision == "SELF_RISK":
#             proposed_label = ""
#         elif decision == "INDICATOR":
#             proposed_label = "Indicator"
#         else:
#             proposed_label = "Unsupportive"


#         return {
#             "proposed_label": proposed_label,
#             "path": [path_item],
#             "errors": [*state.get("errors", []), *errors],
#             "traces": [
#                 *state.get("traces", []),
#                 {"event": "perspective_complete", **path_item},
#             ],
#             "llm_calls": [*state.get("llm_calls", []), *calls],
#         }

#     def _route_after_perspective(self, state: WorkflowState) -> str:
#         if state.get("final_label") == "Error":
#             return "judge"
#         return "judge" if state.get("proposed_label") else "ideation_action"

#     def _ideation_action_node(self, state: WorkflowState) -> dict[str, Any]:
#         parsed, calls, errors = self._call_llm(
#             role="ideation_action_agent",
#             system_prompt=IDEATION_ACTION_SYSTEM_PROMPT,
#             user_prompt_text=user_prompt(state["text"]),
#             schema=IdeationActionDecision,
#         )

#         if parsed is None:
#             return self._failed_node_update(
#                 state,
#                 role="ideation_action_agent",
#                 calls=calls,
#                 errors=errors,
#             )

#         decision = parsed.decision.strip().upper()
#         if decision not in {"IDEATION", "ACTION_PATHWAY"}:
#             return self._invalid_decision_update(
#                 state,
#                 role="ideation_action_agent",
#                 decision=decision,
#                 calls=calls,
#             )

#         path_item = {
#             "agent": "ideation_action_agent",
#             "decision": decision,
#             "confidence": parsed.confidence,
#             "reasoning": parsed.reasoning,
#         }

#         proposed_label = "Ideation" if decision == "IDEATION" else ""

#         return {
#             "proposed_label": proposed_label,
#             "path": [*state.get("path", []), path_item],
#             "errors": [*state.get("errors", []), *errors],
#             "traces": [
#                 *state.get("traces", []),
#                 {"event": "ideation_action_complete", **path_item},
#             ],
#             "llm_calls": [*state.get("llm_calls", []), *calls],
#         }

#     def _route_after_ideation_action(self, state: WorkflowState) -> str:
#         if state.get("final_label") == "Error":
#             return "judge"
#         return "judge" if state.get("proposed_label") else "behavior_attempt"

#     def _behavior_attempt_node(self, state: WorkflowState) -> dict[str, Any]:
#         parsed, calls, errors = self._call_llm(
#             role="behavior_attempt_agent",
#             system_prompt=BEHAVIOR_ATTEMPT_SYSTEM_PROMPT,
#             user_prompt_text=user_prompt(state["text"]),
#             schema=BehaviorAttemptDecision,
#         )

#         if parsed is None:
#             return self._failed_node_update(
#                 state,
#                 role="behavior_attempt_agent",
#                 calls=calls,
#                 errors=errors,
#             )

#         decision = parsed.decision.strip().upper()
#         if decision not in {"BEHAVIOR", "ATTEMPT"}:
#             return self._invalid_decision_update(
#                 state,
#                 role="behavior_attempt_agent",
#                 decision=decision,
#                 calls=calls,
#             )

#         path_item = {
#             "agent": "behavior_attempt_agent",
#             "decision": decision,
#             "confidence": parsed.confidence,
#             "reasoning": parsed.reasoning,
#         }

#         proposed_label = "Attempt" if decision == "ATTEMPT" else "Behavior"

#         return {
#             "proposed_label": proposed_label,
#             "path": [*state.get("path", []), path_item],
#             "errors": [*state.get("errors", []), *errors],
#             "traces": [
#                 *state.get("traces", []),
#                 {"event": "behavior_attempt_complete", **path_item},
#             ],
#             "llm_calls": [*state.get("llm_calls", []), *calls],
#         }

#     def _judge_node(self, state: WorkflowState) -> dict[str, Any]:
#         if state.get("final_label") == "Error" or state.get("errors"):
#             return {
#                 "judge_label": "Error",
#                 "final_label": "Error",
#                 "review_reasons": ["previous_node_error"],
#                 "traces": [
#                     *state.get("traces", []),
#                     {"event": "judge_skipped", "reason": "previous_node_error"},
#                 ],
#             }

#         proposed_label = state.get("proposed_label")
#         if proposed_label not in FINAL_LABELS:
#             error = {
#                 "role": "judge_agent",
#                 "error_type": "MissingProposedLabel",
#                 "error_message": (
#                     f"Expected a valid proposed label, received {proposed_label!r}."
#                 ),
#             }
#             return {
#                 "judge_label": "Error",
#                 "final_label": "Error",
#                 "errors": [*state.get("errors", []), error],
#                 "review_reasons": ["missing_proposed_label"],
#                 "traces": [
#                     *state.get("traces", []),
#                     {"event": "judge_skipped", "reason": "missing_proposed_label"},
#                 ],
#             }

#         parsed, calls, errors = self._call_llm(
#             role="judge_agent",
#             system_prompt=JUDGE_SYSTEM_PROMPT,
#             user_prompt_text=judge_user_prompt(
#                 text=state["text"],
#                 proposed_label=proposed_label,
#                 path=state.get("path", []),
#             ),
#             schema=JudgeDecision,
#         )

#         if parsed is None:
#             return self._failed_node_update(
#                 state,
#                 role="judge_agent",
#                 calls=calls,
#                 errors=errors,
#             )

#         review_reasons = self._review_reasons(
#             state=state,
#             judge=parsed,
#         )
#         margin = abs(parsed.confidence - parsed.competing_confidence)

#         trace = {
#             "event": "judge_complete",
#             "label_name": parsed.label_name,
#             "confidence": parsed.confidence,
#             "competing_label": parsed.competing_label,
#             "competing_confidence": parsed.competing_confidence,
#             "margin": margin,
#             "review_required": bool(review_reasons),
#             "review_reasons": review_reasons,
#             "reasoning": parsed.reasoning,
#         }

#         return {
#             "judge_label": parsed.label_name,
#             "judge_confidence": parsed.confidence,
#             "competing_label": parsed.competing_label,
#             "competing_confidence": parsed.competing_confidence,
#             "review_reasons": review_reasons,
#             "errors": [*state.get("errors", []), *errors],
#             "traces": [*state.get("traces", []), trace],
#             "llm_calls": [*state.get("llm_calls", []), *calls],
#         }
        

#     def _route_after_judge(self, state: WorkflowState) -> str:
#         if state.get("judge_label") == "Error" or state.get("final_label") == "Error":
#             return "finalize"
#         return "secondary_review" if state.get("review_reasons") else "finalize"

#     def _secondary_review_node(self, state: WorkflowState) -> dict[str, Any]:
#         parsed, calls, errors = self._call_llm(
#             role="secondary_review_agent",
#             system_prompt=SECONDARY_REVIEW_SYSTEM_PROMPT,
#             user_prompt_text=secondary_review_user_prompt(
#                 text=state["text"],
#                 judge_label=state["judge_label"],
#                 judge_confidence=state["judge_confidence"],
#                 competing_label=state["competing_label"],
#                 competing_confidence=state["competing_confidence"],
#                 review_reasons=state.get("review_reasons", []),
#                 path=state.get("path", []),
#             ),
#             schema=ReviewDecision,
#         )

#         if parsed is None:
#             # Keep the judge label rather than throwing away an otherwise valid case.
#             trace = {
#                 "event": "secondary_review_failed",
#                 "fallback_label": state.get("judge_label") or "Error",
#             }
#             return {
#                 "final_label": state.get("judge_label") or "Error",
#                 "escalation_flag": True,
#                 "errors": [*state.get("errors", []), *errors],
#                 "traces": [*state.get("traces", []), trace],
#                 "llm_calls": [*state.get("llm_calls", []), *calls],
#             }

#         trace = {
#             "event": "secondary_review_complete",
#             "original_judge_label": state["judge_label"],
#             "reviewed_label": parsed.label_name,
#             "confidence": parsed.confidence,
#             "escalation_flag": parsed.escalation_flag,
#             "review_reasons": state.get("review_reasons", []),
#             "reasoning": parsed.reasoning,
#         }

#         return {
#             "final_label": parsed.label_name,
#             "escalation_flag": parsed.escalation_flag,
#             "errors": [*state.get("errors", []), *errors],
#             "traces": [*state.get("traces", []), trace],
#             "llm_calls": [*state.get("llm_calls", []), *calls],
#         }

    

    

#     def _finalize_node(self, state: WorkflowState) -> dict[str, Any]:
#         final_label = (
#             state.get("final_label")
#             or state.get("judge_label")
#             or state.get("proposed_label")
#             or "Error"
#         )

#         if final_label != "Error":
#             final_label = canonicalize_label(final_label)

#         trace = {
#             "event": "workflow_finalized",
#             "final_label": final_label,
#             "secondary_review_used": any(
#                 item.get("event") in {
#                     "secondary_review_complete",
#                     "secondary_review_failed",
#                 }
#                 for item in state.get("traces", [])
#             ),
#             "escalation_flag": state.get("escalation_flag", False),
#         }

#         return {
#             "final_label": final_label,
#             "traces": [*state.get("traces", []), trace],
#         }

#     def _review_reasons(
#         self,
#         *,
#         state: WorkflowState,
#         judge: JudgeDecision,
#     ) -> list[str]:
#         reasons: list[str] = []

#         low_path_agents = [
#             str(item.get("agent"))
#             for item in state.get("path", [])
#             if _safe_float(item.get("confidence")) < self.confidence_threshold
#         ]
#         if low_path_agents:
#             reasons.append(
#                 "low_node_confidence:" + ",".join(low_path_agents)
#             )

#         if judge.confidence < self.confidence_threshold:
#             reasons.append("low_judge_confidence")

#         margin = abs(judge.confidence - judge.competing_confidence)
#         boundary_pair = frozenset(
#             {judge.label_name, judge.competing_label}
#         )
#         known_pairs = {
#             frozenset({"Indicator", "Ideation"}),
#             frozenset({"Behavior", "Attempt"}),
#         }

#         margin_is_actionable = (
#             not self.review_known_boundaries_only
#             or boundary_pair in known_pairs
#         )

#         if margin_is_actionable and margin < self.margin_threshold:
#             reasons.append(
#                 f"small_top2_margin:{judge.label_name}_vs_"
#                 f"{judge.competing_label}:{margin:.4f}"
#             )


#         return reasons

#     def _failed_node_update(
#         self,
#         state: WorkflowState,
#         *,
#         role: str,
#         calls: list[dict[str, Any]],
#         errors: list[dict[str, Any]],
#     ) -> dict[str, Any]:
#         return {
#             "final_label": "Error",
#             "errors": [*state.get("errors", []), *errors],
#             "traces": [
#                 *state.get("traces", []),
#                 {"event": "node_failed", "role": role},
#             ],
#             "llm_calls": [*state.get("llm_calls", []), *calls],
#         }

#     def _invalid_decision_update(
#         self,
#         state: WorkflowState,
#         *,
#         role: str,
#         decision: str,
#         calls: list[dict[str, Any]],
#     ) -> dict[str, Any]:
#         error = {
#             "role": role,
#             "error_type": "InvalidDecision",
#             "error_message": f"Unexpected decision={decision!r}.",
#         }
#         return {
#             "final_label": "Error",
#             "errors": [*state.get("errors", []), error],
#             "traces": [
#                 *state.get("traces", []),
#                 {
#                     "event": "invalid_decision",
#                     "role": role,
#                     "decision": decision,
#                 },
#             ],
#             "llm_calls": [*state.get("llm_calls", []), *calls],
#         }

#     def run_case(self, *, sample_id: str, text: str) -> CaseResult:
#         initial_state: WorkflowState = {
#             "sample_id": sample_id,
#             "text": text,
#             "proposed_label": "",
#             "final_label": "",
#             "judge_label": "",
#             "judge_confidence": 0.0,
#             "competing_label": "",
#             "competing_confidence": 0.0,
#             "review_reasons": [],
#             "escalation_flag": False,
#             "specialist_answers": {},
#             "path": [],
#             "errors": [],
#             "traces": [],
#             "llm_calls": [],
#         }
    

#         result = self.graph.invoke(initial_state)
#         final_label = result.get("final_label") or "Error"

#         suicidal_intent = (
#             label_to_suicidal_intent(final_label)
#             if final_label in FINAL_LABELS
#             else False
#         )

#         return CaseResult(
#             sample_id=sample_id,
#             text=text,
#             final_label=final_label,
#             suicidal_intent=suicidal_intent,
#             errors=result.get("errors", []),
#             traces=result.get("traces", []),
#             llm_calls=result.get("llm_calls", []),
#         )


# def _safe_float(value: Any) -> float:
#     try:
#         return float(value)
#     except (TypeError, ValueError):
#         return 0.0


# def _validate_unit_interval(value: float, name: str) -> float:
#     numeric = float(value)
#     if not 0.0 <= numeric <= 1.0:
#         raise ValueError(f"{name} must be between 0.0 and 1.0.")
#     return numeric



"""Confidence-gated hierarchical LangGraph workflow.

The workflow preserves the same public input/output contract as the existing
framework while adding:

1. a Judge Agent that returns a candidate label and competing label;
2. a deterministic uncertainty gate;
3. a Secondary Review Agent invoked only for ambiguous cases.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

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
    IDEATION_ACTION_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    PERSPECTIVE_SYSTEM_PROMPT,
    SECONDARY_REVIEW_SYSTEM_PROMPT,
    judge_user_prompt,
    secondary_review_user_prompt,
    user_prompt,
)
from .schemas import (
    BehaviorAttemptDecision,
    IdeationActionDecision,
    JudgeDecision,
    PerspectiveDecision,
    ReviewDecision,
)

print("***** USING GRAPH.PY I JUST EDITED *****")

class WorkflowState(TypedDict):
    sample_id: str
    text: str

    proposed_label: str
    final_label: str

    judge_label: str
    judge_confidence: float
    competing_label: str
    competing_confidence: float
    review_reasons: list[str]
    escalation_flag: bool

    specialist_answers: dict[str, bool]
    path: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]


@dataclass
class CaseResult:
    """Public output contract expected by the shared evaluator."""

    sample_id: str
    text: str
    final_label: str
    suicidal_intent: bool
    errors: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]





def _sanitize_error_message(message: str) -> str:
    """Remove provider-side user identifiers from stored errors."""

    sanitized = re.sub(
        r"('user_id'\s*:\s*)'[^']+'",
        r"\1'[redacted]'",
        message,
    )
    sanitized = re.sub(
        r'("user_id"\s*:\s*)"[^"]+"',
        r'\1"[redacted]"',
        sanitized,
    )
    return sanitized


class MultiAgentWorkflow:
    """Confidence-gated hierarchical multi-agent workflow."""

    def __init__(
        self,
        *,
        model: InvokableModel,
        model_name: str = "unknown",
        max_retries: int = 2,
        confidence_threshold: float = 0.85,
        margin_threshold: float = 0.25,
        review_known_boundaries_only: bool = False,
        mode: str = "bool",
        models_by_role: dict[str, InvokableModel] | None = None,
        model_names_by_role: dict[str, str] | None = None,
    ):
        self.model = model
        self.model_name = model_name
        self.max_retries = max(0, int(max_retries))
        self.confidence_threshold = _validate_unit_interval(
            confidence_threshold,
            "confidence_threshold",
        )
        self.margin_threshold = _validate_unit_interval(
            margin_threshold,
            "margin_threshold",
        )
        self.review_known_boundaries_only = bool(review_known_boundaries_only)
        self.mode = mode
        self.models_by_role = models_by_role or {}
        self.model_names_by_role = model_names_by_role or {}
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkflowState)

        builder.add_node("perspective", self._perspective_node)
        builder.add_node("ideation_action", self._ideation_action_node)
        builder.add_node("behavior_attempt", self._behavior_attempt_node)
        builder.add_node("judge", self._judge_node)
        builder.add_node("secondary_review", self._secondary_review_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "perspective")

        builder.add_conditional_edges(
            "perspective",
            self._route_after_perspective,
            {
                "ideation_action": "ideation_action",
                "judge": "judge",
            },
        )

        builder.add_conditional_edges(
            "ideation_action",
            self._route_after_ideation_action,
            {
                "behavior_attempt": "behavior_attempt",
                "judge": "judge",
            },
        )

        builder.add_edge("behavior_attempt", "judge")

        builder.add_conditional_edges(
            "judge",
            self._route_after_judge,
            {
                "secondary_review": "secondary_review",
                "finalize": "finalize",
            },
        )

        builder.add_edge("secondary_review", "finalize")
        builder.add_edge("finalize", END)

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
        final_errors: list[dict[str, Any]] = []

        selected_model = self.models_by_role.get(role, self.model)
        selected_model_name = self.model_names_by_role.get(role, self.model_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_text},
        ]

        attempts = self.max_retries + 1

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
                response = selected_model.invoke(messages)
                latency_s = time.perf_counter() - started
                raw = _response_text(response)
                usage = _usage(response)
                parsed = schema.model_validate(extract_json_object(raw))

                calls.append(
                    {
                        "role": role,
                        "attempt": attempt,
                        "model_name": selected_model_name,
                        "_input_messages": [dict(message) for message in messages],
                        "latency_s": latency_s,
                        "raw_output": raw,
                        "parsed": parsed.model_dump(),
                        **usage,
                    }
                )
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

                calls.append(
                    {
                        "role": role,
                        "attempt": attempt,
                        "model_name": selected_model_name,
                        "_input_messages": [dict(message) for message in messages],
                        "latency_s": latency_s,
                        "raw_output": raw,
                        **usage,
                        "error": error,
                    }
                )

                if attempt >= attempts:
                    final_errors.append(error)
                    return None, calls, final_errors

        return None, calls, final_errors

    def _perspective_node(self, state: WorkflowState) -> dict[str, Any]:
        parsed, calls, errors = self._call_llm(
            role="perspective_agent",
            system_prompt=PERSPECTIVE_SYSTEM_PROMPT,
            user_prompt_text=user_prompt(state["text"]),
            schema=PerspectiveDecision,
        )

        if parsed is None:
            return self._failed_node_update(
                state,
                role="perspective_agent",
                calls=calls,
                errors=errors,
            )

        decision = parsed.decision.strip().upper()
        if decision not in {"SELF_RISK", "INDICATOR", "UNSUPPORTIVE"}:
            return self._invalid_decision_update(
                state,
                role="perspective_agent",
                decision=decision,
                calls=calls,
            )

        path_item = {
            "agent": "perspective_agent",
            "decision": decision,
            "confidence": parsed.confidence,
            "reasoning": parsed.reasoning,
        }

        if decision == "SELF_RISK":
            proposed_label = ""
        elif decision == "INDICATOR":
            proposed_label = "Indicator"
        else:
            proposed_label = "Unsupportive"


        return {
            "proposed_label": proposed_label,
            "path": [path_item],
            "errors": [*state.get("errors", []), *errors],
            "traces": [
                *state.get("traces", []),
                {"event": "perspective_complete", **path_item},
            ],
            "llm_calls": [*state.get("llm_calls", []), *calls],
        }

    def _route_after_perspective(self, state: WorkflowState) -> str:
        if state.get("final_label") == "Error":
            return "judge"
        return "judge" if state.get("proposed_label") else "ideation_action"

    def _ideation_action_node(self, state: WorkflowState) -> dict[str, Any]:
        parsed, calls, errors = self._call_llm(
            role="ideation_action_agent",
            system_prompt=IDEATION_ACTION_SYSTEM_PROMPT,
            user_prompt_text=user_prompt(state["text"]),
            schema=IdeationActionDecision,
        )

        if parsed is None:
            return self._failed_node_update(
                state,
                role="ideation_action_agent",
                calls=calls,
                errors=errors,
            )

        decision = parsed.decision.strip().upper()
        if decision not in {"IDEATION", "ACTION_PATHWAY"}:
            return self._invalid_decision_update(
                state,
                role="ideation_action_agent",
                decision=decision,
                calls=calls,
            )

        path_item = {
            "agent": "ideation_action_agent",
            "decision": decision,
            "confidence": parsed.confidence,
            "reasoning": parsed.reasoning,
        }

        proposed_label = "Ideation" if decision == "IDEATION" else ""

        return {
            "proposed_label": proposed_label,
            "path": [*state.get("path", []), path_item],
            "errors": [*state.get("errors", []), *errors],
            "traces": [
                *state.get("traces", []),
                {"event": "ideation_action_complete", **path_item},
            ],
            "llm_calls": [*state.get("llm_calls", []), *calls],
        }

    def _route_after_ideation_action(self, state: WorkflowState) -> str:
        if state.get("final_label") == "Error":
            return "judge"
        return "judge" if state.get("proposed_label") else "behavior_attempt"

    def _behavior_attempt_node(self, state: WorkflowState) -> dict[str, Any]:
        parsed, calls, errors = self._call_llm(
            role="behavior_attempt_agent",
            system_prompt=BEHAVIOR_ATTEMPT_SYSTEM_PROMPT,
            user_prompt_text=user_prompt(state["text"]),
            schema=BehaviorAttemptDecision,
        )

        if parsed is None:
            return self._failed_node_update(
                state,
                role="behavior_attempt_agent",
                calls=calls,
                errors=errors,
            )

        decision = parsed.decision.strip().upper()
        if decision not in {"BEHAVIOR", "ATTEMPT"}:
            return self._invalid_decision_update(
                state,
                role="behavior_attempt_agent",
                decision=decision,
                calls=calls,
            )

        path_item = {
            "agent": "behavior_attempt_agent",
            "decision": decision,
            "confidence": parsed.confidence,
            "reasoning": parsed.reasoning,
        }

        proposed_label = "Attempt" if decision == "ATTEMPT" else "Behavior"

        return {
            "proposed_label": proposed_label,
            "path": [*state.get("path", []), path_item],
            "errors": [*state.get("errors", []), *errors],
            "traces": [
                *state.get("traces", []),
                {"event": "behavior_attempt_complete", **path_item},
            ],
            "llm_calls": [*state.get("llm_calls", []), *calls],
        }

    def _judge_node(self, state: WorkflowState) -> dict[str, Any]:
        if state.get("final_label") == "Error" or state.get("errors"):
            return {
                "judge_label": "Error",
                "final_label": "Error",
                "review_reasons": ["previous_node_error"],
                "traces": [
                    *state.get("traces", []),
                    {"event": "judge_skipped", "reason": "previous_node_error"},
                ],
            }

        proposed_label = state.get("proposed_label")
        if proposed_label not in FINAL_LABELS:
            error = {
                "role": "judge_agent",
                "error_type": "MissingProposedLabel",
                "error_message": (
                    f"Expected a valid proposed label, received {proposed_label!r}."
                ),
            }
            return {
                "judge_label": "Error",
                "final_label": "Error",
                "errors": [*state.get("errors", []), error],
                "review_reasons": ["missing_proposed_label"],
                "traces": [
                    *state.get("traces", []),
                    {"event": "judge_skipped", "reason": "missing_proposed_label"},
                ],
            }

        parsed, calls, errors = self._call_llm(
            role="judge_agent",
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt_text=judge_user_prompt(
                text=state["text"],
                proposed_label=proposed_label,
                path=state.get("path", []),
            ),
            schema=JudgeDecision,
        )

        if parsed is None:
            return self._failed_node_update(
                state,
                role="judge_agent",
                calls=calls,
                errors=errors,
            )

        review_reasons = self._review_reasons(
            state=state,
            judge=parsed,
        )
        margin = abs(parsed.confidence - parsed.competing_confidence)

        trace = {
            "event": "judge_complete",
            "label_name": parsed.label_name,
            "confidence": parsed.confidence,
            "competing_label": parsed.competing_label,
            "competing_confidence": parsed.competing_confidence,
            "margin": margin,
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "reasoning": parsed.reasoning,
        }

        return {
            "judge_label": parsed.label_name,
            "judge_confidence": parsed.confidence,
            "competing_label": parsed.competing_label,
            "competing_confidence": parsed.competing_confidence,
            "review_reasons": review_reasons,
            "errors": [*state.get("errors", []), *errors],
            "traces": [*state.get("traces", []), trace],
            "llm_calls": [*state.get("llm_calls", []), *calls],
        }

    def _route_after_judge(self, state: WorkflowState) -> str:
        if state.get("judge_label") == "Error" or state.get("final_label") == "Error":
            return "finalize"
        return "secondary_review" if state.get("review_reasons") else "finalize"

    def _secondary_review_node(self, state: WorkflowState) -> dict[str, Any]:
        parsed, calls, errors = self._call_llm(
            role="secondary_review_agent",
            system_prompt=SECONDARY_REVIEW_SYSTEM_PROMPT,
            user_prompt_text=secondary_review_user_prompt(
                text=state["text"],
                judge_label=state["judge_label"],
                judge_confidence=state["judge_confidence"],
                competing_label=state["competing_label"],
                competing_confidence=state["competing_confidence"],
                review_reasons=state.get("review_reasons", []),
                path=state.get("path", []),
            ),
            schema=ReviewDecision,
        )

        if parsed is None:
            # Keep the judge label rather than throwing away an otherwise valid case.
            trace = {
                "event": "secondary_review_failed",
                "fallback_label": state.get("judge_label") or "Error",
            }
            return {
                "final_label": state.get("judge_label") or "Error",
                "escalation_flag": True,
                "errors": [*state.get("errors", []), *errors],
                "traces": [*state.get("traces", []), trace],
                "llm_calls": [*state.get("llm_calls", []), *calls],
            }

        trace = {
            "event": "secondary_review_complete",
            "original_judge_label": state["judge_label"],
            "reviewed_label": parsed.label_name,
            "confidence": parsed.confidence,
            "escalation_flag": parsed.escalation_flag,
            "review_reasons": state.get("review_reasons", []),
            "reasoning": parsed.reasoning,
        }

        return {
            "final_label": parsed.label_name,
            "escalation_flag": parsed.escalation_flag,
            "errors": [*state.get("errors", []), *errors],
            "traces": [*state.get("traces", []), trace],
            "llm_calls": [*state.get("llm_calls", []), *calls],
        }

    def _finalize_node(self, state: WorkflowState) -> dict[str, Any]:
        final_label = (
            state.get("final_label")
            or state.get("judge_label")
            or state.get("proposed_label")
            or "Error"
        )

        if final_label != "Error":
            final_label = canonicalize_label(final_label)

        trace = {
            "event": "workflow_finalized",
            "final_label": final_label,
            "secondary_review_used": any(
                item.get("event") in {
                    "secondary_review_complete",
                    "secondary_review_failed",
                }
                for item in state.get("traces", [])
            ),
            "escalation_flag": state.get("escalation_flag", False),
        }

        return {
            "final_label": final_label,
            "traces": [*state.get("traces", []), trace],
        }

    def _review_reasons(
        self,
        *,
        state: WorkflowState,
        judge: JudgeDecision,
    ) -> list[str]:
        reasons: list[str] = []

        low_path_agents = [
            str(item.get("agent"))
            for item in state.get("path", [])
            if _safe_float(item.get("confidence")) < self.confidence_threshold
        ]
        if low_path_agents:
            reasons.append(
                "low_node_confidence:" + ",".join(low_path_agents)
            )

        if judge.confidence < self.confidence_threshold:
            reasons.append("low_judge_confidence")

        margin = abs(judge.confidence - judge.competing_confidence)
        boundary_pair = frozenset(
            {judge.label_name, judge.competing_label}
        )
        known_pairs = {
            frozenset({"Indicator", "Ideation"}),
            frozenset({"Behavior", "Attempt"}),
        }

        margin_is_actionable = (
            not self.review_known_boundaries_only
            or boundary_pair in known_pairs
        )

        if margin_is_actionable and margin < self.margin_threshold:
            reasons.append(
                f"small_top2_margin:{judge.label_name}_vs_"
                f"{judge.competing_label}:{margin:.4f}"
            )


        proposed = state.get("proposed_label")

        if proposed and proposed != judge.label_name:
            reasons.append(
                f"judge_hierarchy_disagreement:{proposed}_vs_{judge.label_name}"
            )

        return reasons

    def _failed_node_update(
        self,
        state: WorkflowState,
        *,
        role: str,
        calls: list[dict[str, Any]],
        errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "final_label": "Error",
            "errors": [*state.get("errors", []), *errors],
            "traces": [
                *state.get("traces", []),
                {"event": "node_failed", "role": role},
            ],
            "llm_calls": [*state.get("llm_calls", []), *calls],
        }

    def _invalid_decision_update(
        self,
        state: WorkflowState,
        *,
        role: str,
        decision: str,
        calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        error = {
            "role": role,
            "error_type": "InvalidDecision",
            "error_message": f"Unexpected decision={decision!r}.",
        }
        return {
            "final_label": "Error",
            "errors": [*state.get("errors", []), error],
            "traces": [
                *state.get("traces", []),
                {
                    "event": "invalid_decision",
                    "role": role,
                    "decision": decision,
                },
            ],
            "llm_calls": [*state.get("llm_calls", []), *calls],
        }

    def run_case(self, *, sample_id: str, text: str) -> CaseResult:
        initial_state: WorkflowState = {
            "sample_id": sample_id,
            "text": text,
            "proposed_label": "",
            "final_label": "",
            "judge_label": "",
            "judge_confidence": 0.0,
            "competing_label": "",
            "competing_confidence": 0.0,
            "review_reasons": [],
            "escalation_flag": False,
            "specialist_answers": {},
            "path": [],
            "errors": [],
            "traces": [],
            "llm_calls": [],
        }

        result = self.graph.invoke(initial_state)
        final_label = result.get("final_label") or "Error"

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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _validate_unit_interval(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")
    return numeric
