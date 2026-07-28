"""Single-model workflows compatible with the shared evaluation contract."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from domain.final_output import extract_json_object
from domain.labels import canonicalize_label, label_to_suicidal_intent
from llm.invoker import response_text, usage_summary
from llm.protocol import InvokableModel

from .config import DEFAULT_MODE, MODE_NAMES
from .prompts import (
    MULTI_CLASS_CHANCE_SYSTEM_PROMPT,
    SINGLE_CLASS_SYSTEM_PROMPT,
    binary_system_prompt,
    per_class_system_prompt,
    user_prompt,
)


EVALRING_LABEL_ORDER = (
    "Unsupportive",
    "Ideation",
    "Behavior",
    "Indicator",
    "Attempt",
)


@dataclass
class CaseResult:
    sample_id: str
    final_label: str
    suicidal_intent: bool | None
    errors: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]


class SingleAgentWorkflow:
    """Run one model using one of EvalRing's non-host classification modes."""

    def __init__(
        self,
        *,
        model: InvokableModel,
        model_name: str = "unknown-model",
        max_retries: int = 3,
        mode: str = DEFAULT_MODE,
        base_class: str = "Unsupportive",
    ) -> None:
        if mode not in MODE_NAMES:
            raise ValueError(f"single_agent mode must be one of {list(MODE_NAMES)}; got {mode!r}")
        self.model = model
        self.model_name = model_name
        self.max_retries = max(0, int(max_retries))
        self.mode = mode
        self.base_class = canonicalize_label(base_class)

    def _invoke(
        self,
        *,
        role: str,
        system_prompt: str,
        text: str,
        parser: Callable[[str], Any],
    ) -> tuple[Any | None, list[dict[str, Any]], list[dict[str, Any]]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt(text)},
        ]
        calls: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

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
                raw = response_text(response)
                usage = usage_summary(response)
                parsed = parser(raw)
                calls.append(
                    {
                        "role": role,
                        "attempt": attempt,
                        "model_name": self.model_name,
                        "_input_messages": [dict(message) for message in messages],
                        "latency_s": latency_s,
                        "raw_output": raw,
                        "parsed": parsed,
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
                        "_input_messages": [dict(message) for message in messages],
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

    @staticmethod
    def _parse_label(raw: str) -> str:
        clean = raw.strip('\'". \n').lower()
        for label in EVALRING_LABEL_ORDER:
            if label.lower() in clean:
                return label
        raise ValueError(f"Could not parse a valid label from output: {raw}")

    @staticmethod
    def _parse_chances(raw: str) -> dict[str, float]:
        parsed = extract_json_object(raw)
        scores: dict[str, float] = {}
        aliases = {label.lower(): label for label in EVALRING_LABEL_ORDER}
        for raw_label, raw_score in parsed.items():
            label = aliases.get(str(raw_label).strip().lower())
            if label is None:
                continue
            try:
                scores[label] = max(0.0, float(raw_score))
            except (TypeError, ValueError):
                continue
        for label in EVALRING_LABEL_ORDER:
            scores.setdefault(label, 0.0)
        total = sum(scores.values())
        if total <= 0:
            raise ValueError("Class probabilities must sum to more than zero")
        return {label: score / total for label, score in scores.items()}

    @staticmethod
    def _score_parser(target_class: str) -> Callable[[str], float]:
        def parse(raw: str) -> float:
            score = 1.0
            try:
                parsed = extract_json_object(raw)
                raw_score = parsed.get(target_class.lower(), parsed.get(target_class))
                if raw_score is not None:
                    score = float(raw_score)
            except (TypeError, ValueError):
                pass
            return max(1.0, min(10.0, score))

        return parse

    @staticmethod
    def _binary_parser(class_a: str, class_b: str) -> Callable[[str], str]:
        def parse(raw: str) -> str:
            return class_b if class_b.lower() in raw.lower() else class_a

        return parse

    @staticmethod
    def _winner(scores: dict[str, float]) -> str:
        return max(EVALRING_LABEL_ORDER, key=lambda label: scores[label])

    def run_case(self, *, sample_id: str, text: str) -> CaseResult:
        node_started_at_s = time.perf_counter()
        calls: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        final_label = "Error"

        if self.mode == "single-class":
            parsed, mode_calls, mode_errors = self._invoke(
                role="single_agent",
                system_prompt=SINGLE_CLASS_SYSTEM_PROMPT,
                text=text,
                parser=self._parse_label,
            )
            calls.extend(mode_calls)
            errors.extend(mode_errors)
            if parsed is not None:
                final_label = parsed

        elif self.mode == "multi-class-chance":
            parsed, mode_calls, mode_errors = self._invoke(
                role="single_agent",
                system_prompt=MULTI_CLASS_CHANCE_SYSTEM_PROMPT,
                text=text,
                parser=self._parse_chances,
            )
            calls.extend(mode_calls)
            errors.extend(mode_errors)
            if parsed is not None:
                final_label = self._winner(parsed)

        elif self.mode == "per-class-score":
            scores: dict[str, float] = {}
            for target_class in EVALRING_LABEL_ORDER:
                parsed, mode_calls, mode_errors = self._invoke(
                    role=f"{target_class}_score",
                    system_prompt=per_class_system_prompt(target_class),
                    text=text,
                    parser=self._score_parser(target_class),
                )
                calls.extend(mode_calls)
                errors.extend(mode_errors)
                if parsed is None:
                    break
                scores[target_class] = parsed
            if len(scores) == len(EVALRING_LABEL_ORDER):
                final_label = self._winner(scores)

        else:
            current_winner = self.base_class
            for target_class in EVALRING_LABEL_ORDER:
                if target_class == self.base_class:
                    continue
                parsed, mode_calls, mode_errors = self._invoke(
                    role=f"{current_winner}_vs_{target_class}",
                    system_prompt=binary_system_prompt(current_winner, target_class),
                    text=text,
                    parser=self._binary_parser(current_winner, target_class),
                )
                calls.extend(mode_calls)
                errors.extend(mode_errors)
                if parsed is None:
                    break
                current_winner = parsed
            if not errors:
                final_label = current_winner

        node_finished_at_s = time.perf_counter()
        successful = final_label != "Error"
        suicidal_intent = label_to_suicidal_intent(final_label) if successful else None
        return CaseResult(
            sample_id=str(sample_id),
            final_label=final_label,
            suicidal_intent=suicidal_intent,
            errors=errors,
            traces=[
                {
                    "event": "single_agent_complete",
                    "mode": self.mode,
                    "ok": successful,
                    "node_started_at_s": node_started_at_s,
                    "node_finished_at_s": node_finished_at_s,
                    "node_duration_s": node_finished_at_s - node_started_at_s,
                }
            ],
            llm_calls=calls,
        )
