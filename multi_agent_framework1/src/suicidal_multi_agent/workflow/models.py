"""Shared model construction used by CLI-registered workflows."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from ..core.labels import FINAL_LABELS, VALID_LABELS


class DeterministicFakeModel:
    """Small fake model for smoke tests and artifact verification."""

    def invoke(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        if "Perspective / Risk-Relevance Agent" in system:
            return json.dumps(
                {
                    "decision": "INDICATOR",
                    "confidence": 0.95,
                    "reasoning": "Synthetic fake-model perspective decision.",
                }
            )
        if "Ideation vs Action Agent" in system:
            return json.dumps(
                {
                    "decision": "IDEATION",
                    "confidence": 0.95,
                    "reasoning": "Synthetic fake-model pathway decision.",
                }
            )
        if "Behavior vs Attempt Agent" in system:
            return json.dumps(
                {
                    "decision": "BEHAVIOR",
                    "confidence": 0.95,
                    "reasoning": "Synthetic fake-model action decision.",
                }
            )
        if "Final Evaluator / Safety Checker" in system:
            return json.dumps(
                {
                    "label_name": "Indicator",
                    "confidence": 0.95,
                    "escalation_flag": False,
                    "reasoning": "Synthetic fake-model evaluator decision.",
                }
            )
        if "Judge Agent in a hierarchical" in system:
            return json.dumps(
                {
                    "label_name": "Indicator",
                    "confidence": 0.95,
                    "competing_label": "Ideation",
                    "competing_confidence": 0.05,
                    "reasoning": "Synthetic fake-model judge decision.",
                }
            )
        if "Secondary Review Agent" in system:
            return json.dumps(
                {
                    "label_name": "Indicator",
                    "confidence": 0.95,
                    "escalation_flag": False,
                    "reasoning": "Synthetic fake-model review decision.",
                }
            )
        if "final judge" in system.lower():
            return json.dumps({"label_name": "Indicator", "suicidal_intent": True})
        for label in FINAL_LABELS:
            if f"Evidence family: {label}" in system:
                return json.dumps({"evidence": []})
        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                answer = label == "Indicator"
                if '"reasoning"' in system:
                    return json.dumps(
                        {
                            "answer": answer,
                            "reasoning": (
                                f"Synthetic fake-model rationale for {label}; "
                                f"answer is {str(answer).lower()}."
                            ),
                        }
                    )
                return json.dumps({"answer": answer})
        return json.dumps({"label_name": "Indicator", "suicidal_intent": True})


def openai_base_url() -> str:
    load_dotenv()
    return os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def resolve_model_name_for_base_url(model_name: str, base_url: str) -> str:
    """Resolve common OpenAI aliases for OpenRouter's provider/model syntax."""
    if "openrouter.ai" in base_url.lower() and "/" not in model_name and model_name.startswith("gpt-"):
        return f"openai/{model_name}"
    return model_name


def build_real_model(model_name: str, temperature: float, max_completion_tokens: int):
    load_dotenv()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for real LLM runs. Install with: pip install -e ."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env, or use --fake-llm for artifact smoke runs.")
    base_url = openai_base_url()
    resolved_model_name = resolve_model_name_for_base_url(model_name, base_url)
    return ChatOpenAI(
        model=resolved_model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        max_completion_tokens=max_completion_tokens,
        max_retries=0,
    )
