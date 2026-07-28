"""Deterministic model used by CLI smoke tests."""

from __future__ import annotations

import json
import re

from domain.labels import FINAL_LABELS


class DeterministicFakeModel:
    """Return contract-valid fixtures for every built-in agent role."""

    def invoke(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        if system.startswith("Classify the post into exactly ONE class"):
            return "Indicator"
        if system.startswith("You are a suicide-risk classification assistant."):
            score_match = re.search(
                r'evaluate whether the post matches the class: (\w+)',
                system,
            )
            if score_match:
                label = score_match.group(1)
                return json.dumps({label.lower(): 10 if label == "Indicator" else 1})
            if 'Include ALL 5 keys' in system:
                return json.dumps(
                    {
                        "unsupportive": 0.0,
                        "ideation": 0.0,
                        "behavior": 0.0,
                        "indicator": 1.0,
                        "attempt": 0.0,
                    }
                )
        if system.startswith("You are a suicide-risk classifier."):
            match = re.search(r"Compare the two classes: '([^']+)' vs '([^']+)'", system)
            if match:
                class_a, class_b = match.groups()
                return "Indicator" if "Indicator" in (class_a, class_b) else class_a
        if "Perspective / Risk-Relevance Agent" in system:
            return json.dumps({"decision": "INDICATOR", "confidence": 0.95, "reasoning": "Synthetic fake-model perspective decision."})
        if "Ideation vs Action Agent" in system:
            return json.dumps({"decision": "IDEATION", "confidence": 0.95, "reasoning": "Synthetic fake-model pathway decision."})
        if "Behavior vs Attempt Agent" in system:
            return json.dumps({"decision": "BEHAVIOR", "confidence": 0.95, "reasoning": "Synthetic fake-model action decision."})
        if "Final Evaluator / Safety Checker" in system:
            return json.dumps({"label_name": "Indicator", "confidence": 0.95, "escalation_flag": False, "reasoning": "Synthetic fake-model evaluator decision."})
        if "Judge Agent in a hierarchical" in system:
            return json.dumps({"label_name": "Indicator", "confidence": 0.95, "competing_label": "Ideation", "competing_confidence": 0.05, "reasoning": "Synthetic fake-model judge decision."})
        if "Secondary Review Agent" in system:
            return json.dumps({"label_name": "Indicator", "confidence": 0.95, "escalation_flag": False, "reasoning": "Synthetic fake-model review decision."})
        if "final judge" in system.lower():
            return json.dumps({"label_name": "Indicator", "suicidal_intent": True})
        for label in FINAL_LABELS:
            if f"Evidence family: {label}" in system:
                return json.dumps({"evidence": []})
        for label in FINAL_LABELS:
            if f"Target label: {label}" in system:
                answer = label == "Indicator"
                if '"reasoning"' in system:
                    return json.dumps({"answer": answer, "reasoning": f"Synthetic fake-model rationale for {label}; answer is {str(answer).lower()}."})
                return json.dumps({"answer": answer})
        return json.dumps({"label_name": "Indicator", "suicidal_intent": True})
