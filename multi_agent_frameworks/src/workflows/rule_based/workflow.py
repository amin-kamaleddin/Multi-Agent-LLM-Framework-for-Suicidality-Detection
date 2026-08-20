"""Context-aware regex baseline for the repository's five final labels."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Final

from domain.labels import UNSUPPORTIVE_LABEL, label_to_suicidal_intent


@dataclass(frozen=True)
class RuleMatch:
    """One deterministic label match retained for evaluation auditability."""

    label_name: str
    rule_name: str
    sentence: str


@dataclass
class CaseResult:
    sample_id: str
    final_label: str
    suicidal_intent: bool
    errors: list[dict[str, str]]
    traces: list[dict[str, object]]
    llm_calls: list[dict[str, object]]
    agent_outputs: dict[str, dict[str, str]]


_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?;\n])\s+")
_SELF: Final[str] = r"(?:i|i'm|im|i've|ive|i'd|id|me|my|myself)"
_OTHER_PERSON: Final[re.Pattern[str]] = re.compile(
    r"\b(?:my\s+(?:friend|brother|sister|mother|father|mom|dad|partner|wife|husband|child)|"
    r"(?:he|she|they|someone|another\s+person|a\s+friend|family\s+member))\b"
)
_SUICIDE_CONTEXT: Final[re.Pattern[str]] = re.compile(
    r"\b(?:suicid(?:e|al|ality)?|self[ -]?harm|kill\s+(?:myself|yourself|himself|herself|themselves)|"
    r"end\s+(?:my|your|his|her|their)\s+life)\b"
)
_BENIGN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:suicide\s+squad|career\s+suicide|political\s+suicide|suicide\s+doors?|"
    r"suicide\s+king|dead\s+tired|could\s+die\s+laughing|kill\s+time)\b"
)
_DENIAL: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:{_SELF})\b.{{0,30}}\b(?:am\s+not|not|never|don't|do\s+not|won't|will\s+not)\b"
    r".{0,30}\b(?:suicidal|suicide|kill\s+myself|end\s+my\s+life|self[ -]?harm)\b"
)

_ATTEMPT_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "self_reported_attempt",
        re.compile(
            rf"\b(?:{_SELF})\b.{{0,40}}\b(?:attempted\s+suicide|tried\s+to\s+"
            r"(?:kill\s+myself|end\s+my\s+life|die)|made\s+a\s+suicide\s+attempt)\b"
        ),
    ),
    (
        "personal_attempt_history",
        re.compile(r"\bmy\s+(?:suicide\s+)?attempt\b"),
    ),
    (
        "attempt_aftermath",
        re.compile(r"\b(?:survived|after)\s+(?:my\s+)?(?:suicide\s+)?attempt\b"),
    ),
)
_BEHAVIOR_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "self_suicide_plan",
        re.compile(
            rf"\b(?:{_SELF})\b.{{0,40}}\b(?:have|had|made|making|planned|planning)\b"
            r".{0,25}\b(?:suicide\s+plan|plan\s+(?:to\s+)?(?:kill\s+myself|end\s+my\s+life|die))\b"
        ),
    ),
    (
        "preparatory_note",
        re.compile(
            rf"\b(?:{_SELF})\b.{{0,30}}\b(?:wrote|writing|write|prepared|preparing)\b"
            r".{0,25}\b(?:suicide|goodbye)\s+note\b"
        ),
    ),
    (
        "preparation_for_death",
        re.compile(
            rf"\b(?:{_SELF})\b.{{0,40}}\b(?:prepared|preparing|get\s+ready|getting\s+ready|"
            r"gave|giving)\b.{0,40}\b(?:to\s+die|for\s+death|things\s+away)\b"
        ),
    ),
    (
        "cut_or_burn_self",
        re.compile(
            rf"\b(?:{_SELF})\b.{{0,25}}\b(?:cut|burn(?:ed)?)\b"
            r".{0,25}\b(?:myself|my\s+(?:arm|arms|leg|legs|body|skin))\b"
        ),
    ),
    (
        "intentional_self_harm",
        re.compile(
            rf"\b(?:{_SELF})\b.{{0,25}}\b(?:hurt|injure(?:d)?|harm(?:ed)?)\b"
            r".{0,25}\b(?:myself|my\s+(?:arm|arms|leg|legs|body|skin))\b.{0,25}\bon\s+purpose\b"
        ),
    ),
)
_IDEATION_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "passive_death_wish",
        re.compile(
            rf"\b(?:{_SELF})\b.{{0,25}}\b(?:wish(?:ed)?\s+(?:i\s+)?(?:were|was)\s+dead|"
            r"wish(?:ed)?\s+(?:i\s+)?(?:would|could)\s+(?:not\s+)?wake\s+up|"
            r"would\s+be\s+better\s+off\s+dead|want\s+to\s+die|don't\s+want\s+to\s+live|"
            r"do\s+not\s+want\s+to\s+live)\b"
        ),
    ),
    (
        "active_suicidal_thought",
        re.compile(
            rf"\b(?:{_SELF})\b.{{0,35}}\b(?:am|feel|been|being|thinking|think|thought|want|"
            r"wanted)\b.{0,35}\b(?:suicidal|suicide|kill\s+myself|end\s+my\s+life)\b"
        ),
    ),
    (
        "self_directed_suicide_phrase",
        re.compile(r"\b(?:kill\s+myself|end\s+my\s+life|want\s+to\s+die|suicidal\s+thoughts?)\b"),
    ),
)


def _normalize(text: str) -> str:
    return " ".join(
        str(text).lower().replace("’", "'").replace("`", "'").split()
    )


def _sentences(text: str) -> list[str]:
    normalized = _normalize(text)
    return [sentence for sentence in _SENTENCE_BOUNDARY.split(normalized) if sentence]


def classify_text(text: str) -> RuleMatch:
    """Classify text using auditable context-sensitive regex precedence."""
    indicator_match: RuleMatch | None = None
    phase_matches: dict[str, RuleMatch] = {}
    for sentence in _sentences(text):
        if _BENIGN.search(sentence):
            continue
        if _DENIAL.search(sentence):
            indicator_match = indicator_match or RuleMatch("Indicator", "explicit_self_denial", sentence)
            continue
        for label_name, rules in (
            ("Attempt", _ATTEMPT_RULES),
            ("Behavior", _BEHAVIOR_RULES),
            ("Ideation", _IDEATION_RULES),
        ):
            for rule_name, pattern in rules:
                if pattern.search(sentence):
                    phase_matches.setdefault(
                        label_name,
                        RuleMatch(label_name, rule_name, sentence),
                    )
                    break
        if _SUICIDE_CONTEXT.search(sentence):
            rule_name = "third_party_or_general_suicide_context"
            if _OTHER_PERSON.search(sentence):
                rule_name = "third_party_suicide_context"
            indicator_match = indicator_match or RuleMatch("Indicator", rule_name, sentence)
    for label_name in ("Attempt", "Behavior", "Ideation"):
        if label_name in phase_matches:
            return phase_matches[label_name]
    return indicator_match or RuleMatch(UNSUPPORTIVE_LABEL, "no_suicide_or_self_harm_evidence", "")


class RuleBasedWorkflow:
    """Shared-evaluator-compatible deterministic workflow with zero LLM calls."""

    def run_case(self, *, sample_id: str, text: str) -> CaseResult:
        started_at = time.perf_counter()
        match = classify_text(text)
        finished_at = time.perf_counter()
        return CaseResult(
            sample_id=str(sample_id),
            final_label=match.label_name,
            suicidal_intent=label_to_suicidal_intent(match.label_name),
            errors=[],
            traces=[
                {
                    "event": "rule_based_classification_complete",
                    "label_name": match.label_name,
                    "rule_name": match.rule_name,
                    "matched_sentence": match.sentence,
                    "node_started_at_s": started_at,
                    "node_finished_at_s": finished_at,
                    "node_duration_s": finished_at - started_at,
                }
            ],
            llm_calls=[],
            agent_outputs={
                "rule_based_classifier": {
                    "label_name": match.label_name,
                    "rule_name": match.rule_name,
                    "matched_sentence": match.sentence,
                }
            },
        )
