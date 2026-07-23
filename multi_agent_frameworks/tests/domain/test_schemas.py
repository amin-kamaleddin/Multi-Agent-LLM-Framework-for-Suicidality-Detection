import pytest
from pydantic import ValidationError

from domain.evidence import EvidenceCollectorAnswer, validate_evidence_answer
from domain.final_output import JudgeAnswer, extract_json_object
from workflows.framework_1_parallel_no_evidence.schemas import (
    SpecialistAnswer,
    SpecialistReasonedAnswer,
)


def test_specialist_answer_accepts_strict_json_bool_only():
    assert SpecialistAnswer.model_validate({"answer": True}).answer is True
    assert SpecialistAnswer.model_validate({"answer": False}).answer is False


def test_specialist_answer_rejects_extra_label_and_string_answer():
    with pytest.raises(ValidationError):
        SpecialistAnswer.model_validate({"label_name": "Ideation", "answer": True})
    with pytest.raises(ValidationError):
        SpecialistAnswer.model_validate({"answer": "true"})
    with pytest.raises(ValidationError):
        SpecialistAnswer.model_validate({"answer": "yes"})


def test_reasoned_specialist_answer_accepts_bool_and_reasoning_string():
    parsed = SpecialistReasonedAnswer.model_validate(
        {
            "answer": True,
            "reasoning": "Sentence one. Sentence two. Sentence three. Sentence four.",
        }
    )

    assert parsed.answer is True
    assert parsed.reasoning == "Sentence one. Sentence two. Sentence three. Sentence four."


def test_reasoned_specialist_answer_rejects_string_answer_and_extra_keys():
    with pytest.raises(ValidationError):
        SpecialistReasonedAnswer.model_validate(
            {
                "answer": "true",
                "reasoning": "String booleans are invalid.",
            }
        )
    with pytest.raises(ValidationError):
        SpecialistReasonedAnswer.model_validate(
            {
                "answer": True,
                "reasoning": "Extra labels stay invalid.",
                "label_name": "Attempt",
            }
        )
    with pytest.raises(ValidationError):
        SpecialistReasonedAnswer.model_validate(
            {
                "answer": True,
                "reasoning": "Evidence is included in reasoning.",
                "evidence": ["separate evidence is not allowed"],
            }
        )
    with pytest.raises(ValidationError):
        SpecialistReasonedAnswer.model_validate({"answer": False, "reasoning": " "})


def test_evidence_collector_answer_accepts_compact_five_field_items():
    parsed = EvidenceCollectorAnswer.model_validate(
        {
            "evidence": [
                {
                    "quote": "I wrote a note",
                    "type": "preparation",
                    "subject": "author",
                    "time": "past",
                    "observation": "The author reports writing a note.",
                }
            ]
        }
    )

    validate_evidence_answer(
        parsed,
        allowed_types={"suicide_plan", "preparation", "self_harm"},
    )
    assert parsed.model_dump()["evidence"][0]["type"] == "preparation"


def test_evidence_collector_answer_rejects_extra_fields_limits_and_bad_type():
    valid_item = {
        "quote": "I wrote a note",
        "type": "preparation",
        "subject": "author",
        "time": "past",
        "observation": "The author reports writing a note.",
    }
    with pytest.raises(ValidationError):
        EvidenceCollectorAnswer.model_validate(
            {"evidence": [{**valid_item, "confidence": 0.9}]}
        )
    with pytest.raises(ValidationError):
        EvidenceCollectorAnswer.model_validate({})
    with pytest.raises(ValidationError):
        EvidenceCollectorAnswer.model_validate({"evidence": [valid_item] * 4})

    parsed = EvidenceCollectorAnswer.model_validate({"evidence": [valid_item]})
    with pytest.raises(ValueError, match="not allowed"):
        validate_evidence_answer(
            parsed,
            allowed_types={"attempt_report"},
        )


def test_evidence_quote_and_observation_allow_100_words_but_reject_101():
    hundred_words = " ".join(f"word{i}" for i in range(100))
    item = {
        "quote": hundred_words,
        "type": "preparation",
        "subject": "author",
        "time": "past",
        "observation": hundred_words,
    }

    assert EvidenceCollectorAnswer.model_validate({"evidence": [item]}).evidence
    too_long = f"{hundred_words} extra"
    with pytest.raises(ValidationError, match="quote must contain at most 100 words"):
        EvidenceCollectorAnswer.model_validate(
            {"evidence": [{**item, "quote": too_long}]}
        )
    with pytest.raises(ValidationError, match="observation must contain at most 100 words"):
        EvidenceCollectorAnswer.model_validate(
            {"evidence": [{**item, "observation": too_long}]}
        )


def test_judge_answer_requires_known_label_only():
    parsed = JudgeAnswer.model_validate({"label_name": "attempt", "suicidal_intent": True})
    assert parsed.label_name == "Attempt"
    assert parsed.suicidal_intent is True
    assert JudgeAnswer.model_validate(
        {"label_name": "unsupportive", "suicidal_intent": False}
    ).label_name == "Unsupportive"
    with pytest.raises(ValidationError):
        JudgeAnswer.model_validate({"label_name": "No-Risk", "suicidal_intent": False})
    with pytest.raises(ValidationError):
        JudgeAnswer.model_validate({"label_name": "Indicator", "suicidal_intent": False})
    with pytest.raises(ValidationError):
        JudgeAnswer.model_validate({"label_name": "Unsupportive", "suicidal_intent": True})


def test_extract_json_object_accepts_wrapped_json():
    parsed = extract_json_object('Here is JSON: {"label_name": "Indicator"}')
    assert parsed == {"label_name": "Indicator"}
    assert extract_json_object('{"evidence":[]}}') == {"evidence": []}
