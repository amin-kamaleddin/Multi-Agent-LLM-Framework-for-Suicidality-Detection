import pytest
from pydantic import ValidationError

from suicidal_multi_agent.core.schemas import (
    JudgeAnswer,
    SpecialistAnswer,
    SpecialistReasonedAnswer,
    extract_json_object,
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
