import pytest
from pydantic import ValidationError

from suicidal_multi_agent.core.schemas import JudgeAnswer, SpecialistAnswer, extract_json_object


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


def test_judge_answer_requires_known_label_only():
    assert JudgeAnswer.model_validate({"label_name": "attempt"}).label_name == "Attempt"
    with pytest.raises(ValidationError):
        JudgeAnswer.model_validate({"label_name": "No-Risk"})


def test_extract_json_object_accepts_wrapped_json():
    parsed = extract_json_object('Here is JSON: {"label_name": "Indicator"}')
    assert parsed == {"label_name": "Indicator"}
