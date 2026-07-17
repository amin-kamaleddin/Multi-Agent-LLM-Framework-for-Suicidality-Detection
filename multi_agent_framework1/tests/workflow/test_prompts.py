from suicidal_multi_agent.core.labels import LABEL_DEFINITIONS, VALID_LABELS
from suicidal_multi_agent.core.schemas import JudgeAnswer, SpecialistAnswer, SpecialistReasonedAnswer
from suicidal_multi_agent.workflow.prompts import (
    ATTEMPT_SPECIALIST_PROMPT,
    BEHAVIOR_SPECIALIST_PROMPT,
    IDEATION_SPECIALIST_PROMPT,
    INDICATOR_SPECIALIST_PROMPT,
    LABEL_TO_AGENT_NAME,
    REASONED_SPECIALIST_AGENT_PROMPTS,
    SPECIALIST_AGENT_PROMPTS,
    WORKFLOW_MODES,
    specialist_system_prompt,
)


def test_static_prompt_variables_exist_for_each_label():
    prompt_by_label = {
        "Indicator": INDICATOR_SPECIALIST_PROMPT,
        "Ideation": IDEATION_SPECIALIST_PROMPT,
        "Behavior": BEHAVIOR_SPECIALIST_PROMPT,
        "Attempt": ATTEMPT_SPECIALIST_PROMPT,
    }

    assert set(prompt_by_label) == set(VALID_LABELS)
    for label, prompt in prompt_by_label.items():
        assert f"Target label: {label}" in prompt
        assert '"label_name"' not in prompt
        assert '"answer": true' in prompt
        assert "JSON boolean" in prompt
        assert "Input format:" in prompt
        assert "Output format:" in prompt
        assert LABEL_DEFINITIONS[label] in prompt


def test_agent_to_prompt_mapping_is_one_to_one():
    assert set(SPECIALIST_AGENT_PROMPTS) == {
        "Indicator Agent",
        "Ideation Agent",
        "Behavior Agent",
        "Attempt Agent",
    }
    assert LABEL_TO_AGENT_NAME == {
        "Indicator": "Indicator Agent",
        "Ideation": "Ideation Agent",
        "Behavior": "Behavior Agent",
        "Attempt": "Attempt Agent",
    }

    for label in VALID_LABELS:
        agent_name = LABEL_TO_AGENT_NAME[label]
        config = SPECIALIST_AGENT_PROMPTS[agent_name]
        assert config.label_name == label
        assert config.agent_name == agent_name
        assert config.system_prompt == specialist_system_prompt(label)
        assert config.parallel_group == "four_label_specialists"


def test_workflow_modes_reuse_same_agent_names_with_different_schemas():
    assert set(WORKFLOW_MODES) == {"bool", "reasoned"}
    assert WORKFLOW_MODES["bool"].specialist_schema is SpecialistAnswer
    assert WORKFLOW_MODES["bool"].judge_schema is JudgeAnswer
    assert WORKFLOW_MODES["reasoned"].specialist_schema is SpecialistReasonedAnswer
    assert WORKFLOW_MODES["reasoned"].judge_schema is JudgeAnswer
    assert set(REASONED_SPECIALIST_AGENT_PROMPTS) == set(SPECIALIST_AGENT_PROMPTS)
    for mode in WORKFLOW_MODES.values():
        judge_prompt = mode.judge_system_prompt()
        assert "Unsupportive" in judge_prompt
        assert '"suicidal_intent": true' in judge_prompt


def test_reasoned_prompts_request_reasoning_without_separate_evidence_or_final_label():
    for label in VALID_LABELS:
        agent_name = LABEL_TO_AGENT_NAME[label]
        config = REASONED_SPECIALIST_AGENT_PROMPTS[agent_name]
        prompt = config.system_prompt

        assert f"Target label: {label}" in prompt
        assert LABEL_DEFINITIONS[label] in prompt
        assert '"answer": true' in prompt
        assert '"reasoning"' in prompt
        assert '"evidence"' not in prompt
        assert "one to three sentences max" in prompt
        assert '"label_name"' not in prompt
        assert "JSON boolean" in prompt
        assert config.output_format == '{"answer": true|false, "reasoning": string}'
