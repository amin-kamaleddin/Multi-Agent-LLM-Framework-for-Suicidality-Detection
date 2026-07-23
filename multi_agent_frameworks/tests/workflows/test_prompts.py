from domain.evidence import EvidenceCollectorAnswer
from domain.final_output import JudgeAnswer
from domain.labels import FINAL_LABELS, LABEL_DEFINITIONS
from workflows.framework_1_parallel_no_evidence.schemas import SpecialistAnswer, SpecialistReasonedAnswer
from workflows.shared.parallel.prompts import (
    ATTEMPT_SPECIALIST_PROMPT,
    BEHAVIOR_SPECIALIST_PROMPT,
    EVIDENCE_COLLECTOR_PROMPTS,
    EVIDENCE_TYPE_DEFINITIONS,
    IDEATION_SPECIALIST_PROMPT,
    INDICATOR_SPECIALIST_PROMPT,
    LABEL_TO_AGENT_NAME,
    REASONED_SPECIALIST_AGENT_PROMPTS,
    SPECIALIST_AGENT_PROMPTS,
    UNSUPPORTIVE_SPECIALIST_PROMPT,
    WORKFLOW_MODES,
    specialist_system_prompt,
)


def test_static_prompt_variables_exist_for_each_label():
    prompt_by_label = {
        "Indicator": INDICATOR_SPECIALIST_PROMPT,
        "Ideation": IDEATION_SPECIALIST_PROMPT,
        "Behavior": BEHAVIOR_SPECIALIST_PROMPT,
        "Attempt": ATTEMPT_SPECIALIST_PROMPT,
        "Unsupportive": UNSUPPORTIVE_SPECIALIST_PROMPT,
    }

    assert set(prompt_by_label) == set(FINAL_LABELS)
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
        "Unsupportive Agent",
    }
    assert LABEL_TO_AGENT_NAME == {
        "Indicator": "Indicator Agent",
        "Ideation": "Ideation Agent",
        "Behavior": "Behavior Agent",
        "Attempt": "Attempt Agent",
        "Unsupportive": "Unsupportive Agent",
    }

    for label in FINAL_LABELS:
        agent_name = LABEL_TO_AGENT_NAME[label]
        config = SPECIALIST_AGENT_PROMPTS[agent_name]
        assert config.label_name == label
        assert config.agent_name == agent_name
        assert config.system_prompt == specialist_system_prompt(label)
        assert config.parallel_group == "five_label_specialists"


def test_workflow_modes_reuse_same_agent_names_with_different_schemas():
    assert set(WORKFLOW_MODES) == {"bool", "reasoned", "evidence"}
    assert WORKFLOW_MODES["bool"].specialist_schema is SpecialistAnswer
    assert WORKFLOW_MODES["bool"].judge_schema is JudgeAnswer
    assert WORKFLOW_MODES["reasoned"].specialist_schema is SpecialistReasonedAnswer
    assert WORKFLOW_MODES["reasoned"].judge_schema is JudgeAnswer
    assert WORKFLOW_MODES["evidence"].specialist_schema is EvidenceCollectorAnswer
    assert WORKFLOW_MODES["evidence"].judge_schema is JudgeAnswer
    assert set(REASONED_SPECIALIST_AGENT_PROMPTS) == set(SPECIALIST_AGENT_PROMPTS)
    for mode in WORKFLOW_MODES.values():
        judge_prompt = mode.judge_system_prompt()
        assert "Unsupportive" in judge_prompt
        assert '"suicidal_intent": true' in judge_prompt
    assert "five specialist answers" in WORKFLOW_MODES["bool"].judge_system_prompt()
    assert "five specialist answers" in WORKFLOW_MODES["reasoned"].judge_system_prompt()


def test_evidence_prompts_define_five_non_voting_collectors_and_compact_contract():
    assert {config.label_name for config in EVIDENCE_COLLECTOR_PROMPTS.values()} == set(
        FINAL_LABELS
    )
    for config in EVIDENCE_COLLECTOR_PROMPTS.values():
        prompt = config.system_prompt
        assert LABEL_DEFINITIONS[config.label_name] in prompt
        assert set(EVIDENCE_TYPE_DEFINITIONS[config.label_name])
        for evidence_type, description in EVIDENCE_TYPE_DEFINITIONS[config.label_name].items():
            assert evidence_type in prompt
            assert description in prompt
        assert "zero to three evidence items" in prompt
        assert "at most 100 words" in prompt
        assert "do not classify" in prompt.lower()
        assert config.parallel_group == "five_evidence_collectors"
        assert config.role_suffix == "collector"
        assert config.produces_answer is False


def test_evidence_judge_prompt_contains_all_label_and_evidence_type_descriptions():
    prompt = WORKFLOW_MODES["evidence"].judge_system_prompt()
    for label in FINAL_LABELS:
        assert LABEL_DEFINITIONS[label] in prompt
        for evidence_type, description in EVIDENCE_TYPE_DEFINITIONS[label].items():
            assert evidence_type in prompt
            assert description in prompt
    assert "A packet is not a vote" in prompt
    assert "Empty evidence" in prompt


def test_reasoned_prompts_request_reasoning_without_separate_evidence_or_final_label():
    for label in FINAL_LABELS:
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
