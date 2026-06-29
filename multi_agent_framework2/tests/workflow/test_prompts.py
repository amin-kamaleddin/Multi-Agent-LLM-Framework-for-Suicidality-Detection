from suicidal_multi_agent.core.labels import VALID_LABELS
from suicidal_multi_agent.workflow.prompts import (
    ATTEMPT_SPECIALIST_PROMPT,
    BEHAVIOR_SPECIALIST_PROMPT,
    IDEATION_SPECIALIST_PROMPT,
    INDICATOR_SPECIALIST_PROMPT,
    LABEL_TO_AGENT_NAME,
    SPECIALIST_AGENT_PROMPTS,
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
