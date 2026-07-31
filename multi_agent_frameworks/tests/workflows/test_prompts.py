from domain.evidence import EvidenceCollectorAnswer
from domain.final_output import JudgeAnswer
from domain.labels import FINAL_LABELS, LABEL_DEFINITIONS
from workflows.framework_1_parallel_no_evidence.schemas import SpecialistAnswer
from workflows.framework_3_hierarchical_no_reviewer import prompts as framework_3_prompts
from workflows.framework_4_hierarchical_with_reviewer import prompts as framework_4_prompts
from workflows.shared.parallel.prompts import (
    ATTEMPT_SPECIALIST_PROMPT,
    BEHAVIOR_SPECIALIST_PROMPT,
    EVIDENCE_COLLECTOR_PROMPTS,
    EVIDENCE_TYPE_DEFINITIONS,
    IDEATION_SPECIALIST_PROMPT,
    INDICATOR_SPECIALIST_PROMPT,
    LABEL_TO_AGENT_NAME,
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
    assert LABEL_DEFINITIONS == {
        "Indicator": (
            "The post does NOT indicate the author's own suicidal risk, but might be a "
            "window into their inner thoughts related to suicide. Includes: discussion "
            "of another person's suicidality, general discussion/news, or explicit "
            "denial of own suicidal intent. Person does not seem at any risk, but suicide "
            "was mentioned or discussed, even if in an explicit denial, so it should be "
            "flagged accordingly. Also includes concern for someone else at risk."
        ),
        "Ideation": (
            "The author expresses suicidal thoughts, wishes, or desire to die/end life. "
            "Includes passive or active suicidal thinking, including hypothetical or "
            "unrealistic methods. No concrete self-harm/suicide action has been taken in "
            "the post."
        ),
        "Behavior": (
            "The author describes preparatory or self-harm behavior beyond verbal "
            "thoughts. Includes planning/preparation actions (for example, gathering "
            "means, writing notes, preparing for death). Includes non-suicidal self-harm "
            "when self-inflicted harm is described, even without explicit intent to die."
        ),
        "Attempt": (
            "The post mentions a suicide attempt (current or past). A past attempt should "
            "still be labeled Attempt, even if the current post does not express ongoing "
            "suicidal thoughts."
        ),
        "Unsupportive": (
            "The post does NOT indicate the author's own suicidal risk or any connection "
            "to suicidal ideation. Includes: discussion of topics unrelated to suicidal "
            "ideation or self harm, such as hobbies, nature, and any other topic. However, "
            "also includes discussions about different emotions, including negative ones "
            "such as anger, sadness and even depression, as long as the author does not "
            "indicate at all to be thinking or discussing suicide or self-harm."
        ),
    }
    assert "four suicide-risk labels" not in UNSUPPORTIVE_SPECIALIST_PROMPT


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


def test_workflow_modes_keep_bool_specialists_separate_from_evidence_collectors():
    assert set(WORKFLOW_MODES) == {"bool", "evidence"}
    assert WORKFLOW_MODES["bool"].specialist_schema is SpecialistAnswer
    assert WORKFLOW_MODES["bool"].judge_schema is JudgeAnswer
    assert WORKFLOW_MODES["evidence"].specialist_schema is EvidenceCollectorAnswer
    assert WORKFLOW_MODES["evidence"].judge_schema is JudgeAnswer
    for mode in WORKFLOW_MODES.values():
        judge_prompt = mode.judge_system_prompt()
        assert "Unsupportive" in judge_prompt
        assert '"suicidal_intent": true' in judge_prompt
    assert "five specialist answers" in WORKFLOW_MODES["bool"].judge_system_prompt()


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
    assert LABEL_DEFINITIONS["Unsupportive"] in prompt
    assert "supports Indicator rather than Unsupportive" in prompt


def test_hierarchical_prompts_distinguish_unsupportive_from_indicator():
    for prompt in (
        framework_3_prompts.PERSPECTIVE_SYSTEM_PROMPT,
        framework_4_prompts.PERSPECTIVE_SYSTEM_PROMPT,
    ):
        for label in FINAL_LABELS:
            assert LABEL_DEFINITIONS[label] in prompt
        assert "Use UNSUPPORTIVE when the text does not match the suicide-risk task" in prompt

    for prompts in (framework_3_prompts, framework_4_prompts):
        assert LABEL_DEFINITIONS["Ideation"] in prompts.IDEATION_ACTION_SYSTEM_PROMPT
        assert LABEL_DEFINITIONS["Behavior"] in prompts.BEHAVIOR_ATTEMPT_SYSTEM_PROMPT
        assert LABEL_DEFINITIONS["Attempt"] in prompts.BEHAVIOR_ATTEMPT_SYSTEM_PROMPT

    assert "UNSUPPORTIVE is not meaningfully suicide-related" in (
        framework_4_prompts.JUDGE_SYSTEM_PROMPT
    )
    assert "UNSUPPORTIVE is unrelated to the suicide-risk task" in (
        framework_4_prompts.SECONDARY_REVIEW_SYSTEM_PROMPT
    )
