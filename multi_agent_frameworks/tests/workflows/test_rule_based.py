import pytest

from workflows.rule_based.workflow import RuleBasedWorkflow, classify_text


@pytest.mark.parametrize(
    ("text", "expected_label", "expected_rule"),
    [
        ("I tried to kill myself last year.", "Attempt", "self_reported_attempt"),
        ("I wrote a goodbye note because I planned to die.", "Behavior", "preparatory_note"),
        ("I wish I would not wake up tomorrow.", "Ideation", "passive_death_wish"),
        ("My friend is suicidal and I am worried about them.", "Indicator", "third_party_suicide_context"),
        ("I am not suicidal.", "Indicator", "explicit_self_denial"),
        ("I watched Suicide Squad after work.", "Unsupportive", "no_suicide_or_self_harm_evidence"),
        ("I have a plan for my weekend.", "Unsupportive", "no_suicide_or_self_harm_evidence"),
    ],
)
def test_rule_based_classifies_representative_contexts(text, expected_label, expected_rule):
    match = classify_text(text)

    assert match.label_name == expected_label
    assert match.rule_name == expected_rule


def test_rule_based_uses_global_label_precedence():
    match = classify_text("I want to die. I tried to kill myself last year.")

    assert match.label_name == "Attempt"


def test_rule_based_workflow_returns_shared_contract_without_llm_calls():
    result = RuleBasedWorkflow().run_case(
        sample_id="case-1",
        text="I do not want to live.",
    )

    assert result.sample_id == "case-1"
    assert result.final_label == "Ideation"
    assert result.suicidal_intent is True
    assert result.errors == []
    assert result.llm_calls == []
    assert result.traces[0]["event"] == "rule_based_classification_complete"
    assert result.agent_outputs["rule_based_classifier"]["rule_name"] == "passive_death_wish"
