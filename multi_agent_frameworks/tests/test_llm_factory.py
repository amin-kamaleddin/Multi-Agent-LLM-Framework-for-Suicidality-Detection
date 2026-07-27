import sys
from types import SimpleNamespace

import pytest

from llm import factory


class CapturingChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def chat_openai_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=CapturingChatOpenAI),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://us.api.openai.com/v1",
        "https://eu.api.openai.com/v1",
    ],
)
def test_direct_openai_base_url_detection(base_url):
    assert factory.is_direct_openai_base_url(base_url) is True


def test_standard_and_flex_map_to_openai_service_tier(monkeypatch, chat_openai_module):
    monkeypatch.setattr(factory, "openai_base_url", lambda: "https://api.openai.com/v1")

    standard = factory.build_real_model("gpt-4o-mini", 0.0, 128)
    flex = factory.build_real_model("gpt-4o-mini", 0.0, 128, service_tier="flex")

    assert standard.kwargs["service_tier"] == "default"
    assert flex.kwargs["service_tier"] == "flex"


def test_standard_omits_openai_service_tier_for_compatible_provider(
    monkeypatch,
    chat_openai_module,
):
    monkeypatch.setattr(factory, "openai_base_url", lambda: "https://openrouter.ai/api/v1")

    model = factory.build_real_model("gpt-4o-mini", 0.0, 128)

    assert model.kwargs["model"] == "openai/gpt-4o-mini"
    assert "service_tier" not in model.kwargs


def test_flex_rejects_non_openai_endpoint(monkeypatch, chat_openai_module):
    monkeypatch.setattr(factory, "openai_base_url", lambda: "https://openrouter.ai/api/v1")

    with pytest.raises(ValueError, match="direct OpenAI API endpoint"):
        factory.build_real_model("gpt-4o-mini", 0.0, 128, service_tier="flex")


def test_batch_is_not_accepted_as_a_synchronous_service_tier(
    monkeypatch,
    chat_openai_module,
):
    monkeypatch.setattr(factory, "openai_base_url", lambda: "https://api.openai.com/v1")

    with pytest.raises(ValueError, match="service_tier must be one of"):
        factory.build_real_model("gpt-4o-mini", 0.0, 128, service_tier="batch")
