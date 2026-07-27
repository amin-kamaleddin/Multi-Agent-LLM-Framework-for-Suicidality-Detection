"""Construction and provider-name resolution for real chat models."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv


OPENAI_SERVICE_TIERS = ("standard", "flex")


def openai_base_url() -> str:
    load_dotenv()
    return os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def is_direct_openai_base_url(base_url: str) -> bool:
    """Return whether requests are sent directly to an OpenAI API endpoint."""
    hostname = (urlparse(base_url).hostname or "").lower()
    return hostname == "api.openai.com" or hostname.endswith(".api.openai.com")


def resolve_model_name_for_base_url(model_name: str, base_url: str) -> str:
    """Resolve common OpenAI aliases for OpenRouter's provider/model syntax."""
    if "openrouter.ai" in base_url.lower() and "/" not in model_name and model_name.startswith("gpt-"):
        return f"openai/{model_name}"
    return model_name


def build_real_model(
    model_name: str,
    temperature: float,
    max_completion_tokens: int,
    *,
    service_tier: str = "standard",
):
    load_dotenv()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for real LLM runs. Install with: pip install -e ."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env, or use --fake-llm for artifact smoke runs.")
    base_url = openai_base_url()
    resolved_model_name = resolve_model_name_for_base_url(model_name, base_url)
    if service_tier not in OPENAI_SERVICE_TIERS:
        raise ValueError(
            f"service_tier must be one of {list(OPENAI_SERVICE_TIERS)}; got {service_tier!r}"
        )
    if service_tier == "flex" and not is_direct_openai_base_url(base_url):
        raise ValueError(
            "OpenAI Flex processing requires a direct OpenAI API endpoint "
            "(for example, OPENAI_BASE_URL=https://api.openai.com/v1)."
        )

    model_kwargs = {
        "model": resolved_model_name,
        "temperature": temperature,
        "api_key": api_key,
        "base_url": base_url,
        "max_completion_tokens": max_completion_tokens,
        "max_retries": 0,
    }
    if is_direct_openai_base_url(base_url):
        model_kwargs["service_tier"] = "flex" if service_tier == "flex" else "default"
    return ChatOpenAI(
        **model_kwargs,
    )
