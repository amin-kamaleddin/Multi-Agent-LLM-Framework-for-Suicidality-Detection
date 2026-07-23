"""Provider-response normalization shared by workflow implementations."""

from __future__ import annotations

from typing import Any


def response_text(response: Any) -> str:
    """Normalize LangChain responses and simple string test doubles to text."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(part) for part in content)
    return str(content)


def usage_summary(response: Any) -> dict[str, Any]:
    """Extract the common token and cost fields exposed by supported providers."""
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    usage = usage if isinstance(usage, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    token_usage = metadata.get("token_usage") or {}
    token_usage = token_usage if isinstance(token_usage, dict) else {}

    prompt_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
        or 0
    )
    completion_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or token_usage.get("completion_tokens")
        or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(
            usage.get("total_tokens")
            or token_usage.get("total_tokens")
            or prompt_tokens + completion_tokens
        ),
        "cost": usage.get("cost") or token_usage.get("cost"),
    }
