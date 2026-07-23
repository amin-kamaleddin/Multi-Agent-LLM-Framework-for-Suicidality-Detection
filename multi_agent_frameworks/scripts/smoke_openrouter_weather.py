"""Smoke-test OpenRouter credentials and base URL with one chat completion."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_OUTPUT = Path("logs/tests/openrouter_gpt4o_mini_weather_smoke.json")
DEFAULT_PROMPT = "How is the weather today in Montreal?"


def run_smoke_test(
    *,
    model: str,
    prompt: str,
    output_path: Path,
    max_completion_tokens: int,
) -> dict[str, Any]:
    load_dotenv()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is not configured in .env")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in .env")

    started = dt.datetime.now(dt.UTC)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_completion_tokens=max_completion_tokens,
    )
    ended = dt.datetime.now(dt.UTC)

    message = response.choices[0].message.content if response.choices else ""
    usage = response.usage.model_dump() if response.usage is not None else {}
    payload = {
        "test_name": "openrouter_gpt4o_mini_weather_smoke",
        "status": "response_received" if message else "empty_response",
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "latency_seconds": (ended - started).total_seconds(),
        "base_url": base_url,
        "model": model,
        "prompt": prompt,
        "max_completion_tokens": max_completion_tokens,
        "api_key_configured": bool(api_key),
        "response_text": message,
        "usage": usage,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Smoke-test OpenRouter API configuration.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-completion-tokens", type=int, default=200)
    args = parser.parse_args()

    try:
        payload = run_smoke_test(
            model=args.model,
            prompt=args.prompt,
            output_path=args.output,
            max_completion_tokens=args.max_completion_tokens,
        )
    except Exception as exc:
        payload = {
            "test_name": "openrouter_gpt4o_mini_weather_smoke",
            "status": "error",
            "started_at": dt.datetime.now(dt.UTC).isoformat(),
            "ended_at": dt.datetime.now(dt.UTC).isoformat(),
            "base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "model": args.model,
            "prompt": args.prompt,
            "max_completion_tokens": args.max_completion_tokens,
            "api_key_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: payload[k] for k in ("test_name", "status", "base_url", "model", "error_type")}, indent=2))
        print(f"result_path={args.output}")
        return 1

    print(json.dumps({k: payload[k] for k in ("test_name", "status", "base_url", "model", "latency_seconds")}, indent=2))
    print(f"result_path={args.output}")
    return 0 if payload["status"] == "response_received" else 1


if __name__ == "__main__":
    raise SystemExit(main())
