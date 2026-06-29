# Multi-Agent Framework 1

This folder vendors the standalone naive multi-agent suicide-risk classifier from `suicidal_multi_agent` into the legacy `Suicide-Detection-Agent` repository without changing the legacy code paths.

## Purpose

The framework evaluates this v1 workflow:

```text
input text -> 4 parallel binary specialist agents -> JSON answers -> final judge -> one label
```

Labels are exactly:

- `Indicator`
- `Ideation`
- `Behavior`
- `Attempt`

Specialist output schema:

```json
{"answer": true}
```

Judge output schema:

```json
{"label_name": "Ideation"}
```

## Folder Index

- `pyproject.toml`, `uv.lock`, `.python-version`: self-contained `uv` project configuration.
- `.env.example`: environment template; copy to `.env` locally and fill API settings.
- `src/suicidal_multi_agent/`: source package.
- `tests/`: pytest suite for data rules, schemas, prompts, graph behavior, metrics, evaluator, and reporting.
- `scripts/smoke_openrouter_weather.py`: optional OpenRouter smoke test.
- `dataset/`: active anonymized dataset and processed evaluation samples used by the CLI.

Generated `logs/`, local `.env`, virtual environments, caches, and nested Git metadata are intentionally not committed here.

## Setup

From this folder:

```powershell
uv sync
Copy-Item .env.example .env
```

Then edit `.env` with an OpenRouter/OpenAI-compatible key. The default base URL is OpenRouter-compatible.

## Useful Commands

Run tests:

```powershell
uv run python -m pytest tests/core tests/data tests/workflow tests/evaluation
```

Prepare the canonical cleaned dataset and balanced 500-row sample:

```powershell
uv run python -m suicidal_multi_agent.cli prepare-data --sample-size 500 --seed 42
```

Run a fake-LLM smoke evaluation:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --fake-llm --limit 4
```

Run the real 500-row `gpt-4o` evaluation:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --model gpt-4o --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --dataset-audit-path dataset/processed/dataset1_audit.json --max-completion-tokens 256 --run-id gpt4o_dataset1_500
```

Compile a completed run:

```powershell
uv run python -m suicidal_multi_agent.cli compile-results --run-dir logs/runs/<run_folder>
```

## Notes

- Use `--max-completion-tokens 256` for `gpt-4o`; the previous 500,000-token default exceeds the model/provider limit.
- Real runs write artifacts under `logs/runs/`; those are local generated evidence and are not committed in this folder.
- This is research evaluation code, not clinical deployment software.
