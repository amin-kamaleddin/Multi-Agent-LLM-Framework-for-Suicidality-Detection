# Multi-Agent Framework 1

This folder vendors the standalone naive multi-agent suicide-risk classifier from `suicidal_multi_agent` into the legacy `Suicide-Detection-Agent` repository without changing the legacy code paths.

## Purpose

The framework evaluates this v1 workflow:

```text
input text -> 4 parallel binary specialist agents -> JSON answers -> final judge -> label + suicidal_intent
```

The four specialist labels are:

- `Indicator`
- `Ideation`
- `Behavior`
- `Attempt`

The final judge may also return `Unsupportive`.

Default specialist output schema (`--mode bool`):

```json
{"answer": true}
```

Reasoned specialist output schema (`--mode reasoned`):

```json
{"answer": true, "reasoning": "1-3 sentence explanation including supporting evidence"}
```

Judge output schema:

```json
{"label_name": "Ideation", "suicidal_intent": true}
```

## Folder Index

- `pyproject.toml`, `uv.lock`, `.python-version`: self-contained `uv` project configuration.
- `.env.example`: environment template; copy to `.env` locally and fill API settings.
- `configs/framework1.example.json`: copyable shared-CLI configuration for framework1.
- `src/suicidal_multi_agent/`: source package.
- `src/suicidal_multi_agent/workflow/prompts.py`: prompt/schema mode registry. `bool` and `reasoned` modes reuse the same graph.
- `src/suicidal_multi_agent/workflow/registry.py`: CLI workflow-name registry and built-in framework1 runner.
- `src/suicidal_multi_agent/evaluation/ADAPTING_WORKFLOWS.md`: copyable guide for connecting another agentic workflow to the shared evaluator.
- `src/suicidal_multi_agent/evaluation/EVALUATION_CONTRACT.md`: exhaustive input, output, metric, guardrail, and artifact reference.
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
uv run python -m suicidal_multi_agent.cli list-workflows
uv run python -m suicidal_multi_agent.cli evaluate --workflow framework1 --fake-llm --limit 4
```

Select a dataset, workflow JSON configuration, and log base directory:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --workflow framework1 --workflow-config configs/framework1.example.json --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --artifact-base-dir logs/experiments --run-id framework1_dataset1_500
```

Run a fake-LLM smoke evaluation with specialist reasoning:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --fake-llm --limit 4 --mode reasoned
```

Run the real 500-row `gpt-4o` evaluation:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --model gpt-4o --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --dataset-audit-path dataset/processed/dataset1_audit.json --max-completion-tokens 256 --run-id gpt4o_dataset1_500
```

Run the real 500-row `gpt-4o` evaluation with specialist reasoning:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --model gpt-4o --mode reasoned --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --dataset-audit-path dataset/processed/dataset1_audit.json --max-completion-tokens 256 --run-id gpt4o_reasoned_dataset1_500
```

Compile a completed run:

```powershell
uv run python -m suicidal_multi_agent.cli compile-results --run-dir logs/runs/<run_folder>
```

## Notes

- The default maximum completion-token setting is 256. Override it only with a value supported by the selected model and provider.
- Framework1 compatibility runs contain `specialist_answers` for metrics and `specialist_outputs` for the full per-label JSON in `predictions.csv`. In `reasoned` mode they also include per-label `*_reasoning` columns. Architecture-neutral evaluator runs omit those framework-specific fields unless an adapter supplies them.
- `run` and `evaluate` automatically write `compiled_results.csv` and `compiled_results.md` after final artifacts are complete. Use `compile-results` only to regenerate reports for an older run.
- Real runs write artifacts under `logs/runs/` by default; `evaluate --artifact-base-dir` selects another base directory. These are local generated evidence and are not committed in this folder.
- This is research evaluation code, not clinical deployment software.
