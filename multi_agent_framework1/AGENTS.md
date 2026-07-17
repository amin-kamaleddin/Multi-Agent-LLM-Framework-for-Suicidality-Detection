# Multi-Agent Frameworks

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

The second workflow is evidence-first and has exactly two stages:

```text
input text -> 5 parallel evidence collectors -> validate and aggregate -> final judge -> label + suicidal_intent
```

The five collectors cover `Indicator`, `Ideation`, `Behavior`, `Attempt`, and `Unsupportive`. They collect evidence only; they do not return predictions, booleans, scores, confidence, citations, or review status.

Evidence collector output schema (`framework2`, fixed `evidence` mode):

```json
{
  "evidence": [
    {
      "quote": "relevant excerpt or concise paraphrase",
      "type": "collector-specific evidence type",
      "subject": "author",
      "time": "current",
      "observation": "What the quote factually shows."
    }
  ]
}
```

Each collector returns zero to three items. Quotes and observations are each limited to 100 words; quotes may be normalized or paraphrased and are not checked as exact input substrings. Each collector has only three allowed evidence types, defined with their interpretations in `workflow/prompts.py` and repeated in the final judge prompt.

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
- `configs/framework1.example.json`, `configs/framework2.example.json`: copyable shared-CLI configurations.
- `src/suicidal_multi_agent/`: source package.
- `src/suicidal_multi_agent/workflow/prompts.py`: prompt/schema registry for `bool`, `reasoned`, and `evidence` modes.
- `src/suicidal_multi_agent/workflow/registry.py`: CLI registry for framework1 and framework2.
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
uv run python -m suicidal_multi_agent.cli evaluate --workflow framework2 --fake-llm --limit 4
```

Select a dataset, workflow JSON configuration, and log base directory:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --workflow framework1 --workflow-config configs/framework1.example.json --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --artifact-base-dir logs/experiments --run-id framework1_dataset1_500
uv run python -m suicidal_multi_agent.cli evaluate --workflow framework2 --workflow-config configs/framework2.example.json --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --artifact-base-dir logs/experiments --run-id framework2_dataset1_500
```

Run a fake-LLM smoke evaluation with specialist reasoning:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --fake-llm --limit 4 --mode reasoned
```

Run the real 500-row `gpt-4o` evaluation:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --model gpt-4o --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --dataset-audit-path dataset/processed/dataset1_audit.json --max-completion-tokens 1024 --run-id gpt4o_dataset1_500
```

Run the real 500-row `gpt-4o` evaluation with specialist reasoning:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate --model gpt-4o --mode reasoned --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --dataset-audit-path dataset/processed/dataset1_audit.json --max-completion-tokens 1024 --run-id gpt4o_reasoned_dataset1_500
```

Compile a completed run:

```powershell
uv run python -m suicidal_multi_agent.cli compile-results --run-dir logs/runs/<run_folder>
```

## Notes

- The default maximum completion-token setting is 1024 to accommodate evidence packets. Override it only with a value supported by the selected model and provider.
- Framework1 compatibility runs contain `specialist_answers` for metrics and `specialist_outputs` for the full per-label JSON in `predictions.csv`. In `reasoned` mode they also include per-label `*_reasoning` columns. Architecture-neutral evaluator runs omit those framework-specific fields unless an adapter supplies them.
- Framework2 runs contain `collector_outputs` and role-keyed `agent_outputs`. Collector outputs are never converted into predictions or one-vs-rest metrics; only the final judge is evaluated.
- `run` and `evaluate` automatically write `compiled_results.csv` and `compiled_results.md` after final artifacts are complete. Use `compile-results` only to regenerate reports for an older run.
- Real runs write artifacts under `logs/runs/` by default; `evaluate --artifact-base-dir` selects another base directory. These are local generated evidence and are not committed in this folder.
- This is research evaluation code, not clinical deployment software.
