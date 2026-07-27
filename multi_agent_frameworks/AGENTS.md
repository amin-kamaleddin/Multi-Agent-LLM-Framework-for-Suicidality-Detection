# Multi-Agent Frameworks

This folder contains the four standalone multi-agent suicide-risk research workflows. Python modules are flattened directly under `src/`; there is no intermediate application package.

## Purpose

The framework evaluates this v1 workflow:

```text
input text -> 5 parallel binary specialist agents -> JSON answers -> final judge -> label + suicidal_intent
```

The five specialist labels are:

- `Indicator`
- `Ideation`
- `Behavior`
- `Attempt`
- `Unsupportive`

The first four specialists represent the positive grouping; the `Unsupportive` specialist represents the negative grouping. All five agents independently return whether the text matches their assigned label, and the final judge resolves their outputs into one final label.

The second workflow is evidence-first and has exactly two stages:

```text
input text -> 5 parallel evidence collectors -> validate and aggregate -> final judge -> label + suicidal_intent
```

The five collectors cover `Indicator`, `Ideation`, `Behavior`, `Attempt`, and `Unsupportive`. They collect evidence only; they do not return predictions, booleans, scores, confidence, citations, or review status.

Evidence collector output schema (`framework_2_parallel_with_evidence`, fixed `evidence` mode):

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

Each collector returns zero to three items. Quotes and observations are each limited to 100 words; quotes may be normalized or paraphrased and are not checked as exact input substrings. Each collector has only three allowed evidence types, defined with their interpretations in `src/workflows/framework_2_parallel_with_evidence/prompts.py` and repeated in the final judge prompt.

Default specialist output schema (`--mode bool`):

```json
{"answer": true}
```

Judge output schema:

```json
{"label_name": "Ideation", "suicidal_intent": true}
```

## Folder Index

- `pyproject.toml`, `uv.lock`, `.python-version`: self-contained `uv` project configuration.
- `.env.example`: environment template; copy to `.env` locally and fill API settings.
- `configs/framework_1_parallel_no_evidence.example.json`, `configs/framework_2_parallel_with_evidence.example.json`: copyable shared-CLI configurations.
- `src/`: source package.
- `src/domain/`: labels, evidence items, and the shared final judge contract.
- `src/llm/`: model protocol, real-model factory, and deterministic smoke-test model.
- `src/workflows/registry.py`: CLI registry for all four canonical framework names.
- `src/workflows/framework_1_parallel_no_evidence/`: parallel decision-making specialists.
- `src/workflows/framework_2_parallel_with_evidence/`: parallel evidence collectors.
- `src/workflows/framework_3_hierarchical_no_reviewer/`: hierarchy without a secondary reviewer.
- `src/workflows/framework_4_hierarchical_with_reviewer/`: hierarchy with uncertainty review.
- `src/evaluation/ADAPTING_WORKFLOWS.md`: copyable guide for connecting another agentic workflow to the shared evaluator.
- `src/evaluation/EVALUATION_CONTRACT.md`: exhaustive input, output, metric, guardrail, and artifact reference.
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
uv run python -m pytest
```

Prepare the canonical cleaned dataset and balanced 500-row sample:

```powershell
uv run python -m cli prepare-data --sample-size 500 --seed 42
```

Run a fake-LLM smoke evaluation:

```powershell
uv run python -m cli list-workflows
uv run python -m cli evaluate --workflow framework_1_parallel_no_evidence --fake-llm --limit 4
uv run python -m cli evaluate --workflow framework_2_parallel_with_evidence --fake-llm --limit 4
uv run python -m cli evaluate --workflow framework_3_hierarchical_no_reviewer --fake-llm --limit 4
uv run python -m cli evaluate --workflow framework_4_hierarchical_with_reviewer --fake-llm --limit 4
```

Select a dataset, workflow JSON configuration, and log base directory:

```powershell
uv run python -m cli evaluate --workflow framework_1_parallel_no_evidence --workflow-config configs/framework_1_parallel_no_evidence.example.json --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --artifact-base-dir logs/experiments --run-id framework_1_parallel_no_evidence_dataset1_500
uv run python -m cli evaluate --workflow framework_2_parallel_with_evidence --workflow-config configs/framework_2_parallel_with_evidence.example.json --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --artifact-base-dir logs/experiments --run-id framework_2_parallel_with_evidence_dataset1_500
```

Run the real 500-row `gpt-4o` evaluation:

```powershell
uv run python -m cli evaluate --model gpt-4o --sample-path dataset/processed/dataset1_balanced_500_seed42.csv --dataset-audit-path dataset/processed/dataset1_audit.json --max-completion-tokens 1024 --run-id gpt4o_dataset1_500
```

Compile a completed run:

```powershell
uv run python -m cli compile-results --run-dir logs/runs/<run_folder>
```

## Notes

- The default maximum completion-token setting is 1024 to accommodate evidence packets. Override it only with a value supported by the selected model and provider.
- `framework_1_parallel_no_evidence` is bool-only and preserves `specialist_answers`, full per-label `specialist_outputs`, and per-label `*_answer` columns in `predictions.csv`. Intermediate outputs are retained for inspection but are not evaluated; metrics cover only the final judge.
- Direct OpenAI evaluations accept `--service-tier standard` or `--service-tier flex`. OpenAI Batch is a separate asynchronous API and is not a service-tier value in this evaluator.
- `evaluate --stop-after-tokens N` stops new case scheduling after the current invocation reports at least `N` prompt plus completion tokens. Resuming the run starts a fresh token counter; cumulative runtime metrics still include earlier invocations.
- `framework_2_parallel_with_evidence` runs contain `collector_outputs` and role-keyed `agent_outputs`. Collector outputs are never converted into predictions or one-vs-rest metrics; only the final judge is evaluated.
- `run` and `evaluate` automatically write `compiled_results.csv` and `compiled_results.md` after final artifacts are complete. Use `compile-results` only to regenerate reports for an older run.
- Real runs write artifacts under `logs/runs/` by default; `evaluate --artifact-base-dir` selects another base directory. These are local generated evidence and are not committed in this folder.
- This is research evaluation code, not clinical deployment software.
