# Suicide-Risk Multi-Agent Evaluator

This package contains two LangGraph workflows and an architecture-neutral evaluator:

- `framework1`: four parallel binary or reasoned specialists, then a final judge
- `framework2`: five parallel non-decision-making evidence collectors, then a final judge

Both workflows accept the same input and return the same strict final JSON contract. The evaluator can also run any workflow that accepts one `sample_id` and text at a time and returns the shared final-result and telemetry fields.

Start here:

- [Adapting another workflow](src/suicidal_multi_agent/evaluation/ADAPTING_WORKFLOWS.md) provides a copyable adapter and runner.
- [Evaluation contract](src/suicidal_multi_agent/evaluation/EVALUATION_CONTRACT.md) defines all input/output schemas, metrics, guardrails, and artifacts.
- [Shared CLI usage](../docs/USAGE.md#shared-agent-workflow-evaluator-cli) explains workflow selection, JSON configuration, datasets, log directories, and registry extension.
- [Framework1 example configuration](configs/framework1.example.json) can be copied and adjusted per run.
- [Framework2 example configuration](configs/framework2.example.json) selects the evidence workflow.
- [Framework1 guide](AGENTS.md) documents the built-in four-specialist commands.

## Setup

From this directory:

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` before using a real OpenAI-compatible endpoint. A local fake-model smoke run requires no API key:

```powershell
uv run python -m suicidal_multi_agent.cli list-workflows
uv run python -m suicidal_multi_agent.cli evaluate --workflow framework1 --fake-llm --limit 4
uv run python -m suicidal_multi_agent.cli evaluate --workflow framework2 --fake-llm --limit 4
```

Run the test suite with:

```powershell
uv run python -m pytest tests/core tests/data tests/workflow tests/evaluation
```

This repository is research evaluation code, not a clinical assessment or deployment system. The `suicidal_intent` field means membership in the four-label positive dataset grouping; it is not by itself a claim that the author personally expresses suicidal intent.
