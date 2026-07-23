# Suicide-Risk Multi-Agent Evaluator

This package contains four LangGraph workflows and an architecture-neutral evaluator:

- `framework_1_parallel_no_evidence`: five parallel binary or reasoned specialists, then a final judge
- `framework_2_parallel_with_evidence`: five parallel non-decision-making evidence collectors, then a final judge
- `framework_3_hierarchical_no_reviewer`: hierarchical routing followed by a final evaluator
- `framework_4_hierarchical_with_reviewer`: hierarchical routing, an uncertainty gate, and an optional secondary reviewer

All four workflows accept the same input and return the same strict final JSON contract. The evaluator can also run any workflow that accepts one `sample_id` and text at a time and returns the shared final-result and telemetry fields.

Start here:

- [Adapting another workflow](src/evaluation/ADAPTING_WORKFLOWS.md) provides a copyable adapter and runner.
- [Evaluation contract](src/evaluation/EVALUATION_CONTRACT.md) defines all input/output schemas, metrics, guardrails, and artifacts.
- [Shared CLI usage](../docs/USAGE.md#shared-agent-workflow-evaluator-cli) explains workflow selection, JSON configuration, datasets, log directories, and registry extension.
- [Framework 1 example configuration](configs/framework_1_parallel_no_evidence.example.json) can be copied and adjusted per run.
- [Framework 2 example configuration](configs/framework_2_parallel_with_evidence.example.json) selects the evidence workflow.
- [Framework guide](AGENTS.md) documents the built-in workflows and commands.

## How the layers fit together

The command path for every registered framework is:

```text
src/cli.py
  -> src/workflows/registry.py
  -> src/workflows/framework_<n>_<name>/evaluation.py
  -> src/workflows/framework_<n>_<name>/graph.py::run_case
  -> src/evaluation/evaluator.py
  -> run artifacts
```

The workflow owns model orchestration, intermediate schemas, prompts, and traces. The shared evaluator owns dataset iteration, the five-label confusion matrix, grouped `suicidal_intent` metrics, runtime telemetry, guardrails, and artifacts.

## Directory structure

```text
multi_agent_frameworks/
├── configs/        Example JSON configurations for CLI-selectable workflows
├── dataset/        Source, processed, and frozen evaluation datasets
├── logs/           Generated local runs; ignored by Git
├── scripts/        Dataset and provider smoke-test utilities
├── src/            Runtime source code
├── test-results/   Deliberately retained benchmark snapshots
├── tests/          Unit, integration, graph, prompt, and reporting tests
├── .env.example    Environment-variable template
├── pyproject.toml  Python project, dependencies, and test configuration
└── uv.lock         Locked dependency versions
```

Local `.env`, `.venv`, caches, and `logs/` are generated development state rather than source code.

### `src/`

| Path | Responsibility |
|---|---|
| `src/cli.py` | Implements `evaluate`, `list-workflows`, dataset commands, and the legacy framework-1-only `run` shortcut. New frameworks should integrate with `evaluate` through the registry. |
| `src/data/` | Loads, audits, cleans, samples, and validates datasets. |
| `src/domain/` | Defines concepts shared by every architecture: the five labels, label-to-`suicidal_intent` mapping, strict final output, and evidence contracts. |
| `src/llm/` | Defines the model protocol, real-client factory, retry/structured-output invocation, and deterministic fake model. |
| `src/evaluation/` | Evaluates any workflow that satisfies `workflows/contracts.py`, then writes metrics, reports, traces, and other artifacts. |
| `src/workflows/` | Contains the shared workflow boundary, CLI registry, reusable workflow internals, and one directory per runnable agentic framework. |

### `src/domain/`

| Module | Responsibility |
|---|---|
| `labels.py` | Canonical label constants and deterministic boolean grouping. |
| `final_output.py` | Strict final judge JSON schema and label/boolean consistency validation. |
| `evidence.py` | Evidence item, evidence packet, and evidence-type validation used by framework 2. |

Domain modules describe architecture-independent data. They must not import a particular workflow.

### `src/llm/`

| Module | Responsibility |
|---|---|
| `protocol.py` | Minimal interface a real or fake model must implement. |
| `invoker.py` | Structured-output invocation, retries, validation, and call telemetry. |
| `factory.py` | Builds the configured OpenAI-compatible model client. |
| `fakes.py` | Deterministic responses for tests and smoke runs. |

### `src/evaluation/`

| Module | Responsibility |
|---|---|
| `evaluator.py` | Architecture-neutral dataset loop and per-record result persistence. |
| `metrics.py` | Five-label confusion audit and the only predictive score set: grouped `suicidal_intent` metrics. |
| `artifacts.py` | Writes JSON, CSV, Markdown, JSONL, text, and SVG run artifacts. |
| `reporting.py` | Compiles one run directory into review-oriented CSV and Markdown reports. |
| `EVALUATION_CONTRACT.md` | Complete input, output, metric, guardrail, and artifact contract. |
| `ADAPTING_WORKFLOWS.md` | Detailed adapter example for an arbitrary workflow architecture. |

### `src/workflows/`

```text
workflows/
├── contracts.py
├── registry.py
├── shared/
│   ├── parallel/
│   │   ├── graph.py
│   │   └── prompts.py
│   └── hierarchical/
│       └── schemas.py
├── framework_1_parallel_no_evidence/
├── framework_2_parallel_with_evidence/
├── framework_3_hierarchical_no_reviewer/
└── framework_4_hierarchical_with_reviewer/
```

`contracts.py` defines the small architecture-neutral boundary used by the evaluator:

- `Workflow.run_case(sample_id, text)`;
- the required case-result fields;
- the request and registration objects used by the CLI registry.

`registry.py` maps stable lowercase names such as `framework_2_parallel_with_evidence` to runner functions. `python -m cli evaluate --workflow ...` can select only workflows registered there.

Each `framework_*` directory is a runnable architecture and uses the same module roles:

| Module | Framework-owned responsibility |
|---|---|
| `config.py` | Stable workflow name, default mode, and accepted JSON configuration keys. |
| `schemas.py` | Intermediate agent outputs unique to this framework. The final output always uses `domain/final_output.py`. |
| `prompts.py` | Framework prompt API. It may define prompts locally or re-export a shared implementation. |
| `graph.py` | Graph topology, nodes, routing, state, and the public workflow class. |
| `evaluation.py` | Thin adapter that builds the graph and passes it to the shared evaluator. It may preserve framework-specific outputs but must not calculate intermediate-agent metrics. |
| `__init__.py` | Small public export surface for the workflow name and class. |

## What `workflows/shared/` means

`shared/` contains implementation details reused by at least two workflow architectures. It is not a fifth workflow and cannot be selected from the CLI.

- `shared/parallel/graph.py` is the common LangGraph engine used by frameworks 1 and 2. It creates the parallel intermediate nodes, joins them at the judge, invokes models, validates structured outputs, and records traces and calls.
- `shared/parallel/prompts.py` contains the prompt/mode catalogue consumed by that engine. Framework 1 selects the boolean or reasoned specialist modes; framework 2 selects the evidence mode.
- `shared/hierarchical/schemas.py` contains the three routing-decision schemas whose meanings are identical in frameworks 3 and 4.

Code belongs in `shared/` only when both its behavior and contract are genuinely identical across multiple frameworks. A prompt, schema, node, or routing rule used by one framework should remain in that framework's directory.

### Why are `parallel/` and `hierarchical/` different inside `shared/`?

Both architecture families now have explicit namespaces:

```text
shared/
├── parallel/
│   ├── graph.py    shared parallel graph engine
│   └── prompts.py  shared parallel prompt modes
└── hierarchical/
    └── schemas.py  shared hierarchical routing schemas
```

The contents still differ because the reuse boundary differs. Frameworks 1 and 2 share their complete fan-out/join graph implementation and prompt-mode catalogue. Frameworks 3 and 4 share only routing-decision schemas; their graph topologies and prompts remain framework-owned because the reviewer changes the nodes, routing, and finalization behavior.

## Adding a new agentic framework

Use a new stable lowercase name, for example `framework_5_debate_with_judge`.

### 1. Create the framework package

Create:

```text
src/workflows/framework_5_debate_with_judge/
├── __init__.py
├── config.py
├── evaluation.py
├── graph.py
├── prompts.py
└── schemas.py
```

In `config.py`, declare at least:

```python
WORKFLOW_NAME = "framework_5_debate_with_judge"
DEFAULT_MODE = "debate"
CONFIG_KEYS = (
    "model",
    "temperature",
    "max_completion_tokens",
    "max_retries",
    "mode",
    "fake_llm",
)
```

Add an `agents` key and role-specific configuration only if the new framework supports separate model settings per role.

### 2. Implement the workflow boundary

The public class in `graph.py` must expose:

```python
run_case(*, sample_id: str, text: str) -> CaseResult
```

The returned object must provide:

```text
sample_id
final_label
suicidal_intent
errors
traces
llm_calls
```

It may also provide `agent_outputs` or other framework-specific data for persistence. The final successful result must validate against:

```json
{
  "label_name": "Ideation",
  "suicidal_intent": true
}
```

The boolean is deterministic: `Indicator`, `Ideation`, `Behavior`, and `Attempt` map to `true`; `Unsupportive` maps to `false`.

If the new workflow has exactly the same fan-out/join behavior as frameworks 1 and 2, it may subclass `workflows.shared.parallel.MultiAgentWorkflow`. The shared engine currently accepts modes registered in `workflows/shared/parallel/prompts.py::WORKFLOW_MODES`. If the topology or intermediate contract differs, implement a separate graph in the new framework instead of adding framework-specific branches to the shared engine.

For a hierarchical workflow, reuse `shared/hierarchical/schemas.py` only when the routing decisions have exactly the existing meanings. New routing semantics require local schemas.

### 3. Add the evaluation adapter

In the new `evaluation.py`:

1. construct the workflow with the resolved model and retry settings;
2. copy the shared `EvaluationConfig` while setting `workflow_name`;
3. call `evaluation.evaluator.run_evaluation`;
4. optionally pass a `prediction_enricher` to persist extra outputs.

The adapter should not iterate the dataset, calculate metrics, or write artifacts itself. Those concerns stay in `src/evaluation/`.

### 4. Register the workflow

In `src/workflows/registry.py`:

1. import the new `CONFIG_KEYS` and evaluation runner;
2. add a private runner that validates workflow-config keys and modes;
3. build the fake or real model;
4. call the framework's evaluation adapter;
5. add one `WorkflowRegistration` to `WORKFLOW_REGISTRY`.

The generic `evaluate` command reads its choices from the registry automatically. Do not add another top-level CLI command unless the framework needs genuinely new command-line behavior. The existing `run` command is a legacy shortcut for framework 1, not the extension point.

### 5. Add configuration and tests

- Add `configs/framework_5_debate_with_judge.example.json` when the workflow accepts configuration beyond common CLI defaults.
- Add prompt and schema tests.
- Add graph tests for every route and failure path.
- Add evaluator tests that verify the shared final contract and preserved optional outputs.
- Update `tests/workflows/test_registry.py` to expect the new registration.
- Extend `llm/fakes.py` only when the existing deterministic fake cannot produce the new intermediate schema.

Verify the integration:

```powershell
uv run python -m pytest
uv run python -m cli list-workflows
uv run python -m cli evaluate --workflow framework_5_debate_with_judge --fake-llm --limit 4
```

The workflow is integrated when it appears in `list-workflows`, passes the fake-model smoke run, returns the shared final output, preserves its traces and calls, and produces the standard evaluator artifacts.

## Setup

From this directory:

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` before using a real OpenAI-compatible endpoint. A local fake-model smoke run requires no API key:

```powershell
uv run python -m cli list-workflows
uv run python -m cli evaluate --workflow framework_1_parallel_no_evidence --fake-llm --limit 4
uv run python -m cli evaluate --workflow framework_2_parallel_with_evidence --fake-llm --limit 4
uv run python -m cli evaluate --workflow framework_3_hierarchical_no_reviewer --fake-llm --limit 4
uv run python -m cli evaluate --workflow framework_4_hierarchical_with_reviewer --fake-llm --limit 4
```

Run the test suite with:

```powershell
uv run python -m pytest
```

This repository is research evaluation code, not a clinical assessment or deployment system. The `suicidal_intent` field means membership in the four-label positive dataset grouping; it is not by itself a claim that the author personally expresses suicidal intent.
