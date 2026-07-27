# Suicide-Risk Multi-Agent Evaluator

This package contains four LangGraph workflows and an architecture-neutral evaluator:

- `framework_1_parallel_no_evidence`: five parallel boolean specialists, then a final judge
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
| `src/llm/` | Defines the model protocol, real-client factory, retry/structured-output invocation, deterministic fake model, and run-wide request limiter. |
| `src/evaluation/` | Evaluates any workflow that satisfies `workflows/contracts.py`, coordinates concurrent cases, persists resumable progress, then writes metrics and artifacts. |
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
| `limiter.py` | Applies one shared `max_in_flight_llm` semaphore to every model used by a run. |

### `src/evaluation/`

| Module | Responsibility |
|---|---|
| `evaluator.py` | Architecture-neutral concurrent case scheduler, result aggregation, coverage validation, and reporting handoff. |
| `progress.py` | SQLite progress store, manifest fingerprints, atomic checkpoints, and exact ID coverage checks. |
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

`registry.py` maps stable lowercase names such as `framework_2_parallel_with_evidence` to runner functions. `uv run python -m cli evaluate --workflow ...` can select only workflows registered there.

Each `framework_*` directory is a runnable architecture and uses the same module roles:

| Module | Framework-owned responsibility |
|---|---|
| `config.py` | Stable workflow name, default mode, and accepted JSON configuration keys. |
| `schemas.py` | Intermediate agent outputs unique to this framework. The final output always uses `domain/final_output.py`. |
| `prompts.py` | Framework prompt API. It may define prompts locally or re-export a shared implementation. |
| `PROMPTS.md` | Fully formatted system and user messages for every LLM agent, with runtime input text represented by `{{INPUT_TEXT}}`. |
| `graph.py` | Graph topology, nodes, routing, state, and the public workflow class. |
| `evaluation.py` | Thin adapter that builds the graph and passes it to the shared evaluator. It may preserve framework-specific outputs but must not calculate intermediate-agent metrics. |
| `__init__.py` | Small public export surface for the workflow name and class. |

## What `workflows/shared/` means

`shared/` contains implementation details reused by at least two workflow architectures. It is not a fifth workflow and cannot be selected from the CLI.

- `shared/parallel/graph.py` is the common LangGraph engine used by frameworks 1 and 2. It creates the parallel intermediate nodes, joins them at the judge, invokes models, validates structured outputs, and records traces and calls.
- `shared/parallel/prompts.py` contains the prompt/mode catalogue consumed by that engine. Framework 1 selects the boolean specialist mode; framework 2 selects the evidence mode.
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

## CLI reference

From the repository root, enter the standalone multi-agent project and install
its locked dependencies:

```console
cd multi_agent_frameworks
uv sync
```

All commands below must be run from `multi_agent_frameworks/` and use the
cross-platform `uv run python -m cli ...` form:

```console
uv run python -m cli --help
uv run python -m cli evaluate --help
```

Multi-line examples use PowerShell backticks for readability. The `uv run`
command itself is the same on Windows, macOS, and Linux; use the continuation
syntax appropriate for the current shell or place the command on one line.

### Configure the endpoint and default model settings

Copy `.env.example` to `.env` and set at least:

```dotenv
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=<your-key>
```

For OpenRouter, a bare OpenAI model such as `gpt-4o-mini` is resolved to
`openai/gpt-4o-mini`. Other OpenRouter models should normally use their complete
`provider/model` name. For another OpenAI-compatible endpoint, use the model name
expected by that endpoint.

The evaluator recognizes these environment defaults:

| Variable | Built-in default | Used by |
|---|---:|---|
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | All real-model commands |
| `OPENAI_API_KEY` | None; required for real models | All real-model commands |
| `OPENAI_MODEL` | `openai/gpt-4o-mini` | Legacy `run` shortcut only |
| `OPENAI_TEMPERATURE` | `0` | `run` and `evaluate` |
| `OPENAI_SERVICE_TIER` | `standard` | `evaluate`; direct OpenAI endpoints only |
| `MAX_COMPLETION_TOKENS` | `1024` | `run` and `evaluate` |
| `MAX_RETRIES` | `3` | `run` and `evaluate` |
| `EVAL_CASE_WORKERS` | `4` | `evaluate` |
| `EVAL_MAX_IN_FLIGHT_LLM` | `20` | `evaluate` |
| `EVAL_CHECKPOINT_EVERY` | `250` | `evaluate` |
| `EVAL_ERROR_STOP_FAILED_LLM_ATTEMPTS` | `10` | `evaluate` |
| `EVAL_ERROR_STOP_FAILED_RECORDS` | `3` | `evaluate` |
| `EVAL_ERROR_STOP_RATE` | `0.5` | `evaluate` |
| `EVAL_ERROR_STOP_MIN_RECORDS` | `10` | `evaluate` |

For common workflow settings, precedence is:

```text
explicit CLI option -> workflow-config JSON -> environment/default
```

`evaluate` defaults to `gpt-4o` when neither `--model` nor a workflow JSON model
is supplied. `OPENAI_MODEL` applies only to the legacy `run` shortcut, so
production `evaluate` commands should normally specify `--model` explicitly.

For a direct OpenAI endpoint, `evaluate` accepts `--service-tier standard` or
`--service-tier flex`. `standard` sends the API value `service_tier=default`;
`flex` sends `service_tier=flex`. Flex is rejected for OpenRouter and other
OpenAI-compatible endpoints because it is an OpenAI processing tier, not a
portable chat-completions option. Standard requests to compatible providers
omit the OpenAI-specific field.

OpenAI Batch is not a `service_tier` value. It is a separate asynchronous API
that uploads a JSONL request file and returns results later. The current
case-by-case LangGraph evaluator therefore supports Standard and Flex but not
Batch; adding Batch requires a separate staged scheduler that batches each
independent graph layer and reconnects results before downstream judge calls.

### Inspect available workflows

```powershell
uv run python -m cli list-workflows
```

The registered workflow names and modes are:

| Workflow | Valid mode |
|---|---|
| `framework_1_parallel_no_evidence` | `bool` |
| `framework_2_parallel_with_evidence` | `evidence` |
| `framework_3_hierarchical_no_reviewer` | `bool` or `reasoned` |
| `framework_4_hierarchical_with_reviewer` | `bool` or `reasoned` |

### Schedule a new evaluation run

This example supplies every execution, retry, and early-stop control commonly
needed for a large run:

```powershell
uv run python -m cli evaluate `
  --workflow framework_1_parallel_no_evidence `
  --sample-path dataset/suicidality_full_dataset_posneg.csv `
  --artifact-base-dir logs/runs `
  --run-id framework1_full_posneg `
  --model gpt-5.6-luna `
  --temperature 0 `
  --service-tier standard `
  --mode bool `
  --case-workers 4 `
  --max-in-flight-llm 20 `
  --checkpoint-every 10 `
  --stop-after-tokens 1000000 `
  --max-completion-tokens 2000 `
  --max-retries 3 `
  --error-stop-failed-llm-attempts 200 `
  --error-stop-failed-records 10 `
  --error-stop-rate 0.10 `
  --error-stop-min-records 100
```

`--dataset-audit-path` is optional. For a custom dataset without an audit file,
omit it. The run directory is created as:

```text
<artifact-base-dir>/<timestamp>_<run-id>/
```

The input CSV must contain unique `ID` values and the required `text` and
`sentiment` columns. Legacy fixtures may use unique `source_row` values instead
of `ID`.

### Complete `evaluate` option reference

| Option | Default | Meaning |
|---|---|---|
| `--workflow` | `framework_1_parallel_no_evidence` | Registered workflow to execute. |
| `--workflow-config PATH` | None | JSON configuration for workflow/model defaults and supported role overrides. |
| `--sample-path PATH` | `dataset/processed/dataset1_balanced_500_seed42.csv` | Frozen CSV evaluated by a new run. |
| `--dataset-audit-path PATH` | Paired audit for the default sample; otherwise none | Audit JSON copied into new-run artifacts. |
| `--artifact-base-dir PATH` | `logs/runs` | Parent directory for a new timestamped run. |
| `--run-id TEXT` | Generated suffix | Human-readable suffix for a new run directory. |
| `--resume-run-dir PATH` | None | Resume the exact existing directory instead of creating a run. |
| `--fake-llm` | Off | Use deterministic local responses; no API key or provider calls. |
| `--model NAME` | Workflow JSON or `gpt-4o` | Default model for every role without a role override. |
| `--temperature FLOAT` | `OPENAI_TEMPERATURE` or `0` | Default model temperature. |
| `--max-completion-tokens N` | `MAX_COMPLETION_TOKENS` or `1024` | Positive generated-token ceiling for each LLM attempt. |
| `--max-retries N` | `MAX_RETRIES` or `3` | Retries after the initial failed or malformed attempt; may be zero. |
| `--mode NAME` | Workflow-specific | Intermediate output/prompt mode from the workflow table above. |
| `--service-tier NAME` | `OPENAI_SERVICE_TIER` or `standard` | Direct OpenAI processing tier: `standard` or `flex`. Standard maps to API `default`; Flex requires a direct OpenAI endpoint. |
| `--limit N` | None | Evaluate only the first positive `N` rows for a smoke test. Not allowed with resume. |
| `--case-workers N` | `EVAL_CASE_WORKERS` or `4` | Dataset records whose workflows may execute concurrently. |
| `--max-in-flight-llm N` | `EVAL_MAX_IN_FLIGHT_LLM` or `20` | Run-wide ceiling across every worker, role, retry, and role-specific model. |
| `--checkpoint-every N` | `EVAL_CHECKPOINT_EVERY` or `250` | Completed cases between readable `manifest.json` progress refreshes. SQLite still commits every completed case. |
| `--stop-after-tokens N` | None | Stop scheduling new cases after at least `N` prompt plus completion tokens finish in the current invocation. The counter resets to zero on every resume. |
| `--error-stop-failed-llm-attempts N` | Environment or `10` | Stop when cumulative failed LLM attempts reach `N`, including recovered retry attempts. |
| `--error-stop-failed-records N` | Environment or `3` | Stop when records with no valid final label reach `N`. |
| `--error-stop-rate FLOAT` | Environment or `0.5` | Stop when `failed_records / completed_records` reaches this value. |
| `--error-stop-min-records N` | Environment or `10` | Do not apply the failure-rate guard until this many records have completed. |

The absolute-count guardrails and failure counters are cumulative across
resumes. Select thresholds appropriate for dataset size and provider behavior;
for example, resuming a run that already contains 10 failed attempts requires
`--error-stop-failed-llm-attempts` to be greater than 10.

The process returns exit code `2` for a controlled `stopped_early` run, `0` for
normal completion, and a nonzero Python error exit for an unhandled failure.

### Configure one default model

Use CLI options when all roles should share one model:

```powershell
uv run python -m cli evaluate `
  --workflow framework_2_parallel_with_evidence `
  --sample-path dataset/processed/dataset1_balanced_500_seed42.csv `
  --model gpt-4o-mini `
  --temperature 0 `
  --max-completion-tokens 2000 `
  --max-retries 3 `
  --mode evidence
```

### Configure different models by agent role

Frameworks 1 and 2 accept an `agents` object in `--workflow-config`. Common
top-level keys are `model`, `temperature`, `max_completion_tokens`,
`max_retries`, `mode`, `service_tier`, and `fake_llm`. Each role override accepts `agent_name`,
`model`, `temperature`, and `max_completion_tokens`.

Example `configs/framework_1_mixed_models.json`:

```json
{
  "model": "gpt-4o-mini",
  "temperature": 0.0,
  "max_completion_tokens": 1000,
  "max_retries": 3,
  "mode": "bool",
  "service_tier": "standard",
  "fake_llm": false,
  "agents": {
    "Indicator_specialist": {
      "model": "gpt-4o-mini",
      "temperature": 0.0
    },
    "judge": {
      "agent_name": "Final Judge",
      "model": "gpt-5.6-luna",
      "temperature": 0.0,
      "max_completion_tokens": 2000
    }
  }
}
```

Run it with:

```powershell
uv run python -m cli evaluate `
  --workflow framework_1_parallel_no_evidence `
  --workflow-config configs/framework_1_mixed_models.json `
  --sample-path dataset/processed/dataset1_balanced_500_seed42.csv
```

Valid framework-1 roles are:

```text
Indicator_specialist
Ideation_specialist
Behavior_specialist
Attempt_specialist
Unsupportive_specialist
judge
```

Valid framework-2 roles replace `_specialist` with `_collector`:

```text
Indicator_collector
Ideation_collector
Behavior_collector
Attempt_collector
Unsupportive_collector
judge
```

Frameworks 3 and 4 currently accept only the top-level model settings; they do
not accept the `agents` object. Explicit common CLI options override the matching
top-level JSON settings, while role-specific entries remain role-specific.

### Smoke-test a workflow without provider calls

```powershell
uv run python -m cli evaluate `
  --workflow framework_4_hierarchical_with_reviewer `
  --fake-llm `
  --limit 4 `
  --run-id framework4_smoke
```

### Resume an existing evaluation run

Use the exact generated run directory, not the original `--run-id`:

```powershell
uv run python -m cli evaluate `
  --workflow framework_1_parallel_no_evidence `
  --resume-run-dir "logs/runs/20260724_042628_framework1_full_posneg_gpt_5.6_luna_concurrent" `
  --model gpt-5.6-luna `
  --temperature 0 `
  --mode bool `
  --max-completion-tokens 2000 `
  --max-retries 3 `
  --case-workers 4 `
  --max-in-flight-llm 20 `
  --checkpoint-every 10 `
  --stop-after-tokens 1000000 `
  --error-stop-failed-llm-attempts 200
```

Do not pass `--sample-path`, `--artifact-base-dir`, `--run-id`, or `--limit`
when resuming. Resume reads the run's frozen `sampled_dataset.csv` and
`progress.sqlite`, resets abandoned `running` rows to `pending`, and schedules
only pending IDs.

These behavior-defining settings must match the original run:

- workflow name and complete workflow-config JSON;
- requested and resolved model names, including every role override;
- temperature;
- mode;
- service tier;
- maximum completion tokens;
- retry count;
- fake versus real model selection.

These execution controls may be changed on resume:

- case workers;
- maximum in-flight LLM requests;
- checkpoint frequency;
- stop-after token threshold;
- early-stop thresholds.

`--stop-after-tokens` is intentionally session-scoped. A resumed invocation
starts its counter at zero and excludes all token usage stored by earlier
invocations of the same run. The cumulative token total in `metrics.json`
continues to cover every completed request in the run, while
`token_stop.current_session_total_tokens` reports only the latest invocation.

If resume reports `workflow/model configuration does not match`, compare the
original `manifest.json` workflow identity and `run_config.json` with the resume
command. Also ensure `OPENAI_BASE_URL` still resolves the requested model name
the same way.

### Concurrency, checkpoints, and resume behavior

The evaluator uses one coordinator and a configurable pool of case workers. Workers
receive records selected by the coordinator; they do not read the CSV or SQLite
database. Before submitting work, the coordinator atomically changes that unique
dataset ID from `pending` to `running`. Only the coordinator commits results.

The default settings are four concurrent cases and at most 20 simultaneous LLM
requests across every worker, agent role, and model override:

```powershell
uv run python -m cli evaluate `
  --workflow framework_1_parallel_no_evidence `
  --sample-path dataset/my_30000_rows.csv `
  --case-workers 4 `
  --max-in-flight-llm 20 `
  --checkpoint-every 250
```

There are no fixed shards. Faster workers take the next pending ID dynamically.
`progress.sqlite` is the authoritative per-run status and result store; each
completed case is committed transactionally. `manifest.json` records the frozen
dataset fingerprint, workflow/model fingerprint, execution limits, progress, and
coverage. A human-readable manifest snapshot is refreshed every
`--checkpoint-every` completions.

The token threshold is checked after a case finishes and its calls report token
usage. Cases that were already in flight are allowed to finish, so the final
current-invocation total can exceed `--stop-after-tokens`, especially with
multiple case workers. Once the threshold is observed, no new cases are
scheduled.

Completed row payloads are committed to `progress.sqlite`, but derived files
such as `predictions.csv`, `llm_calls.jsonl`, metrics, and compiled reports are
normally rebuilt only when an evaluator invocation finishes normally or reaches
a controlled early-stop threshold. Ctrl+C preserves committed SQLite results
but bypasses that final artifact pass.

Export a consistent report snapshot at any time, including while an evaluator
is running:

```powershell
uv run python -m cli snapshot-results `
  --run-dir "logs/runs/<run-folder>"
```

Each invocation creates a timestamped directory under
`logs/runs/<run-folder>/snapshots/`. The snapshot includes every row committed
as `completed` when its read-only SQLite transaction begins. Currently running
rows are excluded. The source run, its progress database, and its existing
artifacts are not modified.

Runtime reports keep both `agent_execution_duration_s` (the cumulative sum of
per-case worker durations) and `real_execution_duration_s` (the concurrency-aware
span from the first agent-call start to the last agent-call finish). Resumed
execution sessions are added without counting time while the run was stopped.

Resume always loads that run's `sampled_dataset.csv`, resets abandoned `running`
rows to `pending`, and schedules only pending IDs. The database does not store an
`input_position`: the dataset's unique `ID` (or legacy `source_row`) is the sole
case key. Final artifact order is recovered by iterating the frozen dataset and
joining completed results by that ID.

A run is marked `completed` only when completed output IDs exactly match the
input IDs, with no missing, unexpected, or duplicate IDs. SQLite prevents two
local workers from committing or scheduling the same ID. After a machine/process
crash, a provider request that completed before its local transaction was
committed may be retried; the local result store is exactly-once, but external
provider billing cannot be guaranteed exactly-once without provider idempotency.

### Dataset and reporting commands

The dataset-oriented commands expose these common options:

| Option | Default | Meaning |
|---|---|---|
| `--data-path PATH` | `dataset/anonymizing_suicidal_dataset.csv` | Active anonymized source CSV. |
| `--seed N` | `42` | Reproducible balanced-sampling seed. |
| `--sample-size N` | `500` | Balanced sample size; must be divisible by four. |

Audit the source dataset and write audit artifacts:

```powershell
uv run python -m cli audit `
  --data-path dataset/anonymizing_suicidal_dataset.csv `
  --seed 42 `
  --sample-size 500
```

`audit` uses `--data-path`; `--seed` and `--sample-size` are accepted through
the shared parser but do not change an audit-only result.

Create a temporary balanced sample inside a generated run directory:

```powershell
uv run python -m cli sample `
  --data-path dataset/anonymizing_suicidal_dataset.csv `
  --seed 42 `
  --sample-size 500
```

Create persistent cleaned and balanced files. `prepare-data` adds
`--output-dir`, whose default is `dataset/processed`:

```powershell
uv run python -m cli prepare-data `
  --data-path dataset/anonymizing_suicidal_dataset.csv `
  --seed 42 `
  --sample-size 500 `
  --output-dir dataset/processed
```

Regenerate compiled CSV and Markdown reports without rerunning agents:

```powershell
uv run python -m cli compile-results `
  --run-dir logs/runs/<run-folder>
```

`compile-results` requires only `--run-dir`.

Unlike `compile-results`, `snapshot-results` reads the authoritative
`progress.sqlite` and regenerates partial JSONL, prediction, metric,
visualization, and compiled report artifacts. Snapshot metrics use
`evaluation_status: "snapshot"` and report coverage against the full frozen
dataset.

### Legacy framework-1 `run` shortcut

`run` audits and balances the active dataset in memory and executes only
`framework_1_parallel_no_evidence`. Prefer `evaluate` for persistent datasets,
workflow selection, custom artifact locations, concurrency tuning, and resume.

```powershell
uv run python -m cli run `
  --data-path dataset/anonymizing_suicidal_dataset.csv `
  --seed 42 `
  --sample-size 500 `
  --model gpt-4o-mini `
  --temperature 0 `
  --max-completion-tokens 1024 `
  --max-retries 3 `
  --mode bool
```

The `run`-only options are `--fake-llm`, `--model`, `--temperature`,
`--max-completion-tokens`, `--max-retries`, and `--mode bool`, in
addition to the three common dataset options. It does not expose
`--workflow`, resume, artifact-directory, concurrency, checkpoint, or early-stop
arguments.

## Setup

From the repository root:

```console
cd multi_agent_frameworks
uv sync
```

Copy `.env.example` to `.env`, then edit `.env` before using a real
OpenAI-compatible endpoint. A local fake-model smoke run requires no API key:

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
