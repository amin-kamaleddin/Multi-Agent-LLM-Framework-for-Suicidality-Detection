# EvalRing Usage Guide

This guide documents how to operate the current EvalRing system in this repository.

## 1. Environment

Primary working directory:

- [EvalRing](../)

Recommended environment variables in [EvalRing/.env](../.env):

- `OPEN_ROUTER_KEY`
- `OPEN_ROUTER_MODEL` (optional)
- `OPENAI_API_KEY` (fallback)
- `OPENAI_MODEL` (optional)

Current provider behavior:

- if `OPEN_ROUTER_KEY` exists, the system defaults to OpenRouter
- if not, it falls back to OpenAI

---

## 2. Main Entry Points

### 2.1 Standard evaluation runner

File:

- [src/example/suicide_detection/evaluate_rsd15k_main.py](../src/example/suicide_detection/evaluate_rsd15k_main.py)

Use for:

- one-model benchmark runs
- report generation
- in-place retry of failed error rows
- switching prediction modes

### 2.2 Sequential model suite runner

File:

- [src/example/suicide_detection/run_model_suite.py](../src/example/suicide_detection/run_model_suite.py)

Use for:

- testing multiple models one after another
- stopping the sequence when a model run is not error-free

### 2.3 Dataset translator for user-round splits

File:

- [src/example/suicide_detection/data_translator.py](../src/example/suicide_detection/data_translator.py)

Use for:

- creating `*_simple.csv` (users appearing exactly once)
- creating `*_multi_round.csv` (users appearing more than once)
- preserving original source dataset unchanged

Command:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/data_translator.py
```

---

## 3. Prediction Modes

The suicide detection runner supports multiple modes via `--agent-mode`.

### 3.1 `single-class`

- Model returns one label directly.
- Expected labels: `Indicator`, `Ideation`, `Behavior`, `Attempt`.

### 3.2 `multi-class-chance`

- Model returns structured JSON probabilities for all four classes.
- Example:

```json
{"ideation": 0.8, "behavior": 0.1, "indicator": 0.05, "attempt": 0.05}
```

- Evaluator resolves top class automatically for accuracy/F1 metrics.

### 3.3 `base-vs-rest-binary`

- You declare one class as `--base-class`.
- The agent runs a knock-out tournament. It evaluates `base vs target_1`, taking the winner, and evaluates `winner vs target_2`, and so on.
- Output is a sequence of single-word knockout decisions until a final winner is determined.
- Evaluator returns the final tournament surviving class natively.

### 3.4 `per-class-score`

- Model returns independent scores from 1 to 10 for each of the four classes based on a rubric.
- Example:

```json
{"ideation": 8, "behavior": 2, "indicator": 1, "attempt": 1}
```

- Evaluator resolves the top class automatically by picking the highest-scored class.

---

## 4. Standard Evaluation Usage

From the EvalRing root:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 100 --max-workers 10
```

### Supported flags

- `--n-samples`: limit dataset size for the run
- `--max-workers`: number of parallel worker threads
- `--max-retries`: retries per sample inside evaluator execution
- `--seed`: recorded in metadata for reproducibility
- `--agent-mode`: `single-class`, `multi-class-chance`, `base-vs-rest-binary`, `multi-agent-host`, or `per-class-score`
- `--base-class`: used by `base-vs-rest-binary`; one of `Indicator`, `Ideation`, `Behavior`, `Attempt`
- `--retry-failed`: rerun only error cases from an existing run
- `--meta-path`: specific `Meta.json` to use with `--retry-failed`

---

## 4.1 Sample IDs (CRITICAL)

EvalRing assumes every dataset row maps to exactly one evaluation request.

- **Use `ID` (per-row question/message ID) as the dataset `id_field`.**
- **Do NOT use `users` as `id_field`.** `users` is an author identifier and is not unique.

Why this matters:

- Resume logic and report merging key off `sample_id`.
- Non-unique IDs can cause incorrect resume/merge behavior (e.g., skipping remaining rows or overwriting/merging unrelated rows).

RSD_15K in this repo is maintained with an explicit leading `ID` column:

- [src/example/suicide_detection/data/rsd_15k.csv](../src/example/suicide_detection/data/rsd_15k.csv)

---

## 4.2 Sampling Policy (Deterministic)

When `--n-samples N` is set, the runners take the first `N` rows in file order (equivalent to `head(N)`):

- Sample IDs should be `0..N-1` for RSD_15K
- After sampling, the runner verifies there are no duplicate sample IDs (set size must equal `N`)
- Even with parallel workers, per-sample results are emitted in deterministic order (sorted by `sample_id`) so runs do not *look* randomly sampled.

### Examples

Single-class:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode single-class
```

Multi-class chance:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode multi-class-chance
```

Base-vs-rest with Indicator as base:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Indicator
```

Base-vs-rest with Attempt as base:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Attempt
```

---

## 5. Output Files

Each fresh run creates a directory under:

- [src/example/suicide_detection/_EvalRing](../src/example/suicide_detection/_EvalRing)

Example layout:

```text
run_YYYYMMDD_HHMMSS/
|-- result.md
|-- all_cases.txt
|-- all_cases.csv
|-- failed_cases.txt
`-- Meta.json
```

### 5.1 `result.md`

Human-readable run summary:

- model used
- mode used
- base class (when relevant)
- sample count
- aggregate accuracy / precision / recall / F1
- per-class metrics
- confusion matrix
- per-sample summary table

### 5.2 `all_cases.csv`

Primary machine-readable table for downstream analysis.

Key columns include:

- `sample_id`
- `ground_truth`
- `prediction`
- `correct`
- `prediction_confidence`
- `class_scores`
- `ttft`
- `tps`
- `total_time`
- `generation_time`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `error`
- `text`

### 5.3 `failed_cases.txt`

Convenience text file listing failed predictions that remain after the run.

### 5.4 `Meta.json`

Run manifest and reproducibility record.

Contains:

- `run_config`
- `model_config` (includes `agent_mode`, `base_class`)
- `dataset_config`
- `aggregate_metrics`
- `per_class_metrics`
- `confusion_matrix`
- `retry_history`

---

## 6. Retry Failed Cases In Place

This is the preferred workflow when a small number of samples failed due to provider/API instability.

### What it does

`--retry-failed`:

1. loads an existing run's `Meta.json`
2. reuses that run's configuration
3. reads original `all_cases.csv`
4. selects only rows where:
   - prediction is `Error`, or
   - `error` is non-empty
5. reruns only those rows
6. updates the original run directory in place

### Retry latest run

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

### Retry a specific run

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed --meta-path src/example/suicide_detection/_EvalRing/run_20260313_185114/Meta.json
```

### Files updated during retry

In the original run directory:

- `result.md`
- `all_cases.txt`
- `all_cases.csv`
- `failed_cases.txt`
- `Meta.json`

---

## Shared Agent-Workflow Evaluator CLI

The architecture-neutral evaluator under `multi_agent_framework1` has a registry-backed CLI. Dataset selection, audit selection, artifact location, run IDs, limits, guardrails, metrics, and report compilation are common to every registered workflow. Each workflow runner is responsible only for constructing its architecture and interpreting its workflow-specific JSON configuration.

Run these commands from `multi_agent_framework1`.

### Discover valid workflow names

```powershell
uv run python -m suicidal_multi_agent.cli list-workflows
```

The value passed to `--workflow` must be one of the exact lowercase names printed by this command. These are predetermined registry keys, not Python paths or arbitrary class names. The current registry contains:

```text
framework1
```

`framework1` is the default when `--workflow` is omitted. An unknown name is rejected by argument parsing. Registering another workflow, such as `hierarchical`, automatically adds that name to `list-workflows` and the accepted `--workflow` choices.

### Select workflow, dataset, configuration, and log location

```powershell
uv run python -m suicidal_multi_agent.cli evaluate `
  --workflow framework1 `
  --workflow-config configs/framework1.example.json `
  --sample-path dataset/positive-negative-final-datasets/dataset1_positive_balanced_100_seed42.csv `
  --dataset-audit-path path/to/paired-audit.json `
  --artifact-base-dir logs/experiments `
  --run-id framework1-positive-100
```

The run is written under:

```text
<artifact-base-dir>/YYYYMMDD_HHMMSS_<run-id>/
```

The evaluator automatically writes `compiled_results.csv` and `compiled_results.md` into that directory. To regenerate them later:

```powershell
uv run python -m suicidal_multi_agent.cli compile-results --run-dir logs/experiments/<run-folder>
```

### Workflow configuration

`--workflow-config` accepts a path to a JSON object. Start from `configs/framework1.example.json`. API keys do not belong in this file; keep credentials in `.env`.

Framework1 accepts this shape:

```json
{
  "model": "gpt-4o",
  "temperature": 0.0,
  "max_completion_tokens": 256,
  "max_retries": 2,
  "mode": "reasoned",
  "fake_llm": false,
  "agents": {
    "Indicator_specialist": {
      "model": "gpt-4o-mini",
      "temperature": 0.0,
      "max_completion_tokens": 128
    },
    "judge": {
      "model": "gpt-4o",
      "temperature": 0.0,
      "max_completion_tokens": 256
    }
  }
}
```

Valid framework1 agent keys are `Indicator_specialist`, `Ideation_specialist`, `Behavior_specialist`, `Attempt_specialist`, and `judge`. The `agents` object is optional; omitted roles inherit the top-level model configuration.

For common model settings, precedence is:

```text
explicit CLI option > workflow-config JSON > environment/default
```

Dataset, audit, output directory, run ID, limits, and guardrails remain common CLI arguments rather than workflow-specific JSON fields.

### Registry location

The registry and built-in framework1 registration are in:

- [multi_agent_framework1/src/suicidal_multi_agent/workflow/registry.py](../multi_agent_framework1/src/suicidal_multi_agent/workflow/registry.py)

The registry maps a stable name to a `WorkflowRegistration` containing:

- `name`: exact lowercase value accepted by `--workflow`;
- `description`: displayed by `list-workflows`;
- `runner`: callable accepting `WorkflowRunRequest`;
- `config_keys`: workflow-specific JSON keys shown to users.

### Add a workflow to the registry

1. Create an adapter implementing the shared `run_case(sample_id=..., text=...)` contract.
2. Add a runner that accepts `WorkflowRunRequest`, builds the workflow from `request.workflow_config`, and calls `run_evaluation` with the common sample, configuration, run ID, and artifact directory.
3. Register the runner with a unique lowercase name in `workflow/registry.py`.
4. Add tests for registration, configuration validation, and one fake-model artifact run.
5. Confirm the name appears under `list-workflows` before using it with `--workflow`.

Minimal registration:

```python
from ..evaluation.evaluator import run_evaluation
from .hierarchical import HierarchicalAdapter


def _run_hierarchical(request: WorkflowRunRequest) -> EvaluationResult:
    workflow = HierarchicalAdapter.from_config(request.workflow_config)
    return run_evaluation(
        sample_df=request.sample_df,
        workflow=workflow,
        config=request.config,
        run_id=request.run_id,
        artifact_base_dir=request.artifact_base_dir,
    )


WORKFLOW_REGISTRY.register(
    WorkflowRegistration(
        name="hierarchical",
        description="Router and variable specialist hierarchy.",
        runner=_run_hierarchical,
        config_keys=("router", "specialists", "judge"),
    )
)
```

After registration, run it without changing the evaluator:

```powershell
uv run python -m suicidal_multi_agent.cli evaluate `
  --workflow hierarchical `
  --workflow-config configs/hierarchical.json `
  --sample-path dataset/combined_200.csv `
  --artifact-base-dir logs/experiments
```

The complete input/output and telemetry adapter contract is documented in [ADAPTING_WORKFLOWS.md](../multi_agent_framework1/src/suicidal_multi_agent/evaluation/ADAPTING_WORKFLOWS.md).

---

## 7. Sequential Multi-Model Suite Usage

Runner utility:

- [src/EvalRing/utils/suite_runner.py](../src/EvalRing/utils/suite_runner.py)

### CLI Usage

```powershell
.\.venv_EvalRing\Scripts\python.exe src/EvalRing/utils/suite_runner.py --eval-script src/example/suicide_detection/evaluate_rsd15k_main.py --models-file src/example/suicide_detection/model_list.json --n-samples 100 --max-workers 10
```

### Automatic Visualizations

If the suite completes, the runner automatically delegates the generated `suite_report_*.json` into the visualization utility which produces comparison graphs and a composite Markdown presentation inside a bounded `visuals/` directory.

---

## 8. LLM Judge Evaluation

Runner file:

- [src/example/suicide_detection/evaluate_rsd15k_llm_judge.py](../src/example/suicide_detection/evaluate_rsd15k_llm_judge.py)

### Usage

Use the LLM-as-a-Judge API to score another language model's reasoning and classification alignment against the ground truth. This mode requires configuring rubrics (like "reasoning_quality", "classification_correctness", etc.) to dictate how a master LLM judge should grade responses.

Run the judge evaluator:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_llm_judge.py
```

---

## 9. Reusable Core Behavior for Future Tasks

Core module:

- [src/EvalRing/agent/classification.py](../src/EvalRing/agent/classification.py)

Reusable utilities include:

- `resolve_classification_prediction`
- `normalize_probability_distribution`
- `aggregate_base_vs_rest_probabilities`

Implication for future domains:

- You can return either label strings or class-score mappings from agents.
- Evaluator-level metric calculation can stay unchanged.
- Base-vs-rest strategies can be adapted to emotion/intent/categorization tasks with minimal code changes.

---

## 10. Suggested Commands

Fast sanity run:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 2 --max-workers 2 --agent-mode single-class
```

Structured chance run:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode multi-class-chance
```

Base-vs-rest run:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode base-vs-rest-binary --base-class Indicator
```

Retry broken rows only:

```powershell
.\.venv_EvalRing\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```
