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
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/data_translator.py
```

---

## 3. Prediction Modes

The suicide detection runner supports three modes via `--agent-mode`.

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
- The agent runs three binary calls: `base vs target` for each non-base class.
- Pairwise probabilities are aggregated into one 4-class distribution by core utility logic.
- Evaluator resolves top class automatically for standard classification metrics.

---

## 4. Standard Evaluation Usage

From the EvalRing root:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 100 --max-workers 10
```

### Supported flags

- `--n-samples`: limit dataset size for the run
- `--max-workers`: number of parallel worker threads
- `--max-retries`: retries per sample inside evaluator execution
- `--seed`: recorded in metadata for reproducibility
- `--agent-mode`: `single-class`, `multi-class-chance`, or `base-vs-rest-binary`
- `--base-class`: used by `base-vs-rest-binary`; one of `Indicator`, `Ideation`, `Behavior`, `Attempt`
- `--retry-failed`: rerun only error cases from an existing run
- `--meta-path`: specific `Meta.json` to use with `--retry-failed`

### Examples

Single-class:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode single-class
```

Multi-class chance:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode multi-class-chance
```

Base-vs-rest with Indicator as base:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Indicator
```

Base-vs-rest with Attempt as base:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Attempt
```

---

## 5. Output Files

Each fresh run creates a directory under:

- [src/example/suicide_detection/.EvalRing](../src/example/suicide_detection/.EvalRing)

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
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

### Retry a specific run

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed --meta-path src/example/suicide_detection/.EvalRing/run_20260313_185114/Meta.json
```

### Files updated during retry

In the original run directory:

- `result.md`
- `all_cases.txt`
- `all_cases.csv`
- `failed_cases.txt`
- `Meta.json`

---

## 7. Sequential Multi-Model Suite Usage

Runner file:

- [src/example/suicide_detection/run_model_suite.py](../src/example/suicide_detection/run_model_suite.py)

### Programmatic call

```python
from example.suicide_detection.run_model_suite import run_suite

run_suite(
    n_samples=100,
    max_workers=10,
    seed=42,
    agent_mode="base-vs-rest-binary",
    base_class="Indicator",
)
```

### Stop conditions

The suite stops before running the next model if the current model has any of the following:

- process exit failure
- missing run directory
- `execution_failures` recorded in `Meta.json`
- per-sample prediction errors in `all_cases.csv`

### Suite report

If all models succeed, the suite writes a JSON summary under:

- [src/example/suicide_detection/.EvalRing](../src/example/suicide_detection/.EvalRing)

with file name pattern:

- `suite_report_YYYYMMDD_HHMMSS.json`

---

## 8. Reusable Core Behavior for Future Tasks

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

## 9. Suggested Commands

Fast sanity run:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 2 --max-workers 2 --agent-mode single-class
```

Structured chance run:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode multi-class-chance
```

Base-vs-rest run:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode base-vs-rest-binary --base-class Indicator
```

Retry broken rows only:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```
