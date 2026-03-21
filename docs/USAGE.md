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

### 2.2 Sequential model suite runner

File:

- [src/example/suicide_detection/run_model_suite.py](../src/example/suicide_detection/run_model_suite.py)

Use for:

- testing multiple models one after another
- stopping the sequence when a model is not error-free

---

## 3. Standard Evaluation Usage

From the EvalRing root:

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 100 --max-workers 10
```

### Supported flags

- `--n-samples`: limit dataset size for the run
- `--max-workers`: number of parallel worker threads
- `--max-retries`: retries per sample inside evaluator execution
- `--seed`: recorded in metadata for reproducibility
- `--retry-failed`: rerun only error cases from an existing run
- `--meta-path`: specific `Meta.json` to use with `--retry-failed`

### Example

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 32 --max-retries 3 --seed 42
```

---

## 4. Output Files

Each fresh run creates a directory under:

- [src/example/suicide_detection/.EvalRing](../src/example/suicide_detection/.EvalRing)

Example layout:

```text
run_YYYYMMDD_HHMMSS/
├── result.md
├── all_cases.txt
├── all_cases.csv
├── failed_cases.txt
└── Meta.json
```

### `result.md`

Human-readable run summary:

- model used
- sample count
- aggregate accuracy / precision / recall / F1
- per-class metrics
- confusion matrix
- per-sample summary table

### `all_cases.csv`

Primary machine-readable table for downstream analysis.

Use this file for:

- pandas analysis
- error audits
- retry selection
- model comparison

### `failed_cases.txt`

Convenience text file listing failed predictions that remain after the run.

### `Meta.json`

Run manifest and reproducibility record.

Contains:

- `run_config`
- `model_config`
- `dataset_config`
- `aggregate_metrics`
- `per_class_metrics`
- `confusion_matrix`
- `retry_history`

---

## 5. Retry Failed Cases In Place

This is the preferred workflow when a small number of samples failed due to provider/API instability.

### What it does

`--retry-failed`:

1. loads an existing run’s `Meta.json`
2. reuses that run’s configuration
3. reads original `all_cases.csv`
4. selects only rows where:
   - prediction is `Error`, or
   - `error` is non-empty
5. reruns only those rows
6. updates the original run directory in place

### Why it exists

Without retry mode, one transient provider error can force a full rerun of 100 or 1000 samples.

Retry mode keeps the original run as the canonical record while repairing only the broken rows.

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

### Retry audit trail

Each retry appends to `retry_history` in `Meta.json` with fields such as:

- `retried_cases`
- `resolved_errors`
- `remaining_error_cases`
- `source_meta_path`

---

## 6. Sequential Multi-Model Suite Usage

Runner file:

- [src/example/suicide_detection/run_model_suite.py](../src/example/suicide_detection/run_model_suite.py)

Current embedded model list:

```python
model_names = [
    "openai/gpt-5-mini",
    "openai/gpt-5-mini:provider-azure",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3-flash-preview",
    "x-ai/grok-3-mini",
]
```

### Current test behavior

The script is currently configured to run:

- `n_samples = 2`
- `max_workers = 32`

### Run it

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/run_model_suite.py
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

## 7. Programmatic Usage

### Minimal classification example

```python
from EvalRing.dataset import CSVDataset
from EvalRing.evaluator import ClassificationEvaluator
from llm_agent import OpenAISuicideDetectionAgent


dataset = CSVDataset(name="demo")
dataset.load_data(
    source="data.csv",
    text_field="text",
    label_field="label",
    id_field="id",
)

agent = OpenAISuicideDetectionAgent()
agent.initialize()

evaluator = ClassificationEvaluator()
result = evaluator.evaluate(
    agent=agent,
    dataset=dataset,
    task_name="classification",
    version="v1",
    max_workers=8,
    max_retries=3,
)

print(result.metrics.get_metric("accuracy"))
```

### DataFrame retry-subset pattern

For partial reruns, the current system uses `DataFrameDataset` with an explicit row ID column.

This pattern is useful whenever:

- dataset IDs are not unique
- only a subset of rows should be re-evaluated
- row-position stability matters more than source ID

---

## 8. Research Workflow Recommendation

For publishable experiments, use this cycle:

1. run the benchmark with a fixed sample count and model
2. inspect `all_cases.csv` and `failed_cases.txt`
3. if any rows are provider errors, run `--retry-failed`
4. only after retry, treat the run folder as final
5. archive or compare final `Meta.json` and `all_cases.csv`

Suggested artifacts to keep under versioned experiment storage:

- final `Meta.json`
- final `all_cases.csv`
- final `result.md`
- prompt version / model version notes

---

## 9. Known Operational Caveats

### Duplicate sample IDs in the source dataset

The current suicide detection example uses `users` as `id_field`, which is not guaranteed unique.

Implication:

- do not assume `sample_id` is a primary key across the dataset
- use CSV row position for repair workflows when necessary

### Root README mismatch

The root [README.md](../README.md) does not fully describe the current code path.

For day-to-day operation, prefer:

- [docs/DESIGN_SPEC.md](DESIGN_SPEC.md)
- [docs/USAGE.md](USAGE.md)

---

## 10. Suggested Commands

### Fast sanity run

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 2 --max-workers 2
```

### Medium run

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 100 --max-workers 10
```

### Retry broken rows only

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

### Multi-model smoke suite

```powershell
.\.venv_evalring\Scripts\python.exe src/example/suicide_detection/run_model_suite.py
```
