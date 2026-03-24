# Suicide Detection Sample Guide

This document is a complete usage guide for the suicide detection sample under EvalRing.

It covers:

- environment setup
- single-run evaluation
- all three prediction modes
- retry-in-place workflow
- model-suite workflow
- output artifacts
- adaptation guidance for future tasks

## 1. Scope and Files

Core package and sample files:

- [src/EvalRing](../../../src/EvalRing)
- [src/example/suicide_detection/llm_agent.py](../../../src/example/suicide_detection/llm_agent.py)
- [src/example/suicide_detection/evaluate_rsd15k_main.py](../../../src/example/suicide_detection/evaluate_rsd15k_main.py)
- [src/example/suicide_detection/run_model_suite.py](../../../src/example/suicide_detection/run_model_suite.py)
- [src/example/suicide_detection/data/rsd_15k.csv](../../../src/example/suicide_detection/data/rsd_15k.csv)

One can access original dataset from: https://drive.google.com/file/d/1DrWVF28hEj70x3Yxtfonk3pzS0hPOQBT/view?usp=sharing

Alternative datasets: https://drive.google.com/drive/folders/1c8wL7woT-td9SFVr78wqiIYMCjXHRsoh?usp=drive_link

Reusable classification helpers used by this sample:

- [src/EvalRing/agent/classification.py](../../../src/EvalRing/agent/classification.py)

## 2. Environment Setup

From repository root:

```powershell
Set-Location "EvalRing"
```

Activate virtual environment (example path):

```powershell
.\.venv_evalring\Scripts\Activate.ps1
```

Required environment key (at least one):

- `OPEN_ROUTER_KEY`
- `OPENAI_API_KEY`

Optional model override keys:

- `OPEN_ROUTER_MODEL`
- `OPENAI_MODEL`

The runner will prefer OpenRouter when `OPEN_ROUTER_KEY` is available.

## 3. Standard Evaluation Command

Main runner:

- [src/example/suicide_detection/evaluate_rsd15k_main.py](../../../src/example/suicide_detection/evaluate_rsd15k_main.py)

Basic command:

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 100 --max-workers 10
```

Supported arguments:

- `--n-samples`
- `--max-workers`
- `--max-retries`
- `--seed`
- `--agent-mode`
- `--base-class`
- `--retry-failed`
- `--meta-path`

## 4. Agent Modes

### 4.1 single-class

Model returns one label.

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode single-class
```

### 4.2 multi-class-chance

Model returns one JSON object with probabilities for all four classes.

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode multi-class-chance
```

Expected JSON style:

```json
{"ideation": 0.80, "behavior": 0.10, "indicator": 0.05, "attempt": 0.05}
```

### 4.3 base-vs-rest-binary

One class is declared as base, and the agent runs three binary decisions: `base vs each non-base class`.

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Indicator
```

Other base-class examples:

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Attempt
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Behavior
```

Behind the scenes:

1. agent runs one binary call per non-base class
2. pairwise probabilities are collected
3. core aggregation converts pairwise probabilities into one multi-class probability distribution
4. evaluator resolves top class for standard metrics

## 5. Retry Failed Rows In Place

Retry mode avoids rerunning all rows when only a few API/provider calls failed.

Retry latest run:

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

Retry specific run:

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed --meta-path src/example/suicide_detection/.EvalRing/run_YYYYMMDD_HHMMSS/Meta.json
```

Behavior summary:

1. load previous `Meta.json`
2. identify rows with `Error` prediction or non-empty `error`
3. rerun only those rows
4. patch the original run folder files in place

## 6. Multi-Model Suite

Runner:

- [src/example/suicide_detection/run_model_suite.py](../../../src/example/suicide_detection/run_model_suite.py)

Default script behavior is controlled in that file's `run_suite(...)` call.

Programmatic example:

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

Suite stop conditions:

- subprocess exits non-zero
- run directory not found
- `execution_failures` found in `Meta.json`
- `prediction == Error` or non-empty `error` found in `all_cases.csv`

## 7. Artifacts Produced Per Run

Run directories are created at:

- [src/example/suicide_detection/.EvalRing](../../../src/example/suicide_detection/.EvalRing)

Each run contains:

- `result.md`
- `all_cases.txt`
- `all_cases.csv`
- `failed_cases.txt` (if failures exist)
- `Meta.json`

Important fields for structured modes:

- `prediction_confidence`
- `class_scores`
- `model_config.agent_mode`
- `model_config.base_class`

## 8. How Metrics Are Computed With Structured Outputs

The evaluator accepts either:

- plain string labels
- mapping outputs (class to score)

Core resolver in [src/EvalRing/agent/classification.py](../../../src/EvalRing/agent/classification.py) selects the top class for accuracy/precision/recall/F1 while preserving confidence and class scores in per-sample metadata.

## 9. Practical Experiment Recipes

Quick smoke test:

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 2 --max-workers 2 --agent-mode single-class
```

Structured mode sanity:

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode multi-class-chance
```

Base-vs-rest sanity:

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode base-vs-rest-binary --base-class Indicator
```

Recover transient errors:

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

## 10. Adapting This Pattern to Other Tasks

For a new classification task (for example emotion categorization):

1. create a task-specific agent class similar to [src/example/suicide_detection/llm_agent.py](../../../src/example/suicide_detection/llm_agent.py)
2. define class labels and output prompt format
3. optionally add base-vs-rest decomposition if binary comparisons are preferred
4. return either label strings or class-score mapping outputs
5. reuse [src/EvalRing/evaluator/implementations.py](../../../src/EvalRing/evaluator/implementations.py) without changing metric logic

This keeps the evaluation flow stable while allowing task-specific prompting and parsing.
