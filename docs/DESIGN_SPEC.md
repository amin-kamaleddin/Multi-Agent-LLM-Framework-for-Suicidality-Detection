# EvalRing Design Specification

## 1. Purpose

EvalRing is a lightweight evaluation framework for model and agent benchmarking.

In the current repository state, the system has two layers:

1. **Core framework layer** in [src/EvalRing](../src/EvalRing)
2. **Task-specific evaluation applications** in [src/example](../src/example), especially [src/example/suicide_detection](../src/example/suicide_detection)

The framework standardizes:

- how datasets are loaded
- how agents are called
- how evaluation metrics are computed
- how per-sample and aggregate results are persisted
- how failed executions are retried without rerunning the whole experiment

---

## 2. System Goals

### Primary goals

- Provide a common interface for benchmarking heterogeneous agents
- Preserve reproducibility through saved run configuration and metadata
- Support structured result persistence for research workflows
- Support incremental recovery from transient model/API failures

### Non-goals

- Full experiment orchestration platform
- Dataset labeling or annotation tooling
- Dashboarding or visualization server
- Distributed job execution manager

---

## 3. High-Level Architecture

## 3.1 Core modules

### Agent layer

Location: [src/EvalRing/agent](../src/EvalRing/agent)

Responsibilities:

- define the agent contract
- wrap model backends
- normalize prediction outputs into a common response object

Key public types:

- `BaseAgent`
- `AgentResponse`
- `ClassificationPrediction`
- `MockAgent`
- `RuleBasedAgent`
- `OpenAIAgent`

### Dataset layer

Location: [src/EvalRing/dataset](../src/EvalRing/dataset)

Responsibilities:

- load raw evaluation data
- normalize rows into `DataSample`
- expose iterable samples to the evaluator

Key public types:

- `BaseDataset`
- `DataSample`
- `JSONDataset`
- `CSVDataset`
- `DataFrameDataset`

### Evaluator layer

Location: [src/EvalRing/evaluator](../src/EvalRing/evaluator)

Responsibilities:

- run agents against datasets
- coordinate concurrency and retries
- compute task metrics
- emit evaluation result objects
- resolve top-class labels from either plain labels or structured class-score outputs

Key public types:

- `BaseEvaluator`
- `EvaluationMetrics`
- `EvaluationResult`
- `ClassificationEvaluator`

### LLM-as-a-Judge layer

Location: [src/EvalRing/evaluator/llm_judge](../src/EvalRing/evaluator/llm_judge)

Responsibilities:

- rubric-based scoring
- templated judgment prompts
- evaluation pipelines where an LLM judges another output

Key public types exported through [src/EvalRing/evaluator/__init__.py](../src/EvalRing/evaluator/__init__.py):

- `Rubric`
- `RubricLevel`
- `ScoringCriteria`
- `JudgeVerdict`
- `EvalSteps`
- `JudgeTemplate`
- `JudgeMetric`
- `LLMJudge`
- `OpenAIJudge`
- `LLMJudgeEvaluator`

---

## 4. Object Model

## 4.1 `DataSample`

Canonical dataset row representation.

Fields:

- `id`: unique or semi-unique sample identifier
- `input_text`: model input text
- `target_output`: ground-truth label or expected output
- `metadata`: row-level auxiliary fields

## 4.2 `AgentResponse`

Canonical prediction response returned by agents.

Fields:

- `input_id`
- `input_text`
- `output`
- `confidence`
- `metadata`
- `processing_time`
- `error`

For model-backed agents, `metadata` may include runtime telemetry such as:

- `raw_output`
- `model`
- `base_url`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `ttft`
- `generation_time`
- `tps`
- `total_time`

## 4.3 `EvaluationResult`

Top-level result container returned by evaluators.

Fields:

- `agent_name`
- `dataset_name`
- `metrics`
- `duration`
- `timestamp`
- `task_name`
- `version`
- `metadata`

---

## 5. Runtime Flow

Standard classification flow:

1. Load dataset into a `BaseDataset` implementation
2. Initialize the agent
3. Create evaluator instance
4. Run `evaluate()`
5. Collect per-sample predictions
6. Compute aggregate metrics
7. Persist run artifacts

In the suicide detection example, this flow is implemented in [src/example/suicide_detection/evaluate_rsd15k_main.py](../src/example/suicide_detection/evaluate_rsd15k_main.py).

---

## 6. Concurrency and Retry Model

## 6.1 First-pass evaluation

`ClassificationEvaluator` uses a `ThreadPoolExecutor` to process samples concurrently.

Configurable controls:

- `max_workers`
- `max_retries`

Retry behavior during first pass:

- each sample can be retried up to `max_retries`
- exponential backoff is used for repeated failures
- failed predictions are recorded as `prediction = "Error"`

## 6.2 In-place retry mode

The suicide detection application adds a second retry layer beyond the evaluator:

- CLI flag: `--retry-failed`
- optional pointer: `--meta-path <path to Meta.json>`

Behavior:

1. load the original run’s `Meta.json`
2. reuse original run config (`n_samples`, `max_workers`, `max_retries`, `seed`)
3. scan original `all_cases.csv`
4. select only rows with:
   - `prediction == Error`, or
   - non-empty `error`
5. rerun only those rows
6. patch original run artifacts in place

This prevents rerunning the entire experiment for one or two transient API failures.

---

## 7. Provider and Model Abstraction

## 7.1 Generic model client

`OpenAIAgent` in [src/EvalRing/agent/implementations.py](../src/EvalRing/agent/implementations.py) is the common model-backed client.

Current behavior:

- API key preference order:
  1. `OPEN_ROUTER_KEY`
  2. `OPENAI_API_KEY`
- Base URL behavior:
  - when `OPEN_ROUTER_KEY` is present, default base URL is OpenRouter
  - otherwise the default OpenAI endpoint is used

## 7.2 Task-specific agent

The suicide detection task uses `OpenAISuicideDetectionAgent` in [src/example/suicide_detection/llm_agent.py](../src/example/suicide_detection/llm_agent.py).

Responsibilities:

- define task prompt
- define label space
- construct user messages
- support multiple output modes:
  - single-class label
  - multi-class probabilities
  - base-vs-rest binary decomposition

### 7.2.1 Suicide detection modes

Current `OpenAISuicideDetectionAgent` modes:

- `single-class`
- `multi-class-chance`
- `base-vs-rest-binary`

`base-vs-rest-binary` takes a configurable `base_class`, runs one binary inference per non-base label, then aggregates the pairwise probabilities into one multi-class distribution.

### 7.2.2 Reusable classification output utilities

Location: [src/EvalRing/agent/classification.py](../src/EvalRing/agent/classification.py)

Core reusable functions:

- `resolve_classification_prediction`
- `normalize_probability_distribution`
- `aggregate_base_vs_rest_probabilities`

Design intent:

- future tasks can emit structured class scores without changing evaluator interfaces
- future tasks can reuse base-vs-rest decomposition and aggregation logic

## 7.3 Default model resolution

Current defaults:

- OpenRouter: `OPEN_ROUTER_MODEL` or `openai/gpt-4o-mini`
- OpenAI: `OPENAI_MODEL` or `gpt-5.2`

---

## 8. Suicide Detection Application Design

Location: [src/example/suicide_detection](../src/example/suicide_detection)

This application provides a concrete research workflow around the RSD_15K dataset.

### Inputs

- dataset: `data/rsd_15k.csv`
- environment keys from `EvalRing/.env`
- optional model override via environment variables

### Outputs per run

Created under:

- [src/example/suicide_detection/_EvalRing](../src/example/suicide_detection/_EvalRing)

Each run produces a timestamped folder containing:

- `result.md`
- `all_cases.txt`
- `all_cases.csv`
- `failed_cases.txt` (only when failures remain)
- `Meta.json`

### Metadata recorded

`Meta.json` stores:

- run configuration
- model configuration
- mode configuration (`agent_mode`, `base_class`)
- dataset configuration
- aggregate metrics
- per-class metrics
- confusion matrix
- estimated token cost
- retry history

---

## 9. Sequential Multi-Model Runner

Location: [src/example/suicide_detection/run_model_suite.py](../src/example/suicide_detection/run_model_suite.py)

Purpose:

- benchmark multiple models in sequence
- stop immediately when a model run is not error-free

Current guarantees:

- models are run one-by-one in declared order
- next model runs only if previous model has:
  - zero process failures
  - zero `Meta.json` execution failures
  - zero per-sample prediction errors in `all_cases.csv`

This script is intended for controlled benchmarking rather than maximum throughput.

---

## 10. Persistence Contract

## 10.1 `all_cases.csv`

Machine-readable source of truth for sample-level results.

Columns:

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

## 10.2 `Meta.json`

Machine-readable source of truth for run-level state.

Used for:

- reproducibility
- downstream analysis
- retry recovery
- suite-level validation

## 10.3 `result.md`

Human-readable report containing:

- run summary
- aggregate metrics
- per-class metrics
- confusion matrix
- per-sample summary table
- retry update summary when applicable

---

## 11. Known Constraints

### Duplicate sample IDs

The suicide detection dataset currently uses `users` as sample ID, which is not globally unique in all observed rows.

Impact:

- duplicate identifiers may appear in reports
- row position is sometimes a more reliable retry key than dataset-provided ID

The retry system therefore uses row position from `all_cases.csv` for in-place error recovery.

### Outdated top-level README

The repository root README describes an earlier interface and does not fully match the current implementation.

The docs in this folder should be treated as the more accurate operational reference.

---

## 12. Extension Points

### Add a new dataset loader

Implement a new `BaseDataset` subclass and normalize all rows to `DataSample`.

### Add a new agent backend

Implement a new `BaseAgent` subclass or adapt `OpenAIAgent` for a different API surface.

### Add a new evaluator

Implement a `BaseEvaluator` subclass for tasks such as:

- generation quality
- ranking
- retrieval evaluation
- judge-based rubric scoring

### Add a new task application

Pattern to follow:

1. create a task-specific folder under [src/example](../src/example)
2. add a task-specific agent wrapper if needed
3. add an application runner that loads data, runs evaluation, and writes structured outputs
4. optionally add retry and suite orchestration

---

## 13. Design Principles

The current implementation follows these practical principles:

- **simple contracts** over framework complexity
- **file-based reproducibility** over database dependence
- **recoverability** over full rerun cost
- **research traceability** over minimal logging
- **task-specific applications** built on top of reusable primitives
