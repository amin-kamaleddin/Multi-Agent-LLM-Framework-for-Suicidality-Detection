# Evaluation Contracts, Metrics, and Guardrails

This document maps every requirement in the evaluation checklist to the code and run artifact that implements it. The architecture-neutral contract is produced by `evaluation/evaluator.py::run_evaluation`; each framework owns its graph construction and evaluation adapter. The shared `evaluate` command dispatches through `workflows/registry.py`.

For a copyable adapter and runner, start with [Adapting another workflow](ADAPTING_WORKFLOWS.md). This document is the exhaustive schema and artifact reference.

## Important label and binary-metric interpretation

The final label set is now:

- `Indicator`
- `Ideation`
- `Behavior`
- `Attempt`
- `Unsupportive`

Framework 1 runs five parallel label specialists. Four cover `Indicator`, `Ideation`, `Behavior`, and `Attempt`, which form the positive grouping. The fifth covers `Unsupportive`, the negative grouping. Each specialist makes only its assigned binary decision; the final judge receives all five outputs and selects one final label.

The required deterministic mapping is implemented in `core/labels.py::label_to_suicidal_intent`:

```text
Indicator | Ideation | Behavior | Attempt -> suicidal_intent = true
Unsupportive                              -> suicidal_intent = false
```

This mapping deserves care: the repository's existing `Indicator` definition says that the author's own suicidal risk is not indicated, but the requested mapping treats `Indicator` as `suicidal_intent = true`. Therefore, `suicidal_intent` means **membership in the dataset's four suicide-related positive labels**, not necessarily a clinically validated claim that the author expresses intent. The field name is retained for compatibility. If it is instead intended to mean the author's own self-risk, whether `Indicator` should map to `false` requires a separate ontology decision.

The frozen 500-row evaluation sample contains only the original four labels. Under the required mapping, all 500 ground-truth boolean values are `true`; it contains no negative examples. The new seeded benchmark cohorts under `dataset/positive-negative-final-datasets/` add 100 positive examples and 100 `Unsupportive` examples. The latter is balanced between 50 real and 50 synthetic negatives. Evaluating only one cohort still triggers `has_both_ground_truth_classes = false`; a meaningful two-class comparison must contain rows from both cohorts.

## Requirement status

| Checklist component | Status | Implementation |
|---|---|---|
| Input `run_case(sample_id, text)` | Satisfied | `workflows/contracts.py::Workflow` defines the shared boundary; framework 1 implements it in `workflows/framework_1_parallel_no_evidence/graph.py::MultiAgentWorkflow.run_case`. |
| Dataset text plus ground-truth label | Satisfied | `evaluation/evaluator.py::load_evaluation_sample` validates canonical RSD15K fields or the legacy `source_row` fixture schema. |
| Five final labels | Satisfied | `domain/labels.py::FINAL_LABELS`; framework 1 assigns one specialist to each label. |
| Temperature 0 and per-run configuration | Satisfied | CLI default and `EvaluationConfig.temperature`; persisted in `run_config.json`. |
| Per-agent model configuration | Satisfied | The shared CLI loads per-workflow JSON; framework_1_parallel_no_evidence builds matching role-keyed model objects from its optional `agents` object. The Python API also accepts arbitrary role-keyed metadata in `EvaluationConfig.agent_configuration`. |
| Markdown metric summary | Satisfied | Includes grouped `suicidal_intent` metrics plus runtime and guardrail telemetry. |
| Text files with wrong-evaluation IDs | Satisfied | `misclassified_ids.txt` and `suicidal_intent_misclassified_ids.txt`. |
| JSON with complete results and metrics | Satisfied | `results.json`; written by `evaluation/evaluator.py::_complete_results_payload`. |
| Cost/precision/etc. visualizations | Satisfied | `suicidal_intent_metrics.svg` and `runtime_metrics.svg`; written by `ArtifactWriter.write_visualizations`. |
| Accuracy, F1, precision/recall | Satisfied | Only positive-oriented `suicidal_intent` metrics are emitted. |
| Final-label aggregate or per-class scores | Not emitted by design | The evaluator retains the five-label confusion matrix without calculating final-label accuracy, precision, recall, or F1. |
| Confusion matrix | Satisfied | Final-label and binary matrices in `metrics.json` and compiled reports. |
| Failed records/failure rate | Satisfied | `classification.failed_records` and `classification.failure_rate`. |
| LLM calls and latency | Satisfied | `runtime.call_count`, `llm_request_count`, and `latency_s`. |
| Token totals | Satisfied | `runtime.tokens.prompt_total`, `completion_total`, and `total`. |
| Average cost per call | Satisfied | `runtime.cost_usd.avg_per_call`. |
| JSON/parsing errors | Satisfied | `runtime.json_parsing_errors` and `runtime.errors_by_type`. |

## Input contracts

### Per-case Python boundary

Consumed by `evaluation/evaluator.py::run_evaluation`; framework 1 implements the boundary in `workflows/framework_1_parallel_no_evidence/graph.py::MultiAgentWorkflow.run_case`:

```json
{
  "sample_id": "10",
  "text": "Text to classify"
}
```

Equivalent JSON Schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": ["sample_id", "text"],
  "properties": {
    "sample_id": {"type": "string"},
    "text": {"type": "string"}
  }
}
```

### Evaluation dataset

Validated in `evaluation/evaluator.py::load_evaluation_sample` and copied to each run as `sampled_dataset.csv`.

The canonical schema shared with the existing single-framework runner is:

| Column | Type | Meaning |
|---|---|---|
| `ID` | unique ID | Stable zero-based RSD15K sample identifier; preferred as `sample_id`. |
| `users` | author ID | Original, non-unique author identifier. |
| `text` | string | Model input. |
| `sentiment` | enum | Ground truth in `Indicator`, `Ideation`, `Behavior`, `Attempt`, or `Unsupportive`. |
| `time` | integer | Original Unix timestamp, or `0` for synthetic rows without a real timestamp. |
| `real` | optional boolean | Provenance: real (`true`) or synthetic (`false`). |

For existing fixtures, `source_row,text,sentiment[,real]` remains accepted as a legacy schema. `ID` takes precedence when present. All input columns are preserved in `sampled_dataset.csv` and `predictions.csv`, while `sample_id` is added as the common artifact identifier. Full-dataset comparisons should use RSD15K with its original fields rather than the stripped Dataset1-derived export.

## Model configuration

`evaluation/evaluator.py::EvaluationConfig` is serialized into `run_config.json`. It records the selected workflow name, raw workflow JSON configuration, resolved and requested model names, temperature, maximum completion tokens, retry thresholds, prompt/schema mode, dataset paths, early-stop guardrails, and an optional arbitrary role-keyed `agent_configuration` mapping.

The normal `framework_1_parallel_no_evidence` CLI configures one model for six calls per case: five specialists and the judge. `framework_2_parallel_with_evidence` also uses six calls: five collectors and the judge. Both default to temperature `0`; their optional workflow JSON can override model settings per allowed role. The shared evaluator itself does not maintain an agent-role allowlist.

For independent models, use the role keys `Indicator_specialist`, `Ideation_specialist`, `Behavior_specialist`, `Attempt_specialist`, `Unsupportive_specialist`, and `judge`:

```python
config = EvaluationConfig(
    ...,
    agent_configuration={
        "Attempt_specialist": {"agent_name": "Attempt Agent", "model_name": "attempt-model", "temperature": 0.0},
        "Unsupportive_specialist": {"agent_name": "Unsupportive Agent", "model_name": "negative-model", "temperature": 0.0},
        "judge": {"agent_name": "Final Judge", "model_name": "judge-model", "temperature": 0.0},
    },
)
from workflows.framework_1_parallel_no_evidence.evaluation import run_evaluation as run_framework_1

result = run_framework_1(
    ...,
    config=config,
    agent_models={
        "Attempt_specialist": attempt_model,
        "Unsupportive_specialist": negative_model,
        "judge": judge_model,
    },
)
```

The caller constructs each model object with the recorded temperature. Frameworks 1 and 2 each validate six roles, and both route overrides in `workflows/shared/parallel/graph.py::MultiAgentWorkflow._call_and_parse`. Other workflows may use any role names in `agent_configuration` and `llm_calls.jsonl`.

## Architecture-neutral workflow contract

`run_evaluation` accepts an injected object with `run_case(sample_id: str, text: str)`. Each case result must expose:

```text
sample_id, final_label, suicidal_intent, errors, traces, llm_calls
```

Successful outputs use a valid final label and a boolean `suicidal_intent`; failed outputs use `final_label = "Error"` and `suicidal_intent = null`. A workflow may additionally expose a role-keyed `agent_outputs` object. Agent outputs are stored but never interpreted or converted into one-vs-rest predictions by the shared evaluator.

```python
result = run_evaluation(
    sample_df=sample,
    workflow=hierarchical_workflow,
    config=EvaluationConfig(
        ...,
        agent_configuration={
            "perspective_agent": {"model_name": "model-a", "temperature": 0.0},
            "evaluator_agent": {"model_name": "model-b", "temperature": 0.0},
        },
    ),
)
```

The optional prediction enricher may be supplied by a workflow adapter to preserve additional intermediate outputs. These values are stored for inspection but are not evaluated. Metrics are calculated only from the final judge's label and grouped binary decision.

## Agent output contracts

The shared Pydantic contracts live in `domain/`; intermediate contracts and prompts live inside the corresponding `workflows/framework_*` package.

### Specialist, boolean mode

```json
{"answer": true}
```

`answer` must be a JSON boolean. Extra keys are forbidden.

### Specialist, reasoned mode

```json
{
  "answer": true,
  "reasoning": "One to three sentence explanation containing any supporting evidence."
}
```

`reasoning` must be a nonblank string. Extra keys, including a separate `evidence` key, are forbidden.

### Evidence collector, framework_2_parallel_with_evidence

```json
{
  "evidence": [
    {
      "quote": "I wrote a note",
      "type": "preparation",
      "subject": "author",
      "time": "past",
      "observation": "The author reports writing a note."
    }
  ]
}
```

The top-level object has only `evidence`. It contains zero to three items with exactly five fields. `quote` is a relevant excerpt or concise paraphrase of at most 100 words and is not required to be an exact input substring; `observation` is at most 100 words. `subject` and `time` use the enums enforced in `domain/evidence.py`. Each collector accepts only its three evidence types from `workflows/framework_2_parallel_with_evidence/prompts.py`. Schema and collector-specific type failures are retried. Collectors do not return decisions, scores, confidence, citations, or review status.

### Final judge

```json
{
  "label_name": "Ideation",
  "suicidal_intent": true
}
```

The strict JSON Schema is:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": ["label_name", "suicidal_intent"],
  "properties": {
    "label_name": {
      "enum": ["Indicator", "Ideation", "Behavior", "Attempt", "Unsupportive"]
    },
    "suicidal_intent": {
      "type": "boolean",
      "description": "Membership in the dataset's four-label positive class; not a separate clinical inference."
    }
  },
  "allOf": [
    {
      "if": {"properties": {"label_name": {"const": "Unsupportive"}}},
      "then": {"properties": {"suicidal_intent": {"const": false}}},
      "else": {"properties": {"suicidal_intent": {"const": true}}}
    }
  ]
}
```

`domain/final_output.py::JudgeAnswer` enforces the conditional mapping. A missing boolean, string boolean, unknown label, extra key, or inconsistent label/boolean pair is a schema-validation failure and is retried according to `max_retries`. Failed cases are represented in artifacts as `prediction = "Error"` and `suicidal_intent = null`; `Error` is not a valid successful judge output.

## Per-record and aggregate output formats

`evaluation/evaluator.py::build_prediction_row` creates each `predictions.csv` row. Shared columns are:

- original input fields plus `sample_id`
- `ground_truth`, `ground_truth_suicidal_intent`
- `prediction`, `suicidal_intent`
- `correct`, `suicidal_intent_correct`
- optional `agent_outputs`
- `error`

The framework 1 wrapper additionally preserves `specialist_answers`, `specialist_outputs`, and one `*_answer`/`*_reasoning` column per specialist. Framework 2 preserves `collector_outputs`; collector packets are stored but not interpreted as predictions and do not produce agent classification metrics.

`results.json` combines complete records and metrics:

```json
{
  "schema_version": "1.1",
  "results": [
    {
      "ID": 10,
      "sample_id": 10,
      "ground_truth_output": {
        "label_name": "Ideation",
        "suicidal_intent": true
      },
      "final_output": {
        "label_name": "Ideation",
        "suicidal_intent": true
      },
      "agent_outputs": {}
    }
  ],
  "metrics": {}
}
```

## Metrics and calculations

The calculations are implemented without an external metrics package in `evaluation/metrics.py`.

### Final-label confusion matrix

`compute_final_label_confusion_matrix` records the five final labels without calculating aggregate or per-label model scores. Invalid or failed predictions go into an `Error` prediction column. The persisted matrix has actual labels as rows and predicted labels plus `Error` as columns.

`metrics.json::classification` contains only:

- `confusion_matrix`;
- `total`;
- `failed_records`, counting predictions outside the final label set;
- `failure_rate = failed_records / total`.

There is no final-label accuracy, precision, recall, F1, `model_summary`, `model_metrics.csv`, or `label_metrics.csv`. Exact-label correctness remains available per record in `predictions.csv`, and the five-label matrix remains available for inspection.

### `suicidal_intent` metrics

The evaluator keeps two different views of the final judge result:

1. `metrics.json::classification` is a five-label audit view containing the confusion matrix and failure guardrail counts only.
2. `metrics.json::suicidal_intent` is one fixed binary task. `Indicator`, `Ideation`, `Behavior`, and `Attempt` are always Positive (`true`); only `Unsupportive` is Negative (`false`).

`compute_suicidal_intent_metrics` derives both actual and predicted binary values from the final labels and then calls `compute_binary_classification_metrics`. The required judge boolean is validated against the same mapping by `domain/final_output.py::JudgeAnswer`; it is not a separate classification decision.

The grouped confusion matrix uses actual values as rows and predicted values as columns:

| Actual \ Predicted | `false` / Negative | `true` / Positive | `Error` |
|---|---:|---:|---:|
| `false` / Unsupportive | TN | FP | failed negative record |
| `true` / any of the four suicidality labels | FN | TP | failed positive record |

The canonical counts are stored in `suicidal_intent.positive_class_counts`:

```text
TP = actual is one of the four positive labels and prediction is one of the four positive labels
TN = actual is Unsupportive and prediction is Unsupportive
FP = actual is Unsupportive and prediction is one of the four positive labels
FN = actual is one of the four positive labels and prediction is Unsupportive
```

Failed predictions stay in the separate `Error` column and are excluded from these four canonical cell counts. They remain included in `failed_records`, `failure_rate`, and overall accuracy. Therefore, a dataset whose ground truth is entirely `Unsupportive` can never have binary TP or FN values above zero; its valid predictions can only contribute TN or FP.

The binary payload does not calculate a second one-vs-rest row with `false` treated as positive. It produces one summary whose TP, FP, TN, and FN fields always use the fixed Positive definition above:

- overall accuracy;
- positive-class precision, recall, and F1;
- positive and negative ground-truth support;
- a `false`/`true` confusion matrix with an `Error` prediction column;
- the fixed positive and negative label mapping;
- canonical positive-oriented TP, TN, FP, and FN counts;
- failed-record count and failure rate;
- ground-truth class counts and the single-class limitation flag.

The values are only fully informative when the evaluated ground truth contains both `true` and `false`. With a positive-only cohort, specificity/negative recall cannot be established. With a negative-only cohort, sensitivity/positive recall cannot be established and canonical TP is necessarily zero.

### Intermediate agent outputs are not evaluated

The shared evaluator does not calculate accuracy, confusion matrices, precision, recall, or F1 for specialists, collectors, routers, reviewers, or other intermediate agents. Framework 1 still stores its five specialist answers and complete outputs in `predictions.csv` and `results.json`:

```text
specialist_answers -> stored for trace inspection
specialist_outputs -> stored for trace inspection
final judge output -> evaluated against ground truth
```

Consequently, `metrics.json` contains no `agent_metrics` or `specialist_metrics` payload, and runs do not create `agent_metrics.csv` or a per-agent metrics section in Markdown reports.

## Runtime metrics and guardrails

Every attempt, including retries, is written to `llm_calls.jsonl`. `evaluation/metrics.py::summarize_runtime` calculates:

- total agent execution duration, measured as wall-clock time spent only inside each `workflow.run_case(...)` invocation;
- call count and failed attempt count;
- latency minimum, maximum, average, and total;
- input, output, total, cached, and cache-write token totals and per-call averages;
- total cost and average cost per call;
- provider timing when exposed: TTFT, generation time, provider latency, moderation latency, and throughput;
- end-to-end completion throughput;
- error counts by type;
- JSON/schema parsing errors, counted from JSON decoding, Pydantic validation, and LangChain structured-output validation failures.

`runtime.agent_execution_duration_s` accumulates the per-record workflow wall-clock durations. It includes all agent work, retries, and parallel orchestration performed inside `run_case`, but excludes dataset preparation, evaluator bookkeeping, metric calculation, artifact writing, and report compilation. Because parallel agents overlap, their shared wall-clock span is counted once rather than summing their individual call latencies.

Transport or provider metadata may omit tokens, cost, or provider timing. Missing values remain zero or `null` rather than being estimated. Local end-to-end latency is always measured.

`evaluation/evaluator.py::_stop_reason` stops abnormal runs when any configured threshold is reached:

- failed LLM attempts;
- failed records;
- failed-record rate after a minimum number of processed records.

Thresholds and the stop reason are persisted in `run_config.json`, `metrics.json`, and `trace_summary.json`. Full parse/transport error records are also written to `errors.jsonl`.

## Run artifact inventory

| Artifact | Created by | Contents |
|---|---|---|
| `run_config.json` | `run_evaluation` | Run and arbitrary role-keyed agent configuration; the built-in wrappers record their resolved five- or six-role configuration. |
| `dataset_audit.json` | `run_evaluation` | Source audit or explicit audit-unavailable marker. |
| `sampled_dataset.csv` | `run_evaluation` | Exact evaluated inputs and ground truth. |
| `predictions.csv` | `_write_evaluation_artifacts` | Source fields, shared outputs, correctness, errors, and optional workflow data. |
| `results.json` | `_complete_results_payload` | Complete nested per-record results plus all metrics. |
| `metrics.json` | `_write_evaluation_artifacts` | Five-label confusion audit, grouped-binary metrics, agent execution duration, and runtime metrics. |
| `suicidal_intent_metrics.csv` | `_write_evaluation_artifacts` | One positive-oriented binary summary; negative rows are never reinterpreted as TP/FN. |
| `metrics.md` | `ArtifactWriter.write_metrics_md` | Grouped-binary metrics plus runtime and guardrail summary. |
| `compiled_results.csv` | `compile_run_results` | Long-form machine-readable report. |
| `compiled_results.md` | `compile_run_results` | Review-oriented complete Markdown report. |
| `misclassified_ids.txt` | `_write_evaluation_artifacts` | IDs whose final label is wrong. |
| `suicidal_intent_misclassified_ids.txt` | `_write_evaluation_artifacts` | IDs whose boolean output is wrong. |
| `suicidal_intent_metrics.svg` | `ArtifactWriter.write_visualizations` | Grouped-binary accuracy, precision, recall, and F1 bars. |
| `runtime_metrics.svg` | `ArtifactWriter.write_visualizations` | Agent execution duration, calls, latency, tokens, cost, failures, and parse errors. |
| `llm_calls.jsonl` | `run_evaluation` | Every workflow-reported request/retry with role, model, usage, cost, timing, raw output, parse result, and error. |
| `traces.jsonl` | `run_evaluation` | Workflow-supplied node events and durations. |
| `errors.jsonl` | `run_evaluation` | Final unrecovered workflow-reported case errors. |
| `trace_summary.json` | `_write_evaluation_artifacts` | Compact status, runtime, guardrail, and trace-file summary. |
