# Evaluation Contracts, Metrics, and Guardrails

This document maps every requirement in the evaluation checklist to the code and run artifact that implements it. The architecture-neutral contract is produced by `run_evaluation`; `run_workflow_evaluation` is the compatibility wrapper that adds framework1's specialist outputs and metrics. Both the `run` and `evaluate` CLI commands use that wrapper.

For a copyable adapter and runner, start with [Adapting another workflow](ADAPTING_WORKFLOWS.md). This document is the exhaustive schema and artifact reference.

## Important label and binary-metric interpretation

The final label set is now:

- `Indicator`
- `Ideation`
- `Behavior`
- `Attempt`
- `Unsupportive`

The four existing specialist agents are unchanged. There is no separate `Unsupportive` specialist; the final judge may select `Unsupportive` when none of the four risk labels applies.

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
| Input `run_case(sample_id, text)` | Satisfied | `evaluation/evaluator.py::EvaluationWorkflow` defines the shared boundary; framework1 implements it in `workflow/graph.py::MultiAgentWorkflow.run_case`. |
| Dataset text plus ground-truth label | Satisfied | `evaluation/evaluator.py::load_evaluation_sample` validates canonical RSD15K fields or the legacy `source_row` fixture schema. |
| Current four risk labels | Satisfied | `core/labels.py::VALID_LABELS`; `Unsupportive` is an additional final-only label in `FINAL_LABELS`. |
| Temperature 0 and per-run configuration | Satisfied | CLI default and `EvaluationConfig.temperature`; persisted in `run_config.json`. |
| Per-agent model configuration | Satisfied through the Python API | Record arbitrary role-keyed metadata in `EvaluationConfig.agent_configuration`; framework1 also accepts matching role-keyed model objects through `agent_models`. |
| Markdown summary per agent | Conditional | Included when a workflow supplies comparable agent metrics. Shared final-output metrics are always present. |
| Text files with wrong-evaluation IDs | Satisfied | `misclassified_ids.txt` and `suicidal_intent_misclassified_ids.txt`. |
| JSON with complete results and metrics | Satisfied | `results.json`; written by `evaluation/evaluator.py::_complete_results_payload`. |
| Cost/precision/etc. visualizations | Satisfied | `classification_metrics.svg` and `runtime_metrics.svg`; written by `ArtifactWriter.write_visualizations`. |
| Accuracy, macro F1, macro precision/recall | Satisfied | Final-label and binary payloads in `metrics.json`. |
| Per-class F1/precision/recall | Satisfied | `label_metrics.csv`, `suicidal_intent_metrics.csv`, and `metrics.json`. |
| Confusion matrix | Satisfied | Final-label and binary matrices in `metrics.json` and compiled reports. |
| Failed records/failure rate | Satisfied | `classification.failed_records` and `classification.failure_rate`. |
| LLM calls and latency | Satisfied | `runtime.call_count`, `llm_request_count`, and `latency_s`. |
| Token totals | Satisfied | `runtime.tokens.prompt_total`, `completion_total`, and `total`. |
| Average cost per call | Satisfied | `runtime.cost_usd.avg_per_call`. |
| JSON/parsing errors | Satisfied | `runtime.json_parsing_errors` and `runtime.errors_by_type`. |

## Input contracts

### Per-case Python boundary

Consumed by `evaluation/evaluator.py::run_evaluation`; framework1 implements the boundary in `workflow/graph.py::MultiAgentWorkflow.run_case`:

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

`evaluation/evaluator.py::EvaluationConfig` is serialized into `run_config.json`. It records the resolved and requested model names, temperature, maximum completion tokens, retry thresholds, prompt/schema mode, dataset paths, early-stop guardrails, and an optional arbitrary role-keyed `agent_configuration` mapping.

The normal framework1 CLI configures one model for all five calls in a case and defaults to temperature `0`. Its compatibility wrapper expands the resolved configuration for the four specialists and final judge. The shared evaluator itself does not maintain an agent-role allowlist.

For independent models, use the role keys `Indicator_specialist`, `Ideation_specialist`, `Behavior_specialist`, `Attempt_specialist`, and `judge`:

```python
config = EvaluationConfig(
    ...,
    agent_configuration={
        "Attempt_specialist": {"agent_name": "Attempt Agent", "model_name": "attempt-model", "temperature": 0.0},
        "judge": {"agent_name": "Final Judge", "model_name": "judge-model", "temperature": 0.0},
    },
)
result = run_workflow_evaluation(
    ...,
    config=config,
    agent_models={"Attempt_specialist": attempt_model, "judge": judge_model},
)
```

The caller constructs each model object with the recorded temperature. Framework1 validates only its own five roles and routes overrides in `workflow/graph.py::MultiAgentWorkflow._call_and_parse`. Other workflows may use any role names in `agent_configuration` and `llm_calls.jsonl`.

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

The optional prediction enricher and agent-metrics provider are supplied by a workflow adapter only when its internal tasks have defensible ground truth. Without them, final-label, binary grouping, runtime, token, cost, parse-error, and failure metrics remain complete, `agent_metrics` is empty, and no `agent_metrics.csv` is written.

## Agent output contracts

The Pydantic contracts live in `core/schemas.py`; prompts that request these shapes live in `workflow/prompts.py`.

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

`core/schemas.py::JudgeAnswer` enforces the conditional mapping. A missing boolean, string boolean, unknown label, extra key, or inconsistent label/boolean pair is a schema-validation failure and is retried according to `max_retries`. Failed cases are represented in artifacts as `prediction = "Error"` and `suicidal_intent = null`; `Error` is not a valid successful judge output.

## Per-record and aggregate output formats

`evaluation/evaluator.py::build_prediction_row` creates each `predictions.csv` row. Shared columns are:

- original input fields plus `sample_id`
- `ground_truth`, `ground_truth_suicidal_intent`
- `prediction`, `suicidal_intent`
- `correct`, `suicidal_intent_correct`
- optional `agent_outputs`
- `error`

The framework1 wrapper additionally preserves `specialist_answers`, `specialist_outputs`, and one `*_answer`/`*_reasoning` column per specialist.

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

### Final-label metrics

`compute_classification_metrics` evaluates the five final labels. Invalid or failed predictions go into an `Error` prediction column.

For each class, one-vs-rest counts are calculated:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
accuracy  = (TP + TN) / N
```

A zero denominator returns `0.0`. Macro precision, recall, and F1 are the unweighted means of the corresponding per-class values. Overall accuracy is `correct / N`. `failed_records` counts predictions outside the final label set; `failure_rate = failed_records / N`.

### `suicidal_intent` metrics

`compute_binary_classification_metrics` runs the same calculation over explicit `false` and `true` classes. It produces:

- overall accuracy;
- macro precision, recall, and F1;
- false-class and true-class precision, recall, F1, support, TP, FP, TN, and FN;
- positive-class (`true`) precision, recall, and F1;
- a `false`/`true` confusion matrix with an `Error` prediction column;
- failed-record count and failure rate;
- ground-truth class counts and the single-class limitation flag.

Yes, the same metric family can be added to the original metrics, and it is now present under `metrics.json::suicidal_intent`. The values are only fully informative when the evaluated ground truth contains both `true` and `false`. With the current four-label sample, positive-class recall and accuracy can still be calculated, but specificity/negative recall and two-class macro scores cannot establish negative-class performance.

### Optional workflow agent metrics

The shared evaluator does not assume that internal agents are classifiers. Framework1's optional provider uses `evaluation/evaluator.py::_compute_specialist_metrics` to evaluate its specialists as binary one-vs-rest classifiers:

```text
specialist ground truth = (row ground-truth label == specialist label)
specialist prediction   = specialist output.answer
```

For framework1, the full payload is stored under `metrics.json::specialist_metrics`. A flat summary is stored in `agent_metrics.csv`, `metrics.json::agent_metrics`, and the Markdown reports. Workflows without comparable agent-level ground truth omit these artifacts and must not fabricate specialist answers from the final label.

## Runtime metrics and guardrails

Every attempt, including retries, is written to `llm_calls.jsonl`. `evaluation/metrics.py::summarize_runtime` calculates:

- call count and failed attempt count;
- latency minimum, maximum, average, and total;
- input, output, total, cached, and cache-write token totals and per-call averages;
- total cost and average cost per call;
- provider timing when exposed: TTFT, generation time, provider latency, moderation latency, and throughput;
- end-to-end completion throughput;
- error counts by type;
- JSON/schema parsing errors, counted from JSON decoding, Pydantic validation, and LangChain structured-output validation failures.

Transport or provider metadata may omit tokens, cost, or provider timing. Missing values remain zero or `null` rather than being estimated. Local end-to-end latency is always measured.

`evaluation/evaluator.py::_stop_reason` stops abnormal runs when any configured threshold is reached:

- failed LLM attempts;
- failed records;
- failed-record rate after a minimum number of processed records.

Thresholds and the stop reason are persisted in `run_config.json`, `metrics.json`, and `trace_summary.json`. Full parse/transport error records are also written to `errors.jsonl`.

## Run artifact inventory

| Artifact | Created by | Contents |
|---|---|---|
| `run_config.json` | `run_evaluation` | Run and arbitrary role-keyed agent configuration; framework1's wrapper records its resolved five-role configuration. |
| `dataset_audit.json` | `run_evaluation` | Source audit or explicit audit-unavailable marker. |
| `sampled_dataset.csv` | `run_evaluation` | Exact evaluated inputs and ground truth. |
| `predictions.csv` | `_write_evaluation_artifacts` | Source fields, shared outputs, correctness, errors, and optional workflow data. |
| `results.json` | `_complete_results_payload` | Complete nested per-record results plus all metrics. |
| `metrics.json` | `_write_evaluation_artifacts` | Mandatory final-label, binary, and runtime metrics plus optional workflow agent metrics. |
| `model_metrics.csv` | `_write_evaluation_artifacts` | Final-judge aggregate label metrics. |
| `label_metrics.csv` | `_write_evaluation_artifacts` | Per-final-label one-vs-rest metrics. |
| `agent_metrics.csv` | optional workflow provider | Workflow-defined per-agent metrics; emitted for framework1's comparable specialist tasks and omitted otherwise. |
| `suicidal_intent_metrics.csv` | `_write_evaluation_artifacts` | False/true per-class metrics. |
| `metrics.md` | `ArtifactWriter.write_metrics_md` | Main metric and per-agent Markdown summary. |
| `compiled_results.csv` | `compile_run_results` | Long-form machine-readable report. |
| `compiled_results.md` | `compile_run_results` | Review-oriented complete Markdown report. |
| `misclassified_ids.txt` | `_write_evaluation_artifacts` | IDs whose final label is wrong. |
| `suicidal_intent_misclassified_ids.txt` | `_write_evaluation_artifacts` | IDs whose boolean output is wrong. |
| `classification_metrics.svg` | `ArtifactWriter.write_visualizations` | Label and binary accuracy/precision/recall/F1 bars. |
| `runtime_metrics.svg` | `ArtifactWriter.write_visualizations` | Calls, latency, tokens, cost, failures, and parse errors. |
| `llm_calls.jsonl` | `run_evaluation` | Every workflow-reported request/retry with role, model, usage, cost, timing, raw output, parse result, and error. |
| `traces.jsonl` | `run_evaluation` | Workflow-supplied node events and durations. |
| `errors.jsonl` | `run_evaluation` | Final unrecovered workflow-reported case errors. |
| `trace_summary.json` | `_write_evaluation_artifacts` | Compact status, runtime, guardrail, and trace-file summary. |
