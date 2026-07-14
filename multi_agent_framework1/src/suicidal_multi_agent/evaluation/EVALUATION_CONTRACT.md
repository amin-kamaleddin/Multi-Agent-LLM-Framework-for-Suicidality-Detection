# Evaluation Contracts, Metrics, and Guardrails

This document maps every requirement in the evaluation checklist to the code and run artifact that implements it. The full contract is produced by `run_workflow_evaluation` and by both the `run` and `evaluate` CLI commands.

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

This mapping deserves care: the repository's existing `Indicator` definition says that the author's own suicidal risk is not indicated, but the requested mapping treats `Indicator` as `suicidal_intent = true`. Therefore, the new boolean should be interpreted as the requested four-label-versus-`Unsupportive` grouping, not necessarily as a clinically validated self-risk flag. If `suicidal_intent` is intended to mean the author's own self-risk, whether `Indicator` should instead map to `false` requires a separate ontology decision.

The frozen 500-row evaluation sample contains only the original four labels. Under the required mapping, all 500 ground-truth boolean values are `true`; it contains no negative examples. The new seeded benchmark cohorts under `dataset/positive-negative-final-datasets/` add 100 positive examples and 100 `Unsupportive` examples. The latter is balanced between 50 real and 50 synthetic negatives. Evaluating only one cohort still triggers `has_both_ground_truth_classes = false`; a meaningful two-class comparison must contain rows from both cohorts.

## Requirement status

| Checklist component | Status | Implementation |
|---|---|---|
| Input `run_case(example_id, text)` | Satisfied | `workflow/graph.py::MultiAgentWorkflow.run_case` accepts `sample_id` and `text`. |
| Dataset text plus ground-truth label | Satisfied | `evaluation/evaluator.py::load_evaluation_sample` validates `source_row`, `text`, and `sentiment`. |
| Current four risk labels | Satisfied | `core/labels.py::VALID_LABELS`; `Unsupportive` is an additional final-only label in `FINAL_LABELS`. |
| Temperature 0 and per-run configuration | Satisfied | CLI default and `EvaluationConfig.temperature`; persisted in `run_config.json`. |
| Per-agent model configuration | Satisfied through the Python API | Pass role-keyed model objects as `agent_models` and record matching names/temperatures in `EvaluationConfig.agent_model_names` and `agent_temperatures`. |
| Markdown summary per agent | Satisfied | `metrics.md` and `compiled_results.md`; built in `evaluation/artifacts.py` and `evaluation/reporting.py`. |
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

Created and consumed in `workflow/graph.py::MultiAgentWorkflow.run_case`:

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

| Column | Type | Meaning |
|---|---|---|
| `source_row` | unique ID | Stable example identifier. |
| `text` | string | Text sent to every specialist and the final judge. |
| `sentiment` | enum | Ground truth in `Indicator`, `Ideation`, `Behavior`, `Attempt`, or `Unsupportive`. |
| `real` | optional boolean | Provenance flag used by the new cohorts: real (`true`) or synthetic (`false`). |

The bundled balanced sampler still produces the four labels present in the source dataset. A custom evaluation CSV can contain `Unsupportive` rows, and the evaluator accepts and preserves the optional `real` provenance column.

## Model configuration

`evaluation/evaluator.py::EvaluationConfig` is serialized into `run_config.json`. It records the resolved and requested model names, temperature, maximum completion tokens, retry thresholds, prompt/schema mode, dataset paths, and early-stop guardrails.

The normal CLI configures one model for all five calls in a case and defaults to temperature `0`. The resolved configuration is expanded under `agent_configuration` for the four specialists and final judge.

For independent models, use the role keys `Indicator_specialist`, `Ideation_specialist`, `Behavior_specialist`, `Attempt_specialist`, and `judge`:

```python
config = EvaluationConfig(
    ...,
    agent_model_names={"Attempt_specialist": "attempt-model", "judge": "judge-model"},
    agent_temperatures={"Attempt_specialist": 0.0, "judge": 0.0},
)
result = run_workflow_evaluation(
    ...,
    config=config,
    agent_models={"Attempt_specialist": attempt_model, "judge": judge_model},
)
```

The caller constructs each model object with the recorded temperature. `workflow/graph.py::MultiAgentWorkflow._call_and_parse` routes each role to its override and writes the actual role-level model name into `llm_calls.jsonl`.

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
    "suicidal_intent": {"type": "boolean"}
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

`evaluation/evaluator.py::build_prediction_row` creates each `predictions.csv` row. Important columns are:

- `source_row`, `text`
- `ground_truth`, `ground_truth_suicidal_intent`
- `prediction`, `suicidal_intent`
- `correct`, `suicidal_intent_correct`
- `specialist_answers`, `specialist_outputs`
- one `*_answer` and `*_reasoning` column per specialist
- `error`

`results.json` combines complete records and metrics:

```json
{
  "schema_version": "1.0",
  "results": [
    {
      "source_row": 10,
      "ground_truth_output": {
        "label_name": "Ideation",
        "suicidal_intent": true
      },
      "final_output": {
        "label_name": "Ideation",
        "suicidal_intent": true
      },
      "specialist_answers": {},
      "specialist_outputs": {}
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

### Specialist-agent metrics

`evaluation/evaluator.py::_compute_specialist_metrics` evaluates each specialist as a binary one-vs-rest classifier:

```text
specialist ground truth = (row ground-truth label == specialist label)
specialist prediction   = specialist output.answer
```

The full payload is stored under `metrics.json::specialist_metrics`. A flat summary for all specialists plus the final judge's label and boolean tasks is stored in `agent_metrics.csv`, `metrics.json::agent_metrics`, and the Markdown reports.

## Runtime metrics and guardrails

Every attempt, including retries, is written to `llm_calls.jsonl`. `evaluation/metrics.py::summarize_runtime` calculates:

- call count and failed attempt count;
- latency minimum, maximum, average, and total;
- input, output, total, cached, and cache-write token totals and per-call averages;
- total cost and average cost per call;
- provider timing when exposed: TTFT, generation time, provider latency, moderation latency, and throughput;
- end-to-end completion throughput;
- error counts by type;
- JSON/schema parsing errors, counted from `JSONDecodeError` and Pydantic `ValidationError` attempts.

Transport or provider metadata may omit tokens, cost, or provider timing. Missing values remain zero or `null` rather than being estimated. Local end-to-end latency is always measured.

`evaluation/evaluator.py::_stop_reason` stops abnormal runs when any configured threshold is reached:

- failed LLM attempts;
- failed records;
- failed-record rate after a minimum number of processed records.

Thresholds and the stop reason are persisted in `run_config.json`, `metrics.json`, and `trace_summary.json`. Full parse/transport error records are also written to `errors.jsonl`.

## Run artifact inventory

| Artifact | Created by | Contents |
|---|---|---|
| `run_config.json` | `run_workflow_evaluation` | Run and resolved per-agent configuration. |
| `dataset_audit.json` | `run_workflow_evaluation` | Source audit or explicit audit-unavailable marker. |
| `sampled_dataset.csv` | `run_workflow_evaluation` | Exact evaluated inputs and ground truth. |
| `predictions.csv` | `_write_evaluation_artifacts` | Flat per-record inputs, outputs, correctness, and specialist data. |
| `results.json` | `_complete_results_payload` | Complete nested per-record results plus all metrics. |
| `metrics.json` | `_write_evaluation_artifacts` | Final-label, binary, specialist, per-agent, and runtime metrics. |
| `model_metrics.csv` | `_write_evaluation_artifacts` | Final-judge aggregate label metrics. |
| `label_metrics.csv` | `_write_evaluation_artifacts` | Per-final-label one-vs-rest metrics. |
| `agent_metrics.csv` | `_write_evaluation_artifacts` | One summary row per specialist task and two final-judge tasks. |
| `suicidal_intent_metrics.csv` | `_write_evaluation_artifacts` | False/true per-class metrics. |
| `metrics.md` | `ArtifactWriter.write_metrics_md` | Main metric and per-agent Markdown summary. |
| `compiled_results.csv` | `compile_run_results` | Long-form machine-readable report. |
| `compiled_results.md` | `compile_run_results` | Review-oriented complete Markdown report. |
| `misclassified_ids.txt` | `_write_evaluation_artifacts` | IDs whose final label is wrong. |
| `suicidal_intent_misclassified_ids.txt` | `_write_evaluation_artifacts` | IDs whose boolean output is wrong. |
| `classification_metrics.svg` | `ArtifactWriter.write_visualizations` | Label and binary accuracy/precision/recall/F1 bars. |
| `runtime_metrics.svg` | `ArtifactWriter.write_visualizations` | Calls, latency, tokens, cost, failures, and parse errors. |
| `llm_calls.jsonl` | `run_workflow_evaluation` | Every request/retry with role, model, usage, cost, timing, raw output, parse result, and error. |
| `traces.jsonl` | `run_workflow_evaluation` | Specialist/judge node events and durations. |
| `errors.jsonl` | `run_workflow_evaluation` | Final unrecovered case errors. |
| `trace_summary.json` | `_write_evaluation_artifacts` | Compact status, runtime, guardrail, and trace-file summary. |
