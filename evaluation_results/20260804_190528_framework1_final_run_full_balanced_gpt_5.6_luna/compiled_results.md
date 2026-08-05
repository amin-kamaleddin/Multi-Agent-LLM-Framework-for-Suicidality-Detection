# Compiled Evaluation Results

- Run directory: `logs\runs\20260804_190528_framework1_final_run_full_balanced_gpt_5.6_luna`
- Workflow: `framework_1_parallel_no_evidence`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `bool`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `60387014`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 28792 |
| Total records | 28792 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 172761 |
| Failed LLM attempts | 9 |
| JSON/schema parsing errors | 4 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-04T23:05:28.757849+00:00 | 2026-08-05T03:17:20.532690+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 224 | 1432 | 250 | 83 | 2561 | 0 |
| Ideation | 135 | 5722 | 642 | 273 | 250 | 0 |
| Behavior | 8 | 1015 | 883 | 122 | 11 | 0 |
| Attempt | 2 | 72 | 169 | 541 | 1 | 0 |
| Unsupportive | 53 | 127 | 25 | 7 | 14184 | 0 |

## Positive-Label Subclassification Metrics

- Scope: records whose ground-truth and predicted labels are both Indicator, Ideation, Behavior, or Attempt.
- Precision, recall, and F1 are macro averages across the four positive labels.
- Correct: included records on the diagonal of the 4x4 matrix.
- Support: all records represented in the 4x4 matrix.

Formulas for each label `L`:
- `TP(L) = actual L and predicted L`.
- `FP(L) = actual another positive label and predicted L`.
- `FN(L) = actual L and predicted another positive label`.
- `Precision(L) = TP(L) / (TP(L) + FP(L))`.
- `Recall(L) = TP(L) / (TP(L) + FN(L))`.
- `F1(L) = 2 * Precision(L) * Recall(L) / (Precision(L) + Recall(L))`.
- `Overall accuracy = correct / support`.
- `Macro metric = sum of the four per-label metric values / 4`.

| Accuracy | F1 Score | Precision | Recall | Support | Correct | Averaging |
|---|---|---|---|---|---|---|
| 0.636827 | 0.499247 | 0.571628 | 0.520756 | 11573 | 7370 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 224 | 1432 | 250 | 83 |
| Ideation | 135 | 5722 | 642 | 273 |
| Behavior | 8 | 1015 | 883 | 122 |
| Attempt | 2 | 72 | 169 | 541 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| Indicator | 224 | 145 | 1765 | 1989 | 0.607046 | 0.112619 | 0.189992 |
| Ideation | 5722 | 2519 | 1050 | 6772 | 0.694333 | 0.844950 | 0.762273 |
| Behavior | 883 | 1061 | 1145 | 2028 | 0.454218 | 0.435404 | 0.444612 |
| Attempt | 541 | 478 | 243 | 784 | 0.530913 | 0.690051 | 0.600111 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 11573 |
| TN | 14184 |
| FP | 212 |
| FN | 2823 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.894589 | 0.884076 | 0.982011 | 0.803904 | 14396 | 14396 | 11573 | 212 | 14184 | 2823 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 14184 | 212 | 0 |
| true | 2823 | 11573 | 0 |

## Label Distributions

### Ground Truth

| Label | Count |
|---|---:|
| Attempt | 785 |
| Behavior | 2039 |
| Ideation | 7022 |
| Indicator | 4550 |
| Unsupportive | 14396 |

### Predictions

| Predicted Label | Count |
|---|---:|
| Attempt | 1026 |
| Behavior | 1969 |
| Ideation | 8368 |
| Indicator | 422 |
| Unsupportive | 17007 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 136610.571418 |
| Real execution duration seconds | 15094.755306 |
| Total latency seconds | 301548.276120 |
| Average latency seconds per request | 1.745465 |
| Min latency seconds | 0.265482 |
| Max latency seconds | 7200.128454 |
| Total input tokens | 55278990 |
| Total output tokens | 5108024 |
| Total tokens | 60387014 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 1090256 |
| Average input tokens per request | 319.973779 |
| Average output tokens per request | 29.566997 |
| Average total tokens per request | 349.540776 |
| Total cost USD | 8.650027 |
| Average cost USD per request | 0.000050 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 16.768568 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| APIConnectionError | 4 |
| JSONDecodeError | 4 |
| ValueError | 1 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 6207 | Attempt_specialist | 1 | JSONDecodeError |
| 6212 | Ideation_specialist | 1 | JSONDecodeError |
| 6653 | Attempt_specialist | 1 | JSONDecodeError |
| 7733 | judge | 1 | JSONDecodeError |
| 8736 | Unsupportive_specialist | 1 | APIConnectionError |
| 10688 | Unsupportive_specialist | 1 | ValueError |
| 11403 | Behavior_specialist | 1 | APIConnectionError |
| 11910 | Indicator_specialist | 1 | APIConnectionError |
| 12109 | Behavior_specialist | 1 | APIConnectionError |

## Trace Files

| Artifact | Purpose |
|---|---|
| `predictions.csv` | Per-record source fields, shared outputs, correctness, errors, and optional workflow-supplied agent outputs. |
| `suicidal_intent_metrics.csv` | One positive-oriented binary summary with TP, FP, TN, and FN. |
| `positive_subclassification_metrics.csv` | Aggregate accuracy and macro precision, recall, and F1 across the four positive labels. |
| `metrics.json` | Complete metrics payload. |
| `results.json` | Complete per-record outputs and metrics payload. |
| `misclassified_ids.txt` | IDs with an incorrect final-label evaluation. |
| `suicidal_intent_misclassified_ids.txt` | IDs with an incorrect binary evaluation. |
| `suicidal_intent_metrics.svg` | Accuracy, precision, recall, and F1 for the suicidal-intent grouping. |
| `runtime_metrics.svg` | Cost, token, latency, call, failure, and parse-error visualization. |
| `llm_calls.jsonl` | Every LLM request attempt and retry. |
| `traces.jsonl` | Workflow-supplied execution events. |
| `trace_summary.json` | Runtime, token, and failure summary. |
